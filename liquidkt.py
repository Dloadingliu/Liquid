import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Embedding, Dropout
from .que_base_model import QueBaseModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GCNConv(nn.Module):
    def __init__(self, in_dim, out_dim, p):
        super(GCNConv, self).__init__()
        self.w = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.w.weight)
        self.dropout = nn.Dropout(p=p)

    def forward(self, x, adj_norm):
        x = torch.sparse.mm(adj_norm, x)
        x = self.w(x)
        x = self.dropout(x)
        return x


class GraphQueEmbed(nn.Module):
    def __init__(self, num_q, num_c, d, p, edge_index, num_layers=2):
        super(GraphQueEmbed, self).__init__()
        self.num_layers = num_layers
        self.num_q = num_q
        self.num_c = num_c

        self.q_embed = nn.Embedding(num_q, d)
        self.c_embed = nn.Embedding(num_c, d)
        nn.init.xavier_uniform_(self.q_embed.weight)
        nn.init.xavier_uniform_(self.c_embed.weight)

        self.gcn_layers = nn.ModuleList([
            GCNConv(d, d, p) for _ in range(num_layers)
        ])

        self.layer_weights = nn.Parameter(torch.ones(num_layers + 1) / (num_layers + 1))

        total_nodes = num_q + num_c
        current_device = edge_index.device if edge_index is not None else torch.device('cpu')

        self_loops = torch.arange(total_nodes, device=current_device)
        self_loop_edges = torch.stack([self_loops, self_loops], dim=0)

        if edge_index is not None:
            mask = (edge_index[0] < total_nodes) & (edge_index[1] < total_nodes)
            safe_edge_index = edge_index[:, mask]
            safe_edge_index = torch.cat([safe_edge_index, self_loop_edges], dim=1)
        else:
            safe_edge_index = self_loop_edges

        values = torch.ones(safe_edge_index.size(1), device=current_device)
        A = torch.sparse_coo_tensor(safe_edge_index, values, size=(total_nodes, total_nodes)).coalesce()

        indices = A.indices()
        A_values = A.values()
        degree = torch.sparse.sum(A, dim=1).to_dense()
        d_inv_sqrt = torch.pow(degree, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0

        row, col = indices
        norm_values = A_values * d_inv_sqrt[row] * d_inv_sqrt[col]
        adj_norm = torch.sparse_coo_tensor(indices, norm_values, size=(total_nodes, total_nodes)).coalesce()

        self.register_buffer('adj_norm', adj_norm)
        self.register_buffer('edge_index_clean', safe_edge_index)

    def forward(self, q_diff_vec):
        all_q = self.q_embed.weight
        all_c = self.c_embed.weight

        if q_diff_vec is not None:
            all_q = all_q + q_diff_vec

        x_all = torch.cat([all_q, all_c], dim=0)
        embs = [x_all]

        for layer in self.gcn_layers:
            gcn_out = layer(x_all, self.adj_norm)
            gcn_activated = F.relu(gcn_out)
            x_all = x_all + gcn_activated
            embs.append(x_all)

        emb_stack = torch.stack(embs, dim=0)
        weights = F.softmax(self.layer_weights, dim=0)
        x_final = torch.sum(emb_stack * weights.view(-1, 1, 1), dim=0)

        src, dst = self.edge_index_clean[0], self.edge_index_clean[1]
        emb_src = F.normalize(x_final[src], p=2, dim=-1)
        emb_dst = F.normalize(x_final[dst], p=2, dim=-1)
        loss_graph = (emb_src - emb_dst).pow(2).sum(dim=1).mean()

        final_q_embeds = x_final[:self.num_q, :]
        return final_q_embeds, loss_graph


class GatedInputEnhancement(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid()
        )

    def forward(self, q_emb, r_emb, time_vec):
        gate = self.gate_net(torch.cat([q_emb, r_emb], dim=-1))
        interaction = gate * (q_emb * r_emb)
        return q_emb + r_emb + time_vec + interaction


class LiquidCell(nn.Module):

    def __init__(self, hidden_size, input_size, dropout_p=0.1):
        super().__init__()
        self.hidden_size = hidden_size

        self.W_inf = nn.Linear(2 * input_size, hidden_size)
        self.W_h = nn.Linear(hidden_size, hidden_size)
        self.W_x = nn.Linear(input_size, hidden_size)
        self.W_decay = nn.Linear(hidden_size + input_size, hidden_size)
        self.W_up = nn.Linear(hidden_size + input_size, hidden_size)
        self.W_gate = nn.Linear(hidden_size + input_size, hidden_size)
        self.ln_h = nn.LayerNorm(hidden_size)

        self.dropout = nn.Dropout(p=dropout_p)
        self.activation = torch.tanh

        nn.init.xavier_uniform_(self.W_decay.weight, gain=0.5)
        nn.init.xavier_uniform_(self.W_up.weight, gain=0.5)

    def forward(self, h, x, dt_per_dim, q_emb, r_emb):
        h_ln = self.ln_h(h)
        h_inf = torch.tanh(self.W_inf(torch.cat([q_emb, r_emb], dim=-1)))

        hx_decay_in = torch.cat([h_ln, q_emb], dim=-1)
        tau_decay = F.softplus(self.W_decay(hx_decay_in)) + 0.1

        decay_factor = torch.exp(-dt_per_dim / tau_decay)
        h_decayed = h_inf + (h - h_inf) * decay_factor

        hx_now = torch.cat([h_decayed, x], dim=-1)
        tau_up = F.softplus(self.W_up(hx_now)) + 0.1
        gate = torch.sigmoid(self.W_gate(hx_now))
        signal = self.activation(self.W_h(h_decayed) + self.W_x(x))
        target = gate * signal

        h_next = (h_decayed * tau_up + target) / (1.0 + tau_up)
        return self.dropout(h_next)


class GLUOutputHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.fc_value = nn.Linear(input_dim, hidden_dim)
        self.fc_gate = nn.Linear(input_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        value = self.fc_value(x)
        gate = torch.sigmoid(self.fc_gate(x))
        h = value * gate
        h = self.ln(h)
        h = self.dropout(h)
        return self.fc_out(h).squeeze(-1)


class LiquidKTNet(nn.Module):
    def __init__(self, num_q, num_c, emb_size, dropout=0.1, dropout1=0.4, emb_type='qid',
                 dpath="", device='cpu'):
        super().__init__()
        self.model_name = "liquidkt"
        self.num_q = num_q
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.device = device

        dname = dpath.split("/")[-1]
        graph_path = os.path.join(dpath, f"graph_{dname}.pt")
        try:
            self.matrix = torch.load(graph_path).to(device)
        except:
            self.matrix = None

        self.ans_embed = Embedding(2, self.emb_size)

        self.que_embed_layer = GraphQueEmbed(
            num_q, num_c, self.emb_size, dropout1, self.matrix, num_layers=2
        ).to(device)

        self.diff_encoder = nn.Linear(1, self.emb_size)
        self.register_buffer('q_total', torch.zeros(num_q))
        self.register_buffer('q_error', torch.zeros(num_q))

        self.liquid_cell = LiquidCell(self.hidden_size, self.emb_size)

        self.log_time_scale = nn.Parameter(torch.zeros(self.emb_size))

        self.time_encoder = nn.Sequential(
            nn.Linear(1, self.emb_size),
            nn.ReLU(),
            nn.Linear(self.emb_size, self.emb_size)
        )
        self.time_norm = nn.LayerNorm(self.emb_size)

        self.input_enhance = GatedInputEnhancement(self.emb_size)

        self.film_gamma = nn.Linear(self.hidden_size, self.hidden_size)
        self.film_beta = nn.Linear(self.hidden_size, self.hidden_size)

        self.interaction_gate = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Sigmoid()
        )

        self.out_layer = GLUOutputHead(3 * self.hidden_size, self.hidden_size, dropout)

    def get_dynamic_diff(self, q_indices, update=False, responses=None):
        if update and responses is not None:
            with torch.no_grad():
                q_flat = q_indices.view(-1)
                r_flat = responses.view(-1)
                mask = q_flat >= 0
                self.q_total.index_add_(0, q_flat[mask], torch.ones_like(q_flat[mask], dtype=torch.float))
                self.q_error.index_add_(0, q_flat[mask], 1.0 - r_flat[mask].float())

        diff = self.q_error / (self.q_total + 1e-7)
        global_mean = diff[self.q_total > 0].mean() if (self.q_total > 0).any() else torch.tensor(0.5, device=self.device)
        diff[self.q_total == 0] = global_mean

        return self.diff_encoder(diff.unsqueeze(-1))

    def forward(self, q, r, qshft, t, sm):
        q_diff_vec = self.get_dynamic_diff(q, update=self.training, responses=r)
        pro_embed, loss_graph = self.que_embed_layer(q_diff_vec)

        q_emb = F.embedding(q, pro_embed)
        qshft_emb = F.embedding(qshft, pro_embed)
        r_emb = self.ans_embed(r)

        t_sec = t.float() / 1000.0
        dt_raw = torch.zeros_like(t_sec)
        dt_raw[:, 1:] = t_sec[:, 1:] - t_sec[:, :-1]
        dt_raw = torch.clamp(dt_raw, min=0.0)

        dt_log = torch.log1p(dt_raw)
        time_scale = torch.exp(self.log_time_scale)
        dt_per_dim = dt_log.unsqueeze(-1) / (time_scale + 1e-8)

        time_vec = self.time_encoder(dt_log.unsqueeze(-1))
        time_vec = self.time_norm(time_vec)


        x = self.input_enhance(q_emb, r_emb, time_vec)


        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, self.hidden_size, device=self.device)
        hs = []
        for step in range(seq_len):
            h = self.liquid_cell(
                h, x[:, step, :], dt_per_dim[:, step, :],
                q_emb[:, step, :], r_emb[:, step, :]
            )
            hs.append(h.unsqueeze(1))
        h_seq = torch.cat(hs, dim=1)

        gamma = self.film_gamma(h_seq)
        beta = self.film_beta(h_seq)
        q_conditioned = qshft_emb * (1.0 + gamma) + beta

        interaction_raw = h_seq * qshft_emb
        gate_input = torch.cat([h_seq, qshft_emb], dim=-1)
        gate_weight = self.interaction_gate(gate_input)
        interaction = gate_weight * interaction_raw

        out = torch.cat([h_seq, q_conditioned, interaction], dim=-1)
        logits = self.out_layer(out)

        return logits, loss_graph


class LiquidKT(QueBaseModel):
    def __init__(self, num_q, num_c, emb_size, dropout=0.1, dropout1=0.4, emb_type='qid',
                 dpath="", emb_path="", pretrain_dim=768, device='cpu', seed=0,
                 other_config={}, **kwargs):

        super().__init__(model_name="liquidkt", emb_type=emb_type,
                         emb_path=emb_path, pretrain_dim=pretrain_dim,
                         device=device, seed=seed)

        self.model = LiquidKTNet(num_q=num_q, num_c=num_c, emb_size=emb_size,
                                 dropout=dropout, dropout1=dropout1,
                                 emb_type=emb_type, dpath=dpath,
                                 device=device)
        self.model.to(device)
        self.loss_func = nn.BCELoss()

        self.global_step = 0
        self.alpha_init = 0.01
        self.decay_steps = 50000.0

    def get_loss(self, logits, rshft, sm):
        y = torch.masked_select(logits, sm)
        t = torch.masked_select(rshft, sm)
        loss = F.binary_cross_entropy_with_logits(y.double(), t.double())
        return loss

    def train_one_step(self, data, process=True, **kwargs):
        outputs, data_new, loss_graph = self.predict_one_step(
            data, return_details=True, process=process
        )
        loss_pred = self.get_loss(outputs, data_new['rshft'], data_new['sm'])

        with torch.no_grad():
            progress = min(1.0, self.global_step / self.decay_steps)
            alpha = self.alpha_init * 0.5 * (1.0 + math.cos(math.pi * progress))
            self.global_step += 1

        total_loss = loss_pred + alpha * loss_graph
        return outputs, total_loss

    def predict_one_step(self, data, return_details=False, process=True, **kwargs):
        data_new = self.batch_to_device(data, process=process)
        logits, loss_g = self.model(
            data_new['q'].long(),
            data_new['r'].long(),
            data_new['qshft'].long(),
            data_new['t'].long(),
            data_new['sm']
        )
        if return_details:
            return logits, data_new, loss_g
        else:
            return torch.sigmoid(logits)