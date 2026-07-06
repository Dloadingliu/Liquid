The above contains the model code for LiquidKT. We leverage the pyKT benchmark to conduct model training and prediction, and adopt Weights & Biases (WandB) for optimal hyperparameter search. To reproduce the experimental results, register LiquidKT into the benchmark and follow the dataset preprocessing, training, and inference pipelines defined by the pyKT benchmark.

Datasets: Assistments2012, Assistments2017, XES3G5M

Install Python dependencies strictly in accordance with the officially recommended versions and dependency specifications of pyKT.

#### Training command
```python
python wandb_liquidkt_train.py  --dataset_name assist2017 --use_wandb 1 --dropout 0.1 --dropout1 0.3 --learning_rate 0.005 --seed 3407 --emb_size 256 --emb_type qid
```
