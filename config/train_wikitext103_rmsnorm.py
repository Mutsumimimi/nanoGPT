# RMSNorm experiment

out_dir = 'out-wikitext103-rmsnorm'
eval_interval = 500
eval_iters = 200
log_interval = 10

dataset = 'wikitext103'
gradient_accumulation_steps = 1
batch_size = 32
block_size = 2048

# model
n_layer = 8
n_head = 12
n_embd = 768
dropout = 0.0
bias = False
norm_type = 'rmsnorm'

# training
max_iters = 500
lr_decay_iters = 1000
learning_rate = 6e-4
min_lr = 6e-5
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

warmup_iters = 100
always_save_checkpoint = False
wandb_log = False
