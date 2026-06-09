# config for training on WikiText-103
# dataset: ~100M tokens, good for benchmarking attention variants

out_dir = 'out-wikitext103'
eval_interval = 500
eval_iters = 200
log_interval = 10

dataset = 'wikitext103'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 2048

# model: ~25M params, small enough for quick experiments
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.0
bias = False

# training
max_iters = 3000
lr_decay_iters = 10000
learning_rate = 6e-4
min_lr = 6e-5
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# GQA config:
attn_type = 'gqa'
n_kv_head = 2

warmup_iters = 100
always_save_checkpoint = False
wandb_log = False
