import torch

d_model = 512
n_heads = 8
n_layers = 6
d_k = 64
d_v = 64
d_ff = 2048
dropout = 0.1
padding_idx = 0
bos_idx = 2
eos_idx = 3
src_vocab_size = 8000
tgt_vocab_size = 8000
batch_size = 16
epoch_num = 80
lr = 3e-4
warmup = 10000
max_len = 60
beam_size = 3
early_stop = 8
bleu_gap = 2
use_smoothing = False
use_noamopt = True

i_exp = 26
data_dir = './data'
train_data_path = './data/json/training.json'
dev_data_path = './data/json/validation.json'
test_data_path = './data/json/testing.json'
model_path = f'./exp{i_exp}/model.pth'
log_path = f'./exp{i_exp}/train.log'
output_path = f'./exp{i_exp}/output.txt'
cfg_path = f'./exp{i_exp}/config.json'

gpu_id = '0'
device = torch.device(f"cuda:{gpu_id}")

