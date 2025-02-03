import torch

checkpoint = torch.load('/data/lsd_nm_experiments/02_train/3M-APP-SCN/mtlsd_04/model_checkpoint_200000')
print(checkpoint['model_state_dict'].keys())