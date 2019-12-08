import torch

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
NOISE_DIM = 100
IMG_SIZE = 32
IMG_CH = 1
CLASS_NUM = 10
BATCH_SIZE = 64
#IdeaMaker hyperparameter
IM_HIDDEN_UNIT_NUM = 100

#Encoder hyperparameter
NGF = 64
#Decoder hyperparameter
NDF = 64
#Big_D hyperparameter
BIG_D_LR = 5e-5