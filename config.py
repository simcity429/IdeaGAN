import torch

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
NOISE_DIM = 10
IMG_SIZE = 32
IMG_CH = 1
CLASS_NUM = 10
BATCH_SIZE = 64
EPOCH = 100
#IdeaMaker hyperparameter
IM_HIDDEN_UNIT_NUM = NOISE_DIM
IDEAMAKER_LR = 1e-3
#Encoder hyperparameter
NGF = 64
ENCODER_LR = 5e-4
RECON_COEF = 1
#Decoder hyperparameter
DECODER_LR = 5e-4
NDF = 64
#Little_D hyperparameter
LITTLE_D_LR = 5e-5
CLASSIFIER_COEF = 0
KL_COEF = 0.5
COV_COEF = 1
#Big_D hyperparameter
BIG_D_LR = 5e-5
BIG_D_UPDATE_NUM = 5