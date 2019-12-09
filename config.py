import torch

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
NOISE_DIM = 100
IMG_SIZE = 32
IMG_CH = 1
CLASS_NUM = 10
BATCH_SIZE = 64
EPOCH = 10
#IdeaMaker hyperparameter
IM_HIDDEN_UNIT_NUM = 100
IDEAMAKER_LR = 1e-3
#Encoder hyperparameter
NGF = 64
ENCODER_LR = 1e-3
#Decoder hyperparameter
DECODER_LR = 1e-3
NDF = 64
#Little_D hyperparameter
LITTLE_D_LR = 1e-3
CLASSIFIER_COEF = 0.2
#Big_D hyperparameter
BIG_D_LR = 5e-5
BIG_D_UPDATE_NUM = 5