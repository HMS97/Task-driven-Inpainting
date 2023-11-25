from easydict import EasyDict as edict
import socket
from datetime import date
today = date.today()

__C     = edict()
cfg     = __C

# Common
# __C                               = edict()
__C.NUM_WORKER                    = 4                     # number of data workers
__C.EPOCH                    = 101
__C.LR                       = 2e-4
__C.BATCH_SIZE               =  4
__C.MOMENTUM                          = 0.5
__C.SHOW_INTERVAL                       = 200
__C.LR_POLICY = 'step' 
__C.STEP_DEC = 30
__C.DATASET_NAME = 'RSSCN7'
__C.UPLOAD = True
__C.SAVE_IMGS = True
__C.ISTRAIN  = True
__C.PARRALEL = True
__C.GPU_IDS = [0]

# __C.CLASSIFIER = 'alexnet'
# __C                              = edict()
__C.SIZE = 256
__C.MEAN = [0.5]*3
__C.STD = [0.5]*3


# __C.DIR                               = edict()
__C.SAVE_DIR = f'saved_models/class_weight_1_'+str(today)
# __C.SAVE_DIR = 'saved_models/fixed_RICE_'+str(today)
cfg.load_epoch = None
# SEG
__C.SEG_IN  =3
__C.SEG_OUT  =1
__C.NUM_DOWNS  =8



# GENERATOR
__C.GEN_IN1  =4
__C.GEN_OUT1 =3
__C.GEN_IN2  =4
__C.GEN_OUT2 =3

# DISCRIMINATOR 1
__C.D1_TYPE = 'basic'
__C.D1_IN = 6
__C.D1_NDF = 64

# DISCRIMINATOR 2 
__C.D2_TYPE = 'basic'
__C.D2_IN = 6
__C.D2_NDF = 64
__C.SEG_THRD = 0.5


# __C                               = edict()
__C.D1 = 0.5
__C.D2 = 0.5

# __C.COARSE_G_L1 = 100
# __C.G_D1 = 1
# __C.G_D2 = 5
# __C.PERCEPTUAL = 10
# __C.REFINE_G_L1 = 100
# __C.SSIM = 0
# __C.CLASS_WEIGHT = 5
# __C.D1_LAYERS = 3
# __C.D2_LAYERS = 3



__C.COARSE_G_L1 = 1
__C.G_D1 = 1
__C.G_D2 = 1
__C.PERCEPTUAL = 1
__C.REFINE_G_L1 = 1
__C.SSIM = 0
__C.CLASS_WEIGHT = 5
__C.D1_LAYERS = 3
__C.D2_LAYERS = 3