from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()

_C.MODEL = CN()
# Dùng cuda hay cpu để train
_C.MODEL.DEVICE = 'cuda'
# ID của GPU (nếu có 1 GPU thì mặc định là 0, nhiều GPU thì có thể là 1, 2, 3)
_C.MODEL.DEVICE_ID = '0'
# Backbone dùng để train
_C.MODEL.NAME = 'resnet50'
# Last stride của backbone (Thường để là 1 để giữ nguyên feature vector)
_C.MODEL.LAST_STRIDE = 1
# Đường dẫn đến pretrained model của backbone
_C.MODEL.PRETRAIN_PATH = ''
_C.MODEL.DIST_TRAIN = False
# Sử dụng ImageNet pretrained model để khởi tạo weight cho backbone hoặc sử dụng self trained model
# để khởi tạo toàn bộ model. Trong cuộc thi chỉ được sử dụng pretrain trên COCO hoặc Imagenet
# Options: 'imagenet' or 'self'
_C.MODEL.PRETRAIN_CHOICE = 'imagenet'
# Nếu train với Batch Normalization Neck, options: 'bnneck' or 'no'
_C.MODEL.NECK = 'bnneck'
# Train với center loss, options: 'yes' or 'no'. Loss với Center Loss sẽ có optimizer config khác nhau
_C.MODEL.IF_WITH_CENTER = 'no'
_C.MODEL.ID_LOSS_TYPE = 'softmax'
_C.MODEL.ID_LOSS_WEIGHT = 1.0
_C.MODEL.TRIPLET_LOSS_WEIGHT = 1.0
# Loss type
# Options: ['triplet'] (không có center loss) or ['center', 'triplet_center'] (có center loss)
_C.MODEL.METRIC_LOSS_TYPE = 'triplet'
_C.MODEL.TRANSFORMER_TYPE = 'None'
_C.MODEL.STRIDE_SIZE = [32, 32]
# For example, if loss type is cross entropy loss + triplet loss + center loss
# the setting should be: _C.MODEL.METRIC_LOSS_TYPE = 'triplet_center' and _C.MODEL.IF_WITH_CENTER = 'yes'

# Nếu train với soft triplet loss, options: 'True', 'False'
_C.MODEL.NO_MARGIN = False
# Train với label smooth (ngăn ngừa overfitting)
_C.MODEL.IF_LABELSMOOTH = 'on'

# If train with arcface loss, options: 'True', 'False'
_C.MODEL.COS_LAYER = False
# Frozen layers of backbone
_C.MODEL.FROZEN = -1

_C.MODEL.POOLING_METHOD = 'avg'

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size của ảnh trong quá trình training. Vì là người nên bbox sẽ có chiều cao > chiều rộng
_C.INPUT.SIZE_TRAIN = [384, 128]
# Size của ảnh trong quá trình test
_C.INPUT.SIZE_TEST = [384, 128]
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Random probability for random erasing
_C.INPUT.RE_PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Value of padding size
_C.INPUT.PADDING = 10
_C.INPUT.RESIZECROP = False
_C.INPUT.COLORJIT_PROB = 0.0

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.NAMES = ('market1501')
# Root directory where datasets should be used (and downloaded if not found)
_C.DATASETS.ROOT_DIR = ('./data')
_C.DATASETS.PLUS_NUM_ID = 100
_C.DATASETS.QUERY_MINING = False

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
# Sampler for data loading
_C.DATALOADER.SAMPLER = 'softmax'
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 16

# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------
_C.SOLVER = CN()
# Optimizer được sử dụng
_C.SOLVER.OPTIMIZER_NAME = 'Adam'
# Số lượng epoch
_C.SOLVER.MAX_EPOCHS = 100
# Base learning rate
_C.SOLVER.BASE_LR = 3e-4
# Whether using larger learning rate for fc layer
_C.SOLVER.LARGE_FC_LR = False
# Factor learning bias
_C.SOLVER.BIAS_LR_FACTOR = 2
_C.SOLVER.SEED = 2112
# Momentum
_C.SOLVER.MOMENTUM = 0.9
# Margin của triplet loss
_C.SOLVER.MARGIN = 0.3
# Margin of cluster ;pss
_C.SOLVER.CLUSTER_MARGIN = 0.3
# Learning rate of SGD to learn the centers of center loss
_C.SOLVER.CENTER_LR = 0.5
# Balanced weight of center loss
_C.SOLVER.CENTER_LOSS_WEIGHT = 0.0005
# Settings of range loss
_C.SOLVER.RANGE_K = 2
_C.SOLVER.RANGE_MARGIN = 0.3
_C.SOLVER.RANGE_ALPHA = 0
_C.SOLVER.RANGE_BETA = 1
_C.SOLVER.RANGE_LOSS_WEIGHT = 1

# Settings of weight decay
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0005

# decay rate of learning rate
_C.SOLVER.GAMMA = 0.1
# decay step of learning rate
_C.SOLVER.STEPS = (40, 70)

# warm up factor
_C.SOLVER.WARMUP_FACTOR = 0.01
#  warm up epochs
_C.SOLVER.WARMUP_EPOCHS = 10
# method of warm up, option: 'constant','linear'
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.COSINE_MARGIN = 0.35
_C.SOLVER.COSINE_SCALE = 64

# epoch number of saving checkpoints
_C.SOLVER.CHECKPOINT_PERIOD = 10
# iteration of display training log
_C.SOLVER.LOG_PERIOD = 100
# epoch number of validation
_C.SOLVER.EVAL_PERIOD = 10
# Số lượng ảnh mỗi batch
# Nếu có 8 GPUs và IMS_PER_BATCH = 16, => mỗi GPU sẽ thực hiện 2 images/batch
_C.SOLVER.IMS_PER_BATCH = 64

# AUTOMATIC MIXED PRECISION
_C.SOLVER.FP16_ENABLED = False
# Freeze backbone
_C.SOLVER.FREEZE_EPOCH = -1
# -----------------------------------------------------------------------------
# TEST
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Số lượng ảnh mỗi batch trong quá trình test
_C.TEST.IMS_PER_BATCH = 128
# Nếu test với re-ranking, option: 'True' or 'False'
_C.TEST.RE_RANKING = False
# Nếu test với track-re-ranking, option: 'True' or 'False'
_C.TEST.RE_RANKING_TRACK = False

# Đường dẫn tới trained model
_C.TEST.WEIGHT = ""
# Feature nào của BNNeck sẽ được dùng để test, trước hay sau BNNeck
# Options: 'before' or 'after'
_C.TEST.NECK_FEAT = 'after'
# Feature được normalize trước test, nếu là 'yes' thì nó sẽ bằng cosine distance
_C.TEST.FEAT_NORM = 'yes'
# Save distance matrix sau khi test
_C.TEST.DIST_MAT = "dist_mat.npy"
# Whether using fliped feature for testing, option: 'on', 'off'
_C.TEST.FLIP_FEATS = 'off'
# Whether calculate the eval score option: 'True', 'False'
_C.TEST.EVAL = False
_C.TEST.FIC = False
_C.TEST.FAC = False
_C.TEST.RM_CAMERA = True
_C.TEST.CROP_TEST= True
_C.TEST.LA = 0.18

# stage2
_C.STAGE2 = CN()
_C.STAGE2.EPS= 0.55
_C.STAGE2.LA = 0.0005
_C.STAGE2.SAVE_CLUSTER_DIST = False

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = ""