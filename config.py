import torch

# ==========================
# Basic Configuration
# ==========================
debug = True
data_path = "/scratch/zl3493/UNav-Dataset/810p/raw/000"

# ==========================
# Training Setup
# ==========================
batch_size = 10
num_workers = 0
epochs = 150
warmup_epochs = 5
resume_training = True

# ==========================
# Model Parameters
# ==========================
model_name = 'MixVPR'
# Image Embedding Configuration
# image_embedding = 1024 # for vit_large_patch14_dinov2
image_embedding = 4096 # for MixVPR
location_embedding = 3
contrastive_dimension = 256 # The embedding dimension that contrastive learning is done in
image_projection_blocks = 1
location_projection_blocks = 1
projection_dropout = 0.1

# Pretrained & Trainability Settings
pretrained = True
trainable = True

# Temperature for Softmax
temperature = 1.0

# ==========================
# Image Size Configuration
# ==========================
# unav images are 1440 x 810
#img_width = 518 # for vit_large_patch14_dinov2
#img_height = 518 # for vit_large_patch14_dinov2
img_width = 1440
img_height = 810

# ==========================
# Optimization Parameters
# ==========================
lr = 0.002
weight_decay = 0.001
patience = 1
factor = 0.2

# ==========================
# Hardware Configuration
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
