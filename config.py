import torch

# ==========================
# Basic Configuration
# ==========================
#data_path = "/scratch/zl3493/UNav-Dataset/810p/raw/000"
data_path = "/scratch/jh7956/Datasets/000/database" # Junjie's datapath

# ==========================
# Training Setup
# ==========================
batch_size = 8
num_workers = 0
epochs = 150
warmup_epochs = 5
resume_training = True
do_train = False # Set to False if already have a trained model
process_data = True # Set to False if already have stored vector

# ==========================
# Model Parameters
# ==========================
model_name = 'resnet50'

# Image Embedding Configuration
# image_embedding = 1024 # for vit_large_patch14_dinov2
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
# Image Encoder Configuration
# ==========================
encoder_output_height = round(img_height / 16)
encoder_output_width = round(img_width / 16)
image_embedding = encoder_output_height * encoder_output_width
channel = 1024

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


# ==========================
# VPR Configuration
# ==========================
vpr_threshold = 0.5