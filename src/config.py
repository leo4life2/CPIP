import torch

# ==========================
# Training Setup
# ==========================
batch_size = 8
num_workers = 0
epochs = 150
warmup_epochs = 5
resume_training = True
vpr_validation_epochs = 3 # run VPR pipeline after this number of epochs

# ==========================
# Model Parameters
# ==========================
cpip_checkpoint_path = "/scratch/zl3493/CPIP/cpip_0.3_04160259.pt"
image_encoder_model_name = "resnet50"
mixvpr_checkpoint_path = "/scratch/zl3493/CPIP/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt"

# Image Embedding Configuration
# image_embedding = 1024 # for vit_large_patch14_dinov2
location_embedding_dim = 3
contrastive_dimension = 256 # The embedding dimension that contrastive learning is done in
image_projection_blocks = 1
location_projection_blocks = 1
projection_dropout = 0.1

# Pretrained & Trainability Settings
pretrained = True
image_encoder_trainable = False
trainable = True

# Temperature for Softmax
temperature = 1.0

# ==========================
# Image Size Configuration
# ==========================
# unav images are 1440 x 810
#img_width = 518 # for vit_large_patch14_dinov2
#img_height = 518 # for vit_large_patch14_dinov2
img_width = 960
img_height = 540

target_img_width = 320 # for mixvpr pretrained weights
target_img_height = 320 # same as above

# ==========================
# Image Encoder Configuration
# ==========================
encoder_output_height = round(target_img_width / 16)
encoder_output_width = round(target_img_width / 16)
image_embedding_dim = encoder_output_height * encoder_output_width
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
top_k = 1
# top_k = 5

# Hyperparameters for grid generation
grid_extent = 4     # number of points to generate in each direction (left, right, up, down)
grid_side_length_percentage = 0.1  # Percentage of average closest distance to use for grid side length

num_rotation_steps = 6  # Number of rotations to cover 360 degrees for synthesizing position descriptors
# Maximum degree of rotation
max_rotation_degrees = 18 # e.g. for yaw datasets, this might be 18, for ICT datasets, this would be 360
