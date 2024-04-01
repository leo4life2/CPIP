import torch

debug = True
data_path = "/scratch/jh7956/CPIP/000"
batch_size = 8
num_workers = 0
lr = 1e-3
weight_decay = 1e-3
patience = 2
factor = 0.5
epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'MixVPR'
#image_embedding = 1024 # for vit_large_patch14_dinov2
image_embedding = 4096
location_embedding = 3

pretrained = True
trainable = True
temperature = 1.0

# image size
# unav images are 1440 x 810
#img_width = 518 # for vit_large_patch14_dinov2
#img_height = 518 # for vit_large_patch14_dinov2
img_width = 1440 # for vit_large_patch14_dinov2
img_height = 810 # for vit_large_patch14_dinov2

# for projection head; used for both image and text encoders
num_projection_layers = 1
projection_dim = 256 
dropout = 0.1