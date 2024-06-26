import torch
from torch import nn
import torch.nn.functional as F

import config as CFG
from modules import ImageEncoder, ProjectionHead, ImageProjectionHead, ShallowConvNet

class CPIPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.image_projection = ImageProjectionHead(
            embedding_dim=CFG.image_embedding_dim,
            projection_dim=CFG.contrastive_dimension,
            dropout=CFG.projection_dropout,
            num_blocks=CFG.image_projection_blocks,
        )
        self.location_projection = ProjectionHead(
            embedding_dim=CFG.location_embedding_dim,
            projection_dim=CFG.contrastive_dimension,
            dropout=CFG.projection_dropout,
            num_blocks=CFG.location_projection_blocks,
        )
        self.cnn = ShallowConvNet(input_c = 1, 
                                  input_h = int(CFG.contrastive_dimension ** 0.5),
                                  input_w = int(CFG.contrastive_dimension ** 0.5),
                                  output_c = 1024, output_h = 20, output_w = 20)
        self.location_projection_2 = ProjectionHead(
            embedding_dim=1024*20*20,
            projection_dim=CFG.contrastive_dimension,
            dropout=CFG.projection_dropout,
            num_blocks=CFG.location_projection_blocks,
        )

    def forward(self, batch):
        # Getting Image Features
        # batch[image] shape:  (batch_size, 3, target_img_width, target_img_height) = [8, 3, 320, 320] / [batchsize, channels, target_img_width, target_img_height]
        image_features = self.image_encoder(batch["image"]) # shape: (batch_size, channels, encoder_output_height, encoder_output_width) = [8, 1024, 20, 20]
        batch_size, channels, encoder_output_height, encoder_output_width = image_features.shape
        image_features = image_features.view(batch_size, channels, encoder_output_height * encoder_output_width)
        # Getting Image and Location Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features) # shape: (batch_size, channels, contrastive_dimension) = [8, 1024, 256]

        # =========== Location branch ===========
        location_info = batch["location"] # shape:  (batch_size, 3) = [8, 3]
        location_embeddings = self.location_projection(location_info) # shape: (batch_size, contrastive_dimension) = [8, 256]

        # reshape location embeddings to: (batch_size, sqrt(contrastive_dimension), sqrt(contrastive_dimension)) = [8, 16, 16]
        location_embeddings = location_embeddings.view(batch_size, int(CFG.contrastive_dimension ** 0.5), int(CFG.contrastive_dimension ** 0.5))
        # reshape to [8, 1, 16, 16]
        location_embeddings = location_embeddings.unsqueeze(1)
        location_descriptors = self.cnn(location_embeddings) #[8, 1024, 20, 20]
        
        # flatten the location_descriptors to [8, 1024*20*20]
        location_descriptors = location_descriptors.view(batch_size, -1)

        # use dot product (a matrxi) to scale back to (batch_size, contrastive_dimension) = [8, 256]
        location_embeddings = self.location_projection_2(location_descriptors) 
        # ==================================================

        # Normalize embeddings
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        location_embeddings = F.normalize(location_embeddings, p=2, dim=-1)

        image_embeddings = image_embeddings.permute(1, 0, 2) # shape: (channels, batch_size, contrastive_dimension) = [1024, 8, 256]
        location_embeddings = location_embeddings.T.unsqueeze(0) #[1, 256, 8]
        location_embeddings = location_embeddings.expand(image_embeddings.size(0), -1, -1) # [1024, 256, 8]

        # do the dot product in batch
        results = torch.bmm(image_embeddings, location_embeddings) 
        logits = results.mean(dim=0) / CFG.temperature # (batch_size * batch_size) = [8, 8]

        # Calculating the Loss
        labels = torch.arange(logits.size(0)).long().to(logits.device)

        img_to_text_loss = F.cross_entropy(logits, labels)
        text_to_img_loss = F.cross_entropy(logits.T, labels)

        loss = (img_to_text_loss + text_to_img_loss) / 2

        return loss, logits