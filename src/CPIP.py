import torch
from torch import nn
import torch.nn.functional as F

import config as CFG
from modules import ImageEncoder, ProjectionHead, ImageProjectionHead

import pdb


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

    def forward(self, batch):
        # Getting Image Features
        # batch[image] shape:  (batch_size, 3, target_img_width, target_img_height) = [8, 3, 320, 320] / [batchsize, channels, target_img_width, target_img_height]
        image_features = self.image_encoder(batch["image"]) # shape: (batch_size, channels, encoder_output_height, encoder_output_width) = [8, 1024, 20, 20]
        batch_size, channels, encoder_output_height, encoder_output_width = image_features.shape
        image_features = image_features.view(batch_size, channels, encoder_output_height * encoder_output_width)
        location_features = batch["location"] # shape:  (batch_size, 3) = [8, 3]

        # Getting Image and Location Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features) # shape: (batch_size, channels, contrastive_dimension) = [8, 1024, 256]
        location_embeddings = self.location_projection(location_features) # shape: (batch_size, contrastive_dimension) = [8, 256]

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