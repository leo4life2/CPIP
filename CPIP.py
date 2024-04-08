import torch
from torch import nn
import torch.nn.functional as F

import config as CFG
from modules import ImageEncoder, ProjectionHead

import pdb


class CPIPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.image_projection = ProjectionHead(
            embedding_dim=CFG.image_embedding,
            projection_dim=CFG.contrastive_dimension,
            dropout=CFG.projection_dropout,
            num_blocks=CFG.image_projection_blocks,
        )
        self.location_projection = ProjectionHead(
            embedding_dim=CFG.location_embedding,
            projection_dim=CFG.contrastive_dimension,
            dropout=CFG.projection_dropout,
            num_blocks=CFG.location_projection_blocks,
        )

    def forward(self, batch):
        # Getting Image Features
        # shape: torch.Size([8, 3, 810, 1440])
        # device(type='cuda', index=0)
        image_features = self.image_encoder(batch["image"])
        location_features = batch["location"]

        # Getting Image and Location Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        location_embeddings = self.location_projection(location_features)

        # Normalize embeddings
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        location_embeddings = F.normalize(location_embeddings, p=2, dim=-1)

        # Calculating the Loss
        logits = image_embeddings @ location_embeddings.T / CFG.temperature
        labels = torch.arange(logits.size(0)).long().to(logits.device)

        img_to_text_loss = F.cross_entropy(logits, labels)
        text_to_img_loss = F.cross_entropy(logits.T, labels)

        loss = (img_to_text_loss + text_to_img_loss) / 2

        return loss, logits
