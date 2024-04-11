import torch
from torch import nn
import torch.nn.functional as F

import config as CFG
from modules import ImageEncoder, ProjectionHead, ImageProjectionHead, Pooling

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
        # shape: torch.Size([8, 3, 810, 1440])
        image_features = self.image_encoder(batch["image"])
        # output shape of image_features: torch.Size([8, 1024, 90, 51])

        image_features = image_features.view(CFG.batch_size, CFG.channel, -1)
        # image_features： torch.Size([8, 1024, 4590]) = [batch_size, channels, w*h]

        location_features = batch["location"]

        # Getting Image and Location Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        # image_embeddings: torch.Size([8, 1024, 256]) = batch_size x channel x contrastive_dimension
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

if __name__ == "__main__":
    model = CPIPModel().to(CFG.device)
    test_input = {
        "image": torch.randn(8, 3, 1440, 810).to(CFG.device),
        "location": torch.randn(8, 3, 1440, 810).to(CFG.device),
    }
    loss, logits = model(test_input)