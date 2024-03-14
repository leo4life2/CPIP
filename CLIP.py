import torch
from torch import nn
import torch.nn.functional as F

import config as CFG
from modules import ImageEncoder, ProjectionHead

class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=CFG.image_embedding,
        location_embedding=CFG.location_embedding,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.location_projection = ProjectionHead(embedding_dim=location_embedding)
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image Features
        image_features = self.image_encoder(batch["image"])
        location_features = batch["location"]
        # Getting Image and Location Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        location_embeddings = self.location_projection(location_features)

        # Normalize embeddings
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        location_embeddings = F.normalize(location_embeddings, p=2, dim=-1)
        
        # Calculating the Loss
        logits = image_embeddings @ location_embeddings.T / self.temperature
        labels = torch.arange(logits.size(0)).long().to(logits.device)
        
        img_to_text_loss = F.cross_entropy(logits, labels)
        text_to_img_loss = F.cross_entropy(logits.T, labels)

        loss = (img_to_text_loss + text_to_img_loss) / 2

        return loss

if __name__ == '__main__':
    images = torch.randn(8, 3, 224, 224)
    locations = torch.randn(8, 3) # locations are 3d vectors
    batch = {
        'image': images,
        'location': locations,
    }

    CLIP = CLIPModel()
    loss = CLIP(batch)
    print(loss)
