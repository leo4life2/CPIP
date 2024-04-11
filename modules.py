from torch import nn
import timm
import config as CFG
import utils
from models import helper
import pdb
from typing import Union, Tuple
import torch

from pipeline import get_mixvpr_model


class MixVPRModel(nn.Module):
    def __init__(
        self,
        backbone_arch="resnet50",
        pretrained=True,
        layers_to_freeze=1,
        layers_to_crop=[],
        agg_arch="MixVPR",
        agg_config={},
        loss_name="MultiSimilarityLoss",
        miner_name="MultiSimilarityMiner",
        miner_margin=0.1
    ):
        super().__init__()
        self.encoder_arch = backbone_arch
        self.pretrained = pretrained
        self.layers_to_freeze = layers_to_freeze
        self.layers_to_crop = layers_to_crop

        self.agg_arch = agg_arch
        self.agg_config = agg_config

        self.loss_name = loss_name
        self.miner_name = miner_name
        self.miner_margin = miner_margin

        self.loss_fn = utils.get_loss(loss_name)
        self.miner = utils.get_miner(miner_name, miner_margin)

        self.backbone = helper.get_backbone(
            backbone_arch, pretrained, layers_to_freeze, layers_to_crop
        )
        self.aggregator = helper.get_aggregator(agg_arch, agg_config)

        self.backbone.to(CFG.device)
        self.aggregator.to(CFG.device)

        for p in self.backbone.parameters():
            p.requires_grad = CFG.image_encoder_trainable


    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregator(x)
        return x


class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self,
        model_name=CFG.image_encoder_model_name,
        pretrained=CFG.pretrained,
        trainable=CFG.image_encoder_trainable,
    ):
        super().__init__()
        if model_name == "MixVPR":
            self.model = MixVPRModel(
                #---- Encoder
                backbone_arch='resnet50',
                pretrained=True,
                layers_to_freeze=2,
                layers_to_crop=[4], # 4 crops the last resnet layer, 3 crops the 3rd, ...etc

                agg_arch='MixVPR',
                agg_config={'in_channels' : 1024,
                        'in_h' : 40,
                        'in_w' : 30,
                        'out_channels' : 1024,
                        'mix_depth' : 4,
                        'mlp_ratio' : 1,
                        'out_rows' : 4}, # the output dim will be (out_rows * out_channels)

                #----- Loss functions
                # example: ContrastiveLoss, TripletMarginLoss, MultiSimilarityLoss,
                # FastAPLoss, CircleLoss, SupConLoss,
                loss_name='MultiSimilarityLoss',
                miner_name='MultiSimilarityMiner', # example: TripletMarginMiner, MultiSimilarityMiner, PairMarginMiner
                miner_margin=0.1
            )
        elif model_name == "resnet50":
            self.model = helper.get_backbone('resnet50', pretrained, layers_to_freeze=2, layers_to_crop=[4])
        else:
            self.model = timm.create_model(
                model_name,
                pretrained,
                num_classes=0,
                # , global_pool="avg"
            )

        self.model.to(CFG.device)
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)


class ProjectionHead(nn.Module):
    def __init__(
        self, 
        embedding_dim, 
        projection_dim=256, 
        dropout=0.1, 
        num_blocks=1
    ):
        super().__init__()
        self.num_blocks = num_blocks
        
        # Linearly transition in width
        steps = num_blocks
        step_size = (projection_dim - embedding_dim) // steps
        dims = [embedding_dim + step_size * i for i in range(steps)] + [projection_dim]
        
        self.projection_layers = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(num_blocks)])
        self.block_middle_parts = nn.ModuleList([
            nn.Sequential(
                nn.GELU(),
                nn.Linear(dims[i+1], dims[i+1]),
                nn.Dropout(dropout),
            ) for i in range(num_blocks)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(dims[i+1]) for i in range(num_blocks)])
    
    def forward(self, x):
        for i in range(self.num_blocks):
            projected = self.projection_layers[i](x)
            x = self.block_middle_parts[i](projected)
            x = x + projected
            x = self.layer_norms[i](x)
        return x
    
    
class ImageProjectionHead(ProjectionHead):
    def __init__(
        self, 
        embedding_dim, 
        projection_dim=256, 
        dropout=0.1, 
        num_blocks=1
    ):
        super().__init__(embedding_dim, projection_dim, dropout, num_blocks)
    
    def forward(self, x):
        batch_size, channels, features = x.shape
        x = x.view(batch_size * channels, features)
        x = super().forward(x)
        x = x.view(batch_size, channels, -1)
        return x

class Pooling():
    def __init__(self):
        model = MixVPRModel(
        #---- Encoder
        backbone_arch='resnet50',
        pretrained=True,
        layers_to_freeze=2,
        layers_to_crop=[4], # 4 crops the last resnet layer, 3 crops the 3rd, ...etc

        agg_arch='MixVPR',
        agg_config={'in_channels' : 256,
                # 'in_h' : 40,
                # 'in_w' : 30,
                'in_h' : 1,
                'in_w' : 1,
                'out_channels' : 1024,
                'mix_depth' : 4,
                'mlp_ratio' : 1,
                'out_rows' : 4}, # the output dim will be (out_rows * out_channels)

        #----- Loss functions
        # example: ContrastiveLoss, TripletMarginLoss, MultiSimilarityLoss,
        # FastAPLoss, CircleLoss, SupConLoss,
        loss_name='MultiSimilarityLoss',
        miner_name='MultiSimilarityMiner', # example: TripletMarginMiner, MultiSimilarityMiner, PairMarginMiner
        miner_margin=0.1)
    
        state_dict = torch.load(CFG.mixvpr_checkpoint_name)
        model.load_state_dict(state_dict)

        self.aggregator = model.aggregator

    def forward(self, x):
        x = self.aggregator(x)
        return x

if __name__ == "__main__":
    test_input = torch.randn(8, 256).to(CFG.device)
    model = Pooling().to(CFG.device)
    output = model(test_input)
    print(output.shape)

# class LocationEncoder(nn.Module):
#     def __init__(self,
#                  embed_dim: int,
#                  # vision
#                  image_resolution: int,
#                  vision_layers: Union[Tuple[int, int, int, int], int],
#                  vision_width: int,
#                  vision_patch_size: int,
#                  # text
#                  context_length: int,
#                  vocab_size: int,
#                  transformer_width: int,
#                  transformer_heads: int,
#                  transformer_layers: int
#                  ):
#         super().__init__()

#         self.context_length = context_length

#         if isinstance(vision_layers, (tuple, list)):
#             vision_heads = vision_width * 32 // 64
#             self.visual = ModifiedResNet(
#                 layers=vision_layers,
#                 output_dim=embed_dim,
#                 heads=vision_heads,
#                 input_resolution=image_resolution,
#                 width=vision_width
#             )
#         else:
#             vision_heads = vision_width // 64
#             self.visual = VisionTransformer(
#                 input_resolution=image_resolution,
#                 patch_size=vision_patch_size,
#                 width=vision_width,
#                 layers=vision_layers,
#                 heads=vision_heads,
#                 output_dim=embed_dim
#             )

#         self.transformer = Transformer(
#             width=transformer_width,
#             layers=transformer_layers,
#             heads=transformer_heads,
#             attn_mask=self.build_attention_mask()
#         )

#         self.vocab_size = vocab_size
#         self.token_embedding = nn.Embedding(vocab_size, transformer_width)
#         self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
#         self.ln_final = LayerNorm(transformer_width)

#         self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
#         self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

#         self.initialize_parameters()

#     def initialize_parameters(self):
#         nn.init.normal_(self.token_embedding.weight, std=0.02)
#         nn.init.normal_(self.positional_embedding, std=0.01)

#         if isinstance(self.visual, ModifiedResNet):
#             if self.visual.attnpool is not None:
#                 std = self.visual.attnpool.c_proj.in_features ** -0.5
#                 nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
#                 nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
#                 nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
#                 nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

#             for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
#                 for name, param in resnet_block.named_parameters():
#                     if name.endswith("bn3.weight"):
#                         nn.init.zeros_(param)

#         proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
#         attn_std = self.transformer.width ** -0.5
#         fc_std = (2 * self.transformer.width) ** -0.5
#         for block in self.transformer.resblocks:
#             nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
#             nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
#             nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
#             nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

#         if self.text_projection is not None:
#             nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

#     def build_attention_mask(self):
#         # lazily create causal attention mask, with full attention between the vision tokens
#         # pytorch uses additive attention mask; fill with -inf
#         mask = torch.empty(self.context_length, self.context_length)
#         mask.fill_(float("-inf"))
#         mask.triu_(1)  # zero out the lower diagonal
#         return mask

#     @property
#     def dtype(self):
#         return self.visual.conv1.weight.dtype

#     def encode_text(self, text):

#         x = x.permute(1, 0, 2)  # NLD -> LND
#         x = self.transformer(x)
#         x = x.permute(1, 0, 2)  # LND -> NLD
#         x = self.ln_final(x).type(self.dtype)

#         # x.shape = [batch_size, n_ctx, transformer.width]
#         # take features from the eot embedding (eot_token is the highest number in each sequence)
#         x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

#         return x