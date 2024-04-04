from torch import nn
import timm
import config as CFG
import utils
from models import helper
import pdb


class MixVPRModel(nn.Module):
    def __init__(self,
                 backbone_arch='resnet50',
                 pretrained=True,
                 layers_to_freeze=1,
                 layers_to_crop=[],
                 agg_arch='ConvAP',
                 agg_config={},
                 loss_name='MultiSimilarityLoss',
                 miner_name='MultiSimilarityMiner',
                 miner_margin=0.1,
                 faiss_gpu=False
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
        self.faiss_gpu = faiss_gpu

        self.loss_fn = utils.get_loss(loss_name)
        self.miner = utils.get_miner(miner_name, miner_margin)

        self.backbone = helper.get_backbone(backbone_arch, pretrained, layers_to_freeze, layers_to_crop)
        self.aggregator = helper.get_aggregator(agg_arch, agg_config)

        self.backbone.to(CFG.device)
        self.aggregator.to(CFG.device)

    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregator(x)
        return x

class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name=CFG.model_name, pretrained=CFG.pretrained, trainable=CFG.trainable
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
                        'in_h' : 90,
                        'in_w' : 51,
                        'out_channels' : 1024,
                        'mix_depth' : 4,
                        'mlp_ratio' : 1,
                        'out_rows' : 4}, # the output dim will be (out_rows * out_channels)

                #----- Loss functions
                # example: ContrastiveLoss, TripletMarginLoss, MultiSimilarityLoss,
                # FastAPLoss, CircleLoss, SupConLoss,
                loss_name='MultiSimilarityLoss',
                miner_name='MultiSimilarityMiner', # example: TripletMarginMiner, MultiSimilarityMiner, PairMarginMiner
                miner_margin=0.1,
                faiss_gpu=False)
        else:
            self.model = timm.create_model(
                model_name, pretrained, num_classes=0
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
        projection_dim=CFG.projection_dim,
        dropout=CFG.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        # x:  torch.Size([8, 4096])
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x
