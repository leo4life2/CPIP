from torch import nn
import timm
import config as CFG
import utils
from models import helper
import torch

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

        self.backbone = helper.get_backbone(backbone_arch, pretrained, layers_to_freeze, layers_to_crop)
        self.aggregator = helper.get_aggregator(agg_arch, agg_config)

    def forward(self, x):
        x = self.backbone(x)
        #pdb.set_trace()
        #shape torch.Size([1, 1024, 90, 51])
        x = self.aggregator(x)
        return x



if __name__ == "__main__":

    #ssl._create_default_https_context = ssl._create_unverified_context

    model = MixVPRModel(
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
    

    # 创建一个测试输入张量，假设使用resnet50标准输入大小为3x224x224
    test_input = torch.randn(1, 3, 1440, 810)  # 假设batch size为1

    # 前向传播以获取输出
    with torch.no_grad():  # 确保不计算梯度
        model.eval()  # 设置模型为评估模式
        output = model(test_input)
        print("Output size:", output.size())  # 输出结果维度