import argparse
import csv
import glob
import json
import os
import pdb
import pickle
import re
from torch.utils.tensorboard import SummaryWriter

import config as CFG

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import CPIPDataset, get_transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils_file import AvgMeter, get_lr
from CPIP import CPIPModel
from train import prepare_data, build_loaders, train_epoch, valid_epoch, calculate_metrics, train
from modules import MixVPRModel

from vpr import do_vpr, get_vpr_descriptors, vpr_recall, compute_topk_match_vector

def get_mixvpr_model():
    model = MixVPRModel(
        #---- Encoder
        backbone_arch='resnet50',
        pretrained=True,
        layers_to_freeze=2,
        layers_to_crop=[4], # 4 crops the last resnet layer, 3 crops the 3rd, ...etc

        agg_arch='MixVPR',
        agg_config={'in_channels' : 1024,
                # 'in_h' : 40,
                # 'in_w' : 30,
                'in_h' : 20,
                'in_w' : 20,
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
    
    state_dict = torch.load(CFG.mixvpr_checkpoint_name)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def main(data_path):
    print("Starting the pipeline...")
    
    image_database_path = data_path + "/database"
    image_query_path = data_path + "/query"
    
    # TODO: full pipeline not implemented yet, this cpip model is currently not used
    # 1. Train the model
    if CFG.do_train:
        print("Start training the model...")
        train(data_path)

    # 2. Process the data
    database_vector_path = os.path.join(data_path, "database_vectors.npy")
    query_vector_path = os.path.join(data_path, "query_vectors.npy")

    if CFG.process_data:
        print("Start processing the data...")

        # Create the loaders
        database_df = prepare_data(image_database_path)
        database_loader = build_loaders(database_df, mode="valid")
        
        query_df = prepare_data(image_query_path)
        query_loader = build_loaders(query_df, mode="valid")
        print("Data loaders created")
        
        mixvpr_model = get_mixvpr_model() 
        # Create descriptors from the images and store them if they do not already exist
        if not os.path.exists(database_vector_path):
            get_vpr_descriptors(mixvpr_model, database_loader, CFG.device, database_vector_path)
            print("Database vectors stored")
        else:
            print("Database vectors already exist.")

        if not os.path.exists(query_vector_path):
            get_vpr_descriptors(mixvpr_model, query_loader, CFG.device, query_vector_path)
            print("Query vectors stored")
        else:
            print("Query vectors already exist.")
    
    # 3. Use the model to do VPR
    database_vectors = np.load(database_vector_path)
    query_vectors = np.load(query_vector_path)
    vpr_success_rates = do_vpr(database_vectors, query_vectors)
    print("VPR success rates:", vpr_success_rates)

def parse_args():
    parser = argparse.ArgumentParser(description="Run the VPR pipeline.")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data.')
    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    main(data_path=args.data_path)
