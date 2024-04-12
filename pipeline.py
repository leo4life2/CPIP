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

def get_mixvpr_descriptors_for_dir(model, image_path, save_path):
    df = prepare_data(image_path)
    loader = build_loaders(df, mode="valid")
    vectors = get_vpr_descriptors(model, loader, CFG.device, save_path)
    df['descriptors'] = vectors
    return df

# Do similarity search in {D_f} with D_q, getting top K descriptors {D_a}. 
def get_average_position(df, matched_indices):
    positions = df['location'].values  # Extract locations as a numpy array

    # Get the positions of the matched images
    # matched_indices is a 2D array where each row corresponds to the top k matches for each query
    matched_positions = np.array([positions[indices] for indices in matched_indices])

    # Calculate the average of all matched positions
    # We average across axis=1, which is the new axis representing the matched positions for each query
    average_positions = np.mean(matched_positions, axis=1)
    return average_positions
# Let P_a be the average of all {D_a} descriptors’s corresponding positions (average theta too)
def get_average_position(database, matched_indices):
    # get the positions of the matched images
    matched_positions = database_positions[matched_indices]
    # calculate the average of all matched positions
    average_positions = np.mean(matched_positions, axis=1)
    return average_positions

# Calculate distance D_1 between P_a and P_q 2. Calculate distance D_2 between P_d and P_q
def calculate_distances(average_positions, query_position):
    distance = np.linalg.norm(average_positions - query_position)
    return distance

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

def cpip_training(data_path):
    if CFG.do_train:
        print("Start CPIP model training...")
        train(data_path)
        
def process_vectors(model, image_path, vector_path, vector_type):
    if os.path.exists(vector_path):
        print(f"{vector_type} vectors at {vector_path} exist, will overwrite with new outputs.")
    df = get_mixvpr_descriptors_for_dir(model, image_path, vector_path)
    print(f"New {vector_type.lower()} vectors stored at {vector_path}.")
    return df

def load_or_generate_dataframes(mixvpr_model, data_path):
    database_df_path = os.path.join(data_path, "database_df.pkl")
    query_df_path = os.path.join(data_path, "query_df.pkl")
    
    if CFG.process_data:
        database_df = process_vectors(mixvpr_model, data_path + "/database", database_df_path, "Database")
        query_df = process_vectors(mixvpr_model, data_path + "/query", query_df_path, "Query")
        database_df.to_pickle(database_df_path)
        query_df.to_pickle(query_df_path)
    else:
        database_df = pd.read_pickle(database_df_path)
        query_df = pd.read_pickle(query_df_path)
    
    return database_df, query_df

def main(data_path):
    print("Starting the pipeline...")
    cpip_training(data_path)
    
    mixvpr_model = get_mixvpr_model()
    # 1. Obtain database and query vectors
    db_df, query_df = load_or_generate_dataframes(mixvpr_model, data_path)
    
    # 2.
    # Get top k closest descriptors {D_d} for all input D_q
    matched_indices = do_similarity_search(database_vectors, query_vectors, k=CFG.top_k)
    
    # Get P_d for all {D_d}
    query_average_positions = get_average_position(db_df, matched_indices)
    
    
    vpr_success_rates = do_vpr(database_vectors, query_vectors)
    print("VPR success rates:", vpr_success_rates)

def parse_args():
    parser = argparse.ArgumentParser(description="Run the VPR pipeline.")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data.')
    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    main(data_path=args.data_path)
