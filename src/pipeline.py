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
import math
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from scipy.spatial.transform import Rotation as R
from dataset import CPIPDataset, get_transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils_file import AvgMeter, get_lr
from CPIP import CPIPModel
from cpip_train import prepare_data, build_loaders, train_epoch, valid_epoch, calculate_metrics, train
from modules import MixVPRModel, ShallowConvNet

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

def get_cpip_model():
    model = CPIPModel().to(CFG.device)

    if os.path.exists(CFG.cpip_checkpoint_name):
        model.load_state_dict(torch.load(best_model_path))
    else:
        print("WARNING: No weights for CPIP model")
        
    return model
        
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

def generate_grid(locations):
    """
    Generates a grid of points around each location, excluding the center point in the horizontal and vertical range,
    and includes diagonal points. Additionally, it generates multiple rotation steps for each grid point.

    Args:
        locations (torch.Tensor): A tensor of locations with shape (N, 3) where each row is (x, y, theta).

    Returns:
        torch.Tensor: A tensor containing grid points with shape (N, ((2*CFG.grid_extent+1)**2 - 1) * CFG.num_rotation_steps, 3).
    """
    # Extract x and y coordinates from the first two dimensions of the input tensor
    x_center, y_center = locations[:, 0], locations[:, 1]
    
    # Create ranges for horizontal and vertical grid points
    range_tensor = torch.arange(-CFG.grid_extent, CFG.grid_extent + 1, device=locations.device)
    
    # Create a meshgrid for x and y ranges
    grid_x, grid_y = torch.meshgrid(range_tensor, range_tensor, indexing='ij')
    
    # Flatten the grid and remove the center point (0,0)
    grid_x_flat = grid_x.flatten()
    grid_y_flat = grid_y.flatten()
    center_mask = (grid_x_flat != 0) | (grid_y_flat != 0)
    grid_x_flat = grid_x_flat[center_mask]
    grid_y_flat = grid_y_flat[center_mask]
    
    # Calculate grid points
    grid_points_x = x_center[:, None] + grid_x_flat * CFG.grid_spacing
    grid_points_y = y_center[:, None] + grid_y_flat * CFG.grid_spacing
    
    # Combine x and y coordinates for grid points
    grid_points = torch.stack((grid_points_x, grid_points_y), dim=2)
    
    # Generate rotation steps
    theta_steps = torch.linspace(0, 2 * math.pi, CFG.num_rotation_steps + 1, device=locations.device)[:-1]  # Exclude the last point to avoid duplication of 0 and 2*pi
    theta_steps = theta_steps.expand(grid_points.size(0), grid_points.size(1), CFG.num_rotation_steps)
    
    # Repeat grid points for each rotation step
    grid_points = grid_points.unsqueeze(2).expand(-1, -1, CFG.num_rotation_steps, -1)
    
    # Combine grid points with theta steps
    grid_points_with_theta = torch.cat((grid_points, theta_steps.unsqueeze(-1)), dim=-1)
    
    # Reshape to final size (N, (2*CFG.grid_extent+1)**2 - 1 * CFG.num_rotation_steps, 3)
    grid_points_with_theta = grid_points_with_theta.reshape(locations.size(0), -1, 3)
    
    return grid_points_with_theta

def get_location_descriptors(mixvpr_agg, cpip_model, locations):
    location_encoder = cpip_model.location_projection
    encoded_locations = location_encoder(locations)  # Encode all location vectors
    encoded_locations = encoded_locations.view(-1, CFG.contrastive_dimension)  # Ensure shape is [N, 256]

    # Reshape to [1, sqrt(256), sqrt(256)] for each vector
    dimension_size = int(CFG.contrastive_dimension**0.5)
    reshaped_locations = encoded_locations.view(-1, 1, dimension_size, dimension_size)

    # Create and apply CNN model
    cnn_model = ShallowConvNet(input_c=1, input_h=dimension_size, input_w=dimension_size, output_c=1024, output_h=20, output_w=20)
    descriptors = cnn_model(reshaped_locations)  # Output shape [N, 1024, 20, 20]

    synthetic_descriptors = mixvpr_agg(descriptors)

    # Create a DataFrame to return locations and their corresponding descriptors
    descriptors_df = pd.DataFrame({
        "locations": [loc.tolist() for loc in locations],
        "descriptors": [desc.squeeze().tolist() for desc in synthetic_descriptors]
    })

    return descriptors_df
    

def main(data_path):
    print("Starting the pipeline...")
    
    mixvpr_model = get_mixvpr_model()
    cpip_model = get_cpip_model()
    
    # 1. Obtain database and query vectors
    db_df, query_df = load_or_generate_dataframes(mixvpr_model, data_path)
    
    # 2.
    # Get top k closest descriptors {D_d} for all input D_q
    matched_indices = do_similarity_search(db_df['descriptors'].tolist(), query_vectors, k=CFG.top_k)
    # Get P_d for all {D_d}
    query_average_positions = get_average_position(db_df, matched_indices)
    # Get grid points for all P_d
    grid_points = generate_grid(query_average_positions)
    
    # 3. Generate synthetic descriptors for all locations
    synthetic_descriptors_df = get_location_descriptors(mixvpr_model.aggregator, cpip_model, grid_points)
    
    # 4. Complete the similarity search for synthetic descriptors against query descriptors
    synthetic_d_match_ixs = do_similarity_search(synthetic_descriptors_df['descriptors'].tolist(), query_df['descriptors'].tolist(), k=CFG.top_k)
    # Get P_a for all {D_a}
    synthetic_query_avg_positions = get_average_position(synthetic_descriptors_df, synthetic_d_match_ixs)

    # 5. Calculate distances between query positions and average positions
    # Calculate distance D_1 between P_a and P_q
    distance_D1 = np.linalg.norm(query_df['positions'].values - synthetic_query_avg_positions, axis=1)
    # Calculate distance D_2 between P_d and P_q
    query_average_positions = get_average_position(db_df, matched_indices)
    distance_D2 = np.linalg.norm(query_df['positions'].values - query_average_positions, axis=1)
    
    # vpr_success_rates = do_vpr(db_df['descriptors'].tolist(), query_df['descriptors'].tolist())
    # print("VPR success rates:", vpr_success_rates)

def parse_args():
    parser = argparse.ArgumentParser(description="Run the VPR pipeline.")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data.')
    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    main(data_path=args.data_path)
