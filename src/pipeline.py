import argparse
import os
import pickle
import config as CFG
import math
import numpy as np
import pandas as pd
import torch
from CPIP import CPIPModel
from tqdm import tqdm
from cpip_utils import prepare_data, build_loaders, train_epoch, valid_epoch, calculate_metrics
from modules import MixVPRModel, ShallowConvNet
from vpr import do_vpr, get_vpr_descriptors, vpr_recall, compute_topk_match_vector

def get_mixvpr_descriptors_for_dir(model, image_path, save_path):
    df = prepare_data(image_path)
    loader = build_loaders(df, mode="valid")
    vectors = get_vpr_descriptors(model, loader, CFG.device, save_path)
    df['descriptors'] = list(vectors)
    return df

def calculate_average_closest_distance(database_df, sample_percentage):
    # Extract 2D coordinates, ignoring the last element of each array in the 'location' column
    coordinates = np.array([loc[:2] for loc in database_df['location']])
    
    # Deduplicate coordinates to remove exact same rows
    coordinates = np.unique(coordinates, axis=0)
    
    # Determine the number of samples to take
    sample_size = int(len(coordinates) * sample_percentage)
    
    # Randomly sample indices from the coordinates array
    sample_indices = np.random.choice(len(coordinates), size=sample_size, replace=False)
    sampled_coordinates = coordinates[sample_indices]
    
    # Initialize a list to store the minimum distances for each sampled point
    min_distances = []
    
    # Compute the closest neighbor distance for each point in the sample
    for sample in sampled_coordinates:
        # Compute distances from the current sample to all points in the dataset
        distances = np.sqrt(np.sum((coordinates - sample) ** 2, axis=1))
        
        # The closest distance is the minimum of these distances, excluding the distance to itself (which is zero)
        # We sort and pick the second smallest since the smallest will be zero (distance to itself)
        closest_distance = np.partition(distances, 1)[1]
        
        # Append the closest distance found to the list
        min_distances.append(closest_distance)
    
    # Calculate the average of the minimum distances
    average_distance = np.mean(min_distances)
    
    return average_distance

# Do similarity search in {D_f} with D_q, getting top K descriptors {D_a}. 
def do_similarity_search(database_vectors, query_vectors, k=5, batch=False):
    return compute_topk_match_vector(query_vectors, database_vectors, k=k)

# Getting top K descriptors {D_a}. 
def get_average_position(database, matched_indices):
    if len(database.shape) == 2:  # Original case: shape (database_rows, 3)
        positions = np.array(database['location'].tolist())  # Convert list of lists to a numpy array of numpy arrays
    elif len(database.shape) == 3:  # New case: shape (batchsize, database_rows, 3)
        positions = database  # Directly use the numpy array

    # Ensure matched_indices is used to index positions correctly
    if len(database.shape) == 2:
        matched_positions = np.array([positions[indices] for indices in matched_indices])
    elif len(database.shape) == 3:
        matched_positions = np.array([positions[i, indices] for i, indices in enumerate(matched_indices)])

    # Calculate the average of all matched positions
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
    
    state_dict = torch.load(CFG.mixvpr_checkpoint_path)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def get_cpip_model(checkpoint_path):
    model = CPIPModel().to(CFG.device)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
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
    
    if not (os.path.isfile(database_df_path) and os.path.isfile(query_df_path)):
        database_df = process_vectors(mixvpr_model, data_path + "/database", database_df_path, "Database")
        query_df = process_vectors(mixvpr_model, data_path + "/query", query_df_path, "Query")
        database_df.to_pickle(database_df_path)
        query_df.to_pickle(query_df_path)
    else:
        database_df = pd.read_pickle(database_df_path)
        query_df = pd.read_pickle(query_df_path)
    
    return database_df, query_df

def generate_grid(locations, grid_spacing):
    """
    Generates a grid of points around each location, excluding the center point in the horizontal and vertical range,
    and includes diagonal points. Additionally, it generates multiple rotation steps for each grid point.

    Args:
        locations (torch.Tensor): A tensor of locations with shape (N, 3) where each row is (x, y, theta).

    Returns:
        torch.Tensor: A tensor containing grid points with shape (N, num_points, 3).
            num_points = ((2*CFG.grid_extent+1)**2 - 1) * CFG.num_rotation_steps
    """
    locations = torch.tensor(locations, device=CFG.device)
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
    grid_points_x = x_center[:, None] + grid_x_flat * grid_spacing
    grid_points_y = y_center[:, None] + grid_y_flat * grid_spacing
    
    # Combine x and y coordinates for grid points
    grid_points_orig = torch.stack((grid_points_x, grid_points_y), dim=2)
    
    # Generate rotation steps
    theta_steps = torch.linspace(0, CFG.max_rotation_degrees, CFG.num_rotation_steps + 1, device=locations.device)[:-1]  # Exclude the last point to avoid duplication of 0 and 2*pi
    theta_steps = theta_steps.expand(grid_points_orig.size(0), grid_points_orig.size(1), CFG.num_rotation_steps)
    
    # Repeat grid points for each rotation step
    grid_points = grid_points_orig.unsqueeze(2).expand(-1, -1, CFG.num_rotation_steps, -1)
    
    # Combine grid points with theta steps
    grid_points_with_theta = torch.cat((grid_points, theta_steps.unsqueeze(-1)), dim=-1)
    
    # Reshape to final size (N, (2*CFG.grid_extent+1)**2 - 1 * CFG.num_rotation_steps, 3)
    grid_points_with_theta = grid_points_with_theta.reshape(locations.size(0), -1, 3)
    return grid_points_with_theta.float(), grid_points_orig.float()

def get_location_descriptors(mixvpr_agg, cpip_model, locations):
    location_encoder = cpip_model.location_projection
    with torch.no_grad():
        encoded_locations = location_encoder(locations)  # Encode all location vectors
        # let num_points = ((2*CFG.grid_extent+1)**2 - 1) * CFG.num_rotation_steps,
        # encoded_locations shape: (b, num_points, CFG.contrastive_dimension)

        # current contrastive_dimension = 256
        # Reshape to [1, sqrt(256), sqrt(256)] for each vector
        b, num_points, _ = encoded_locations.shape
        dimension_size = int(CFG.contrastive_dimension**0.5)
        reshaped_locations = encoded_locations.view(b * num_points, 1, dimension_size, dimension_size)

        # Create and apply CNN model
        cnn_model = ShallowConvNet(input_c=1, input_h=dimension_size, input_w=dimension_size, output_c=1024, output_h=20, output_w=20).to(CFG.device)
        # TODO: cnn weights
        cnn_model.eval()
        descriptors = cnn_model(reshaped_locations)

        synthetic_descriptors = mixvpr_agg(descriptors) # Shape: (b * num_points, out_rows * out_channels)
        # at this point, (800, 1, 16, 16) takes about 20gb vram. 
        synthetic_descriptors = synthetic_descriptors.view(b, num_points, 4 * 1024)
        
        return synthetic_descriptors
    
def get_closest_points(query_positions_2d, grid_points):
    # Compute pairwise Euclidean distances
    # Expand query_positions_2d to match the shape of grid_points for broadcasting
    expanded_query_positions = query_positions_2d[:, np.newaxis, :]  # Shape [5, 1, 2]
    # Calculate distances (using broadcasting)
    distances = np.linalg.norm(grid_points - expanded_query_positions, axis=2)  # Shape [5, 80]
    # Find the index of the minimum distance for each set
    min_indices = np.argmin(distances, axis=1)  # Shape [5]
    # Select the closest points using these indices
    closest_points = grid_points[np.arange(grid_points.shape[0]), min_indices]  # Shape [5, 2]
    
    return closest_points


def main(data_path, cpip_checkpoint_path=CFG.cpip_checkpoint_path):
    print("Starting the pipeline...")
    
    mixvpr_model = get_mixvpr_model()
    cpip_model = get_cpip_model(cpip_checkpoint_path)
    
    print("Obtaining database and query vectors")
    # Obtain database and query vectors
    db_df, query_df = load_or_generate_dataframes(mixvpr_model, data_path)
    
    # Convert the list of numpy arrays in db_df['descriptors'] into a 2D numpy array
    database_vectors = np.stack(db_df['descriptors'].values)
    
    # Compute grid spacing
    avg_closest_dist = calculate_average_closest_distance(db_df, sample_percentage=1.0)
    grid_side_length = CFG.grid_side_length_percentage * avg_closest_dist
    grid_spacing = grid_side_length / (2 * CFG.grid_extent + 1)

    # Initialize lists to collect results
    all_distances_synthetic_emb_best_match = []
    all_distances_mixvpr_best_match = []
    all_distances_synthetic_real_best = []

    # Process in batches of 5 query vectors
    num_queries = len(query_df)
    batch_size = CFG.vpr_batch_size
    for start_idx in tqdm(range(0, num_queries, batch_size), desc="Processing query batches"):
        end_idx = min(start_idx + batch_size, num_queries)
        query_vectors_batch = np.stack(query_df['descriptors'].values[start_idx:end_idx])
        
        # Get top k closest descriptors {D_d} for all input D_q in the batch
        matched_indices = do_similarity_search(database_vectors, query_vectors_batch, k=CFG.top_k)
        # Get P_d for all {D_d}
        mixvpr_average_positions = get_average_position(db_df, matched_indices)
        # Get grid points for all P_d
        grid_points_theta, grid_points_2d = generate_grid(mixvpr_average_positions, grid_spacing)
        
        # Generate synthetic descriptors for all locations
        synthetic_descriptors = get_location_descriptors(mixvpr_model.aggregator, cpip_model, grid_points_theta)
        # Shape: (batchsize, numpoints, 4096)
        
        # Complete the similarity search for synthetic descriptors against query descriptors
        query_vectors_batch = np.stack(query_df['descriptors'].values[start_idx:end_idx])
        synthetic_d_match_ixs = do_similarity_search(synthetic_descriptors.cpu().numpy(), query_vectors_batch, k=CFG.top_k, batch=True)
        
        # Get P_a for all {D_a}
        synthetic_query_avg_positions = get_average_position(grid_points_theta.cpu().numpy(), synthetic_d_match_ixs)
        
        # Calculate distances
        query_positions_batch = query_df['location'].values[start_idx:end_idx]
        query_positions_batch = np.array([np.array(pos) for pos in query_positions_batch])
        
        query_positions_2d = query_positions_batch[:, :2]
        mixvpr_avg_positions_2d = mixvpr_average_positions[:, :2]
        synthetic_avg_positions_2d = synthetic_query_avg_positions[:, :2]
        synthetic_top_positions_2d = synthetic_avg_positions_2d[:1, :]
        # Real 2d best synthetic loc matches from grid points
        synth_loc_real_closest_points = get_closest_points(query_positions_2d, grid_points_2d.cpu().numpy()) # P_hat{a}
        
        distance_synthetic_emb_best_match = np.linalg.norm(query_positions_2d - synthetic_avg_positions_2d, axis=1) # P_q - P_d
        distance_mixvpr_best_match = np.linalg.norm(query_positions_2d - mixvpr_avg_positions_2d, axis=1) # P_q - P_a
        distance_synthetic_real_best_match = np.linalg.norm(query_positions_2d - synth_loc_real_closest_points, axis=1) # P_q - P_hat{a}
        
        # Collect results
        all_distances_synthetic_emb_best_match.extend(distance_synthetic_emb_best_match)
        all_distances_mixvpr_best_match.extend(distance_mixvpr_best_match)
        all_distances_synthetic_real_best.extend(distance_synthetic_real_best_match)

    # Aggregate results
    avg_dist_synthetic_emb_best_match = np.mean(all_distances_synthetic_emb_best_match)
    min_dist_synthetic_emb_best_match = np.min(all_distances_synthetic_emb_best_match)
    max_dist_synthetic_emb_best_match = np.max(all_distances_synthetic_emb_best_match)
    avg_dist_mixvpr_best = np.mean(all_distances_mixvpr_best_match)
    min_dist_mixvpr_best = np.min(all_distances_mixvpr_best_match)
    max_dist_mixvpr_best = np.max(all_distances_mixvpr_best_match)
    avg_dist_synthetic_real_best = np.mean(all_distances_synthetic_real_best)
    min_dist_synthetic_real_best = np.min(all_distances_synthetic_real_best)
    max_dist_synthetic_real_best = np.max(all_distances_synthetic_real_best)

    print("Average distance to synthetic descriptors w/ embedding similarity:", avg_dist_synthetic_emb_best_match)
    print("Minimum distance to synthetic descriptors w/ embedding similarity:", min_dist_synthetic_emb_best_match)
    print("Maximum distance to synthetic descriptors w/ embedding similarity:", max_dist_synthetic_emb_best_match)
    print("Average distance to MixVPR's best match:", avg_dist_mixvpr_best)
    print("Minimum distance to MixVPR's best match:", min_dist_mixvpr_best)
    print("Maximum distance to MixVPR's best match:", max_dist_mixvpr_best)
    print("Average distance to synthetic descriptors w/ gt location:", avg_dist_synthetic_real_best)
    print("Minimum distance to synthetic descriptors w/ gt location:", min_dist_synthetic_real_best)
    print("Maximum distance to synthetic descriptors w/ gt location:", max_dist_synthetic_real_best)

    return avg_dist_synthetic_emb_best_match, avg_dist_mixvpr_best, avg_dist_synthetic_real_best

def parse_args():
    parser = argparse.ArgumentParser(description="Run the VPR pipeline.")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data.')
    parser.add_argument('--cpip_checkpoint_path', type=str, default=CFG.cpip_checkpoint_path, help='Optional checkpoint path for CPIP. Defaults to the path in config.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(data_path=args.data_path, cpip_checkpoint_path=args.cpip_checkpoint_path)
