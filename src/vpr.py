import torch
import numpy as np
import os
import glob
import config as CFG 
import cv2
from tqdm import tqdm
from datetime import datetime
from dataset import prepare_image
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

def compute_topk_match_vector(query_vectors, database_vectors, k=1) -> np.ndarray:
    if len(database_vectors.shape) == 3:
        # Batch mode: database_vectors shape is (batchsize, dbsize, feature_dim)
        batchsize = database_vectors.shape[0]
        dbsize = database_vectors.shape[1]
        matched_indices = np.zeros((batchsize, k), dtype=int)

        for b in range(batchsize):
            # Each query vector is compared only with its corresponding batch of database vectors
            query_vector = query_vectors[b:b+1]  # Shape: (1, feature_dim)
            db_vectors = database_vectors[b]     # Shape: (dbsize, feature_dim)

            # Compute dot product similarity
            prod = np.dot(query_vector, db_vectors.T)  # Shape: (1, dbsize)

            # Find the Top-K matches in the database for the query vector
            indices = np.argsort(-prod, axis=1)[:, :k]
            matched_indices[b, :] = indices.flatten()  # Flatten to fit the expected shape

    else:
        # Non-batch mode: database_vectors shape is (dbsize, feature_dim)
        num_queries = query_vectors.shape[0]
        matched_indices = np.zeros((num_queries, k), dtype=int)

        for start_idx in range(0, num_queries, k):
            end_idx = min(start_idx + k, num_queries)
            batch_query = query_vectors[start_idx:end_idx]

            # Compute dot product similarity
            prod = np.dot(batch_query, database_vectors.T)

            # Find the Top-K matches in the database for each vector in the batch
            indices = np.argsort(-prod, axis=1)[:, :k]
            matched_indices[start_idx:end_idx, :] = indices

    return matched_indices

def vpr_recall(location: np.ndarray, matched_location: np.ndarray, threshold: int) -> bool:
    """
    Simulates the VPR recall mechanism using precomputed matched indices.
    
    Args:
        matched_index (int): The index of the matched image.
        location (np.ndarray): The location associated with the descriptor.
        
    Returns:
        bool: Whether VPR recall was successful.
    """
    if np.linalg.norm(matched_location[:2] - location[:2]) <= threshold:
        return True
    else:
        return False

def do_vpr(database_vectors, query_vectors):
    top_k_values = [1, 2, 5, 10, 20]
    #top_k_values = [1]
    vpr_success_rates = {}
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=f"tensorboard/vpr_{timestamp}")

    for k in top_k_values:
        print(f"Computing VPR for Top-{k}")
        matched_indices = compute_topk_match_vector(query_vectors, database_vectors, k=k)
        
        vpr_success = np.zeros((len(query_vectors), k), dtype=np.bool_)
        distances = np.zeros((len(query_vectors), k), dtype=np.float32)

        for index in tqdm(range(len(query_vectors)), total=len(query_vectors), desc=f'Examining VPR for Top-{k}'):
            for ki in range(k):
                match_index = matched_indices[index, ki]
                distance = np.linalg.norm(query_vectors[index] - database_vectors[match_index])
                distances[index, ki] = distance
                vpr_success[index, ki] = vpr_recall(query_vectors[index], database_vectors[match_index], CFG.vpr_threshold)
        
        vpr_success_rate = np.any(vpr_success, axis=1).mean()
        average_distances = np.mean(distances, axis=1)
        vpr_success_rates[f'Top-{k}'] = vpr_success_rate

        # Log distances
        for ki in range(k):
            writer.add_scalar(f"Average Distance/Top-{k}-{ki+1}", np.mean(distances[:, ki]), k)
        writer.add_scalar(f"Average Distance/Top-{k}-All", np.mean(average_distances), k)
        print(f"{k} VPR successful rate: {vpr_success_rate*100}%")
        
        # write to tensorboard
        writer.add_scalar(f"VPR Success Rate/Top-{k}", vpr_success_rate, k)
    
    writer.close()
    return vpr_success_rates

def get_vpr_descriptors(model, data_loader, device, output_path):
    """
    Compute VPR descriptors using the provided model.

    Args:
        model (torch.nn.Module): The model used to compute VPR descriptors.
        data_loader (torch.utils.data.DataLoader): DataLoader containing the images.
        device (torch.device): The device to perform computation on.
        output_path (str): Path to save the descriptors.
    """

    descriptors = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Extracting Image Descriptors"):
            image_data = batch['image'].to(device)
            vpr_descriptors = model(image_data)
            
            # Convert descriptors to numpy array for storage or further processing
            vpr_descriptors_np = vpr_descriptors.cpu().numpy()
            descriptors.append(vpr_descriptors_np)

    # Concatenate all descriptors into a single numpy array
    descriptors_array = np.concatenate(descriptors, axis=0)
    return descriptors_array