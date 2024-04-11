import torch
import numpy as np
import os
import glob
import config as CFG 
import cv2
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

# ----------------- VPR -----------------
def compute_topk_match_vector(query_vectors, database_vectors, k=1, batch_size=100) -> np.ndarray:
    query_torch = torch.tensor(query_vectors).float().cuda()
    database_torch = torch.tensor(database_vectors).float().cuda()
    
    num_queries = query_vectors.shape[0]
    matched_indices = torch.zeros((num_queries, k), dtype=torch.int64).cuda()

    for start_idx in tqdm(range(0, num_queries, batch_size), desc="Computing matches"):
        end_idx = min(start_idx + batch_size, num_queries)
        batch = query_torch[start_idx:end_idx]

        prod = torch.einsum('ik,jk->ij', batch, database_torch)

        # Find the Top-K matches in the database for each vector in the batch
        _, indices = torch.topk(prod, k, dim=1, largest=True)
        matched_indices[start_idx:end_idx, :] = indices

    return matched_indices.cpu().numpy()

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

    model.eval()
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

    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save(output_path, descriptors_array)
# ---------------------------------------