import torch
import numpy as np
import tqdm
import os
import glob
import cv2
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
    writer = SummaryWriter()

    for k in top_k_values:
        print(f"Computing VPR for Top-{k}")
        matched_indices = compute_topk_match_vector(query_vectors, database_vectors, k=k)
        
        vpr_success = np.zeros((len(query_vectors), k), dtype=np.bool_)

        for index in tqdm(range(len(query_vectors)), total=len(query_vectors), desc=f'Examining VPR for Top-{k}'):
            for ki in range(k):  
                vpr_success[index, ki] = vpr_recall(query_vectors[index], database_vectors[matched_indices[index, ki]], CFG.vpr_threshold)
        
        vpr_success_rate = np.any(vpr_success, axis=1).mean()
        vpr_success_rates[f'Top-{k}'] = vpr_success_rate

        print(f"{k} VPR successful rate: {vpr_success_rate*100}%")
        # write to tensorboard
        
        writer.add_scalar("VPR Success Rate", vpr_success_rates)
    
    writer.close()
    return vpr_success_rates

def generate_query_images(image_path, output_path, decay_factor):
    """
    Processes an image by reducing its resolution by a given decay factor.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Directory to save the processed image.
        decay_factor (int): The factor by which to reduce the image resolution.
    """
    image_paths = glob.glob(os.path.join(image_path, '*.png'))
    feature_list = []

    for image_path in image_paths:
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error opening image {image_path}")
            continue

        # Resize the image based on the decay factor
        resized_frame = cv2.resize(frame, (frame.shape[1] // decay_factor, frame.shape[0] // decay_factor))

        # Save the processed image
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        processed_image_name = os.path.join(output_path, os.path.basename(image_path).replace('.png', '_resized.png'))
        cv2.imwrite(processed_image_name, resized_frame)

def extract_features_from_images(model, image_path, output_path, data_loader, device):
    """
    Get the image features from the model and store it to the vector_path. Used for both database and query images.

    Args:
        model: The model to extract features from.
        image_path (str): Path to the input image.
        output_path (str): Directory to save the features.
    """

    model.eval()
    features = []

    with torch.no_grad(): 
        for batch in data_loader:
            # Assuming 'image' key in batch dict
            image_data = batch['image'].to(device)
            image_features = model.image_encoder(image_data)
            
            # Optionally, convert features to another form, e.g., numpy array
            image_features_np = image_features.cpu().numpy()
            features.append(image_features_np)

    # Save features to file
    features_array = np.concatenate(features, axis=0)
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save(output_path, features_array)
# ---------------------------------------