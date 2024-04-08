import argparse
import csv
import glob
import json
import os
import pdb
import pickle
import re

import config as CFG

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import CPIPDataset, Dataset_for_query, get_transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils_file import AvgMeter, get_lr

from CPIP import CPIPModel

# ----------------- CLIP -----------------
def prepare_data(data_path, test_size=0.1, random_state=42):
    # List all .png files in the directory
    image_files = sorted([f for f in os.listdir(data_path) if f.endswith(".png")])

    # Prepare data
    data = []
    for image_file in image_files:
        json_file = image_file.replace(".png", ".json")
        json_path = os.path.join(data_path, json_file)

        with open(json_path, "r") as f:
            json_data = json.load(f)

        location = json_data["locations"][0]
        yaw = int(re.search(r"yaw(\d+)", image_file).group(1))
        location.append(yaw)

        data.append({"image": image_file, "location": location})

    # Convert to DataFrame for easy handling
    df = pd.DataFrame(data)

    # Split into train and validation sets
    train_df, valid_df = train_test_split(
        df, test_size=test_size, random_state=random_state, shuffle=True
    )

    return train_df, valid_df

def prepare_data_for_query(data_path, test_size=0.1, random_state=42):
    # List all .png files in the directory
    image_files = sorted([f for f in os.listdir(data_path) if f.endswith(".png")])

    # Prepare data
    data = []
    for image_file in image_files:
        data.append({"image": image_file})

    # Convert to DataFrame for easy handling
    df = pd.DataFrame(data)

    # Split into train and validation sets
    train_df, valid_df = train_test_split(
        df, test_size=test_size, random_state=random_state, shuffle=True
    )

    return train_df, valid_df

def build_loaders(dataframe, mode, image_path):
    transforms = get_transforms(mode=mode)
    dataset = CPIPDataset(dataframe, image_path, transforms=transforms)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size= CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader

def build_loaders_for_query(dataframe, mode, image_path):
    transforms = get_transforms(mode=mode)
    dataset = Dataset_for_query(dataframe, image_path, transforms=transforms)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size= CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=False,
    )
    return dataloader

def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    accuracy_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))

    # image input size: torch.Size([3, 518, 518])

    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        optimizer.zero_grad()
        loss, logits = model(batch)# seg fault here
        labels = torch.arange(logits.size(0)).long().to(logits.device)
        accuracy = calculate_accuracy(logits, labels)

        loss.backward()
        optimizer.step()

        if step == "batch":
            lr_scheduler.step()

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)
        accuracy_meter.update(accuracy, count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, train_accuracy=accuracy_meter.avg, lr=get_lr(optimizer))

    return loss_meter.avg, accuracy_meter.avg

def valid_epoch(model, valid_loader):
    model.eval()
    loss_meter = AvgMeter()
    accuracy_meter = AvgMeter()
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))

    with torch.no_grad():
        for batch in tqdm_object:
            batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
            loss, logits = model(batch)
            labels = torch.arange(logits.size(0)).long().to(logits.device)
            accuracy = calculate_accuracy(logits, labels)

            count = batch["image"].size(0)
            loss_meter.update(loss.item(), count)
            accuracy_meter.update(accuracy, count)

            tqdm_object.set_postfix(valid_loss=loss_meter.avg, valid_accuracy=accuracy_meter.avg)

    return loss_meter.avg, accuracy_meter.avg

def calculate_accuracy(logits, labels):
    _, predicted = logits.max(1)
    correct = predicted.eq(labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy
# ---------------------------------------


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
    #top_k_values = [1, 2, 5, 10, 20]
    top_k_values = [1]
    vpr_success_rates = {}

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
        #还要改
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



def main():
    print("Start the pipeline...")

    model_pt = CFG.model_name + ".pt"
    image_database_path = CFG.data_path + "/database"
    image_query_path = CFG.data_path + "/query"

    model = CPIPModel().to(CFG.device)
    if os.path.exists(model_pt):
        model.load_state_dict(torch.load(model_pt))
        print("Loaded Best Model!")
    
    train_df, valid_df = prepare_data(image_database_path)
    
    # 1. Train the model
    if CFG.do_train:

        print("Start training the model...")
        train_loader = build_loaders(train_df, mode="train", image_path =image_database_path)
        valid_loader = build_loaders(valid_df, mode="valid", image_path =image_database_path)
        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=CFG.patience, factor=CFG.factor)
        step = "epoch"
        best_loss = float("inf")
    
        with open('metrics.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write the header to include accuracy
            writer.writerow(["Epoch", "Training Loss", "Validation Loss", "Training Accuracy", "Validation Accuracy"])
            
            for epoch in range(CFG.epochs):
                print(f"Epoch: {epoch + 1}")
                model.train()
                # seg fault here
                train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
                model.eval()
                with torch.no_grad():
                    valid_loss, valid_accuracy = valid_epoch(model, valid_loader)

                # Write the current epoch's losses and accuracies
                writer.writerow([epoch + 1, train_loss, valid_loss, train_accuracy, valid_accuracy])
                print(f"Train Loss: {train_loss}, Valid Loss: {valid_loss}, Train Accuracy: {train_accuracy}, Valid Accuracy: {valid_accuracy}")

                pt_name = f"{CFG.experiment_name}.pt"
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    torch.save(model.state_dict(), pt_name)
                    print("Saved Best Model!")


    # 2. Store the query and database vectors
    database_vector_path = os.path.join(CFG.data_path, "database_vectors.npy")
    query_vector_path = os.path.join(CFG.data_path, "query_vectors.npy")

    # 3. Process the data
    if CFG.process_data == 1:
        print("Start processing the data...")

        # Create the query and database loaders
        #VPR_df, _ = prepare_data_for_query(image_query_path)
        #VPR_query_loader = build_loaders_for_query(VPR_df, mode="valid", image_path =image_query_path) # use images with resolution decay and no shuffle
        VPR_database_loader = build_loaders(train_df, mode="valid", image_path =image_database_path) # use images without resolution decay and no shuffle
        print("Data loader created")

        # Extract features from the database images and store them
        if not os.path.exists(database_vector_path):
            extract_features_from_images(model, image_database_path, database_vector_path, VPR_database_loader, CFG.device)
            print("Database vectors stored")

        # Extract features from the query images and store them
        if not os.path.exists(query_vector_path):
            extract_features_from_images(model, image_query_path, query_vector_path, VPR_query_loader, CFG.device)
            print("Query vectors stored")
    
    # 4. Load the database vectors and query vectors
    database_vectors = np.load(database_vector_path)
    #query_vectors = np.load(query_vector_path)

    print(f"Database vectors: {database_vectors.shape}")
    #print(f"Query vectors: {query_vectors.shape}")

    # 5. Use the model to do VPR on the validation set
    vpr_success_rates = do_vpr(database_vectors, database_vectors)




if __name__=="__main__":
    main()
    