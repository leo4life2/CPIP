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
from dataset import CPIPDataset, get_transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils_file import AvgMeter, get_lr
from CPIP import CPIPModel
from train import prepare_data, build_loaders, train_epoch, valid_epoch, calculate_metrics, train

from vpr import do_vpr, extract_features_from_images, vpr_recall, compute_topk_match_vector




def main():
    print("Starting the pipeline...")

    model_pt = CFG.model_name + ".pt"
    image_database_path = CFG.data_path + "/database"
    image_query_path = CFG.data_path + "/query"

    if os.path.exists(model_pt):
        model.load_state_dict(torch.load(model_pt))
        print("Loaded Best Model!")
    
    # 1. Train the model
    if CFG.do_train:
        print("Start training the model...")
        train()


    # 2. Store the query and database vectors
    database_vector_path = os.path.join(CFG.data_path, "database_vectors.npy")
    #query_vector_path = os.path.join(CFG.data_path, "query_vectors.npy")

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
    