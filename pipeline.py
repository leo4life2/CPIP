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


    # 2. Store the vectors
    vector_path = os.path.join(CFG.data_path, "database_vectors.npy")

    # 3. Process the data
    if CFG.process_data == 1:
        print("Start processing the data...")

        # Create the loaders
        train_df, valid_df = prepare_data(CFG.data_path, test_size=CFG.factor, random_state=42)
        VPR_loader = build_loaders(train_df, mode="valid") # use images without resolution decay and no shuffle
        print("Data loader created")

        # Extract features from the database images and store them
        if not os.path.exists(vector_path):
            extract_features_from_images(model, CFG.data_path, vector_path, VPR_loader, CFG.device)
            print("Database vectors stored")
    
    # 4. Load the database vectors
    database_vectors = np.load(vector_path)

    # 5. Use the model to do VPR on the validation set
    vpr_success_rates = do_vpr(database_vectors, database_vectors)




if __name__=="__main__":
    main()
    