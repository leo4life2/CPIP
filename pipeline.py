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
import rdp
import msgpack
from scipy.spatial.transform import Rotation as R
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

# Step 1: to get the MixVPR descriptor D_q from image query
def get_mixvpr_descriptors(query_image_path, query_vector_path = None):
    print("Step 1: Getting MixVPR descriptors...")
    model = get_mixvpr_model()
    query_df = prepare_data(query_image_path)
    query_loader = build_loaders(query_df, mode="valid")
    if query_vector_path is None:
        query_vector_path = os.path.join(query_image_path, "..", "query_vectors.npy")
    query_vectors = get_vpr_descriptors(model, query_loader, CFG.device, query_vector_path)
    return query_vectors


# Step 4: Do similarity search in {D_f} with D_q, getting top K descriptors {D_a}. 
def do_similarity_search(database_vectors, query_vectors,  k=5):
    print("Step 4: Doing similarity search D_f with D_q")
    matched_indices = compute_topk_match_vector(query_vectors, database_vectors, k=k)
    return matched_indices

#  Let P_a be the average of all {D_a} descriptors’s corresponding positions (average theta too)
def get_average_position(matched_indices, k=5):
    # suppose there is a npy file containing the positions of the database images, with the order of the database_vectors
    database_positions = np.load(npy_file_path)
    # get the positions of the matched images
    matched_positions = database_positions[matched_indices]
    # calculate the average of all matched positions
    average_positions = np.mean(matched_positions, axis=1)
    return average_positions

# Step 5: Calculate distance D_1 between P_a and P_q 2. Calculate distance D_2 between P_d and P_q
def calculate_distances(average_positions, query_position):
    print("Step 5: Calculating distances...")
    distance = np.linalg.norm(average_positions - query_position)
    return distance

def get_orientation(mp1,vp1):
    def nn_line(xx,yy):
        points = np.vstack((xx,yy)).T
        tolerance = 1
        min_angle = np.pi*0.22
        simplified = np.array(rdp.rdp(points.tolist(), tolerance))
        sx, sy = simplified.T
        directions = np.diff(simplified, axis=0)
        theta = angle(directions)
        idx = np.where(theta>min_angle)[0]+1
        org_idx=[]
        for i in range(idx.size):    
            mindist = np.inf
            minidx=0
            for j in range(xx.size):
                d=math.dist([sx[idx[i]],sy[idx[i]]],[xx[j],yy[j]])
                if (d<mindist):
                    mindist=d
                    minidx=j
            org_idx.append(minidx)
        return xx,yy,theta,org_idx

    def angle(dir):
        dir2 = dir[1:]
        dir1 = dir[:-1]
        return np.arccos((dir1*dir2).sum(axis=1)/(np.sqrt((dir1**2).sum(axis=1)*(dir2**2).sum(axis=1))))

    def key_img(msg_path, video_path):

        # Read file as binary and unpack data using MessagePack library
        with open(msg_path, "rb") as f:
            u = msgpack.Unpacker(f)
            msg = u.unpack()

        # The point data is tagged "landmarks"
        key_frames = msg["keyframes"]

        print("Point cloud has {} points.".format(len(key_frames)))

        key_frame = {int(k): v for k, v in key_frames.items()}
        
        video_name = video_path.split("/")[-1][:-4]
        if not os.path.exists(video_name):
            os.mkdir(video_name)

        vidcap = cv2.VideoCapture(video_path)
        fps = int(vidcap.get(cv2.CAP_PROP_FPS)) + 1
        count = 0

        tss=[]
        keyfrm_points=[]
        for key in sorted(key_frame.keys()):
            point = key_frame[key]

            # position capture
            trans_cw = np.matrix(point["trans_cw"]).T
            rot_cw = R.from_quat(point["rot_cw"]).as_matrix()

            rot_wc = rot_cw.T
            trans_wc = - rot_wc * trans_cw
            keyfrm_points.append((trans_wc[0, 0], trans_wc[1, 0], trans_wc[2, 0]))
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, fps * float(point["ts"]))
            tss.append(point["ts"])

            # image capture
            success, image = vidcap.read()

            if not success:
                print("capture failed")
            else:
                cv2.imwrite(os.path.join(video_name, str(count) +".jpg"), image)
            count+=1
        keyfrm_points = np.array(keyfrm_points)
        keyfrm_points = np.delete(keyfrm_points, 1, 1)
        return keyfrm_points,tss


    kp1,ts1=key_img(mp1,vp1)    
    x1,y1,theta1,id1=nn_line(kp1[:,0],kp1[:,1])#
    if len(id1)==0:
        id1.append(len(kp1)-3)
        
    return x1,y1,theta1,id1
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
