import json
import os
import csv
import re
import argparse

import config as CFG 
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from CPIP import CPIPModel
from dataset import CPIPDataset, get_transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
from cpip_utils import AvgMeter, get_lr, prepare_data, build_loaders, train_epoch, valid_epoch, calculate_metrics
from pipeline import main
import pdb
import ssl

ssl._create_default_https_context = ssl._create_unverified_context  # workaround for no local CA trust store issue

USER = os.environ.get("USER")
os.environ['TORCH_HOME'] = f'/scratch/{USER}/pytorch'

def train(data_path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=f"tensorboard/cpip_{timestamp}")
    
    # Define hyperparameters to log
    hparams = {key: value for key, value in vars(CFG).items() if not key.startswith('__')}

    data_df = prepare_data(data_path + "/database")
    # Split into train and validation sets
    train_df, valid_df = train_test_split(
        data_df, test_size=0.2, random_state=42, shuffle=True
    )

    train_loader = build_loaders(train_df, mode="train")
    valid_loader = build_loaders(valid_df, mode="valid")

    model = CPIPModel().to(CFG.device)

    if os.path.exists(CFG.cpip_checkpoint_path) and CFG.resume_training:
        model.load_state_dict(torch.load(CFG.cpip_checkpoint_path))
        print(f"Loaded model weights from {CFG.cpip_checkpoint_path} and resuming training...")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    )
    step = "epoch"

    best_loss = float("inf")
    metrics = {
        'best_loss': best_loss,
        'train_loss': 0,
        'valid_loss': 0,
        'train_accuracy': 0,
        'valid_accuracy': 0,
    }
    
    last_checkpoint_name = CFG.cpip_checkpoint_path if CFG.cpip_checkpoint_path != "" and os.path.isfile(CFG.cpip_checkpoint_path) else None

    for epoch in range(CFG.epochs):
        print(f"Epoch: {epoch + 1}")
        
        if epoch % CFG.vpr_validation_epochs == 0 and last_checkpoint_name: # VPR validation
            print("Performing VPR Validation")
            avg_dist_synthetic, avg_dist_mixvpr_best, avg_dist_synth_gt = main(data_path, cpip_checkpoint_path=last_checkpoint_name)
            
            writer.add_scalar('VPR/synthetic |P_q - P_a|/avg_dist', avg_dist_synthetic, epoch)
            writer.add_scalar('VPR/synthetic_gt |P_q - P_a{hat}|/avg_dist', avg_dist_synth_gt, epoch)
            writer.add_scalar('VPR/mixvpr_best |P_q - P_d|/avg_dist', avg_dist_mixvpr_best, epoch)
            
            writer.add_scalar('VPR/mixvpr_vs_synthetic |P_q - P_d| - |P_q - P_a|', avg_dist_mixvpr_best - avg_dist_synthetic, epoch)
            writer.add_scalar('VPR/mixvpr_vs_synthetic_gt |P_q - P_d| - |P_q - P_a{hat}|', avg_dist_mixvpr_best - avg_dist_synth_gt, epoch)

        train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, lr_scheduler, step, writer, epoch)
        valid_loss, valid_accuracy = valid_epoch(model, valid_loader, writer, epoch)

        # Update metrics
        metrics.update({
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'train_accuracy': train_accuracy,
            'valid_accuracy': valid_accuracy,
        })

        # Update learning rate with ReduceLROnPlateau
        lr_scheduler.step(valid_loss)

        if valid_loss < best_loss:
            best_loss = valid_loss
            metrics['best_loss'] = best_loss
            last_checkpoint_name = f"cpip_val{valid_accuracy:.1f}_{timestamp}.pt"
            torch.save(model.state_dict(), last_checkpoint_name)
            print(f"Saved Best Model as {last_checkpoint_name}!")

    # Log hyperparameters and final metrics
    writer.add_hparams(hparam_dict=hparams, metric_dict=metrics)

    writer.close()
    
def parse_args():
    parser = argparse.ArgumentParser(description="Run the VPR pipeline.")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(data_path=args.data_path)
