import json
import os
import csv
import re

import config as CFG 
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from CPIP import CPIPModel
from dataset import CPIPDataset, get_transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
from utils_file import AvgMeter, get_lr
import pdb
import ssl

ssl._create_default_https_context = ssl._create_unverified_context  # workaround for no local CA trust store issue

USER = os.environ.get("USER")
os.environ['TORCH_HOME'] = f'/scratch/{USER}/pytorch'

def prepare_data(data_path):
    # List all .png files in the directory
    image_files = sorted([f for f in os.listdir(data_path) if f.endswith(".png")])

    # Prepare data
    data = []
    for image_file in image_files:
        full_image_path = os.path.join(data_path, image_file)
        json_file = image_file.replace(".png", ".json")
        json_path = os.path.join(data_path, json_file)

        with open(json_path, "r") as f:
            json_data = json.load(f)

        location = json_data["locations"][0]
        yaw = int(re.search(r"yaw(\d+)", image_file).group(1))
        location.append(yaw)

        data.append({"image": full_image_path, "location": location})

    # Convert to DataFrame for easy handling
    df = pd.DataFrame(data)
    return df

def calculate_metrics(logits, labels):
    _, predicted = torch.max(logits, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    precision = precision_score(labels.cpu(), predicted.cpu(), average='macro', zero_division=0)
    recall = recall_score(labels.cpu(), predicted.cpu(), average='macro', zero_division=0)
    f1 = f1_score(labels.cpu(), predicted.cpu(), average='macro', zero_division=0)
    return accuracy, precision, recall, f1

def build_loaders(dataframe, mode):
    dataset = CPIPDataset(dataframe)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader

def train_epoch(model, train_loader, optimizer, lr_scheduler, step, writer, epoch):
    model.train()
    loss_meter = AvgMeter()
    accuracy_meter = AvgMeter()
    precision_meter = AvgMeter()
    recall_meter = AvgMeter()
    f1_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))

    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        optimizer.zero_grad()
        loss, logits = model(batch)
        labels = torch.arange(logits.size(0)).long().to(logits.device)
        accuracy, precision, recall, f1 = calculate_metrics(logits, labels)

        loss.backward()
        optimizer.step()

        if step == "batch":
            lr_scheduler.step()

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)
        accuracy_meter.update(accuracy, count)
        precision_meter.update(precision, count)
        recall_meter.update(recall, count)
        f1_meter.update(f1, count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, train_accuracy=accuracy_meter.avg, lr=get_lr(optimizer))

    writer.add_scalar('Loss/train', loss_meter.avg, epoch)
    writer.add_scalar('Accuracy/train', accuracy_meter.avg, epoch)
    writer.add_scalar('Precision/train', precision_meter.avg, epoch)
    writer.add_scalar('Recall/train', recall_meter.avg, epoch)
    writer.add_scalar('F1/train', f1_meter.avg, epoch)
    writer.add_scalar('Learning Rate', get_lr(optimizer), epoch)

    return loss_meter.avg, accuracy_meter.avg

def valid_epoch(model, valid_loader, writer, epoch):
    model.eval()
    loss_meter = AvgMeter()
    accuracy_meter = AvgMeter()
    precision_meter = AvgMeter()
    recall_meter = AvgMeter()
    f1_meter = AvgMeter()
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))

    with torch.no_grad():
        for batch in tqdm_object:
            batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
            loss, logits = model(batch)
            labels = torch.arange(logits.size(0)).long().to(logits.device)
            accuracy, precision, recall, f1 = calculate_metrics(logits, labels)

            count = batch["image"].size(0)
            loss_meter.update(loss.item(), count)
            accuracy_meter.update(accuracy, count)
            precision_meter.update(precision, count)
            recall_meter.update(recall, count)
            f1_meter.update(f1, count)

            tqdm_object.set_postfix(valid_loss=loss_meter.avg, valid_accuracy=accuracy_meter.avg)

    writer.add_scalar('Loss/valid', loss_meter.avg, epoch)
    writer.add_scalar('Accuracy/valid', accuracy_meter.avg, epoch)
    writer.add_scalar('Precision/valid', precision_meter.avg, epoch)
    writer.add_scalar('Recall/valid', recall_meter.avg, epoch)
    writer.add_scalar('F1/valid', f1_meter.avg, epoch)

    return loss_meter.avg, accuracy_meter.avg

def train(data_path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=f"tensorboard/cpip_{timestamp}")
    
    # Define hyperparameters to log
    hparams = {key: value for key, value in vars(CFG).items() if not key.startswith('__')}

    data_df = prepare_data(data_path)
    # Split into train and validation sets
    train_df, valid_df = train_test_split(
        data_df, test_size=0.2, random_state=42, shuffle=True
    )

    train_loader = build_loaders(train_df, mode="train")
    valid_loader = build_loaders(valid_df, mode="valid")

    model = CPIPModel().to(CFG.device)

    # Check if best.pt exists and load it
    if os.path.exists(CFG.cpip_checkpoint_name) and CFG.resume_training:
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded model weights from {CFG.cpip_checkpoint_name} and resuming training...")

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

    for epoch in range(CFG.epochs):
        print(f"Epoch: {epoch + 1}")

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
            torch.save(model.state_dict(), "best.pt")
            print("Saved Best Model!")

    # Log hyperparameters and final metrics
    writer.add_hparams(hparam_dict=hparams, metric_dict=metrics)

    writer.close()

if __name__ == "__main__":
    data_path = "/scratch/zl3493/UNav-Dataset/810p/raw/000"
    train(data_path)
