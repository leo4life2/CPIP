import json
import os
import csv
import re

import config as CFG
import pandas as pd
import torch
from CPIP import CPIPModel
from dataset import CPIPDataset, get_transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils_file import AvgMeter, get_lr
import pdb
import ssl

ssl._create_default_https_context = ssl._create_unverified_context # workaround for no local CA trust store issue

USER = os.environ.get("USER")
os.environ['TORCH_HOME'] = f'/scratch/{USER}/pytorch'

def prepare_data(data_path, test_size=0.2, random_state=42):
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


def build_loaders(dataframe, mode):
    transforms = get_transforms(mode=mode)
    dataset = CPIPDataset(dataframe, transforms=transforms)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=True if mode == "train" else False,
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

def main():
    train_df, valid_df = prepare_data(CFG.data_path)

    train_loader = build_loaders(train_df, mode="train")
    valid_loader = build_loaders(valid_df, mode="valid")

    model = CPIPModel().to(CFG.device)
    
    # Check if best.pt exists and load it
    best_model_path = "best.pt"
    if os.path.exists(best_model_path) and CFG.resume_training:
        model.load_state_dict(torch.load(best_model_path))
        print("Loaded model weights from best.pt and resuming training...")
    
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=CFG.patience, factor=CFG.factor
    )
    step = "epoch"

    best_loss = float("inf")
    warmup_epochs = 5  # Number of epochs for the warm-up
    initial_lr = CFG.lr  # Target learning rate after warm-up

    with open('metrics.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Training Loss", "Validation Loss", "Training Accuracy", "Validation Accuracy"])
        
        for epoch in range(CFG.epochs):
            print(f"Epoch: {epoch + 1}")

            # Warm-up: Linearly increase learning rate
            if epoch < warmup_epochs:
                lr = initial_lr * (epoch + 1) / warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            model.train()
            train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
            
            model.eval()
            with torch.no_grad():
                valid_loss, valid_accuracy = valid_epoch(model, valid_loader)

            # Update learning rate with ReduceLROnPlateau after warm-up phase
            if epoch >= warmup_epochs:
                lr_scheduler.step(valid_accuracy)

            writer.writerow([epoch + 1, train_loss, valid_loss, train_accuracy, valid_accuracy])

            if valid_loss < best_loss:
                best_loss = valid_loss
                torch.save(model.state_dict(), "best.pt")
                print("Saved Best Model!")


if __name__ == "__main__":
    main()
