import os
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import config as CFG
from dataset import CPIPDataset

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

def prepare_data(data_path):
    # List all .png files in the directory
    image_files = sorted([f for f in os.listdir(data_path) if f.endswith(".png")])

    # Prepare data
    data = []
    for image_file in image_files:
        image_name, _ = os.path.splitext(image_file)
        full_image_path = os.path.join(data_path, image_file)
        _, x, y, heading = image_name.split("_")
        x, y, heading = float(x), float(y), float(heading)    

        data.append({"image": full_image_path, "location": [x, y, heading]})

    # Convert to DataFrame for easy handling
    df = pd.DataFrame(data)
    return df

def build_loaders(dataframe, mode):
    dataset = CPIPDataset(dataframe)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader

def calculate_metrics(logits, labels):
    _, predicted = torch.max(logits, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    precision = precision_score(labels.cpu(), predicted.cpu(), average='macro', zero_division=0)
    recall = recall_score(labels.cpu(), predicted.cpu(), average='macro', zero_division=0)
    f1 = f1_score(labels.cpu(), predicted.cpu(), average='macro', zero_division=0)
    return accuracy, precision, recall, f1

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