import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from dataset import get_loader
from model import UNET
from loss import DiceLoss
from utils import (
    accuracy_and_dice_score,
    save_checkpoint,
    load_checkpoint,
    save_predictions,
)
import yaml
from tqdm import tqdm
from tabulate import tabulate

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_loop(model, loader, optimizer, loss_fn, writer, step, scaler):
    losses = []
    for images, masks in tqdm(loader):
        images = images.to(DEVICE)
        masks = masks.float().to(DEVICE)

        with torch.cuda.amp.autocast():
            preds = model(images)
            loss = loss_fn(preds, masks)
        losses.append(loss.item())
        writer.add_scalar("Train Loss", loss.item(), global_step=step)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    avg_loss = sum(losses) / len(losses)
    return avg_loss, step


def main(config):
    train_loader, val_loader = get_loader(config)
    model = UNET(config["model"]["in_channels"], config["model"]["out_channels"])
    model = model.to(DEVICE)
    if config["train"]["load_checkpoint"]:
        load_checkpoint(config["train"]["checkpoint_path"], model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["train"]["lr"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, verbose=True)
    loss_fn = DiceLoss()
    scaler = torch.cuda.amp.GradScaler()
    writer = SummaryWriter()
    step = 0

    epochs = config["train"]["num_epochs"]

    best_dice_loss = 100
    for epoch in range(epochs):
        train_loss, step = train_loop(model, train_loader, optimizer, loss_fn, writer, step, scaler)
        scheduler.step(train_loss)

        accuracy, dice_score = accuracy_and_dice_score(model, val_loader, DEVICE)
        writer.add_scalar('dice_score', dice_score, global_step=step)
        writer.add_scalar('accuracy', accuracy, global_step=step)

        if train_loss < best_dice_loss:
            best_dice_loss = train_loss

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, config["train"]["checkpoint_path"])
            save_predictions(val_loader, model, folder="saved_images", device=DEVICE)

        table = [
            ["Epoch", epoch],
            ["Dice Loss", train_loss],
            ["Val Accuracy", accuracy],
            ['Dice Score', dice_score],
            ['Best Dice Loss', best_dice_loss],
        ]
        print(tabulate(table))
    
    print("Done Training!")


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    main(config)
