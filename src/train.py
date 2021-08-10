import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from dataset import get_loader
from model import UNET
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
    loop = tqdm(loader)
    for images, masks in loop:
        images = images.to(DEVICE)
        masks = masks.float().unsqueeze(1).to(DEVICE)

        with torch.cuda.amp.autocast():
            preds = model(images)
            loss = loss_fn(preds, masks)

        writer.add_scalar("Train Loss", loss.item(), global_step=step)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())
    return step


def main(config):
    train_loader, val_loader = get_loader(config)
    model = UNET(config["model"]["in_channels"], config["model"]["out_channels"])
    model = model.to(DEVICE)
    if config["train"]["load_checkpoint"]:
        load_checkpoint(config["train"]["checkpoint_path"], model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["train"]["lr"]))
    loss_fn = torch.nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler()
    writer = SummaryWriter()
    step = 0

    epochs = config["train"]["num_epochs"]

    for epoch in range(epochs):
        step = train_loop(model, train_loader, optimizer, loss_fn, writer, step, scaler)

        accuracy_and_dice_score(model, val_loader, DEVICE)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, config["train"]["checkpoint_path"])
        save_predictions(val_loader, model, folder="saved_images/", device=DEVICE)

    print("Done Training!")


if __name__ == "__main__":
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)
    main(config)
