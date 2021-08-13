import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from dataset import get_loader
import segmentation_models_pytorch as smp
from early_stopping import EarlyStopping
from utils import save_checkpoint, load_checkpoint, save_predictions, iou, dice_coeff
import yaml
from tqdm import tqdm
from tabulate import tabulate

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loss Function
def BCE_dice(output, target, alpha=0.01):
    """Binary Cross-Entropy with soft dice loss"""
    bce = F.binary_cross_entropy_with_logits(output, target)
    soft_dice = 1 - dice_coeff(output, target).mean()
    return bce + alpha * soft_dice


# Training Loop
def train_loop(model, loader, optimizer, loss_fn, writer, step, scaler):
    losses = []

    for images, masks in tqdm(loader):
        images = images.to(DEVICE)
        masks = masks.float().to(DEVICE)
        # forward prop
        with torch.cuda.amp.autocast():
            preds = model(images).squeeze(1)
            loss = loss_fn(preds, masks)
        losses.append(loss.item())

        writer.add_scalar("Train Loss", loss.item(), global_step=step)
        step += 1
        # backward prop
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    train_loss = sum(losses) / len(losses)
    return train_loss, step


def valid_loop(model, loader, loss_fn, writer, step):
    losses = []
    running_dice = 0
    running_iou = 0

    with torch.no_grad():
        for images, masks in tqdm(loader):
            images = images.to(DEVICE)
            masks = masks.float().to(DEVICE)

            preds = model(images).squeeze(1)
            loss = loss_fn(preds, masks)
            # metrics
            running_iou += iou(preds, masks).sum().item()
            running_dice += dice_coeff(preds, masks).sum().item()
            losses.append(loss.item())

            writer.add_scalar("Val Loss", loss.item(), global_step=step)
            step += 1

    val_loss = sum(losses) / len(losses)
    val_iou = running_iou / len(loader.dataset)
    val_dice = running_dice / len(loader.dataset)
    return val_loss, val_iou, val_dice, step


def main(config):
    train_loader, valid_loader, test_loader = get_loader(config)

    model = smp.FPN(
        encoder_name=config["model"]["encoder_name"],
        encoder_weights=config["model"]["encoder_weights"],
        in_channels=config["model"]["in_channels"],
        classes=config["model"]["out_channels"],
        activation=None,
    )
    model = model.to(DEVICE)
    if config["train"]["load_checkpoint"]:
        load_checkpoint(config["train"]["checkpoint_path"], model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.2, patience=3, verbose=True
    )
    early_stopping = EarlyStopping(patience=6)
    loss_fn = BCE_dice
    scaler = torch.cuda.amp.GradScaler()
    writer = SummaryWriter()
    step = 0

    best_dice = 0.00
    for epoch in range(config["train"]["num_epochs"]):
        train_loss, step = train_loop(
            model, train_loader, optimizer, loss_fn, writer, step, scaler
        )
        val_loss, val_iou, val_dice, step = valid_loop(
            model, valid_loader, loss_fn, writer, step
        )

        scheduler.step(val_loss)
        if early_stopping(val_loss, model):
            early_stopping.load_weights(model)
            break

        writer.add_scalar("VAL LOSS", val_loss, global_step=step)
        writer.add_scalar("VAL IOU", val_iou, global_step=step)
        writer.add_scalar("VAL DICE", val_dice, global_step=step)

        if val_dice > best_dice:
            best_dice = val_dice

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "notes": f"EPOCH: {epoch}, IOU: {val_iou}, DICE COEFF: {val_dice}",
            }
            save_checkpoint(checkpoint, config["train"]["checkpoint_path"])
            save_predictions(test_loader, model, folder="saved_images", device="cuda")

        table = [
            ["Epoch", epoch],
            ["Train Loss", train_loss],
            ["Val Loss", val_loss],
            ["Val IOU", val_iou],
            ["Val Dice", val_dice],
            ["Best Val Loss", best_val_loss],
        ]
        print(tabulate(table))

    print("Done Training!")


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    main(config)
