import torch
import torchvision


def save_checkpoint(state, filename=None):
    print("--> Saving checkpoint...")
    torch.save(state, filename)


def load_checkpoint(filename, model):
    print("--> Loading checkpoint...")
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["model_state_dict"])


def dice_coeff(preds, labels, e=1e-7):
    preds = torch.where(preds > 0.5, 1, 0)
    labels = labels.byte()
    intersection = (preds & labels).float().sum((1, 2))
    return ((2 * intersection) + e) / (
        preds.float().sum((1, 2)) + labels.float().sum((1, 2)) + e
    )


def iou(preds, labels, e=1e-7):
    preds = torch.where(preds > 0.5, 1, 0)
    labels = labels.byte()
    intersection = (preds & labels).float().sum((1, 2))
    union = (preds | labels).float().sum((1, 2))

    iou = (intersection + e) / (union + e)
    return iou


def save_predictions(loader, model, folder="saved_images", device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = model(x)
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/{idx}.png")

    model.train()
