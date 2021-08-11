import torch
import yaml
from dataset import get_loader
from tqdm import tqdm


def get_mean_std(loader):
    # VAR[X] = E[X**2] - E[X]**2
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(loader):
        data = data
        data = data.type(torch.FloatTensor)
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    return mean, std


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    train_loader, _ = get_loader(config)
    mean, std = get_mean_std(train_loader)
    print(f"mean: {mean}, std: {std}")
