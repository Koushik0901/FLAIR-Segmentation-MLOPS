import torch
import torchvision
import cv2
from PIL import Image


def inference(
    img_path,
    model_path="/home/koushik/workspace/Projects/MLOPS/saved_models/flair-segmentation.pt",
    save_path="./inference/result.png",
) -> None:

    img = Image.open(img_path)
    img = torchvision.transforms.functional.to_tensor(img).unsqueeze(0)
    model = torch.jit.load(model_path, map_location=torch.device("cpu"))
    prediction = model(img)
    torchvision.utils.save_image(prediction, save_path)


if __name__ == "__main__":
    inference(
        "saved_models/flair-segmentation.pt", "inference/1.tif", "inference/result.png"
    )
