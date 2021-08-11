import torch
import segmentation_models_pytorch as smp
import yaml

def trace(model):
    model.eval()
    x = torch.rand(32, 3, 256, 256)
    traced = torch.jit.trace(model, (x))
    return traced

def main(config):
    print(f"loading model from {config['train']['checkpoint_path']}")
    checkpoint = torch.load(config['train']['checkpoint_path'], map_location=torch.device('cpu'))
    model = smp.FPN(
        encoder_name=config["model"]["encoder_name"],
        encoder_weights=config["model"]["encoder_weights"],
        in_channels=config["model"]["in_channels"],
        classes=config["model"]["out_channels"],
        activation=None,
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    print("tracing model...")
    traced_model = trace(model)
    print(f"saving to {config['train']['optimized_save_path']}")
    traced_model.save(config['train']['optimized_save_path'])
    print("Done!")

if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    main(config)