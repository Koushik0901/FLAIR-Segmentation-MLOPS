import torch.nn as nn

class EarlyStopping:
    def __init__(self, patience: int = 6, min_delta: float = 0, weights_path: str = 'weights.pth.tar') -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.weights_path = weights_path
        self.counter = 0
        self.best_loss = float('inf')
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            torch.save(model.state_dict(), self.weights_path)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def load_weights(self, model: nn.Module):
        return model.load_state_dict(torch.load(self.weights_path))