import torch

def save_checkpoint(state, filename=None):
    print('--> Saving checkpoint...')
    torch.save(state, filename)

def load_checkpoint(filename, model):
    print('--> Loading checkpoint...')
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])


def accuracy_and_dice_score(model, loader, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.unsqueeze(1).to(device)
            preds = torch.round(torch.sigmoid(model(x)))
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds+ y).sum() + 1e-8)
    
    accuracy = num_correct/num_pixels*100:.2f
    model.train()
    return accuracy, dice_score/len(loader)
    
