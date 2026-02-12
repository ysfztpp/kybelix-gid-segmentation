# src/utils.py
import torch
import matplotlib.pyplot as plt

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
    return total_loss / len(loader)

def visualize_step(model, loader, device):
    model.eval()
    images, masks = next(iter(loader))
    with torch.no_grad():
        output = model(images.to(device))
        pred = (torch.sigmoid(output) > 0.5).float()

    plt.figure(figsize=(10,5))
    plt.subplot(1,3,1); plt.imshow(images[0].permute(1,2,0)); plt.title("Görüntü")
    plt.subplot(1,3,2); plt.imshow(masks[0].squeeze(), cmap='gray'); plt.title("Gerçek Maske")
    plt.subplot(1,3,3); plt.imshow(pred[0].squeeze().cpu(), cmap='gray'); plt.title("Tahmin")
    plt.show()