import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_cifar10_dataloaders(batch_size=16, val_split=0.2):
    print("Initializing CIFAR-10 dataloaders...")

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize images to match input size of pre-trained models
        transforms.ToTensor(),         # Convert images to PyTorch tensors
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize to ImageNet mean/std
    ])

    # Load training dataset
    print("Loading training dataset...")
    train_dataset = datasets.CIFAR10(
        root="data/cifar10",
        train=True,
        transform=transform,
        download=True
    )
    print(f"Training dataset loaded with {len(train_dataset)} samples.")

    # Split train_dataset into training and validation sets
    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    print(f"Training subset: {train_size} samples, Validation subset: {val_size} samples.")

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    print(f"Training loader: {len(train_loader)} batches, Validation loader: {len(val_loader)} batches.")

    # Load test dataset
    print("Loading test dataset...")
    test_dataset = datasets.CIFAR10(
        root="data/cifar10",
        train=False,
        transform=transform,
        download=True
    )
    print(f"Test dataset loaded with {len(test_dataset)} samples.")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"Test loader: {len(test_loader)} batches.")

    return train_loader, val_loader, test_loader
