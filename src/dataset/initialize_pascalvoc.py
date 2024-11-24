import os
import tarfile
import torchvision
from torchvision.datasets import VOCDetection
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, random_split

# Define paths
DATA_DIR = "data/pascal_voc"
VOC_YEAR = "2012"

def download_and_prepare_voc(data_dir=DATA_DIR, year=VOC_YEAR):
    """
    Downloads and extracts the PASCAL VOC dataset.
    """
    # Create data directory if not exists
    os.makedirs(data_dir, exist_ok=True)

    # Download VOC dataset
    print(f"Downloading PASCAL VOC {year} dataset...")
    torchvision.datasets.VOCDetection(
        root=data_dir,
        year=year,
        image_set="train",
        download=True,
    )
    print("Download complete.")

def create_dataloaders(batch_size=16, val_split=0.2, data_dir=DATA_DIR, year=VOC_YEAR):
    """
    Prepares train and validation DataLoaders for PASCAL VOC.
    """
    # Define transformations for data augmentation and preprocessing
    transform = transforms.Compose([
        transforms.Resize((300, 300)),  # Resize images to 300x300
        transforms.ToTensor(),          # Convert images to PyTorch tensors
    ])

    # Load VOC dataset
    dataset = VOCDetection(
        root=data_dir,
        year=year,
        image_set="train",
        transform=transform,
    )
    
    # Split dataset into training and validation sets
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader

def collate_fn(batch):
    """
    Custom collate function to handle variable-size targets in the VOC dataset.
    """
    images, targets = zip(*batch)
    return list(images), list(targets)

if __name__ == "__main__":
    # Download and prepare PASCAL VOC
    download_and_prepare_voc()

    # Create data loaders
    train_loader, val_loader = create_dataloaders()
    print("Data loaders created successfully.")
