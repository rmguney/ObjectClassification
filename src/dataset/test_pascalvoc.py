import matplotlib.pyplot as plt
import torchvision
from torchvision.datasets import VOCDetection
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch

# Constants
DATA_DIR = "data/pascal_voc"
VOC_YEAR = "2012"
BATCH_SIZE = 16

def test_pascalvoc_dataset():
    """
    Test the PASCAL VOC dataset by loading and visualizing a batch of images and annotations.
    """
    # Define transformations for the dataset
    transform = transforms.Compose([
        transforms.Resize((300, 300)),  # Resize images to 300x300
        transforms.ToTensor(),          # Convert images to PyTorch tensors
    ])

    # Load the PASCAL VOC dataset
    dataset = VOCDetection(
        root=DATA_DIR,
        year=VOC_YEAR,
        image_set="train",
        transform=transform,
        target_transform=None  # Can be set for transforming annotations
    )

    # DataLoader for the dataset
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # Function to display a batch of images with annotations
    def show_images(images, annotations):
        # Create a grid of images
        grid_img = torchvision.utils.make_grid(images, nrow=4)
        plt.figure(figsize=(10, 5))
        plt.imshow(grid_img.permute(1, 2, 0))  # Convert CHW to HWC
        plt.axis('off')
        plt.title("Sample Images from PASCAL VOC")
        plt.show()

        # Display the annotations for the batch
        for i, annotation in enumerate(annotations):
            print(f"Image {i+1} Annotations: {annotation['annotation']['object']}")

    # Display a batch of images and annotations
    print("Visualizing a batch from the PASCAL VOC dataset...")
    for images, annotations in data_loader:
        print(f"Batch size: {len(images)}, Annotations loaded.")
        show_images(images, annotations)
        break

def collate_fn(batch):
    """
    Custom collate function to handle variable-size annotations in the VOC dataset.
    """
    images, annotations = zip(*batch)
    return torch.stack(images), annotations

if __name__ == "__main__":
    test_pascalvoc_dataset()
