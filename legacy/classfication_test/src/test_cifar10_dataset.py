from src.cifar10_dataset import get_cifar10_dataloaders
import matplotlib.pyplot as plt
import torchvision
import torch

def test_cifar10_dataset():
    # Get CIFAR-10 DataLoaders with batch size 16
    dataloaders = get_cifar10_dataloaders(batch_size=16)  # Adjusted to match consistency

    # Handle different return types
    if isinstance(dataloaders, tuple):
        if len(dataloaders) == 2:
            train_loader, test_loader = dataloaders
        elif len(dataloaders) == 3:
            train_loader, val_loader, test_loader = dataloaders
            print("Validation loader detected.")
        else:
            raise ValueError("Unexpected number of values returned by get_cifar10_dataloaders.")
    elif isinstance(dataloaders, dict):
        train_loader = dataloaders.get("train")
        val_loader = dataloaders.get("val")
        test_loader = dataloaders.get("test")
        print("Dataloader returned as a dictionary.")
    else:
        raise ValueError("get_cifar10_dataloaders returned an unsupported type.")

    # CIFAR-10 class names
    CIFAR10_CLASSES = [
        "Airplane", "Automobile", "Bird", "Cat", "Deer", 
        "Dog", "Frog", "Horse", "Ship", "Truck"
    ]

    # Function to display a batch of images
    def show_images(images, labels, classes):
        # Denormalize images
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        images = images.permute(0, 2, 3, 1)  # Convert CHW to HWC
        images = images * std + mean  # De-normalize the images
        images = images.clip(0, 1)  # Clip to valid range [0, 1]

        # Create and display grid
        grid_img = torchvision.utils.make_grid(images.permute(0, 3, 1, 2), nrow=4)  # Convert back to CHW
        plt.figure(figsize=(10, 5))
        plt.imshow(grid_img.permute(1, 2, 0))  # Convert CHW to HWC
        label_names = [classes[label] for label in labels]
        plt.title(f"Labels: {', '.join(label_names)}")
        plt.axis('off')
        plt.show()

    # Test the train loader
    print("Visualizing a batch from the train dataset...")
    for images, labels in train_loader:
        print(f"Batch size: {images.size()}, Labels: {labels.tolist()}")
        show_images(images, labels, CIFAR10_CLASSES)
        break

    # Test the test loader
    if test_loader:
        print("Visualizing a batch from the test dataset...")
        for images, labels in test_loader:
            print(f"Batch size: {images.size()}, Labels: {labels.tolist()}")
            show_images(images, labels, CIFAR10_CLASSES)
            break

if __name__ == "__main__":
    test_cifar10_dataset()
