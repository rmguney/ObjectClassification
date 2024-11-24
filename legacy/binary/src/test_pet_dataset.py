from torch.utils.data import DataLoader
from torchvision import transforms
from pet_dataset import OxfordPetsDataset

# Define transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the dataset and DataLoader
dataset = OxfordPetsDataset(root_dir='data/images', transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Test dataset loading
def test_dataset():
    for images, labels in dataloader:
        print("Batch of images:", images.shape)  # Should be [batch_size, 3, 128, 128]
        print("Batch of labels:", labels)  # Binary labels for each image in the batch
        break  # Only test one batch

if __name__ == "__main__":
    test_dataset()
