import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader
from pet_dataset import OxfordPetsDataset
from torch import nn
from PIL import Image

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the dataset and DataLoader
dataset = OxfordPetsDataset(root_dir='data/images', transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Load MobileNet model and modify it for binary classification
from torchvision.models import MobileNet_V2_Weights
model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model = model.to(device)  # Move model to the selected device
model.eval()  # Set model to evaluation mode

# Test with one batch of data
def test_mobilenet():
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)  # Move data to selected device
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            print("Predicted labels:", predicted.cpu().numpy())
            print("True labels:", labels.cpu().numpy())
            break  # Only test one batch

if __name__ == "__main__":
    test_mobilenet()
