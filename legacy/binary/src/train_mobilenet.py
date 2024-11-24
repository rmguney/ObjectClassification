import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torch import nn, optim
from pet_dataset import OxfordPetsDataset

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
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Load MobileNet model with the updated weights argument and adjust for binary classification
from torchvision.models import MobileNet_V2_Weights
model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model = model.to(device)  # Move model to the selected device

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_mobilenet(epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)  # Move data to selected device

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Print statistics per epoch
        epoch_loss = running_loss / len(dataloader)
        epoch_accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

    print("MobileNet training complete.")

# Run the training function
if __name__ == "__main__":
    train_mobilenet()
