import torch
from transformers import DetrForObjectDetection, DetrImageProcessor
from torch.utils.data import DataLoader
from torch import nn, optim
from pet_dataset import OxfordPetsDataset
from torchvision import transforms

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define transformations without normalization (handled by processor)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Load the dataset and DataLoader
dataset = OxfordPetsDataset(root_dir='data/images', transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Load the Hugging Face DETR model and processor
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

# Adjust for binary classification (cats and dogs)
model.class_labels_classifier = nn.Linear(model.class_labels_classifier.in_features, 2)
model = model.to(device)  # Move model to the selected device

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
def train_detr(epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in dataloader:
            # Move data to selected device and preprocess images
            images = [img.cpu() for img in images]  # DETR processor works on CPU by default
            labels = labels.to(device)
            inputs = processor(images=images, return_tensors="pt", do_rescale=False).to(device)

            optimizer.zero_grad()
            outputs = model(**inputs)

            # Use only the primary object logits for classification
            logits = outputs.logits[:, 0, :]  # Taking the first detected object
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Print statistics per epoch
        epoch_loss = running_loss / len(dataloader)
        epoch_accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

    print("DETR training complete.")

# Run the training function
if __name__ == "__main__":
    train_detr()
