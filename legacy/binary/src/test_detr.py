import torch
from transformers import DetrForObjectDetection, DetrImageProcessor
from torch.utils.data import DataLoader
from pet_dataset import OxfordPetsDataset
from torchvision import transforms

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define transformations without normalization
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),  # Only convert to tensor; skip normalization
])

# Load the dataset and DataLoader
dataset = OxfordPetsDataset(root_dir='data/images', transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Load the Hugging Face DETR model and processor
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = model.to(device)
model.eval()

# Test with one batch of data
def test_detr():
    with torch.no_grad():
        for images, labels in dataloader:
            # Move each image to CPU if on GPU as processor works with CPU by default
            images = [img.cpu() for img in images]
            
            # Preprocess images with the processor
            inputs = processor(images=images, return_tensors="pt", do_rescale=False).to(device)

            outputs = model(**inputs)

            # Extract predictions
            logits = outputs.logits  # Object classification logits
            pred_boxes = outputs.pred_boxes  # Predicted bounding boxes

            # Print results
            print("Logits:", logits)
            print("Predicted bounding boxes:", pred_boxes)

            # Convert predictions to class labels
            predicted_classes = logits.argmax(-1)  # Get the class with the highest score per prediction
            print("Predicted classes:", predicted_classes.cpu().numpy())
            print("True labels:", labels.cpu().numpy())
            break  # Only test one batch

if __name__ == "__main__":
    test_detr()
