import torch
from transformers import DetrForObjectDetection, DetrImageProcessor
from torchvision import models, transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, Label, Button, OptionMenu, StringVar
from torch import nn

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_model(model_choice):
    if model_choice == "MobileNet":
        from torchvision.models import MobileNet_V2_Weights
        model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    elif model_choice == "DETR":
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        model.class_labels_classifier = nn.Linear(model.class_labels_classifier.in_features, 2)
    model = model.to(device)  # Move model to the selected device
    model.eval()  # Set model to evaluation mode
    return model

def preprocess_image(image_path, model_choice):
    if model_choice == "MobileNet":
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image = Image.open(image_path).convert("RGB")
        return transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device
    elif model_choice == "DETR":
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        return inputs

def predict_image(model, image_tensor, model_choice):
    with torch.no_grad():
        if model_choice == "MobileNet":
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            label = "Dog" if predicted.item() == 1 else "Cat"
        elif model_choice == "DETR":
            outputs = model(**image_tensor)
            logits = outputs.logits[:, 0, :]  # Use the primary object logits for binary classification
            predicted_class = logits.argmax(dim=-1).item()
            label = "Dog" if predicted_class == 1 else "Cat"
    return label

def select_image():
    global img_label, prediction_label

    # Open file dialog to select an image
    image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if not image_path:
        return

    # Load the chosen model
    model = load_model(model_choice.get())

    # Load and preprocess the selected image
    image_tensor = preprocess_image(image_path, model_choice.get())
    label = predict_image(model, image_tensor, model_choice.get())

    # Display the image in the GUI
    image = Image.open(image_path)
    image.thumbnail((200, 200))
    img = ImageTk.PhotoImage(image)
    img_label.config(image=img)
    img_label.image = img

    # Show prediction result
    prediction_label.config(text=f"Predicted Label: {label}")

# Initialize GUI
root = tk.Tk()
root.title("Image Classifier")
root.geometry("300x500")

# Model selection dropdown
model_choice = StringVar(root)
model_choice.set("MobileNet")  # default value
options = ["MobileNet", "DETR"]
model_selector = OptionMenu(root, model_choice, *options)
model_selector.pack(pady=10)

# Add UI elements
img_label = Label(root)
img_label.pack(pady=20)

select_button = Button(root, text="Select Image", command=select_image)
select_button.pack()

prediction_label = Label(root, text="Predicted Label: ", font=("Arial", 14))
prediction_label.pack(pady=20)

# Start the GUI
root.mainloop()
