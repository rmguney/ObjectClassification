import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch import optim
from torchvision.transforms import transforms
from src.utils.benchmark import (
    calculate_mean_average_precision,
    calculate_fps,
    calculate_memory_usage,
    track_training_time,
    Heartbeat,
)
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PASCAL VOC class labels
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

def sanitize_bounding_boxes(targets):
    """
    Ensure all bounding boxes have positive height and width.
    Remove invalid bounding boxes.
    """
    sanitized_targets = []
    for target in targets:
        valid_boxes = []
        valid_labels = []

        for obj in target["annotation"]["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            ymin = float(obj["bndbox"]["ymin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymax = float(obj["bndbox"]["ymax"])

            # Check if the box dimensions are valid
            if xmax > xmin and ymax > ymin:
                valid_boxes.append([xmin, ymin, xmax, ymax])
                valid_labels.append(VOC_CLASSES.index(obj["name"]))

        # If no valid boxes remain, skip the target
        if valid_boxes:
            sanitized_targets.append({
                "boxes": torch.tensor(valid_boxes, dtype=torch.float32),
                "labels": torch.tensor(valid_labels, dtype=torch.int64),
            })

    return sanitized_targets

def train_mobilenet(train_loader, val_loader, epochs=5, learning_rate=0.0001):
    model_name = "MobileNet_PascalVOC"
    best_val_mAP = 0.0

    # Load pre-trained MobileNet-based Faster R-CNN model
    mobilenet = fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")  # Use updated weights parameter

    # Replace the box predictor head with one compatible with Pascal VOC's 21 classes
    num_classes = 21  # 20 Pascal VOC classes + background
    in_features = mobilenet.roi_heads.box_predictor.cls_score.in_features
    mobilenet.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    mobilenet = mobilenet.to(device)

    optimizer = optim.AdamW(mobilenet.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    # Data normalization (ImageNet mean/std)
    normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Initialize Heartbeat
    heartbeat = Heartbeat(timeout=600, model_name=model_name)

    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        logger.info(f"Starting epoch {epoch + 1}/{epochs}...")

        # Training phase
        mobilenet.train()
        running_loss = 0.0
        num_frames = 0

        for batch_idx, (images, targets) in enumerate(train_loader):
            batch_start = time.time()
            heartbeat.check(batch_start)

            # Move images to the device
            images = [normalize_transform(image).to(device) for image in images]

            # Sanitize bounding boxes and move targets to the device
            target_batch = sanitize_bounding_boxes(targets)
            target_batch = [{k: v.to(device) for k, v in t.items()} for t in target_batch]

            optimizer.zero_grad()
            loss_dict = mobilenet(images, target_batch)
            loss = sum(loss for loss in loss_dict.values())

            # Backpropagation
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_frames += len(images)

            logger.info(f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], "
                        f"Loss: {loss.item():.4f}, Time: {time.time() - batch_start:.2f} sec")

        epoch_loss = running_loss / len(train_loader)
        fps = calculate_fps(epoch_start, num_frames)
        memory_usage = calculate_memory_usage()

        logger.info(f"Epoch {epoch + 1}/{epochs} - Training Loss: {epoch_loss:.4f}, FPS: {fps:.2f}, "
                    f"Memory Usage: {memory_usage['current_memory_mb']:.2f} MB (Max: {memory_usage['max_memory_mb']:.2f} MB)")

        # Validation phase
        mobilenet.eval()
        val_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for images, targets in val_loader:
                images = [normalize_transform(image).to(device) for image in images]
                target_batch = sanitize_bounding_boxes(targets)
                target_batch = [{k: v.to(device) for k, v in t.items()} for t in target_batch]

                loss_dict = mobilenet(images, target_batch)
                val_loss += sum(loss for loss in loss_dict.values()).item()

                predictions = mobilenet(images)
                all_predictions.extend(predictions)
                all_targets.extend(target_batch)

        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_mAP = calculate_mean_average_precision(all_predictions, all_targets)

        logger.info(f"Epoch {epoch + 1}/{epochs} - Validation Loss: {epoch_val_loss:.4f}, Validation mAP: {epoch_val_mAP:.4f}")

        if epoch_val_mAP > best_val_mAP:
            best_val_mAP = epoch_val_mAP
            torch.save(mobilenet.state_dict(), f"saved_models/{model_name}_best.pth")

        scheduler.step()

    total_time = track_training_time(start_time)
    logger.info(f"Training completed in {total_time:.2f} seconds.")

    return mobilenet

if __name__ == "__main__":
    from src.dataset.initialize_pascalvoc import create_dataloaders

    # Load data loaders
    train_loader, val_loader = create_dataloaders(batch_size=16)

    # Train the MobileNet model
    mobilenet_model = train_mobilenet(train_loader, val_loader, epochs=10, learning_rate=0.0001)
