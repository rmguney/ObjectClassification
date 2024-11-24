import torch
from transformers import DetrForObjectDetection, DetrImageProcessor
from torch import optim
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

# Pascal VOC class labels
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

def sanitize_bounding_boxes(targets, voc_classes):
    """
    Convert Pascal VOC-style annotations to COCO-style annotations compatible with DETR.
    """
    sanitized_targets = []
    for image_id, target in enumerate(targets):
        annotations = []
        for obj in target["annotation"]["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            ymin = float(obj["bndbox"]["ymin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymax = float(obj["bndbox"]["ymax"])

            if xmax > xmin and ymax > ymin:
                annotations.append({
                    "bbox": [xmin, ymin, xmax, ymax],  # [x_min, y_min, x_max, y_max]
                    "area": (xmax - xmin) * (ymax - ymin),
                    "category_id": voc_classes.index(obj["name"]),
                    "iscrowd": 0
                })
            else:
                logger.warning(f"Invalid bounding box: {xmin}, {ymin}, {xmax}, {ymax}")

        if annotations:
            sanitized_targets.append({"image_id": image_id, "annotations": annotations})
    return sanitized_targets

def train_detr(train_loader, val_loader, epochs=5, learning_rate=0.0001):
    model_name = "DETR_PascalVOC"
    best_val_mAP = 0.0

    # Initialize DETR model and processor
    detr = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", do_rescale=False)

    optimizer = optim.AdamW(detr.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    heartbeat = Heartbeat(timeout=600, model_name=model_name)

    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        logger.info(f"Starting epoch {epoch + 1}/{epochs}...")

        # Training phase
        detr.train()
        running_loss = 0.0
        num_frames = 0

        for batch_idx, (images, targets) in enumerate(train_loader):
            batch_start = time.time()
            heartbeat.check(batch_start)

            images = [image.to(device) for image in images]
            target_batch = sanitize_bounding_boxes(targets, VOC_CLASSES)

            inputs = processor(images=images, annotations=target_batch, return_tensors="pt").to(device)

            optimizer.zero_grad()
            outputs = detr(**inputs)
            loss = outputs.loss

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
        detr.eval()
        val_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for images, targets in val_loader:
                images = [image.to(device) for image in images]
                target_batch = sanitize_bounding_boxes(targets, VOC_CLASSES)

                inputs = processor(images=images, annotations=target_batch, return_tensors="pt").to(device)

                outputs = detr(**inputs)
                val_loss += outputs.loss.item()

                all_predictions.extend(outputs.logits.argmax(dim=-1).cpu().numpy())
                all_targets.extend([t["annotations"] for t in target_batch])

        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_mAP = calculate_mean_average_precision(all_predictions, all_targets)

        logger.info(f"Epoch {epoch + 1}/{epochs} - Validation Loss: {epoch_val_loss:.4f}, Validation mAP: {epoch_val_mAP:.4f}")

        if epoch_val_mAP > best_val_mAP:
            best_val_mAP = epoch_val_mAP
            torch.save(detr.state_dict(), f"saved_models/{model_name}_best.pth")

        scheduler.step()

    total_time = track_training_time(start_time)
    logger.info(f"Training completed in {total_time:.2f} seconds.")

    return detr

if __name__ == "__main__":
    from src.dataset.initialize_pascalvoc import create_dataloaders

    # Load Pascal VOC dataloaders
    train_loader, val_loader = create_dataloaders(batch_size=16)

    # Train the DETR model
    detr_model = train_detr(train_loader, val_loader, epochs=10, learning_rate=0.0001)
