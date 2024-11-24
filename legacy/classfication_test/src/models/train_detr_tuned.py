import torch
from transformers import DetrForObjectDetection, DetrConfig, DetrImageProcessor
from torch import nn, optim
from src.utils.logger import (
    log_message, log_classwise_accuracy, log_confusion_matrix, log_gradients, log_f1_score, log_precision_recall_curve
)
from src.utils.plot_training import plot_training_results
from src.utils.heartbeat import Heartbeat
import os
import time
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_detr(train_loader, val_loader, epochs=5, learning_rate=0.001):
    model_name = "DETR_Tuned"
    best_val_accuracy = 0.0

    # Load DETR model with custom classification head
    config = DetrConfig.from_pretrained("facebook/detr-resnet-50")
    config.num_labels = 10  # Adjust for CIFAR-10 classes
    detr = DetrForObjectDetection(config)
    detr.class_labels_classifier = nn.Linear(detr.class_labels_classifier.in_features, 10)
    detr = detr.to(device)

    optimizer = optim.AdamW(detr.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    heartbeat = Heartbeat(timeout=600, model_name=model_name)

    # Define a normalization transform (normalize CIFAR-10 images to [0, 1])
    normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Typical ImageNet normalization

    training_losses, training_accuracies = [], []
    validation_losses = []  # Fixed initialization
    validation_accuracies = []  # Fixed initialization
    gradient_norms = []

    start_time = time.time()
    log_message(f"Starting {model_name} training with {epochs} epochs, learning rate {learning_rate}.", model_name, epochs, learning_rate)

    for epoch in range(epochs):
        epoch_start = time.time()
        log_message(f"Starting epoch {epoch+1}/{epochs}", model_name, epochs, learning_rate)

        detr.train()
        running_loss, correct, total = 0.0, 0, 0
        layer_gradient_norms = []

        for batch_idx, (images, labels) in enumerate(train_loader):
            batch_start = time.time()
            log_message(f"Processing batch {batch_idx+1}/{len(train_loader)}", model_name)

            # Apply normalization to the images directly
            images = images.to(device)
            images = normalize_transform(images)  # Apply normalization to tensor images
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = detr(images)
            logits = outputs.logits[:, 0, :]
            loss = criterion(logits, labels)
            loss.backward()

            # Log gradient norms for each layer
            for name, param in detr.named_parameters():
                if param.grad is not None:
                    layer_gradient_norms.append(torch.norm(param.grad).item())

            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            heartbeat.check(batch_start)
            log_message(f"Batch {batch_idx+1} processed in {time.time() - batch_start:.2f} sec", model_name)

        # Aggregate and store average gradient norms for the epoch
        gradient_norms.append(sum(layer_gradient_norms) / len(layer_gradient_norms))

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        training_losses.append(epoch_loss)
        training_accuracies.append(epoch_accuracy)

        log_message(f"{model_name} - Epoch [{epoch+1}/{epochs}], Training Loss: {epoch_loss:.4f}, "
                    f"Training Accuracy: {epoch_accuracy:.2f}%", model_name, epochs, learning_rate)

        detr.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_labels, all_predictions = [], []
        all_probabilities = []  # Store probabilities for precision-recall curve

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader):
                log_message(f"Validating batch {batch_idx+1}/{len(val_loader)}", model_name)
                images = images.to(device)
                images = normalize_transform(images)  # Apply normalization to tensor images
                labels = labels.to(device)

                outputs = detr(images)
                logits = outputs.logits[:, 0, :]
                loss = criterion(logits, labels)

                val_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu())
                all_predictions.extend(predicted.cpu())
                all_probabilities.extend(torch.nn.functional.softmax(logits, dim=-1).cpu())  # Store probabilities

        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_accuracy = 100 * val_correct / val_total
        validation_losses.append(epoch_val_loss)
        validation_accuracies.append(epoch_val_accuracy)

        # Log additional metrics
        class_names = train_loader.dataset.dataset.classes  # Access classes from the original dataset
        log_classwise_accuracy(torch.tensor(all_labels), torch.tensor(all_predictions), model_name, class_names)
        log_confusion_matrix(torch.tensor(all_labels), torch.tensor(all_predictions), model_name, num_classes=10)
        log_f1_score(torch.tensor(all_labels), torch.tensor(all_predictions), model_name)

        # Log Precision-Recall Curve
        all_probabilities = torch.stack(all_probabilities)  # Convert list of tensors to a single tensor
        log_precision_recall_curve(torch.tensor(all_labels), all_probabilities, model_name)

        if epoch_val_accuracy > best_val_accuracy:
            best_val_accuracy = epoch_val_accuracy
            torch.save(detr.state_dict(), f"saved_models/{model_name}_best_tuned.pth")

        log_message(f"{model_name} - Epoch [{epoch+1}/{epochs}], Validation Loss: {epoch_val_loss:.4f}, "
                    f"Validation Accuracy: {epoch_val_accuracy:.2f}%", model_name, epochs, learning_rate)

        scheduler.step()

        epoch_time = time.time() - epoch_start
        log_message(f"{model_name} - Epoch [{epoch+1}/{epochs}] completed in {epoch_time:.2f} sec", model_name, epochs, learning_rate)

        torch.save(detr.state_dict(), f"saved_models/{model_name}_epoch_{epoch+1}_tuned.pth")

    total_time = time.time() - start_time
    log_message(f"{model_name} training completed in {total_time:.2f} seconds.", model_name, epochs, learning_rate)

    plot_training_results(
        model_name, training_losses, training_accuracies, validation_losses, validation_accuracies,
        epochs, learning_rate, additional_metrics={"Gradient Norms": gradient_norms}
    )
