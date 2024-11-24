import torch
from torchvision import models
from torch import nn, optim
from src.utils.logger import (
    log_message, log_classwise_accuracy, log_confusion_matrix, log_gradients, log_f1_score
)
from src.utils.plot_training import plot_training_results
from src.utils.heartbeat import Heartbeat
import os
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_mobilenet(train_loader, val_loader, epochs=5, learning_rate=0.001):
    model_name = "MobileNet_Tuned"
    best_val_accuracy = 0.0

    # Load MobileNet model and replace the classifier head
    mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    mobilenet.classifier[1] = nn.Linear(mobilenet.classifier[1].in_features, 10)
    mobilenet = mobilenet.to(device)

    optimizer = optim.AdamW(mobilenet.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    heartbeat = Heartbeat(timeout=600, model_name=model_name)

    training_losses, training_accuracies = [], []
    validation_losses, validation_accuracies = [], []
    gradient_norms = []

    start_time = time.time()
    log_message(f"Starting {model_name} training with {epochs} epochs, learning rate {learning_rate}.", model_name, epochs, learning_rate)

    for epoch in range(epochs):
        epoch_start = time.time()
        log_message(f"Starting epoch {epoch+1}/{epochs}", model_name, epochs, learning_rate)

        mobilenet.train()
        running_loss, correct, total = 0.0, 0, 0
        layer_gradient_norms = []

        for batch_idx, (images, labels) in enumerate(train_loader):
            batch_start = time.time()
            log_message(f"Processing batch {batch_idx+1}/{len(train_loader)}", model_name)

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = mobilenet(images)
            loss = criterion(outputs, labels)
            log_message(f"Batch {batch_idx+1} loss computed: {loss.item():.4f}", model_name)  # Log loss computation
            loss.backward()

            # Log gradient norms for each layer
            for name, param in mobilenet.named_parameters():
                if param.grad is not None:
                    layer_gradient_norms.append(torch.norm(param.grad).item())

            optimizer.step()
            log_message(f"Batch {batch_idx+1} optimizer step completed", model_name)  # Log optimizer step

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
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

        mobilenet.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_labels, all_predictions = [], []

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader):
                log_message(f"Validating batch {batch_idx+1}/{len(val_loader)}", model_name)
                images, labels = images.to(device), labels.to(device)
                outputs = mobilenet(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu())
                all_predictions.extend(predicted.cpu())

        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_accuracy = 100 * val_correct / val_total
        validation_losses.append(epoch_val_loss)
        validation_accuracies.append(epoch_val_accuracy)

        # Log additional metrics
        class_names = train_loader.dataset.dataset.classes  # Access classes from the original dataset
        log_classwise_accuracy(torch.tensor(all_labels), torch.tensor(all_predictions), model_name, class_names)
        log_confusion_matrix(torch.tensor(all_labels), torch.tensor(all_predictions), model_name, num_classes=10)
        log_f1_score(torch.tensor(all_labels), torch.tensor(all_predictions), model_name)

        if epoch_val_accuracy > best_val_accuracy:
            best_val_accuracy = epoch_val_accuracy
            torch.save(mobilenet.state_dict(), f"saved_models/{model_name}_best_tuned.pth")

        log_message(f"{model_name} - Epoch [{epoch+1}/{epochs}], Validation Loss: {epoch_val_loss:.4f}, "
                    f"Validation Accuracy: {epoch_val_accuracy:.2f}%", model_name, epochs, learning_rate)

        scheduler.step()

        epoch_time = time.time() - epoch_start
        log_message(f"{model_name} - Epoch [{epoch+1}/{epochs}] completed in {epoch_time:.2f} sec", model_name, epochs, learning_rate)

        torch.save(mobilenet.state_dict(), f"saved_models/{model_name}_epoch_{epoch+1}_tuned.pth")

    total_time = time.time() - start_time
    log_message(f"{model_name} training completed in {total_time:.2f} seconds.", model_name, epochs, learning_rate)

    plot_training_results(
        model_name, training_losses, training_accuracies, validation_losses, validation_accuracies,
        epochs, learning_rate, additional_metrics={"Gradient Norms": gradient_norms}
    )
