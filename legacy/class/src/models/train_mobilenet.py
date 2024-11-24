import torch
from torchvision import models
from torch import nn, optim
from src.utils.logger import log_message, log_simple_message
from src.utils.plot_training import plot_training_results
from utils.benchmark import Heartbeat
import os
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_mobilenet(train_loader, epochs=5, learning_rate=0.001):
    model_name = "MobileNet"
    mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    mobilenet.classifier[1] = nn.Linear(mobilenet.classifier[1].in_features, 10)  # Adjust for CIFAR-10
    mobilenet = mobilenet.to(device)

    optimizer = optim.Adam(mobilenet.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    heartbeat = Heartbeat(timeout=600, model_name=model_name)

    mobilenet.train()
    losses, accuracies = [], []

    start_time = time.time()
    log_message(f"Starting {model_name} training with {epochs} epochs, learning rate {learning_rate}.", model_name, epochs, learning_rate)

    for epoch in range(epochs):
        epoch_start = time.time()
        running_loss = 0.0
        correct, total = 0, 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            batch_start = time.time()
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = mobilenet(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Check heartbeat
            try:
                heartbeat.check(batch_start)
            except TimeoutError as e:
                log_message(str(e), model_name, epochs, learning_rate)
                raise

            log_message(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], "
                        f"Loss: {loss.item():.4f}, Time: {time.time() - batch_start:.2f} sec", model_name, epochs, learning_rate)

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        losses.append(epoch_loss)
        accuracies.append(epoch_accuracy)
        epoch_time = time.time() - epoch_start

        log_message(f"{model_name} - Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, "
                    f"Accuracy: {epoch_accuracy:.2f}%, Time: {epoch_time:.2f} sec", model_name, epochs, learning_rate)

    total_time = time.time() - start_time
    average_epoch_time = total_time / epochs
    log_message(f"{model_name} training completed in {total_time:.2f} seconds.", model_name, epochs, learning_rate)

    # Log simple summary
    parameter_count = sum(p.numel() for p in mobilenet.parameters())
    summary = f"Parameters: {parameter_count}\n"
    for loss, acc in zip(losses, accuracies):
        summary += f"Loss: {loss:.6f}, Accuracy: {acc:.2f}%\n"
    summary += f"\nTotal Training Time: {total_time:.2f} seconds\n"
    summary += f"Average Time per Epoch: {average_epoch_time:.2f} seconds\n"
    log_simple_message(summary, model_name, epochs, learning_rate)

    plot_training_results(model_name, losses, accuracies, epochs, learning_rate, save_path="plots")
