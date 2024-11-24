import os
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

def plot_training_results(model_name, losses, accuracies, val_losses, val_accuracies, epochs, learning_rate,
                          additional_metrics=None, gradient_norms=None, save_path="plots"):
    """
    Plot training and validation results and additional metrics, saving all visualizations.

    :param model_name: Name of the model (e.g., "MobileNet", "DETR").
    :param losses: List of training losses over epochs.
    :param accuracies: List of training accuracies over epochs.
    :param val_losses: List of validation losses over epochs.
    :param val_accuracies: List of validation accuracies over epochs.
    :param epochs: Number of training epochs.
    :param learning_rate: Learning rate used for training.
    :param additional_metrics: Dictionary of additional metrics (e.g., class-wise accuracy, PR curves, F1 scores).
    :param gradient_norms: List of gradient norms per epoch.
    :param save_path: Directory to save the plots.
    """
    os.makedirs(save_path, exist_ok=True)
    rows = 3 + (len(additional_metrics) if additional_metrics else 0)
    plt.figure(figsize=(12, rows * 3))

    # Training Loss
    plt.subplot(rows, 1, 1)
    plt.plot(range(1, len(losses) + 1), losses, label="Training Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", linestyle="--")
    plt.title(f"{model_name} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Training Accuracy
    plt.subplot(rows, 1, 2)
    plt.plot(range(1, len(accuracies) + 1), accuracies, label="Training Accuracy")
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label="Validation Accuracy", linestyle="--")
    plt.title(f"{model_name} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    # Gradient Norms
    if gradient_norms:
        plt.subplot(rows, 1, 3)
        plt.plot(range(1, len(gradient_norms) + 1), gradient_norms, label="Gradient Norm")
        plt.title(f"{model_name} Gradient Norms")
        plt.xlabel("Epoch")
        plt.ylabel("Gradient Norm (L2)")
        plt.legend()

    # Additional Metrics
    if additional_metrics:
        metric_idx = 4
        for metric_name, metric_values in additional_metrics.items():
            plt.subplot(rows, 1, metric_idx)
            if metric_name == "Confusion Matrix":
                plt.imshow(metric_values, interpolation='nearest', cmap=plt.cm.Blues)
                plt.title(f"{model_name} {metric_name}")
                plt.colorbar()
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
            elif isinstance(metric_values, dict):
                for class_name, values in metric_values.items():
                    plt.plot(values['x'], values['y'], label=class_name)
                plt.title(f"{model_name} {metric_name}")
                plt.xlabel("Class")
                plt.ylabel(metric_name)
                plt.legend()
            else:
                plt.plot(range(1, len(metric_values) + 1), metric_values, label=metric_name)
                plt.title(f"{model_name} {metric_name}")
                plt.xlabel("Epoch")
                plt.ylabel(metric_name)
                plt.legend()
            metric_idx += 1

    plt.tight_layout()

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"{save_path}/{model_name}_epochs_{epochs}_lr_{learning_rate:.5f}_{current_time}_results.png"
    plt.savefig(plot_filename)
    plt.show()
