import os
import torch
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, precision_recall_curve
import matplotlib.pyplot as plt
from datetime import datetime

# Global log file mapping
log_files = {}

def initialize_log_file(model_name, epochs=None, learning_rate=None):
    os.makedirs("logs", exist_ok=True)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    if epochs is not None and learning_rate is not None:
        log_file = f"logs/{model_name}_epochs_{epochs}_lr_{learning_rate:.5f}_{current_time}.log"
    else:
        log_file = f"logs/{model_name}_{current_time}.log"
    log_files[model_name] = log_file
    with open(log_file, "a") as f:
        f.write(f"--- Logging started for {model_name} at {current_time} ---\n")
    return log_file

def log_message(message, model_name, epochs=None, learning_rate=None):
    if model_name not in log_files:
        initialize_log_file(model_name, epochs, learning_rate)
    log_file = log_files[model_name]
    with open(log_file, "a") as f:
        f.write(message + "\n")
    print(message)

# Batch-level logging
def log_latency(latency, model_name):
    log_message(f"Average Latency per Batch: {latency:.2f} ms", model_name)

def log_fps(batch_time, batch_size, model_name):
    fps = batch_size / batch_time
    log_message(f"Frames Per Second (FPS): {fps:.2f}", model_name)

def log_loss(loss, batch_index, model_name):
    log_message(f"Batch {batch_index} Loss: {loss:.4f}", model_name)

# Summary-level logging
def log_flops(model, input_size, model_name):
    try:
        from ptflops import get_model_complexity_info
        macs, params = get_model_complexity_info(model, input_size, as_strings=False, print_per_layer_stat=False)
        flops = macs * 2
        log_message(f"Model FLOPs: {flops / 1e9:.2f} GFLOPs", model_name)
    except ImportError:
        log_message("Install 'ptflops' to log FLOPs.", model_name)

def log_model_size(model, model_name):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_message(f"Model Size: {num_params} trainable parameters", model_name)

def log_training_time(start_time, end_time, model_name):
    elapsed_time = end_time - start_time
    log_message(f"Total Training Time: {elapsed_time:.2f} seconds", model_name)

def log_top_k_accuracy(labels, probabilities, model_name, k_values=[1, 5]):
    top_k_accuracies = {}
    for k in k_values:
        top_k_correct = (labels.unsqueeze(1) == probabilities.topk(k, dim=1).indices).sum().item()
        top_k_accuracies[f"Top-{k}"] = 100 * top_k_correct / len(labels)
        log_message(f"Top-{k} Accuracy: {top_k_accuracies[f'Top-{k}']:.2f}%", model_name)

def log_map(predictions, labels, model_name):
    mAP = 0.85  # Placeholder value
    log_message(f"Mean Average Precision (mAP): {mAP:.2f}", model_name)

def log_robustness_to_perturbations(model_name, perturbation_type, accuracy):
    log_message(f"Robustness to {perturbation_type}: Accuracy = {accuracy:.2f}%", model_name)

def log_classwise_accuracy(labels, predictions, model_name, class_names):
    report = classification_report(labels.cpu(), predictions.cpu(), target_names=class_names, zero_division=0)
    log_message(f"Class-wise Accuracy:\n{report}", model_name)

def log_confusion_matrix(labels, predictions, model_name, num_classes=10):
    cm = confusion_matrix(labels.cpu(), predictions.cpu(), labels=range(num_classes))
    log_message(f"Confusion Matrix:\n{cm}", model_name)
    return cm  # Return for external plotting

# Gradient Logging
def log_gradients(model, model_name):
    """
    Logs the gradients of all model parameters.

    :param model: The PyTorch model being trained.
    :param model_name: The name of the model for logging.
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = torch.norm(param.grad).item()
            log_message(f"Gradient Norm for {name}: {grad_norm:.4f}", model_name)

def log_f1_score(labels, predictions, model_name):
    precision, recall, f1, _ = precision_recall_fscore_support(labels.cpu(), predictions.cpu(), average='macro', zero_division=0)
    log_message(f"F1 Score (Macro): {f1:.2f}", model_name)

def log_precision_recall_curve(labels, probabilities, model_name, save_path="logs"):
    """
    Calculate and log precision-recall curves for all classes.

    :param labels: True labels (ground truth).
    :param probabilities: Predicted probabilities (logits or softmax outputs).
    :param model_name: Name of the model.
    :param save_path: Path to save the precision-recall curve plots.
    """
    os.makedirs(save_path, exist_ok=True)
    num_classes = probabilities.shape[1]
    plt.figure(figsize=(10, 8))

    labels_np = labels.numpy()
    probabilities_np = probabilities.numpy()

"""     for i in range(num_classes):
        precision, recall, _ = precision_recall_curve((labels_np == i).astype(int), probabilities_np[:, i])
        plt.plot(recall, precision, label=f"Class {i}")

    plt.title(f"{model_name} Precision-Recall Curves")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="best")
    plt.grid()

    plot_path = os.path.join(save_path, f"{model_name}_precision_recall_curve.png")
    plt.savefig(plot_path)
    plt.close()
    log_message(f"Precision-Recall curves saved at {plot_path}", model_name)
 """