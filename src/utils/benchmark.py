import torch
import time
import os
import logging
from collections import defaultdict
from typing import List, Dict, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Heartbeat:
    """
    Utility to check batch processing time and prevent timeouts during training.
    """
    def __init__(self, timeout=600, model_name="Model"):
        self.timeout = timeout
        self.model_name = model_name

    def check(self, batch_start_time):
        batch_time = time.time() - batch_start_time
        if batch_time > self.timeout:
            logger.error(f"Batch processing exceeded timeout! ({batch_time:.2f} seconds) {self.model_name} training aborted.")
            raise TimeoutError(
                f"Batch processing exceeded timeout! ({batch_time:.2f} seconds) "
                f"{self.model_name} training aborted."
            )

def calculate_mean_average_precision(predictions: List[Dict], targets: List[Dict], iou_threshold: float = 0.5) -> float:
    """
    Calculate mean Average Precision (mAP) for object detection.
    :param predictions: List of model predictions containing boxes and labels.
    :param targets: List of ground truth boxes and labels.
    :param iou_threshold: Intersection over Union (IoU) threshold for a positive match.
    :return: Mean Average Precision (mAP).
    """
    # Placeholder for actual mAP calculation logic
    # For now, return a dummy value (implement specific mAP logic later)
    logger.debug("Calculating mean Average Precision (mAP)...")
    return 0.0

def calculate_precision_recall_f1(predictions: List[int], targets: List[int]) -> Tuple[float, float, float]:
    """
    Calculate precision, recall, and F1-score for classification or detection.
    :param predictions: Predicted class labels.
    :param targets: Ground truth class labels.
    :return: Precision, Recall, and F1-score.
    """
    logger.debug("Calculating precision, recall, and F1-score...")
    tp = sum(p == t for p, t in zip(predictions, targets))  # True Positives
    fp = len(predictions) - tp  # False Positives
    fn = len(targets) - tp  # False Negatives

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1_score:.4f}")
    return precision, recall, f1_score

def calculate_classwise_metrics(predictions: List[int], targets: List[int], num_classes: int) -> Dict:
    """
    Calculate per-class precision, recall, and F1-score.
    :param predictions: Predicted class labels.
    :param targets: Ground truth class labels.
    :param num_classes: Number of classes.
    :return: Dictionary with metrics for each class.
    """
    logger.debug("Calculating class-wise metrics...")
    metrics = defaultdict(lambda: {"precision": 0, "recall": 0, "f1_score": 0, "count": 0})
    for cls in range(num_classes):
        cls_preds = [1 if p == cls else 0 for p in predictions]
        cls_targets = [1 if t == cls else 0 for t in targets]
        precision, recall, f1_score = calculate_precision_recall_f1(cls_preds, cls_targets)

        metrics[cls]["precision"] = precision
        metrics[cls]["recall"] = recall
        metrics[cls]["f1_score"] = f1_score
        metrics[cls]["count"] = cls_targets.count(1)

    logger.info(f"Class-wise metrics: {metrics}")
    return metrics

def calculate_fps(start_time: float, num_frames: int) -> float:
    """
    Calculate Frames Per Second (FPS).
    :param start_time: Start time of processing.
    :param num_frames: Number of frames processed.
    :return: Frames per second (FPS).
    """
    elapsed_time = time.time() - start_time
    fps = num_frames / elapsed_time if elapsed_time > 0 else 0
    logger.info(f"FPS: {fps:.2f}")
    return fps

def calculate_memory_usage() -> Dict[str, float]:
    """
    Get GPU memory usage during training or inference.
    :return: Dictionary containing max and current memory usage (in MB).
    """
    if torch.cuda.is_available():
        max_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)  # Convert to MB
        curr_mem = torch.cuda.memory_allocated() / (1024 * 1024)  # Convert to MB
        logger.info(f"Max Memory Usage: {max_mem:.2f} MB, Current Memory Usage: {curr_mem:.2f} MB")
        return {"max_memory_mb": max_mem, "current_memory_mb": curr_mem}
    logger.warning("CUDA is not available. Returning zero memory usage.")
    return {"max_memory_mb": 0, "current_memory_mb": 0}

def calculate_model_size(model: torch.nn.Module) -> float:
    """
    Calculate the size of a PyTorch model in megabytes.
    :param model: PyTorch model.
    :return: Model size in MB.
    """
    torch.save(model.state_dict(), "temp.pth")
    model_size_mb = os.path.getsize("temp.pth") / (1024 * 1024)  # Convert to MB
    os.remove("temp.pth")
    logger.info(f"Model size: {model_size_mb:.2f} MB")
    return model_size_mb

def calculate_flops(model: torch.nn.Module, input_size: Tuple[int, int, int, int]) -> int:
    """
    Calculate the FLOPs (Floating Point Operations) for the given model and input size.
    :param model: PyTorch model.
    :param input_size: Tuple representing input size (batch_size, channels, height, width).
    :return: Total FLOPs for a forward pass.
    """
    try:
        from torchprofile import profile_macs
        macs = profile_macs(model, torch.randn(*input_size).to(next(model.parameters()).device))
        logger.info(f"FLOPs: {macs}")
        return macs
    except ImportError:
        logger.error("Install `torchprofile` for FLOPs calculation.")
        return -1

def track_training_time(start_time: float) -> float:
    """
    Track the total training time.
    :param start_time: Start time of training.
    :return: Total training time in seconds.
    """
    total_time = time.time() - start_time
    logger.info(f"Total training time: {total_time:.2f} seconds")
    return total_time
