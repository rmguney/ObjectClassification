import time

class Heartbeat:
    def __init__(self, timeout=600, model_name="Model"):
        """
        Initialize the Heartbeat utility.
        
        :param timeout: Maximum allowable time (in seconds) for processing a single batch.
        :param model_name: Name of the model for logging and debugging.
        """
        self.timeout = timeout
        self.model_name = model_name

    def check(self, batch_start_time):
        """
        Check if the batch processing time exceeds the timeout threshold.

        :param batch_start_time: Start time of the batch processing.
        :raises TimeoutError: If processing time exceeds the timeout threshold.
        """
        batch_time = time.time() - batch_start_time
        if batch_time > self.timeout:
            raise TimeoutError(
                f"Batch processing exceeded timeout! ({batch_time:.2f} seconds) "
                f"{self.model_name} training aborted."
            )
