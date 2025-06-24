class ExperimentLogger:
    """Simple logger for experiment tracking."""

    def __init__(self, log_dir: str):
        """Initialize logging to the given directory.

        TODO: create log files and integrate with wandb or tensorboard.
        """
        self.log_dir = log_dir

    def log(self, message: str):
        """Write a log message to disk or external service."""
        # TODO: implement logging logic
        pass
