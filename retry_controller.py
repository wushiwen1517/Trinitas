# core/retry_controller.py

class RetryController:
    """
    显式重试控制器
    """

    def __init__(self, max_retry: int):
        self.max_retry = max_retry
        self.current_retry = 0

    def can_retry(self) -> bool:
        return self.current_retry < self.max_retry

    def increase(self):
        self.current_retry += 1

    def reset(self):
        self.current_retry = 0