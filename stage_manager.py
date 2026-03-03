# core/stage_manager.py

class StageManager:
    """
    显式阶段状态管理器
    """

    def __init__(self):
        self.current_stage = None

    def set_stage(self, stage_name: str):
        self.current_stage = stage_name

    def get_stage(self):
        return self.current_stage