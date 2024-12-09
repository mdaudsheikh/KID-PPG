from dataclasses import dataclass


@dataclass
class Config:
    dataset: str = "PPG_Dalia"
    root: str = (
        r"C:\Users\sheik\Documents\School\GeorgiaTech\3_2024Fall\DL\KID-PPG\src\data"
    )
    search_type: str = ""
    time_window: int = 8
    input_shape: int = 32 * 8
    batch_size: int = 128
    lr: float = 0.001
    epochs: int = 500
    a: int = 35
    path_PPG_Dalia: str = "./"
    warmup: int = 20
    reg_strength: float = 1e-6
    l2: float = 0.0
    threshold: float = 0.5
    hyst: int = 0
    saving_path: str = "./saved_models_/"

    # parameters MorphNet training
    epochs_MN: int = 350
    batch_size_MN: int = 128

    def __post_init__(self):
        # Update paths based on root and search_type
        self.path_PPG_Dalia = self.root
        self.saving_path = self.root + "saved_models_" + self.search_type + "/"
