from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Configuration for SpectrumModel architecture."""
    embed_dim: int = 256
    num_heads: int = 16
    max_seq_len: int = 40
    penultimate_dim: int = 1024
    dropout_rate: float = 0.2
    num_encoder_layers: int = 1


@dataclass
class TrainConfig:
    """Configuration for training loop."""
    batch_size: int = 512
    learning_rate: float = 1e-4
    weight_decay: float = 1e-3
    epochs: int = 100
    patience: int = 5
    checkpoint_path: str = "checkpoints/best_model.pth"
    num_workers: int = 4


@dataclass
class InferenceConfig:
    """Configuration for inference."""
    batch_size: int = 16
    threshold: float = 1e-4
    output_dir: str = "predicted/"


@dataclass
class DataConfig:
    """Configuration for data loading."""
    data_dir: str = "processed/"
    alphabet_path: str = "amino_acid_alphabet.txt"
    split_dataset: bool = False
    train_files: list = field(default_factory=lambda: [
        'AItrain_LumosSynthetic_2022418v2.ann.txt',
        'AItrain_QEHumanCho_2022418v2.ann.txt',
    ])
    val_files: list = field(default_factory=lambda: [
        'ValidUniq2022418_202333.ann.txt',
    ])
