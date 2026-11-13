"""Benchmark settings. Defaults are sized for a laptop CPU (about 1.5-2 hours for everything)."""
from dataclasses import asdict, dataclass, field
from pathlib import Path

DATASETS = ("pneumoniamnist", "dermamnist", "bloodmnist", "breastmnist")  # octmnist also works (large download)

# method name -> human-readable label (used in tables and figures)
METHODS = {
    "hog_rf": "Handcrafted (HOG+LBP+histogram) + Random Forest",
    "resnet18_finetune": "ResNet-18 fine-tuned (ImageNet init)",
    "resnet18_probe": "Frozen ResNet-18 (ImageNet) + logistic regression",
    "dinov2_probe": "Frozen DINOv2 ViT-S/14 + logistic regression",
    "clip_probe": "Frozen CLIP ViT-B/32 + logistic regression",
    "medsiglip_probe": "Frozen MedSigLIP-448 + logistic regression",
    "random_probe": "Frozen random-init ResNet-18 + logistic regression (smoke test only)",
}
DEFAULT_METHODS = ("hog_rf", "resnet18_finetune", "resnet18_probe", "dinov2_probe", "clip_probe")


@dataclass
class DataConfig:
    datasets: tuple = DATASETS
    size: int = 64                 # MedMNIST+ resolution to download (28, 64, 128 or 224)
    pool_per_class: int = 500      # labelled candidates per class that few-shot draws are taken from
    pool_seed: int = 0


@dataclass
class ProtocolConfig:
    shots: tuple = (1, 5, 10, 50)  # labelled images per class
    draws: int = 5                 # independent random draws per (dataset, shots)
    base_seed: int = 2026


@dataclass
class FinetuneConfig:
    input_size: int = 128          # ResNet-18 input resolution for fine-tuning (CPU-friendly)
    steps: int = 200               # fixed number of optimisation steps, whatever the number of shots
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-4
    rotation_deg: float = 10.0
    scale: float = 0.10
    shift: float = 0.05
    pretrained: bool = True


@dataclass
class ProbeConfig:
    C: float = 1.0                 # inverse L2 regularisation strength of the logistic regression
    max_iter: int = 5000
    batch_size: int = 64           # feature-extraction batch size


@dataclass
class ForestConfig:
    n_estimators: int = 500


@dataclass
class Config:
    out_dir: Path = Path("outputs")
    methods: tuple = DEFAULT_METHODS
    device: str = "auto"
    synthetic: bool = False
    data: DataConfig = field(default_factory=DataConfig)
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    forest: ForestConfig = field(default_factory=ForestConfig)

    def to_dict(self):
        d = asdict(self)
        d["out_dir"] = str(self.out_dir)
        return d

    @property
    def cache_dir(self) -> Path:
        return self.out_dir / "cache"

    @property
    def feature_dir(self) -> Path:
        return self.out_dir / "features"

    @property
    def results_csv(self) -> Path:
        return self.out_dir / "runs.csv"

    @property
    def results_dir(self) -> Path:
        return self.out_dir / "results"


def smoke_config(out_dir: Path) -> Config:
    """Toy data, tiny budgets, no downloads: checks every stage in about a minute."""
    cfg = Config(out_dir=out_dir, synthetic=True,
                 methods=("hog_rf", "resnet18_finetune", "random_probe"))
    cfg.data.datasets = ("toy_grey", "toy_rgb")
    cfg.data.size = 28
    cfg.data.pool_per_class = 30
    cfg.protocol.shots = (1, 5)
    cfg.protocol.draws = 2
    cfg.finetune.steps = 3
    cfg.finetune.input_size = 32
    cfg.finetune.pretrained = False
    cfg.forest.n_estimators = 20
    return cfg
