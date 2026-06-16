"""DiffMethod glue around the upstream CrossCoder/BatchTopKCrossCoder trainer."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from scripts.diffing.base import DiffMethod, DiffPair, register
from scripts.diffing.dictionary_learning_adapter import (
    BatchTopKCrossCoder,
    BatchTopKCrossCoderTrainer,
    CrossCoder,
    CrossCoderTrainer,
    trainSAE,
)
from scripts.diffing.methods.crosscoder.dataloader import (
    get_activation_dim,
    infinite_pair_samples,
    materialize_validation,
    normalizer_for_pair,
)
from scripts.interp.hook_sinks import H5ActivationSink
from scripts.interp.hooks import ActivationRecord


_MODEL_FILENAME = "model_final.pt"
_CONFIG_FILENAME = "config.json"


@register("crosscoder")
class CrossCoderMethod(DiffMethod):
    supports_cross_arch = False
    streaming_ok = True
    output_capture_names = [
        "feature_norm_a",
        "feature_norm_b",
        "feature_dec_norm_diff_a",
        "feature_dec_norm_diff_b",
    ]

    def __init__(
        self,
        *,
        # model
        model_type: str = "relu",
        expansion_factor: int = 8,
        code_normalization: str = "crosscoder",
        same_init_for_all_layers: bool = False,
        norm_init_scale: float = 0.005,
        init_with_transpose: bool = True,
        code_normalization_alpha_sae: float = 1.0,
        code_normalization_alpha_cc: float = 0.1,
        # ReLU+L1
        l1_penalty: float = 0.01,
        use_mse_loss: bool = False,
        # BatchTopK
        k: int = 32,
        k_max: int | None = None,
        k_annealing_steps: int = 0,
        auxk_alpha: float = 1.0 / 32.0,
        threshold_beta: float = 0.999,
        threshold_start_step: int = 1000,
        # training
        steps: int = 10_000,
        batch_size: int = 1024,
        chunk_rows: int = 256,
        lr: float = 1.0e-4,
        warmup_steps: int = 1000,
        decay_start: int | None = None,
        seed: int = 0,
        device: str | None = None,
        target_rms: float = 1.0,
        log_steps: int = 50,
        # validation
        validate_every_n_steps: int | None = None,
        validation_batches: int = 8,
        # wandb
        wandb_enabled: bool = False,
        wandb_entity: str = "",
        wandb_project: str = "Diffing-Game-Crosscoder",
        wandb_group: str = "",
    ):
        if model_type not in ("relu", "batch-top-k"):
            raise ValueError(f"unknown model_type {model_type!r}")
        self.model_type = model_type
        self.expansion_factor = expansion_factor
        self.code_normalization = code_normalization
        self.same_init_for_all_layers = same_init_for_all_layers
        self.norm_init_scale = norm_init_scale
        self.init_with_transpose = init_with_transpose
        self.code_normalization_alpha_sae = code_normalization_alpha_sae
        self.code_normalization_alpha_cc = code_normalization_alpha_cc
        self.l1_penalty = l1_penalty
        self.use_mse_loss = use_mse_loss
        self.k = k
        self.k_max = k_max
        self.k_annealing_steps = k_annealing_steps
        self.auxk_alpha = auxk_alpha
        self.threshold_beta = threshold_beta
        self.threshold_start_step = threshold_start_step
        self.steps = steps
        self.batch_size = batch_size
        self.chunk_rows = chunk_rows
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.decay_start = decay_start
        self.seed = seed
        self.device = device
        self.target_rms = target_rms
        self.log_steps = log_steps
        self.validate_every_n_steps = validate_every_n_steps
        self.validation_batches = validation_batches
        self.wandb_enabled = wandb_enabled
        self.wandb_entity = wandb_entity
        self.wandb_project = wandb_project
        self.wandb_group = wandb_group

        # Populated by fit() / load().
        self.model: CrossCoder | BatchTopKCrossCoder | None = None
        self.activation_dim: int | None = None
        self.dict_size: int | None = None

    @classmethod
    def from_config(cls, cfg) -> "CrossCoderMethod":
        m = cfg.model
        t = cfg.training
        o = cfg.optimization
        wb = cfg.get("wandb", {})
        return cls(
            model_type=m.type,
            expansion_factor=t.expansion_factor,
            code_normalization=m.code_normalization,
            same_init_for_all_layers=m.same_init_for_all_layers,
            norm_init_scale=m.norm_init_scale,
            init_with_transpose=m.init_with_transpose,
            code_normalization_alpha_sae=m.get("code_normalization_alpha_sae", 1.0),
            code_normalization_alpha_cc=m.get("code_normalization_alpha_cc", 0.1),
            l1_penalty=t.get("mu", 0.01),
            use_mse_loss=t.get("use_mse_loss", False),
            k=t.get("k", 32),
            k_max=t.get("k_max", None),
            k_annealing_steps=t.get("k_annealing_steps", 0),
            auxk_alpha=t.get("auxk_alpha", 1.0 / 32.0),
            threshold_beta=o.get("threshold_beta", 0.999),
            threshold_start_step=o.get("threshold_start_step", 1000),
            steps=t.steps,
            batch_size=t.batch_size,
            chunk_rows=t.chunk_rows,
            lr=t.lr,
            warmup_steps=o.warmup_steps,
            decay_start=o.get("decay_start", None),
            seed=t.get("seed", 0),
            device=t.get("device", None),
            target_rms=t.get("target_rms", 1.0),
            log_steps=t.get("log_steps", 50),
            validate_every_n_steps=t.get("validate_every_n_steps", None),
            validation_batches=t.get("validation_batches", 8),
            wandb_enabled=wb.get("enabled", False),
            wandb_entity=wb.get("entity", ""),
            wandb_project=wb.get("project", "Diffing-Game-Crosscoder"),
            wandb_group=wb.get("group", ""),
        )

    def config_tag(self) -> str:
        if self.model_type == "relu":
            tag = (
                f"relu_x{self.expansion_factor}_mu{self.l1_penalty:.0e}"
                f"_lr{self.lr:.0e}_steps{self.steps}"
            )
        else:
            tag = (
                f"topk_x{self.expansion_factor}_k{self.k}"
                f"_lr{self.lr:.0e}_steps{self.steps}"
            )
        return tag.replace("+", "")

    def _resolve_device(self) -> str:
        if self.device is not None:
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _build_trainer_config(
        self,
        pair: DiffPair,
        device: str,
        activation_mean: torch.Tensor,
        activation_std: torch.Tensor,
    ) -> dict:
        common = {
            "activation_dim": self.activation_dim,
            "dict_size": self.dict_size,
            "lr": self.lr,
            "device": device,
            "warmup_steps": self.warmup_steps,
            "layer": pair.a.capture,
            "lm_name": f"crosscoder-{pair.a.capture}",
            "wandb_name": self.config_tag(),
            "activation_mean": activation_mean,
            "activation_std": activation_std,
            "target_rms": self.target_rms,
            "dict_class_kwargs": {
                "same_init_for_all_layers": self.same_init_for_all_layers,
                "norm_init_scale": self.norm_init_scale,
                "init_with_transpose": self.init_with_transpose,
                "encoder_layers": None,
                "code_normalization": self.code_normalization,
                "code_normalization_alpha_sae": self.code_normalization_alpha_sae,
                "code_normalization_alpha_cc": self.code_normalization_alpha_cc,
            },
        }
        if self.model_type == "relu":
            return {
                **common,
                "trainer": CrossCoderTrainer,
                "dict_class": CrossCoder,
                "l1_penalty": self.l1_penalty,
                "pretrained_ae": None,
                "use_mse_loss": self.use_mse_loss,
            }
        return {
            **common,
            "trainer": BatchTopKCrossCoderTrainer,
            "dict_class": BatchTopKCrossCoder,
            "k": self.k,
            "k_max": self.k_max,
            "k_annealing_steps": self.k_annealing_steps,
            "auxk_alpha": self.auxk_alpha,
            "decay_start": self.decay_start,
            "threshold_beta": self.threshold_beta,
            "threshold_start_step": self.threshold_start_step,
            "steps": self.steps,
            "pretrained_ae": None,
        }

    def fit(self, pair: DiffPair) -> None:
        torch.manual_seed(self.seed)
        device = self._resolve_device()
        self.activation_dim = get_activation_dim(pair)
        self.dict_size = self.expansion_factor * self.activation_dim

        mean, std = normalizer_for_pair(pair)
        mean = mean.to(device)
        std = std.to(device)

        trainer_config = self._build_trainer_config(pair, device, mean, std)

        train_data = infinite_pair_samples(
            pair,
            batch_size=self.batch_size,
            chunk_rows=self.chunk_rows,
            device=device,
            shuffle=True,
        )
        validation_data: list[torch.Tensor] = []
        if self.validate_every_n_steps is not None and self.validation_batches > 0:
            validation_data = materialize_validation(
                pair,
                batch_size=self.batch_size,
                chunk_rows=self.chunk_rows,
                device=device,
                n_batches=self.validation_batches,
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            self.model = trainSAE(
                data=train_data,
                trainer_config=trainer_config,
                steps=self.steps,
                log_steps=self.log_steps,
                validate_every_n_steps=self.validate_every_n_steps,
                validation_data=validation_data,
                use_wandb=self.wandb_enabled,
                wandb_entity=self.wandb_entity,
                wandb_project=self.wandb_project,
                wandb_group=self.wandb_group,
                save_dir=tmpdir,
                save_steps=None,
                save_last_eval=False,
                run_wandb_finish=True,
            )

    @torch.no_grad()
    def score(self, pair: DiffPair, sink: H5ActivationSink) -> None:
        if self.model is None:
            raise RuntimeError("score() called before fit() or load()")
        self.model.eval()

        # decoder.weight: (num_layers, dict_size, activation_dim) → norm
        # over activation_dim is the per-(layer, feature) decoder column norm.
        decoder_norms = self.model.decoder.weight.norm(dim=-1).cpu().float()
        norm_a = decoder_norms[0]
        norm_b = decoder_norms[1]
        diff_a = _dec_norm_diff(norm_a, norm_b)
        diff_b = _dec_norm_diff(norm_b, norm_a)

        for name, tensor in [
            ("feature_norm_a", norm_a),
            ("feature_norm_b", norm_b),
            ("feature_dec_norm_diff_a", diff_a),
            ("feature_dec_norm_diff_b", diff_b),
        ]:
            sink.write(ActivationRecord(
                name=name, tensor=tensor.unsqueeze(0),
                metadata_tags={}, per_cell={}, layout="BD",
            ))
        sink.batch_end()

    def save(self, out_dir: Path) -> None:
        if self.model is None:
            raise RuntimeError("save() called before fit()")
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), out_dir / _MODEL_FILENAME)
        with open(out_dir / _CONFIG_FILENAME, "w") as f:
            json.dump(self._serialisable_config(), f, indent=2)

    def _serialisable_config(self) -> dict:
        # Just the kwargs needed to reconstruct __init__, plus the inferred
        # activation_dim and dict_size for sanity-checking on load.
        return {
            "model_type": self.model_type,
            "expansion_factor": self.expansion_factor,
            "code_normalization": self.code_normalization,
            "same_init_for_all_layers": self.same_init_for_all_layers,
            "norm_init_scale": self.norm_init_scale,
            "init_with_transpose": self.init_with_transpose,
            "code_normalization_alpha_sae": self.code_normalization_alpha_sae,
            "code_normalization_alpha_cc": self.code_normalization_alpha_cc,
            "l1_penalty": self.l1_penalty,
            "use_mse_loss": self.use_mse_loss,
            "k": self.k,
            "k_max": self.k_max,
            "k_annealing_steps": self.k_annealing_steps,
            "auxk_alpha": self.auxk_alpha,
            "threshold_beta": self.threshold_beta,
            "threshold_start_step": self.threshold_start_step,
            "steps": self.steps,
            "batch_size": self.batch_size,
            "chunk_rows": self.chunk_rows,
            "lr": self.lr,
            "warmup_steps": self.warmup_steps,
            "decay_start": self.decay_start,
            "seed": self.seed,
            "target_rms": self.target_rms,
            "activation_dim": self.activation_dim,
            "dict_size": self.dict_size,
        }

    @classmethod
    def load(cls, out_dir: Path) -> "CrossCoderMethod":
        with open(out_dir / _CONFIG_FILENAME) as f:
            config = json.load(f)
        activation_dim = config.pop("activation_dim")
        dict_size = config.pop("dict_size")
        obj = cls(**config)
        obj.activation_dim = activation_dim
        obj.dict_size = dict_size

        model_path = str(out_dir / _MODEL_FILENAME)
        kwargs = {"code_normalization": obj.code_normalization}
        if obj.model_type == "relu":
            obj.model = CrossCoder.from_pretrained(model_path, **kwargs)
        else:
            obj.model = BatchTopKCrossCoder.from_pretrained(model_path, **kwargs)
        return obj


def _dec_norm_diff(this: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    # Per-feature specificity score in [0, 1]: 1.0 = feature lives entirely
    # on `this` layer, 0.5 = shared, 0.0 = lives entirely on `other`.
    denom = torch.maximum(this, other).clamp_min(1e-12)
    return 0.5 * ((this - other) / denom + 1.0)
