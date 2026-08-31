#!/usr/bin/env python3
"""Local OpenPI policy server for edge GPUs such as Orin NX.

The edge path deliberately has no dependency on the 4090 training/dashboard
entrypoint.  ``check`` only inspects a staged PyTorch checkpoint and the
hardware/profile contract.  ``serve`` lazily imports OpenPI, builds the small
inference transform stack, and exposes the same WebSocket protocol consumed by
``bimanual_vla.deployment.client``.

The server expects a completed ``model.safetensors`` checkpoint.  Orbax/JAX
training checkpoints are intentionally rejected here: training happens on the
H200 and the finalized PyTorch artifact is what is copied to the edge node.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import http
import json
import logging
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Mapping

import numpy as np

from bimanual_vla.data.action_conventions import (
    absolute_eef_targets_to_chunk_origin,
    DELIVERY_MODEL_ACTION_SEMANTICS,
    DELIVERY_RAW_ACTION_SEMANTICS,
    DELIVERY_CHUNK_ORIGIN_ACTION_CONVENTION,
    JOINT_ACTION_SEMANTICS,
    NEW_GRIPPER_SEMANTICS,
)
from bimanual_vla.data.contract import CONTRACT_VERSION
from bimanual_vla.deployment.rtc_policy import RTCConfig, build_rtc_policy


LOGGER = logging.getLogger(__name__)
DEFAULT_OPENPI_ROOT = Path(os.environ.get("OPENPI_ROOT", "/home/user/vscode/openpi"))
DEFAULT_ACTION_HORIZON = 50
DEFAULT_ACTION_HZ = 20.0


@dataclasses.dataclass(frozen=True)
class OrinNXProfile:
    """Conservative deployment envelope, not a performance guarantee."""

    name: str
    memory_gb: int
    recommended_models: tuple[str, ...]
    experimental_models: tuple[str, ...]
    image_size: tuple[int, int] = (224, 224)
    action_horizon: int = DEFAULT_ACTION_HORIZON
    compile_mode: str | None = None
    minimum_system_memory_gb: int = 12

    def allows(self, model_variant: str) -> bool:
        return model_variant in self.recommended_models or model_variant in self.experimental_models


NX_PROFILES: Mapping[str, OrinNXProfile] = {
    "orin_nx_8gb": OrinNXProfile(
        name="orin_nx_8gb",
        memory_gb=8,
        recommended_models=("pi0", "smolvla"),
        experimental_models=(),
        compile_mode=None,
        minimum_system_memory_gb=10,
    ),
    "orin_nx_16gb": OrinNXProfile(
        name="orin_nx_16gb",
        memory_gb=16,
        recommended_models=("pi0", "smolvla"),
        experimental_models=("pi05",),
        compile_mode=None,
        minimum_system_memory_gb=16,
    ),
}


def get_nx_profile(name: str) -> OrinNXProfile:
    try:
        return NX_PROFILES[str(name).strip().lower()]
    except KeyError as exc:
        raise ValueError(f"unknown Orin NX profile {name!r}; choose {sorted(NX_PROFILES)}") from exc


def _arm_count(arm_mode: str) -> int:
    value = str(arm_mode).strip().lower()
    if value == "single":
        return 1
    if value == "bimanual":
        return 2
    raise ValueError("arm_mode must be 'single' or 'bimanual'")


def _contract_dimensions(schema: str, arm_mode: str) -> tuple[int, int]:
    count = _arm_count(arm_mode)
    schema = str(schema).strip().lower()
    if schema == "joint":
        return 7 * count, 7 * count
    if schema == "delivery":
        return 10 * count, 7 * count
    raise ValueError("schema must be 'joint' or 'delivery'")


def _camera_keys(arm_mode: str, arm_side: str) -> tuple[str, ...]:
    if arm_mode == "bimanual":
        if arm_side != "both":
            raise ValueError("bimanual policy requires arm_side='both'")
        return ("cam_high", "cam_left_wrist", "cam_right_wrist")
    if arm_side not in {"left", "right"}:
        raise ValueError("single-arm policy requires arm_side='left' or 'right'")
    return ("cam_high", f"cam_{arm_side}_wrist")


def build_policy_metadata(
    *,
    schema: str,
    arm_mode: str,
    arm_side: str,
    dataset_id: str,
    model_variant: str,
    backend: str = "pi",
    checkpoint: str | Path,
    profile: str,
    action_horizon: int = DEFAULT_ACTION_HORIZON,
    action_hz: float = DEFAULT_ACTION_HZ,
    rtc_enabled: bool = False,
    rtc_execution_horizon: int = 8,
    rtc_max_guidance_weight: float = 5.0,
    rtc_prefix_attention_schedule: str = "linear",
) -> dict[str, Any]:
    """Create the metadata consumed by the robot client's strict handshake."""

    schema = str(schema).strip().lower()
    arm_mode = str(arm_mode).strip().lower()
    arm_side = "both" if arm_mode == "bimanual" else str(arm_side).strip().lower()
    profile_info = get_nx_profile(profile)
    state_dim, action_dim = _contract_dimensions(schema, arm_mode)
    cameras = _camera_keys(arm_mode, arm_side)
    if int(action_horizon) < 16 or float(action_hz) <= 0:
        raise ValueError("action_horizon must be at least 16 and action_hz must be positive")
    if rtc_enabled:
        if int(rtc_execution_horizon) <= 0:
            raise ValueError("rtc_execution_horizon must be positive")
        if float(rtc_max_guidance_weight) <= 0:
            raise ValueError("rtc_max_guidance_weight must be positive")
        if rtc_prefix_attention_schedule not in {"zeros", "ones", "linear", "exp"}:
            raise ValueError("unsupported RTC prefix attention schedule")

    backend = str(backend).strip().lower()
    if backend not in {"pi", "smolvla"}:
        raise ValueError("backend must be 'pi' or 'smolvla'")
    if not profile_info.allows(model_variant):
        raise ValueError(
            f"{model_variant} is not in the {profile_info.name} edge envelope; "
            f"recommended={profile_info.recommended_models}, experimental={profile_info.experimental_models}"
        )
    if backend == "smolvla" and model_variant != "smolvla":
        raise ValueError("SmolVLA backend requires model_variant='smolvla'")
    if backend == "pi" and model_variant == "smolvla":
        raise ValueError("model_variant='smolvla' requires backend='smolvla'")
    if backend == "smolvla" and schema != "joint":
        raise ValueError("SmolVLA edge backend currently supports the joint schema only")
    if backend == "smolvla" and rtc_enabled:
        raise ValueError("SmolVLA edge backend does not expose model-side RTC; use --no-rtc-enabled")
    if schema == "joint":
        semantics = JOINT_ACTION_SEMANTICS
        convention = "absolute_joint_target"
        raw_semantics = semantics
    else:
        semantics = DELIVERY_MODEL_ACTION_SEMANTICS
        convention = DELIVERY_CHUNK_ORIGIN_ACTION_CONVENTION
        raw_semantics = DELIVERY_RAW_ACTION_SEMANTICS

    return {
        "robot_type": "piper_bimanual" if arm_mode == "bimanual" else "piper_single_arm",
        "deployment_target": "orin_nx",
        "deployment_profile": str(profile),
        "dataset_id": str(dataset_id),
        "checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "model_variant": str(model_variant),
        "inference_backend": backend,
        "transport": "openpi_websocket_v1",
        "schema": schema,
        "arm_mode": arm_mode,
        "arm_side": arm_side,
        "contract_version": CONTRACT_VERSION,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "raw_action_dim": action_dim if schema == "joint" else state_dim,
        "model_action_dim": action_dim,
        "camera_keys": list(cameras),
        "action_hz": float(action_hz),
        "action_horizon": int(action_horizon),
        "action_time_step_s": 1.0 / float(action_hz),
        "action_start_offset_steps": 1,
        "action_offset": 1,
        "model_action_start_offset": 1,
        "minimum_horizon": 16,
        "recommended_inference_launch_hz": 4.0,
        "action_semantics": semantics,
        "wire_action_semantics": semantics,
        "wire_action_convention": convention,
        "raw_action_semantics": raw_semantics,
        "model_action_semantics": semantics,
        "raw_action_convention": convention if schema == "joint" else "absolute_eef_target",
        "model_action_convention": convention,
        "delivery_action_convention": convention if schema == "delivery" else None,
        "raw_gripper_semantics": NEW_GRIPPER_SEMANTICS,
        "model_gripper_semantics": NEW_GRIPPER_SEMANTICS,
        "wire_gripper_semantics": NEW_GRIPPER_SEMANTICS,
        "state_gripper_semantics": NEW_GRIPPER_SEMANTICS,
        "gripper_semantics": NEW_GRIPPER_SEMANTICS,
        "action_source": "next_measured_eef_fallback" if schema == "delivery" else "next_measured_joint_fallback",
        "action_alignment": "next_observation",
        "legacy_delivery_v2": False,
        "legacy_joint_v2": False,
        "rtc_enabled": bool(rtc_enabled),
        "rtc_algorithm": "real_time_chunking_prefix_guidance" if rtc_enabled else None,
        "rtc_backend": "pytorch" if rtc_enabled else None,
        "rtc_execution_horizon": int(rtc_execution_horizon),
        "rtc_max_guidance_weight": float(rtc_max_guidance_weight),
        "rtc_prefix_attention_schedule": str(rtc_prefix_attention_schedule),
        "rtc_physical_action_dim": action_dim,
        "rtc_chunk_origin_reanchoring": bool(rtc_enabled and schema == "joint"),
    }


def _metadata_candidates(checkpoint: Path) -> tuple[Path, ...]:
    return (
        checkpoint / "policy_metadata.json",
        checkpoint / "metadata.json",
        checkpoint.parent / f"{checkpoint.name}.json",
    )


def load_policy_metadata(path: str | Path | None, checkpoint: str | Path) -> dict[str, Any]:
    """Load optional metadata without treating it as a replacement for CLI contract fields."""

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    candidates = (Path(path).expanduser(),) if path else _metadata_candidates(checkpoint_path)
    for candidate in candidates:
        try:
            value = json.loads(candidate.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            continue
        if not isinstance(value, dict):
            raise ValueError(f"policy metadata must be a JSON object: {candidate}")
        return value
    return {}


def find_norm_stats(checkpoint: str | Path, dataset_id: str, explicit: str | Path | None = None) -> Path:
    """Find the normalized-stat directory copied alongside a finalized checkpoint."""

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser().resolve())
    candidates.extend(
        (
            checkpoint_path / "assets" / dataset_id,
            checkpoint_path.parent / "assets" / dataset_id,
            checkpoint_path / "assets" / "norm_stats",
        )
    )
    for candidate in candidates:
        if (candidate / "norm_stats.json").is_file():
            return candidate
    rendered = ", ".join(str(item) for item in candidates)
    raise FileNotFoundError(f"norm stats directory not found; checked: {rendered}")


def _declared_feature_dim(config: Mapping[str, Any], feature_name: str) -> int | None:
    """Read a LeRobot feature shape without importing the model implementation."""

    for container_name in ("output_features", "input_features"):
        container = config.get(container_name)
        if not isinstance(container, Mapping):
            continue
        feature = container.get(feature_name)
        if not isinstance(feature, Mapping):
            continue
        shape = feature.get("shape")
        if isinstance(shape, (list, tuple)) and shape:
            try:
                return int(shape[0])
            except (TypeError, ValueError):
                return None
    return None


def _declared_image_features(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Return LeRobot image feature names without importing the model runtime."""

    features = config.get("input_features")
    if not isinstance(features, Mapping):
        return ()
    return tuple(str(key) for key in features if str(key).startswith("observation.images."))


def inspect_checkpoint(
    checkpoint: str | Path,
    *,
    dataset_id: str | None = None,
    norm_stats: str | Path | None = None,
    backend: str = "pi",
    expected_state_dim: int | None = None,
    expected_action_dim: int | None = None,
    expected_camera_keys: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Return a JSON-safe checkpoint report and fail closed for non-PyTorch artifacts."""

    path = Path(checkpoint).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {path}")
    weights = path / "model.safetensors"
    if not weights.is_file():
        raise ValueError(
            f"edge inference requires {weights}; Orbax/JAX-only checkpoints are not accepted"
        )
    backend = str(backend).strip().lower()
    if backend not in {"pi", "smolvla"}:
        raise ValueError("backend must be 'pi' or 'smolvla'")
    report: dict[str, Any] = {
        "checkpoint": str(path),
        "weights": str(weights),
        "weights_bytes": weights.stat().st_size,
        "weights_gib": round(weights.stat().st_size / (1024**3), 3),
        "format": "lerobot_smolvla_safetensors" if backend == "smolvla" else "pytorch_safetensors",
        "norm_stats": None,
    }
    if backend == "smolvla":
        config_path = path / "config.json"
        if not config_path.is_file():
            raise ValueError(f"SmolVLA checkpoint is missing LeRobot config.json: {config_path}")
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid SmolVLA config.json: {config_path}") from exc
        if not isinstance(config, Mapping):
            raise ValueError(f"SmolVLA config.json must contain an object: {config_path}")
        declared_action_dim = _declared_feature_dim(config, "action")
        declared_state_dim = _declared_feature_dim(config, "observation.state")
        declared_image_features = _declared_image_features(config)
        if expected_action_dim is not None and declared_action_dim is not None and declared_action_dim != expected_action_dim:
            raise ValueError(
                f"SmolVLA checkpoint action dim={declared_action_dim} does not match "
                f"Piper contract action dim={expected_action_dim}"
            )
        if expected_state_dim is not None and declared_state_dim is not None and declared_state_dim != expected_state_dim:
            raise ValueError(
                f"SmolVLA checkpoint state dim={declared_state_dim} does not match "
                f"Piper contract state dim={expected_state_dim}"
            )
        if expected_camera_keys is not None:
            expected_features = tuple(f"observation.images.{key}" for key in expected_camera_keys)
            if set(declared_image_features) != set(expected_features):
                raise ValueError(
                    "SmolVLA checkpoint image features do not match Piper camera contract: "
                    f"declared={list(declared_image_features)!r}, expected={list(expected_features)!r}"
                )
        declared_chunk_size = config.get("chunk_size")
        try:
            declared_chunk_size = int(declared_chunk_size) if declared_chunk_size is not None else None
        except (TypeError, ValueError):
            raise ValueError("SmolVLA config chunk_size must be an integer") from None
        if declared_chunk_size is not None and declared_chunk_size < 16:
            raise ValueError(
                f"SmolVLA checkpoint chunk_size={declared_chunk_size} is below the client minimum of 16"
            )
        report["config"] = str(config_path)
        report["declared_action_dim"] = declared_action_dim
        report["declared_state_dim"] = declared_state_dim
        report["declared_image_features"] = list(declared_image_features)
        report["declared_chunk_size"] = declared_chunk_size
        report["norm_stats"] = "embedded_in_policy_safetensors"
    elif dataset_id:
        report["norm_stats"] = str(find_norm_stats(path, dataset_id, norm_stats))
    return report


def _read_system_memory_gb() -> tuple[float | None, float | None]:
    """Read total and available system memory without making psutil mandatory."""

    try:
        import psutil

        memory = psutil.virtual_memory()
        return float(memory.total) / (1024**3), float(memory.available) / (1024**3)
    except Exception:
        pass
    try:
        values: dict[str, float] = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            parts = value.strip().split()
            if parts and parts[0].replace(".", "", 1).isdigit():
                values[key] = float(parts[0]) * (1024 if len(parts) > 1 and parts[1].lower() == "kb" else 1)
        total = values.get("MemTotal")
        available = values.get("MemAvailable")
        return (
            total / (1024**3) if total is not None else None,
            available / (1024**3) if available is not None else None,
        )
    except Exception:
        return None, None


def check_device(*, profile: str, device: str = "auto", model_variant: str = "pi05", torch_module: Any | None = None) -> dict[str, Any]:
    """Check an edge profile before importing/loading the model."""

    selected = str(device).strip().lower()
    if selected == "auto":
        selected = "cuda" if _torch_cuda_available(torch_module) else "cpu"
    if selected.startswith("cuda") and not _torch_cuda_available(torch_module):
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    profile_info = get_nx_profile(profile)
    if not profile_info.allows(model_variant):
        raise RuntimeError(
            f"{model_variant} is not in the {profile_info.name} edge envelope; "
            f"recommended={profile_info.recommended_models}, experimental={profile_info.experimental_models}"
        )
    result: dict[str, Any] = {
        "profile": profile_info.name,
        "device": selected,
        "model_variant": model_variant,
        "profile_support": (
            "recommended" if model_variant in profile_info.recommended_models else "experimental"
        ),
        "profile_memory_gb": profile_info.memory_gb,
        "memory_check": "not_available",
        "minimum_system_memory_gb": profile_info.minimum_system_memory_gb,
        "system_memory_check": "not_available",
    }
    total_system_gb, available_system_gb = _read_system_memory_gb()
    if total_system_gb is not None:
        result["system_total_memory_gb"] = round(total_system_gb, 2)
        if available_system_gb is not None:
            result["system_available_memory_gb"] = round(available_system_gb, 2)
        if total_system_gb + 1.0 < profile_info.minimum_system_memory_gb:
            raise RuntimeError(
                f"system reports {total_system_gb:.2f} GiB, below profile minimum "
                f"{profile_info.minimum_system_memory_gb:.2f} GiB"
            )
        result["system_memory_check"] = "ok"
    if selected.startswith("cuda"):
        torch = torch_module or _import_torch()
        index = torch.cuda.current_device() if selected == "cuda" else torch.device(selected).index
        index = torch.cuda.current_device() if index is None else index
        props = torch.cuda.get_device_properties(index)
        total_gb = float(props.total_memory) / (1024**3)
        result.update({"cuda_device_name": str(props.name), "cuda_total_memory_gb": round(total_gb, 2)})
        # Jetson reports binary GiB while the SKU is usually advertised in
        # decimal GB. Keep a 1 GiB unit-conversion margin at the profile gate.
        if total_gb + 1.0 < profile_info.memory_gb:
            raise RuntimeError(
                f"GPU reports {total_gb:.2f} GiB, below profile {profile_info.memory_gb} GiB"
            )
        result["memory_check"] = "ok"
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            free_gb = float(free_bytes) / (1024**3)
            result.update(
                {
                    "cuda_free_memory_gb": round(free_gb, 2),
                    "cuda_mem_get_info_total_gb": round(float(total_bytes) / (1024**3), 2),
                    "cuda_free_memory_check": (
                        "ok" if free_gb >= profile_info.minimum_cuda_memory_gb else "low"
                    ),
                }
            )
        except Exception:
            result["cuda_free_memory_check"] = "not_available"
    return result


def _torch_cuda_available(torch_module: Any | None) -> bool:
    try:
        torch = torch_module or _import_torch()
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _import_torch() -> Any:
    import torch

    return torch


def _add_openpi_root(openpi_root: str | Path) -> Path:
    root = Path(openpi_root).expanduser().resolve()
    src = root / "src"
    if not src.is_dir():
        raise FileNotFoundError(f"OpenPI source directory does not exist: {src}")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return root


def _build_openpi_policy(args: argparse.Namespace, metadata: dict[str, Any], checkpoint: Path) -> Any:
    """Build the official OpenPI Policy lazily; this function is never used by ``check``."""

    _add_openpi_root(args.openpi_root)
    import dataclasses as _dataclasses

    from openpi import transforms
    from openpi.models import pi0_config
    from openpi.policies import policy_config
    from openpi.training import config as training_config

    schema = str(metadata["schema"])
    arm_mode = str(metadata["arm_mode"])
    arm_side = str(metadata["arm_side"])
    state_dim = int(metadata["state_dim"])
    model_action_dim = int(metadata["model_action_dim"])
    raw_action_dim = int(metadata["raw_action_dim"])
    camera_keys = tuple(metadata["camera_keys"])
    arm_count = _arm_count(arm_mode)

    def _as_hwc_uint8(image: Any) -> np.ndarray:
        value = np.asarray(image)
        if np.issubdtype(value.dtype, np.floating):
            value = np.clip(value * 255.0, 0, 255).astype(np.uint8)
        elif value.dtype != np.uint8:
            value = np.clip(value, 0, 255).astype(np.uint8)
        if value.ndim != 3:
            raise ValueError(f"image must be rank 3, got {value.shape}")
        if value.shape[0] in (1, 3, 4) and value.shape[-1] not in (1, 3, 4):
            value = np.transpose(value, (1, 2, 0))
        if value.shape[-1] == 4:
            value = value[..., :3]
        if value.shape[-1] != 3:
            raise ValueError(f"image must have three RGB channels, got {value.shape}")
        return value

    class EdgePiperInputs:
        def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
            state = np.asarray(data["state"], dtype=np.float32)
            if state.shape[-1] != state_dim:
                raise ValueError(f"state must be {state_dim}D, got {state.shape}")
            images = data["images"]
            if set(images) != set(camera_keys):
                raise ValueError(f"camera keys must be {sorted(camera_keys)}, got {sorted(images)}")
            high = _as_hwc_uint8(images["cam_high"])
            if arm_mode == "bimanual":
                mapped = {
                    "base_0_rgb": high,
                    "left_wrist_0_rgb": _as_hwc_uint8(images["cam_left_wrist"]),
                    "right_wrist_0_rgb": _as_hwc_uint8(images["cam_right_wrist"]),
                }
                masks = {key: np.True_ for key in mapped}
            else:
                wrist_key = f"cam_{arm_side}_wrist"
                wrist = _as_hwc_uint8(images[wrist_key])
                mapped = {
                    "base_0_rgb": high,
                    "left_wrist_0_rgb": wrist if arm_side == "left" else np.zeros_like(wrist),
                    "right_wrist_0_rgb": wrist if arm_side == "right" else np.zeros_like(wrist),
                }
                masks = {
                    "base_0_rgb": np.True_,
                    "left_wrist_0_rgb": np.bool_(arm_side == "left"),
                    "right_wrist_0_rgb": np.bool_(arm_side == "right"),
                }
            output = {"image": mapped, "image_mask": masks, "state": state}
            if "actions" in data:
                actions = np.asarray(data["actions"], dtype=np.float32)
                if actions.shape[-1] != raw_action_dim:
                    raise ValueError(f"actions must be {raw_action_dim}D, got {actions.shape}")
                if schema == "delivery":
                    actions = absolute_eef_targets_to_chunk_origin(
                        state,
                        actions,
                        arm_count=arm_count,
                    )
                output["actions"] = actions
            if "prompt" in data:
                output["prompt"] = data["prompt"]
            return output

    class EdgePiperOutputs:
        def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
            return {"actions": np.asarray(data["actions"])[..., :model_action_dim]}

    mask = None
    if schema == "joint":
        dims_per_arm = model_action_dim // arm_count
        mask_values: list[bool] = []
        for _ in range(arm_count):
            mask_values.extend([True] * (dims_per_arm - 1))
            mask_values.append(False)
        mask = transforms.make_bool_mask(*mask_values)

    @_dataclasses.dataclass(frozen=True)
    class EdgeDataConfig(training_config.DataConfigFactory):
        contract_name: str = "edge"

        def create(self, assets_dirs: Path, model_config: Any) -> Any:
            data_transforms = transforms.Group(
                inputs=[EdgePiperInputs()],
                outputs=[EdgePiperOutputs()],
            )
            if mask is not None:
                data_transforms = data_transforms.push(
                    inputs=[transforms.DeltaActions(mask)],
                    outputs=[transforms.AbsoluteActions(mask)],
                )
            model_transforms = training_config.ModelTransformFactory(
                default_prompt=args.default_prompt
            )(model_config)
            return _dataclasses.replace(
                self.create_base_config(assets_dirs, model_config),
                data_transforms=data_transforms,
                model_transforms=model_transforms,
                action_sequence_keys=("action",),
            )

    precision = getattr(args, "precision", "auto")
    if precision == "auto":
        precision = "bf16" if str(args.device).startswith("cuda") else "fp32"
    if precision == "fp16":
        raise ValueError(
            "OpenPI pi backend does not support fp16; use --precision bf16 or --precision fp32"
        )
    if precision not in {"bf16", "fp32"}:
        raise ValueError(f"unsupported OpenPI precision {precision!r}")
    model_cfg = pi0_config.Pi0Config(
        dtype="bfloat16" if precision == "bf16" else "float32",
        pi05=args.model_variant == "pi05",
        action_dim=32,
        action_horizon=int(metadata["action_horizon"]),
        paligemma_variant=args.paligemma_variant,
        action_expert_variant=args.action_expert_variant,
        pytorch_compile_mode=None if args.compile_mode == "none" else args.compile_mode,
    )
    data_cfg = EdgeDataConfig(
        repo_id=str(metadata["dataset_id"]),
        assets=training_config.AssetsConfig(
            assets_dir=str(find_norm_stats(checkpoint, str(metadata["dataset_id"]), args.norm_stats).parent),
            asset_id=str(metadata["dataset_id"]),
        ),
        base_config=training_config.DataConfig(prompt_from_task=False),
    )
    train_cfg = training_config.TrainConfig(
        name="edge_inference",
        exp_name="runtime",
        model=model_cfg,
        data=data_cfg,
        assets_base_dir=str(checkpoint / "assets"),
        checkpoint_base_dir=str(checkpoint.parent),
        policy_metadata=dict(metadata),
        wandb_enabled=False,
    )
    policy = policy_config.create_trained_policy(
        train_cfg,
        checkpoint,
        default_prompt=args.default_prompt,
        pytorch_device=args.device,
    )
    if precision == "fp32":
        model = getattr(policy, "_model", None)
        converter = getattr(
            getattr(model, "paligemma_with_expert", None),
            "to_bfloat16_for_selected_params",
            None,
        )
        if not callable(converter):
            raise RuntimeError("OpenPI policy does not expose the precision conversion API required for fp32")
        converter("float32")
    if args.rtc_enabled:
        reanchor_mask = None
        if schema == "joint":
            values: list[bool] = []
            for _ in range(arm_count):
                values.extend([True] * (model_action_dim // arm_count - 1))
                values.append(False)
            reanchor_mask = tuple(values)
        policy = build_rtc_policy(
            policy,
            RTCConfig(
                enabled=True,
                execution_horizon=args.rtc_execution_horizon,
                max_guidance_weight=args.rtc_max_guidance_weight,
                prefix_attention_schedule=args.rtc_prefix_attention_schedule,
                physical_action_dim=model_action_dim,
                reanchor_action_mask=reanchor_mask,
            ),
        )
    return policy


class _SmolVLAPolicyAdapter:
    """Adapt LeRobot SmolVLA to the OpenPI WebSocket BasePolicy interface."""

    def __init__(
        self,
        policy: Any,
        *,
        arm_mode: str,
        arm_side: str,
        action_dim: int,
        image_features: tuple[str, ...],
        instruction: str | None,
    ):
        self.policy = policy
        self.arm_mode = arm_mode
        self.arm_side = arm_side
        self.action_dim = int(action_dim)
        self.image_features = tuple(image_features)
        self.instruction = instruction or ""

    @staticmethod
    def _image(value: Any) -> Any:
        image = np.asarray(value)
        if image.ndim != 3:
            raise ValueError(f"image must be rank 3, got {image.shape}")
        if image.shape[-1] == 3:
            image = np.transpose(image, (2, 0, 1))
        elif image.shape[0] != 3:
            raise ValueError(f"image must be HWC or CHW RGB, got {image.shape}")
        if image.dtype != np.uint8:
            image = np.clip(image * 255.0 if np.issubdtype(image.dtype, np.floating) else image, 0, 255).astype(np.uint8)
        return image.astype(np.float32) / 255.0

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        import torch
        device = next(self.policy.parameters()).device
        dtype = next(self.policy.parameters()).dtype
        images = obs["images"]
        batch: dict[str, Any] = {}
        camera_order = ["cam_high"]
        if self.arm_mode == "bimanual":
            camera_order.extend(["cam_left_wrist", "cam_right_wrist"])
        else:
            camera_order.append(f"cam_{self.arm_side}_wrist")
        if len(self.image_features) != len(camera_order):
            raise RuntimeError(
                f"SmolVLA checkpoint exposes {len(self.image_features)} image feature(s), "
                f"but {self.arm_mode} Piper inference requires exactly {len(camera_order)}"
            )
        for feature_name, key in zip(self.image_features, camera_order, strict=True):
            value = torch.from_numpy(self._image(images[key])).to(device=device, dtype=dtype).unsqueeze(0)
            batch[feature_name] = value
        state = torch.from_numpy(np.asarray(obs["state"], dtype=np.float32)).to(device=device, dtype=dtype).unsqueeze(0)
        batch["observation.state"] = state
        prompt = obs.get("prompt") or self.instruction
        batch["task"] = [str(prompt)]
        with torch.inference_mode():
            actions = self.policy.predict_action_chunk(batch)
        values = actions.detach().float().cpu().numpy()
        if values.ndim == 3:
            values = values[0]
        if values.ndim != 2 or values.shape[1] < self.action_dim:
            raise RuntimeError(f"SmolVLA returned invalid action chunk shape {values.shape}")
        return {"actions": values[:, : self.action_dim]}

    def reset(self) -> None:
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset()


class EdgeWebsocketPolicyServer:
    """Small OpenPI-compatible server used when the edge backend is SmolVLA."""

    def __init__(self, policy: Any, *, host: str, port: int, metadata: dict[str, Any]):
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        try:
            import websockets
            import websockets.asyncio.server as websocket_server
            import websockets.frames
            from openpi_client import msgpack_numpy
        except Exception as exc:
            raise RuntimeError("SmolVLA WebSocket serving requires websockets and openpi-client") from exc

        def health_check(connection: Any, request: Any) -> Any:
            if request.path == "/healthz":
                return connection.respond(http.HTTPStatus.OK, "OK\n")
            return None

        async def handler(websocket: Any) -> None:
            packer = msgpack_numpy.Packer()
            await websocket.send(packer.pack(self._metadata))
            previous_total = None
            while True:
                try:
                    start = time.monotonic()
                    observation = msgpack_numpy.unpackb(await websocket.recv())
                    infer_start = time.monotonic()
                    result = dict(self._policy.infer(observation))
                    infer_elapsed = time.monotonic() - infer_start
                    timing = dict(result.get("server_timing") or {})
                    timing["infer_ms"] = infer_elapsed * 1000.0
                    if previous_total is not None:
                        timing["prev_total_ms"] = previous_total * 1000.0
                    result["server_timing"] = timing
                    await websocket.send(packer.pack(result))
                    previous_total = time.monotonic() - start
                except websockets.ConnectionClosed:
                    return
                except Exception:
                    await websocket.send(traceback.format_exc())
                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason="Internal server error. Traceback included in previous frame.",
                    )
                    raise

        async with websocket_server.serve(
            handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=health_check,
        ) as server:
            await server.serve_forever()


def _build_smolvla_policy(args: argparse.Namespace, metadata: dict[str, Any], checkpoint: Path) -> Any:
    """Load a LeRobot SmolVLA artifact with Jetson-safe ordering and FP16."""

    os.environ.setdefault("PYTORCH_NO_CUDA_MEMORY_CACHING", "1")
    _install_transformers_jetson_workarounds()
    try:
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, standardise_state_dict
    except Exception as exc:
        raise RuntimeError(
            "SmolVLA backend requires LeRobot's smolvla extra and transformers; "
            "install them in the Jetson environment"
        ) from exc

    device = str(args.device)
    # The LeRobot checkpoint stores its normalization buffers in the same
    # safetensors file. Its config also records input/action feature shapes.
    config = SmolVLAConfig.from_pretrained(str(checkpoint))
    config.device = "cpu"
    config.load_vlm_weights = False
    state_feature = config.robot_state_feature
    action_feature = config.action_feature
    if state_feature is None or action_feature is None:
        raise ValueError("SmolVLA config must declare observation.state and action features")
    if int(state_feature.shape[0]) != int(metadata["state_dim"]):
        raise ValueError(
            f"SmolVLA state feature is {state_feature.shape[0]}D, expected {metadata['state_dim']}D"
        )
    if int(action_feature.shape[0]) != int(metadata["action_dim"]):
        raise ValueError(
            f"SmolVLA action feature is {action_feature.shape[0]}D, expected {metadata['action_dim']}D"
        )
    image_features = tuple(config.image_features.keys())
    expected_image_features = tuple(
        f"observation.images.{key}"
        for key in _camera_keys(str(metadata["arm_mode"]), str(metadata["arm_side"]))
    )
    required_images = len(expected_image_features)
    if set(image_features) != set(expected_image_features) or len(image_features) != required_images:
        raise ValueError(
            "SmolVLA checkpoint image features do not match Piper camera contract: "
            f"declared={list(image_features)!r}, expected={list(expected_image_features)!r}"
        )
    for feature_name in expected_image_features:
        shape = tuple(config.image_features[feature_name].shape)
        if len(shape) != 3 or shape[0] != 3:
            raise ValueError(
                "SmolVLA Piper camera features must be channel-first RGB (3,H,W): "
                f"{feature_name} has shape {shape!r}"
            )
    config.chunk_size = max(16, int(config.chunk_size))
    config.n_action_steps = config.chunk_size
    policy = SmolVLAPolicy(config)
    weights_path = checkpoint / "model.safetensors"
    SmolVLAPolicy._load_as_safetensor(policy, str(weights_path), "cpu", strict=False)
    # LeRobot's generic loader intentionally skips normalization buffers. The
    # inference path needs the buffers embedded in this exact artifact.
    from safetensors.torch import load_file

    state_dict = load_file(str(weights_path), device="cpu")
    normalized_state, _ = standardise_state_dict(state_dict, set(policy.state_dict().keys()))
    norm_prefixes = ("normalize_inputs.", "normalize_targets.", "unnormalize_outputs.")
    norm_state = {
        key: value for key, value in normalized_state.items() if key.startswith(norm_prefixes)
    }
    required_prefixes = ("normalize_inputs.", "unnormalize_outputs.")
    missing_required = [
        prefix for prefix in required_prefixes if not any(key.startswith(prefix) for key in norm_state)
    ]
    if missing_required:
        raise ValueError(
            "SmolVLA checkpoint is missing required normalization buffers: "
            + ", ".join(missing_required)
        )
    missing, unexpected = policy.load_state_dict(norm_state, strict=False)
    missing_required = [key for key in missing if key.startswith(required_prefixes)]
    unexpected_norm = [key for key in unexpected if key.startswith(norm_prefixes)]
    if missing_required or unexpected_norm:
        raise RuntimeError(
            "could not restore SmolVLA normalization buffers; "
            f"missing={missing_required!r}, unexpected={unexpected_norm!r}"
        )
    precision = getattr(args, "precision", "auto")
    if precision == "auto":
        precision = "fp16" if device.startswith("cuda") else "fp32"
    if precision == "fp16":
        if not device.startswith("cuda"):
            raise ValueError("--precision fp16 requires a CUDA device")
        policy = policy.half()
    elif precision == "bf16":
        if not device.startswith("cuda"):
            raise ValueError("--precision bf16 requires a CUDA device")
        policy = policy.bfloat16()
    elif precision != "fp32":
        raise ValueError(f"unsupported precision {precision!r}")
    policy = policy.to(device)
    policy.eval()
    return _SmolVLAPolicyAdapter(
        policy,
        arm_mode=str(metadata["arm_mode"]),
        arm_side=str(metadata["arm_side"]),
        action_dim=int(metadata["action_dim"]),
        image_features=expected_image_features,
        instruction=args.default_prompt,
    )


def _install_transformers_jetson_workarounds() -> None:
    """Avoid Jetson unified-memory allocator probes during Transformers import."""

    try:
        import transformers.modeling_utils as modeling_utils

        modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
    except Exception:
        return


def _merge_runtime_metadata(args: argparse.Namespace, checkpoint: Path) -> dict[str, Any]:
    source = load_policy_metadata(args.metadata, checkpoint)
    required = {
        "schema": args.schema,
        "arm_mode": args.arm_mode,
        "arm_side": "both" if args.arm_mode == "bimanual" else args.arm_side,
        "dataset_id": args.dataset_id,
        "model_variant": args.model_variant,
        "backend": args.backend,
    }
    # CLI values are authoritative. A metadata file is useful for inspection,
    # but it cannot silently change the robot contract selected by the operator.
    for key, value in required.items():
        if value is not None:
            advertised = source.get(key)
            if advertised is not None and str(advertised) != str(value):
                raise ValueError(f"metadata {key}={advertised!r} conflicts with CLI value {value!r}")
    metadata = build_policy_metadata(
        **required,
        checkpoint=checkpoint,
        profile=args.profile,
        action_horizon=getattr(args, "action_horizon", DEFAULT_ACTION_HORIZON),
        action_hz=getattr(args, "action_hz", DEFAULT_ACTION_HZ),
        rtc_enabled=getattr(args, "rtc_enabled", False),
        rtc_execution_horizon=getattr(args, "rtc_execution_horizon", 8),
        rtc_max_guidance_weight=getattr(args, "rtc_max_guidance_weight", 5.0),
        rtc_prefix_attention_schedule=getattr(args, "rtc_prefix_attention_schedule", "linear"),
    )
    protected = {
        "deployment_target",
        "deployment_profile",
        "dataset_id",
        "checkpoint",
        "model_variant",
        "inference_backend",
        "transport",
        "schema",
        "arm_mode",
        "arm_side",
        "contract_version",
        "state_dim",
        "action_dim",
        "raw_action_dim",
        "model_action_dim",
        "camera_keys",
        "action_hz",
        "action_horizon",
        "action_time_step_s",
        "action_start_offset_steps",
        "action_offset",
        "model_action_start_offset",
        "minimum_horizon",
        "action_semantics",
        "wire_action_semantics",
        "wire_action_convention",
        "raw_action_semantics",
        "model_action_semantics",
        "raw_action_convention",
        "model_action_convention",
        "delivery_action_convention",
        "raw_gripper_semantics",
        "model_gripper_semantics",
        "wire_gripper_semantics",
        "state_gripper_semantics",
        "gripper_semantics",
        "action_source",
        "action_alignment",
        "legacy_delivery_v2",
        "legacy_joint_v2",
        "rtc_enabled",
        "rtc_algorithm",
        "rtc_backend",
        "rtc_execution_horizon",
        "rtc_max_guidance_weight",
        "rtc_prefix_attention_schedule",
        "rtc_physical_action_dim",
        "rtc_chunk_origin_reanchoring",
    }
    for key in protected:
        source.pop(key, None)
    metadata.update(source)
    return metadata


def _add_common(parser: argparse.ArgumentParser, *, serve: bool) -> None:
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--schema", choices=("joint", "delivery"), required=True)
    parser.add_argument("--arm-mode", choices=("single", "bimanual"), default="bimanual")
    parser.add_argument("--arm-side", choices=("left", "right", "both"), default="both")
    parser.add_argument("--backend", choices=("pi", "smolvla"), default="pi")
    parser.add_argument("--model-variant", choices=("pi0", "pi05", "smolvla"), default="pi05")
    parser.add_argument("--profile", choices=tuple(sorted(NX_PROFILES)), default="orin_nx_16gb")
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--norm-stats", default=None)
    if serve:
        parser.add_argument("--openpi-root", default=str(DEFAULT_OPENPI_ROOT))
        parser.add_argument("--host", default="0.0.0.0")
        parser.add_argument("--port", type=int, default=8000)
        parser.add_argument("--device", default="auto")
        parser.add_argument("--allow-cpu", action="store_true", help="allow a CPU server for smoke tests")
        parser.add_argument(
            "--precision",
            choices=("auto", "fp16", "bf16", "fp32"),
            default="auto",
            help="model dtype; OpenPI uses BF16/FP32, SmolVLA supports FP16/BF16/FP32",
        )
        parser.add_argument("--default-prompt", default=None)
        parser.add_argument("--paligemma-variant", default="gemma_2b_lora")
        parser.add_argument("--action-expert-variant", default="gemma_300m_lora")
        parser.add_argument("--compile-mode", choices=("none", "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"), default="none")
        parser.add_argument("--action-horizon", type=int, default=DEFAULT_ACTION_HORIZON)
        parser.add_argument("--action-hz", type=float, default=DEFAULT_ACTION_HZ)
        parser.add_argument("--rtc-enabled", action=argparse.BooleanOptionalAction, default=None)
        parser.add_argument("--rtc-execution-horizon", type=int, default=8)
        parser.add_argument("--rtc-max-guidance-weight", type=float, default=5.0)
        parser.add_argument("--rtc-prefix-attention-schedule", choices=("zeros", "ones", "linear", "exp"), default="linear")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Orin NX local OpenPI policy node")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="inspect checkpoint, profile, and policy contract")
    _add_common(check, serve=False)
    check.add_argument("--device", default="cpu")
    serve = sub.add_parser("serve", help="load PyTorch OpenPI and serve WebSocket inference")
    _add_common(serve, serve=True)
    args = parser.parse_args(argv)
    if not hasattr(args, "rtc_enabled"):
        args.rtc_enabled = False
    if args.backend == "smolvla":
        args.model_variant = "smolvla"
        if args.rtc_enabled is None:
            args.rtc_enabled = False
        elif args.rtc_enabled:
            parser.error("SmolVLA backend does not support model-side RTC; use --no-rtc-enabled")
    elif args.rtc_enabled is None:
        args.rtc_enabled = True
    elif args.model_variant == "smolvla":
        parser.error("--model-variant smolvla requires --backend smolvla")
    if args.arm_mode == "bimanual" and args.arm_side != "both":
        parser.error("bimanual edge policies require --arm-side both")
    if args.arm_mode == "single" and args.arm_side == "both":
        parser.error("single-arm edge policies require --arm-side left or right")
    return args


def run_check(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if args.backend == "smolvla":
        args.model_variant = "smolvla"
    report = inspect_checkpoint(
        checkpoint,
        dataset_id=args.dataset_id,
        norm_stats=args.norm_stats,
        backend=args.backend,
        expected_state_dim=_contract_dimensions(args.schema, args.arm_mode)[0],
        expected_action_dim=_contract_dimensions(args.schema, args.arm_mode)[1],
        expected_camera_keys=_camera_keys(
            args.arm_mode, "both" if args.arm_mode == "bimanual" else args.arm_side
        ),
    )
    if args.backend == "smolvla" and report.get("declared_chunk_size") is not None:
        args.action_horizon = int(report["declared_chunk_size"])
    device = check_device(profile=args.profile, device=args.device, model_variant=args.model_variant)
    metadata = _merge_runtime_metadata(args, checkpoint)
    report.update({"device": device, "policy_metadata": metadata})
    return report


def run_serve(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if args.backend == "smolvla":
        args.model_variant = "smolvla"
    report = inspect_checkpoint(
        checkpoint,
        dataset_id=args.dataset_id,
        norm_stats=args.norm_stats,
        backend=args.backend,
        expected_state_dim=_contract_dimensions(args.schema, args.arm_mode)[0],
        expected_action_dim=_contract_dimensions(args.schema, args.arm_mode)[1],
        expected_camera_keys=_camera_keys(
            args.arm_mode, "both" if args.arm_mode == "bimanual" else args.arm_side
        ),
    )
    if args.backend == "smolvla" and report.get("declared_chunk_size") is not None:
        args.action_horizon = int(report["declared_chunk_size"])
    device_report = check_device(profile=args.profile, device=args.device, model_variant=args.model_variant)
    args.device = device_report["device"]
    if args.device == "cpu" and not args.allow_cpu:
        raise RuntimeError("CPU serving is disabled by default; pass --allow-cpu only for a smoke test")
    metadata = _merge_runtime_metadata(args, checkpoint)
    LOGGER.info("Loading edge policy: profile=%s device=%s checkpoint=%s", args.profile, args.device, checkpoint)
    policy = (
        _build_smolvla_policy(args, metadata, checkpoint)
        if args.backend == "smolvla"
        else _build_openpi_policy(args, metadata, checkpoint)
    )
    if args.backend == "smolvla":
        server = EdgeWebsocketPolicyServer(
            policy=policy,
            host=args.host,
            port=args.port,
            metadata=metadata,
        )
    else:
        _add_openpi_root(args.openpi_root)
        from openpi.serving import websocket_policy_server

        server = websocket_policy_server.WebsocketPolicyServer(
            policy=policy,
            host=args.host,
            port=args.port,
            metadata=metadata,
        )
    LOGGER.info("Orin NX policy server listening on ws://%s:%d", args.host, args.port)
    server.serve_forever()


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    args = parse_args(argv)
    if args.command == "check":
        print(json.dumps(run_check(args), indent=2, ensure_ascii=False))
    else:
        run_serve(args)


if __name__ == "__main__":
    main()
