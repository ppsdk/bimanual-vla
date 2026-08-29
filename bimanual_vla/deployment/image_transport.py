"""Negotiated image transport for the OpenPI WebSocket protocol."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Mapping

import cv2
import numpy as np


IMAGE_TRANSPORT_VERSION = 1
RAW_IMAGE_TRANSPORT = "raw"
JPEG_IMAGE_TRANSPORT = "jpeg"
DEFAULT_JPEG_QUALITY = 90
MAX_ENCODED_IMAGE_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class ImageTransportConfig:
    encoding: str = RAW_IMAGE_TRANSPORT
    jpeg_quality: int = DEFAULT_JPEG_QUALITY

    def __post_init__(self) -> None:
        if self.encoding not in {RAW_IMAGE_TRANSPORT, JPEG_IMAGE_TRANSPORT}:
            raise ValueError(f"unsupported image transport: {self.encoding!r}")
        if not 1 <= int(self.jpeg_quality) <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")


def server_image_transport_metadata(
    *, preferred: str = JPEG_IMAGE_TRANSPORT, jpeg_quality: int = DEFAULT_JPEG_QUALITY
) -> dict[str, Any]:
    config = ImageTransportConfig(preferred, jpeg_quality)
    return {
        "version": IMAGE_TRANSPORT_VERSION,
        "supported": [RAW_IMAGE_TRANSPORT, JPEG_IMAGE_TRANSPORT],
        "preferred": config.encoding,
        "jpeg_quality": config.jpeg_quality,
    }


def negotiate_image_transport(
    server_metadata: Mapping[str, Any],
    *,
    requested: str = "auto",
    jpeg_quality: int | None = None,
) -> ImageTransportConfig:
    requested = str(requested or "auto").strip().lower()
    if requested not in {"auto", RAW_IMAGE_TRANSPORT, JPEG_IMAGE_TRANSPORT}:
        raise ValueError(f"unsupported requested image transport: {requested!r}")

    advertised = server_metadata.get("image_transport")
    if isinstance(advertised, Mapping):
        version = advertised.get("version")
        if version != IMAGE_TRANSPORT_VERSION:
            raise ValueError(
                f"unsupported server image transport version: {version!r}"
            )
        raw_supported = advertised.get("supported")
        if not isinstance(raw_supported, (list, tuple)):
            raise ValueError("server image transport metadata has no supported list")
        supported = {str(value).strip().lower() for value in raw_supported}
        if not supported or not supported <= {RAW_IMAGE_TRANSPORT, JPEG_IMAGE_TRANSPORT}:
            raise ValueError(f"invalid server image transports: {sorted(supported)!r}")
        preferred = str(advertised.get("preferred") or RAW_IMAGE_TRANSPORT).lower()
        if preferred not in supported:
            raise ValueError("server preferred image transport is not supported")
        advertised_quality = advertised.get("jpeg_quality", DEFAULT_JPEG_QUALITY)
    else:
        # Servers predating this extension only understand raw ndarrays.
        supported = {RAW_IMAGE_TRANSPORT}
        preferred = RAW_IMAGE_TRANSPORT
        advertised_quality = DEFAULT_JPEG_QUALITY

    encoding = preferred if requested == "auto" else requested
    if encoding not in supported:
        raise ValueError(
            f"server does not support requested image transport {encoding!r}; "
            f"supported={sorted(supported)!r}"
        )
    quality = advertised_quality if jpeg_quality is None else jpeg_quality
    return ImageTransportConfig(encoding=encoding, jpeg_quality=int(quality))


def _rgb_hwc(image: Any) -> tuple[np.ndarray, str]:
    value = np.asarray(image)
    if value.dtype != np.uint8:
        raise ValueError(f"transport image must be uint8, got {value.dtype}")
    if value.ndim != 3:
        raise ValueError(f"transport image must be 3D, got {value.shape}")
    if value.shape[0] == 3:
        return np.ascontiguousarray(value.transpose(1, 2, 0)), "chw"
    if value.shape[-1] == 3:
        return np.ascontiguousarray(value), "hwc"
    raise ValueError(f"transport image must be RGB CHW or HWC, got {value.shape}")


def prepare_observation_images(
    observation: Mapping[str, Any], config: ImageTransportConfig
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a shallow observation copy with negotiated image payloads."""
    images = observation.get("images")
    if not isinstance(images, Mapping) or not images:
        raise ValueError("observation images must be a non-empty mapping")

    prepared = dict(observation)
    client_metadata = dict(observation.get("client_metadata") or {})
    raw_bytes = 0
    encoded_bytes = 0
    layouts: dict[str, str] = {}
    shapes: dict[str, list[int]] = {}
    encoded_images: dict[str, Any] = {}
    started = time.monotonic()
    for key, image in images.items():
        rgb, layout = _rgb_hwc(image)
        raw_bytes += int(rgb.nbytes)
        layouts[str(key)] = layout
        shapes[str(key)] = list(rgb.shape)
        if config.encoding == JPEG_IMAGE_TRANSPORT:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            ok, payload = cv2.imencode(
                ".jpg",
                bgr,
                [cv2.IMWRITE_JPEG_QUALITY, int(config.jpeg_quality)],
            )
            if not ok:
                raise RuntimeError(f"JPEG encoding failed for {key}")
            value = payload.tobytes()
            if len(value) > MAX_ENCODED_IMAGE_BYTES:
                raise ValueError(f"encoded image {key} exceeds transport size limit")
            encoded_images[str(key)] = value
            encoded_bytes += len(value)
        else:
            encoded_images[str(key)] = image
            encoded_bytes += int(np.asarray(image).nbytes)
    encode_ms = (time.monotonic() - started) * 1000.0

    client_metadata.update(
        {
            "image_transport": config.encoding,
            "image_transport_version": IMAGE_TRANSPORT_VERSION,
            "image_transport_layouts": layouts,
            "image_transport_shapes": shapes,
            "image_transport_jpeg_quality": (
                config.jpeg_quality if config.encoding == JPEG_IMAGE_TRANSPORT else None
            ),
            "image_transport_raw_bytes": raw_bytes,
            "image_transport_encoded_bytes": encoded_bytes,
        }
    )
    prepared["images"] = encoded_images
    prepared["client_metadata"] = client_metadata
    compression_ratio = raw_bytes / encoded_bytes if encoded_bytes else None
    return prepared, {
        "image_encode_ms": encode_ms,
        "image_raw_bytes": raw_bytes,
        "image_encoded_bytes": encoded_bytes,
        "image_compression_ratio": compression_ratio,
        "image_transport": config.encoding,
    }


def decode_observation_images(
    observation: dict[str, Any], *, expected_hw: tuple[int, int]
) -> dict[str, Any]:
    """Decode JPEG images in place and return bounded transport metrics."""
    client_metadata = observation.get("client_metadata")
    client_metadata = client_metadata if isinstance(client_metadata, dict) else {}
    encoding = str(client_metadata.get("image_transport") or RAW_IMAGE_TRANSPORT).lower()
    if encoding == RAW_IMAGE_TRANSPORT:
        return {"image_transport": encoding, "image_decode_ms": 0.0}
    if encoding != JPEG_IMAGE_TRANSPORT:
        raise ValueError(f"unsupported observation image transport: {encoding!r}")
    if client_metadata.get("image_transport_version") != IMAGE_TRANSPORT_VERSION:
        raise ValueError("JPEG observation has an incompatible image transport version")

    images = observation.get("images")
    layouts = client_metadata.get("image_transport_layouts")
    if not isinstance(images, Mapping) or not images or not isinstance(layouts, Mapping):
        raise ValueError("JPEG observation is missing images or layout metadata")

    decoded: dict[str, np.ndarray] = {}
    encoded_bytes = 0
    started = time.monotonic()
    for key, payload in images.items():
        if isinstance(payload, bytes):
            encoded = np.frombuffer(payload, dtype=np.uint8)
        else:
            encoded = np.asarray(payload, dtype=np.uint8).reshape(-1)
        if not 0 < encoded.size <= MAX_ENCODED_IMAGE_BYTES:
            raise ValueError(f"encoded image {key} has invalid size {encoded.size}")
        encoded_bytes += int(encoded.size)
        bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"JPEG decoding failed for {key}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape != (*expected_hw, 3):
            raise ValueError(
                f"decoded image {key} must have shape {(*expected_hw, 3)}, got {rgb.shape}"
            )
        layout = str(layouts.get(key) or "").lower()
        if layout == "chw":
            decoded[str(key)] = np.ascontiguousarray(rgb.transpose(2, 0, 1))
        elif layout == "hwc":
            decoded[str(key)] = np.ascontiguousarray(rgb)
        else:
            raise ValueError(f"encoded image {key} has invalid layout {layout!r}")
    decode_ms = (time.monotonic() - started) * 1000.0
    observation["images"] = decoded
    return {
        "image_transport": encoding,
        "image_decode_ms": decode_ms,
        "image_encoded_bytes": encoded_bytes,
        "image_raw_bytes": sum(int(value.nbytes) for value in decoded.values()),
    }


class OptimizedPolicyClient:
    """Add image negotiation and wire measurements to an OpenPI client."""

    def __init__(self, policy: Any, config: ImageTransportConfig):
        self._policy = policy
        self._ws = policy._ws
        self._packer = policy._packer
        self._server_metadata = policy.get_server_metadata()
        self.image_transport = config

    def get_server_metadata(self) -> dict[str, Any]:
        return self._server_metadata

    def infer(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        from openpi_client import msgpack_numpy

        prepared, metrics = prepare_observation_images(observation, self.image_transport)
        client_metadata = prepared["client_metadata"]
        request_sent_at = time.time()
        client_metadata["request_sent_at"] = request_sent_at

        pack_started = time.monotonic()
        request = self._packer.pack(prepared)
        pack_finished = time.monotonic()
        wire_send_started_at = time.time()
        request_sent_monotonic = time.monotonic()
        self._ws.send(request)
        response = self._ws.recv()
        response_received_at = time.time()
        response_received_monotonic = time.monotonic()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        unpack_started = time.monotonic()
        result = msgpack_numpy.unpackb(response)
        unpack_finished = time.monotonic()
        if not isinstance(result, dict):
            raise RuntimeError(f"invalid policy result: {type(result).__name__}")
        metrics.update(
            {
                "request_bytes": len(request),
                "response_bytes": len(response),
                "request_pack_ms": (pack_finished - pack_started) * 1000.0,
                "response_unpack_ms": (unpack_finished - unpack_started) * 1000.0,
                "wire_round_trip_ms": (
                    response_received_monotonic - request_sent_monotonic
                )
                * 1000.0,
                "request_sent_at": request_sent_at,
                "request_sent_monotonic": request_sent_monotonic,
                "wire_send_started_at": wire_send_started_at,
                "response_received_at": response_received_at,
                "response_received_monotonic": response_received_monotonic,
            }
        )
        result["_client_wire_metrics"] = metrics
        return result

    def reset(self) -> None:
        reset = getattr(self._policy, "reset", None)
        if callable(reset):
            reset()


class ImageTransportPolicy:
    """Decode negotiated images before invoking the normal Policy stack."""

    def __init__(self, policy: Any, *, expected_hw: tuple[int, int]):
        self.policy = policy
        self.expected_hw = tuple(expected_hw)

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        client_metadata = observation.get("client_metadata")
        if not isinstance(client_metadata, dict):
            client_metadata = {}
            observation["client_metadata"] = client_metadata
        client_metadata["server_transport_received_at"] = time.time()
        metrics = decode_observation_images(observation, expected_hw=self.expected_hw)
        # Keep codec timing available to an outer telemetry wrapper without
        # changing the public observation contract sent to the model.
        client_metadata["_server_image_transport_timing"] = dict(metrics)
        result = dict(self.policy.infer(observation))
        timing = result.get("transport_timing")
        timing = dict(timing) if isinstance(timing, dict) else {}
        timing.update(
            {
                "image_transport": metrics["image_transport"],
                "server_image_decode_ms": metrics["image_decode_ms"],
                "image_encoded_bytes": metrics.get("image_encoded_bytes"),
                "image_raw_bytes": metrics.get("image_raw_bytes"),
            }
        )
        result["transport_timing"] = timing
        return result

    def reset(self) -> None:
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset()
