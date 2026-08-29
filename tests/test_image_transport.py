from __future__ import annotations

import unittest

import numpy as np

from bimanual_vla.data.contract import IMAGE_HW
from bimanual_vla.deployment.image_transport import (
    ImageTransportConfig,
    ImageTransportPolicy,
    OptimizedPolicyClient,
    decode_observation_images,
    negotiate_image_transport,
    prepare_observation_images,
    server_image_transport_metadata,
)


def smooth_rgb_chw(offset: int = 0) -> np.ndarray:
    height, width = IMAGE_HW
    x = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    y = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    image = np.stack(
        [
            np.broadcast_to((x + offset).astype(np.uint8), (height, width)),
            np.broadcast_to((y + offset).astype(np.uint8), (height, width)),
            np.full((height, width), 64 + offset, dtype=np.uint8),
        ],
        axis=0,
    )
    return image


class ImageTransportNegotiationTest(unittest.TestCase):
    def test_auto_prefers_jpeg_and_legacy_server_falls_back_to_raw(self):
        metadata = {"image_transport": server_image_transport_metadata()}
        self.assertEqual(negotiate_image_transport(metadata).encoding, "jpeg")
        self.assertEqual(negotiate_image_transport({}).encoding, "raw")

    def test_explicit_jpeg_fails_closed_for_legacy_server(self):
        with self.assertRaisesRegex(ValueError, "does not support"):
            negotiate_image_transport({}, requested="jpeg")


class ImageTransportCodecTest(unittest.TestCase):
    def observation(self) -> dict:
        return {
            "state": np.zeros(20, dtype=np.float32),
            "images": {
                "cam_high": smooth_rgb_chw(0),
                "cam_left_wrist": smooth_rgb_chw(8),
                "cam_right_wrist": smooth_rgb_chw(16),
            },
            "prompt": "test",
            "client_metadata": {"inference_generation": 3},
        }

    def test_jpeg_round_trip_preserves_keys_shape_layout_and_dtype(self):
        original = self.observation()
        prepared, encode_metrics = prepare_observation_images(
            original, ImageTransportConfig("jpeg", 90)
        )

        self.assertEqual(prepared["client_metadata"]["image_transport"], "jpeg")
        self.assertLess(encode_metrics["image_encoded_bytes"], encode_metrics["image_raw_bytes"])
        self.assertTrue(all(isinstance(value, bytes) for value in prepared["images"].values()))

        decode_metrics = decode_observation_images(prepared, expected_hw=IMAGE_HW)
        self.assertEqual(decode_metrics["image_transport"], "jpeg")
        self.assertEqual(set(prepared["images"]), set(original["images"]))
        for key, decoded in prepared["images"].items():
            self.assertEqual(decoded.shape, (3, *IMAGE_HW))
            self.assertEqual(decoded.dtype, np.uint8)
            mean_error = np.abs(
                decoded.astype(np.int16) - original["images"][key].astype(np.int16)
            ).mean()
            self.assertLess(mean_error, 3.0)

    def test_raw_transport_keeps_array_payloads(self):
        original = self.observation()
        prepared, metrics = prepare_observation_images(
            original, ImageTransportConfig("raw")
        )
        self.assertEqual(metrics["image_raw_bytes"], metrics["image_encoded_bytes"])
        self.assertTrue(all(isinstance(value, np.ndarray) for value in prepared["images"].values()))
        decoded = decode_observation_images(prepared, expected_hw=IMAGE_HW)
        self.assertEqual(decoded["image_transport"], "raw")

    def test_policy_wrapper_decodes_before_model(self):
        class FakePolicy:
            def infer(self, observation):
                return {
                    "actions": np.zeros((10, 14), dtype=np.float32),
                    "decoded_shapes": {
                        key: list(value.shape) for key, value in observation["images"].items()
                    },
                }

        prepared, _ = prepare_observation_images(
            self.observation(), ImageTransportConfig("jpeg", 90)
        )
        result = ImageTransportPolicy(FakePolicy(), expected_hw=IMAGE_HW).infer(prepared)
        self.assertEqual(result["decoded_shapes"]["cam_high"], [3, *IMAGE_HW])
        self.assertGreaterEqual(result["transport_timing"]["server_image_decode_ms"], 0.0)

    def test_optimized_client_reports_actual_message_sizes(self):
        from openpi_client import msgpack_numpy

        class FakeWebsocket:
            def __init__(self):
                self.request = b""

            def send(self, request):
                self.request = request

            def recv(self):
                return msgpack_numpy.packb(
                    {"actions": np.zeros((10, 14), dtype=np.float32)}
                )

        class FakeClient:
            def __init__(self):
                self._ws = FakeWebsocket()
                self._packer = msgpack_numpy.Packer()

            def get_server_metadata(self):
                return {"image_transport": server_image_transport_metadata()}

        base = FakeClient()
        client = OptimizedPolicyClient(base, ImageTransportConfig("jpeg", 90))
        result = client.infer(self.observation())
        metrics = result["_client_wire_metrics"]
        self.assertEqual(metrics["request_bytes"], len(base._ws.request))
        self.assertGreater(metrics["response_bytes"], 0)
        self.assertGreater(metrics["image_compression_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
