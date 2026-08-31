from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from bimanual_vla.deployment.local_policy_server import (
    RTX5060_PROFILES,
    build_policy_metadata,
    check_device,
    get_rtx5060_profile,
    inspect_checkpoint,
    parse_args,
)


class LocalPolicyServerTest(unittest.TestCase):
    def test_rtx5060_profile_is_conservative(self):
        self.assertEqual(get_rtx5060_profile("rtx5060_8gb").recommended_models, ("pi0", "smolvla"))
        self.assertIn("pi05", get_rtx5060_profile("rtx5060_8gb").experimental_models)
        self.assertEqual(set(RTX5060_PROFILES), {"rtx5060_8gb"})

    def test_build_bimanual_joint_metadata_matches_client_contract(self):
        metadata = build_policy_metadata(
            schema="joint",
            arm_mode="bimanual",
            arm_side="both",
            dataset_id="demo",
            model_variant="pi0",
            checkpoint="/tmp/demo",
            profile="rtx5060_8gb",
        )
        self.assertEqual(metadata["state_dim"], 14)
        self.assertEqual(metadata["action_dim"], 14)
        self.assertEqual(metadata["camera_keys"], ["cam_high", "cam_left_wrist", "cam_right_wrist"])
        self.assertEqual(metadata["transport"], "openpi_websocket_v1")
        self.assertEqual(metadata["deployment_target"], "rtx5060")
        self.assertEqual(metadata["rtc_enabled"], False)

    def test_build_single_delivery_metadata(self):
        metadata = build_policy_metadata(
            schema="delivery",
            arm_mode="single",
            arm_side="right",
            dataset_id="demo",
            model_variant="pi05",
            checkpoint="/tmp/demo",
            profile="rtx5060_8gb",
            rtc_enabled=True,
        )
        self.assertEqual(metadata["state_dim"], 10)
        self.assertEqual(metadata["action_dim"], 7)
        self.assertEqual(metadata["raw_action_dim"], 10)
        self.assertEqual(metadata["camera_keys"], ["cam_high", "cam_right_wrist"])
        self.assertTrue(metadata["rtc_enabled"])

    def test_smolvla_metadata_is_piper_joint_only_and_client_compatible(self):
        metadata = build_policy_metadata(
            schema="joint",
            arm_mode="bimanual",
            arm_side="both",
            dataset_id="piper-smol",
            model_variant="smolvla",
            backend="smolvla",
            checkpoint="/tmp/smol",
            profile="rtx5060_8gb",
        )
        self.assertEqual(metadata["inference_backend"], "smolvla")
        self.assertFalse(metadata["rtc_enabled"])
        self.assertEqual(metadata["action_dim"], 14)

        from bimanual_vla.deployment.client import validate_policy_metadata

        protocol = validate_policy_metadata(metadata, "both", "bimanual")
        self.assertEqual(protocol.action_dim, 14)

    def test_smolvla_rejects_delivery_and_rtc_contracts(self):
        with self.assertRaises(ValueError):
            build_policy_metadata(
                schema="delivery",
                arm_mode="single",
                arm_side="right",
                dataset_id="demo",
                model_variant="smolvla",
                backend="smolvla",
                checkpoint="/tmp/smol",
                profile="rtx5060_8gb",
            )
        with self.assertRaises(ValueError):
            build_policy_metadata(
                schema="joint",
                arm_mode="single",
                arm_side="right",
                dataset_id="demo",
                model_variant="smolvla",
                backend="smolvla",
                checkpoint="/tmp/smol",
                profile="rtx5060_8gb",
                rtc_enabled=True,
            )

    def test_smolvla_checkpoint_requires_lerobot_config(self):
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp)
            (checkpoint / "model.safetensors").write_bytes(b"placeholder")
            with self.assertRaises(ValueError):
                inspect_checkpoint(checkpoint, backend="smolvla")
            (checkpoint / "config.json").write_text("{}", encoding="utf-8")
            report = inspect_checkpoint(checkpoint, backend="smolvla")
            self.assertEqual(report["format"], "lerobot_smolvla_safetensors")

    def test_inspect_requires_safetensors_and_norm_stats(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            with self.assertRaises(ValueError):
                inspect_checkpoint(checkpoint)
            (checkpoint / "model.safetensors").write_bytes(b"placeholder")
            with self.assertRaises(FileNotFoundError):
                inspect_checkpoint(checkpoint, dataset_id="demo")
            stats = checkpoint / "assets" / "demo"
            stats.mkdir(parents=True)
            (stats / "norm_stats.json").write_text(json.dumps({"state": {}}), encoding="utf-8")
            report = inspect_checkpoint(checkpoint, dataset_id="demo")
            self.assertEqual(report["format"], "pytorch_safetensors")
            self.assertEqual(Path(report["norm_stats"]), stats)

    def test_cpu_check_does_not_require_openpi(self):
        result = check_device(profile="rtx5060_8gb", device="cpu", model_variant="pi05")
        self.assertEqual(result["device"], "cpu")
        self.assertEqual(result["profile_support"], "experimental")
        self.assertEqual(result["minimum_cuda_memory_gb"], 7.0)
        self.assertEqual(result["memory_check"], "not_available")

    def test_cli_has_separate_check_and_serve_modes(self):
        args = parse_args(
            [
                "check",
                "--checkpoint",
                "/tmp/ckpt",
                "--dataset-id",
                "demo",
                "--schema",
                "joint",
            ]
        )
        self.assertEqual(args.command, "check")
        self.assertEqual(args.profile, "rtx5060_8gb")

        serve_args = parse_args(
            [
                "serve",
                "--checkpoint",
                "/tmp/ckpt",
                "--dataset-id",
                "demo",
                "--schema",
                "joint",
                "--precision",
                "bf16",
            ]
        )
        self.assertEqual(serve_args.precision, "bf16")

        smol_args = parse_args(
            [
                "check",
                "--backend",
                "smolvla",
                "--checkpoint",
                "/tmp/ckpt",
                "--dataset-id",
                "demo",
                "--schema",
                "joint",
                "--arm-mode",
                "single",
                "--arm-side",
                "right",
            ]
        )
        self.assertEqual(smol_args.model_variant, "smolvla")
        self.assertFalse(smol_args.rtc_enabled)


if __name__ == "__main__":
    unittest.main()
