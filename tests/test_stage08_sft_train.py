from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from medenvscale.train.train_sft_lora import (
    _apply_cuda_visible_devices,
    _offset_overlaps_any_span,
    _sft_training_device_map,
    _training_args_eval_strategy_key,
    render_tool_sft_messages,
    render_tool_sft_messages_with_assistant_spans,
    run_train_sft,
)
from medenvscale.train.checkpoints import latest_trainer_checkpoint


class Stage08SftTrainTests(unittest.TestCase):
    def test_render_tool_sft_messages_includes_tool_calls(self) -> None:
        text = render_tool_sft_messages(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "task"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "validate_candidate_code", "arguments": "{\"code\": \"x=1\"}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "name": "validate_candidate_code", "content": "{\"ok\": true}"},
            ]
        )

        self.assertIn("<|system|>", text)
        self.assertIn('"tool_calls"', text)
        self.assertIn('"name": "validate_candidate_code"', text)
        self.assertIn('"arguments": {"code": "x=1"}', text)
        self.assertNotIn('"function"', text)
        self.assertNotIn("<tool_call name=validate_candidate_code>", text)
        self.assertIn("<|tool name=validate_candidate_code|>", text)

    def test_render_tool_sft_messages_marks_only_assistant_loss_spans(self) -> None:
        text, spans = render_tool_sft_messages_with_assistant_spans(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "task"},
                {"role": "assistant", "content": "draft"},
                {"role": "tool", "name": "validate_candidate_code", "content": "{\"ok\": true}"},
                {"role": "assistant", "content": "final"},
            ]
        )

        assistant_positions = [text.index("draft"), text.index("final")]
        tool_position = text.index('"ok"')

        self.assertEqual(len(spans), 2)
        self.assertTrue(all(_offset_overlaps_any_span((pos, pos + 1), [(a, b) for a, b in spans]) for pos in assistant_positions))
        self.assertFalse(_offset_overlaps_any_span((tool_position, tool_position + 1), [(a, b) for a, b in spans]))

    def test_run_train_sft_dry_run_prepares_manifest(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="stage08-sft-"))
        (root / "src" / "medenvscale").mkdir(parents=True)
        (root / "configs").mkdir(parents=True)
        (root / "configs" / "llm.yaml").write_text("cache: {dir: .cache/llm}\n", encoding="utf-8")
        train_dir = root / "data" / "biocoder" / "splits" / "tool_sft" / "teacher"
        train_dir.mkdir(parents=True)
        row = {
            "sample_id": "s1",
            "env_id": "env1",
            "trajectory_type": "oracle_gold_tool_trajectory",
            "messages": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}],
        }
        for split in ("train", "dev"):
            (train_dir / f"tool_sft_{split}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        cfg_path = root / "configs" / "biocoder" / "train_sft.yaml"
        cfg_path.parent.mkdir(parents=True)
        cfg_path.write_text(
            "\n".join(
                [
                    "default_dataset: biocoder",
                    "trainer: trl_sft_lora",
                    "teacher_slug: teacher",
                    "dataset_path: data/splits/tool_sft_train.jsonl",
                    "eval_path: data/splits/tool_sft_dev.jsonl",
                    "output_dir: experiments/tool_sft_lora",
                    "model_name_or_path: dummy-model",
                    "max_steps: 1",
                    "lora:",
                    "  enabled: true",
                    "  r: 4",
                    "  alpha: 8",
                    "  dropout: 0.0",
                ]
            ),
            encoding="utf-8",
        )

        result = run_train_sft(str(cfg_path), dataset="biocoder", dry_run=True)

        self.assertEqual(result["status"], "dry_run")
        self.assertEqual(result["num_train_samples"], 1)
        self.assertTrue(Path(result["prepared_train_path"]).exists())
        prepared_row = json.loads(Path(result["prepared_train_path"]).read_text(encoding="utf-8").splitlines()[0])
        self.assertTrue(prepared_row["assistant_loss_spans"])
        self.assertTrue((Path(result["output_dir"]) / "train_manifest.json").exists())

    def test_training_args_eval_strategy_key_supports_old_and_new_names(self) -> None:
        class OldTrainingArguments:
            def __init__(self, evaluation_strategy=None):
                pass

        class NewTrainingArguments:
            def __init__(self, eval_strategy=None):
                pass

        self.assertEqual(_training_args_eval_strategy_key(OldTrainingArguments), "evaluation_strategy")
        self.assertEqual(_training_args_eval_strategy_key(NewTrainingArguments), "eval_strategy")

    def test_sft_training_device_map_disables_auto_model_parallel(self) -> None:
        self.assertIsNone(_sft_training_device_map({"device_map": "auto"}))
        self.assertIsNone(_sft_training_device_map({"device_map": None}))
        self.assertIsNone(_sft_training_device_map({"device_map": "trainer"}))
        self.assertEqual(_sft_training_device_map({"device_map": {"": 0}}), {"": 0})

    def test_apply_cuda_visible_devices_defaults_to_single_gpu_without_overriding_external_env(self) -> None:
        old = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        try:
            _apply_cuda_visible_devices({})
            self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "0")
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"
            _apply_cuda_visible_devices({"cuda_visible_devices": "0"})
            self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "1")
        finally:
            if old is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = old

    def test_latest_trainer_checkpoint_uses_highest_step(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="trainer-checkpoints-"))
        (root / "checkpoint-2").mkdir()
        (root / "checkpoint-10").mkdir()
        (root / "checkpoint-final").mkdir()

        self.assertEqual(latest_trainer_checkpoint(root), root / "checkpoint-10")


if __name__ == "__main__":
    unittest.main()
