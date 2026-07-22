from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from typing import Any

from medenvscale.config import load_training_config
from medenvscale.distributed import barrier, distributed_metadata, is_distributed, is_main_process, setup_torch_distributed_device
from medenvscale.train.checkpoints import latest_trainer_checkpoint
from medenvscale.utils import read_jsonl, write_jsonl


def run_train_sft(
    config_path: str,
    max_steps: int | None = None,
    dataset: str | None = None,
    *,
    model_name_or_path: str | None = None,
    teacher_slug: str | None = None,
    dry_run: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    root, cfg = load_training_config(config_path, dataset=dataset)
    if max_steps is not None:
        cfg["max_steps"] = int(max_steps)
    if model_name_or_path:
        cfg["model_name_or_path"] = str(model_name_or_path)
    if teacher_slug:
        cfg["teacher_slug"] = str(teacher_slug)

    output_dir = root / str(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = _resolve_sft_path(root, cfg, "dataset_path", "train")
    eval_path = _resolve_sft_path(root, cfg, "eval_path", "dev")
    prepared_train = output_dir / "prepared_train_text.jsonl"
    prepared_eval = output_dir / "prepared_eval_text.jsonl"
    if is_main_process():
        train_rows = _prepare_text_dataset(train_path, prepared_train)
        eval_rows = _prepare_text_dataset(eval_path, prepared_eval) if eval_path.exists() else []
    else:
        train_rows = []
        eval_rows = []
    barrier()
    if not is_main_process():
        train_rows = read_jsonl(prepared_train)
        eval_rows = read_jsonl(prepared_eval) if prepared_eval.exists() else []

    manifest = {
        "trainer": cfg.get("trainer", "trl_sft_lora"),
        "status": "dry_run" if dry_run else "training",
        "resume": bool(resume),
        "resume_from_checkpoint": str(latest_trainer_checkpoint(output_dir) or "") if resume else "",
        "dataset_path": str(train_path),
        "eval_path": str(eval_path) if eval_path.exists() else "",
        "prepared_train_path": str(prepared_train),
        "prepared_eval_path": str(prepared_eval) if eval_rows else "",
        "num_train_samples": len(train_rows),
        "num_eval_samples": len(eval_rows),
        "output_dir": str(output_dir),
        "model_name_or_path": cfg["model_name_or_path"],
        "max_steps": int(cfg.get("max_steps", 50)),
        "learning_rate": float(cfg.get("learning_rate", 2e-4)),
        "per_device_train_batch_size": int(cfg.get("per_device_train_batch_size", 1)),
        "gradient_accumulation_steps": int(cfg.get("gradient_accumulation_steps", 4)),
        "max_seq_length": int(cfg.get("max_seq_length", 4096)),
        "tool_format_version": "simplified_tool_json_v1",
        "lora": cfg.get("lora", {}),
        **distributed_metadata(),
    }
    _write_manifest(output_dir, manifest)
    if dry_run:
        manifest["note"] = "Dry run only: prepared text data and training manifest, no model weights loaded."
        _write_manifest(output_dir, manifest)
        barrier()
        if not is_main_process():
            manifest = json.loads((output_dir / "train_manifest.json").read_text(encoding="utf-8"))
        return manifest

    if not train_rows:
        raise ValueError(f"No SFT training rows found at {train_path}")
    _run_trl_lora_sft(
        root=root,
        cfg=cfg,
        manifest=manifest,
        train_path=prepared_train,
        eval_path=prepared_eval if eval_rows else None,
        resume=resume,
    )
    if is_main_process():
        manifest["status"] = "completed"
        _write_manifest(output_dir, manifest)
    barrier()
    if not is_main_process():
        manifest = json.loads((output_dir / "train_manifest.json").read_text(encoding="utf-8"))
    return manifest


def _resolve_sft_path(root: Path, cfg: dict[str, Any], key: str, split_name: str) -> Path:
    configured = root / str(cfg.get(key) or "")
    dataset = str(cfg.get("active_dataset") or cfg.get("default_dataset") or "").strip()
    teacher_slug = str(cfg.get("teacher_slug") or "").strip()
    if dataset and teacher_slug:
        candidate = root / "data" / dataset / "splits" / "tool_sft" / teacher_slug / f"tool_sft_{split_name}.jsonl"
        if candidate.exists():
            return candidate
        result_candidate = root / "result" / dataset / "07" / teacher_slug / f"tool_sft_{split_name}.jsonl"
        if result_candidate.exists():
            return result_candidate
    if configured.exists():
        return configured
    return configured


def _prepare_text_dataset(input_path: Path, output_path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(input_path) if input_path.exists() else []
    prepared = []
    for row in rows:
        text, assistant_spans = render_tool_sft_messages_with_assistant_spans(row.get("messages") or [])
        if not text.strip():
            continue
        prepared.append(
            {
                "sample_id": row.get("sample_id"),
                "env_id": row.get("env_id"),
                "trajectory_type": row.get("trajectory_type"),
                "text": text,
                "assistant_loss_spans": assistant_spans,
            }
        )
    write_jsonl(output_path, prepared)
    return prepared


def render_tool_sft_messages(messages: list[dict[str, Any]]) -> str:
    text, _ = render_tool_sft_messages_with_assistant_spans(messages)
    return text


def render_tool_sft_messages_with_assistant_spans(messages: list[dict[str, Any]]) -> tuple[str, list[list[int]]]:
    chunks: list[str] = []
    spans: list[list[int]] = []
    cursor = 0
    for message in messages:
        role = str(message.get("role") or "unknown")
        if role == "assistant" and message.get("tool_calls"):
            assistant_payload = {
                "tool_calls": [_simplified_tool_call_payload(call) for call in message.get("tool_calls") or []],
                "content": message.get("content") or "",
            }
            block = "\n".join(["<|assistant|>", json.dumps(assistant_payload, ensure_ascii=False)])
            chunks.append(block)
            spans.append([cursor, cursor + len(block)])
            cursor += len(block) + 1
            continue
        if role == "tool":
            name = str(message.get("name") or "")
            block = "\n".join([f"<|tool name={name}|>", str(message.get("content") or "")])
            chunks.append(block)
            cursor += len(block) + 1
            continue
        block = "\n".join([f"<|{role}|>", str(message.get("content") or "")])
        chunks.append(block)
        if role == "assistant":
            spans.append([cursor, cursor + len(block)])
        cursor += len(block) + 1
    chunks.append("<|end|>")
    return "\n".join(chunks).strip() + "\n", spans


def _simplified_tool_call_payload(call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    name = str(function.get("name") or call.get("name") or "")
    arguments = function.get("arguments", call.get("arguments", "{}"))
    if isinstance(arguments, str):
        try:
            parsed_arguments = json.loads(arguments)
        except json.JSONDecodeError:
            parsed_arguments = {"raw": arguments}
        arguments = parsed_arguments if isinstance(parsed_arguments, dict) else {"value": parsed_arguments}
    elif not isinstance(arguments, dict):
        arguments = {}
    return {
        "name": name,
        "arguments": arguments,
    }


def _tokenize_dataset_with_assistant_loss_mask(dataset: Any, tokenizer: Any, max_seq_length: int) -> Any:
    def tokenize_row(row: dict[str, Any]) -> dict[str, Any]:
        text = str(row.get("text") or "")
        spans = _normalize_loss_spans(row.get("assistant_loss_spans") or [])
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_length,
            return_offsets_mapping=True,
        )
        offsets = encoded.pop("offset_mapping", [])
        input_ids = list(encoded["input_ids"])
        labels = [
            token_id if _offset_overlaps_any_span(offset, spans) else -100
            for token_id, offset in zip(input_ids, offsets)
        ]
        encoded["labels"] = labels
        return encoded

    remove_columns = dataset["train"].column_names if "train" in dataset else []
    return dataset.map(tokenize_row, remove_columns=remove_columns)


def _normalize_loss_spans(raw_spans: Any) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for item in raw_spans or []:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        start = max(0, int(item[0]))
        end = max(start, int(item[1]))
        if end > start:
            spans.append((start, end))
    return spans


def _offset_overlaps_any_span(offset: Any, spans: list[tuple[int, int]]) -> bool:
    if not isinstance(offset, (list, tuple)) or len(offset) != 2:
        return False
    start, end = int(offset[0]), int(offset[1])
    if end <= start:
        return False
    return any(start < span_end and end > span_start for span_start, span_end in spans)


def _sft_training_device_map(cfg: dict[str, Any]) -> Any:
    raw = cfg.get("device_map")
    if raw is None:
        return None
    if isinstance(raw, str) and raw.strip().lower() in {"", "none", "null", "false", "trainer", "auto"}:
        return None
    return raw


def _move_model_to_training_device(model: Any, *, args: Any, device_map: Any) -> None:
    if device_map is not None:
        return
    device = getattr(args, "device", None)
    if device is None:
        return
    model.to(device)


def _run_trl_lora_sft(
    *,
    root: Path,
    cfg: dict[str, Any],
    manifest: dict[str, Any],
    train_path: Path,
    eval_path: Path | None,
    resume: bool,
) -> None:
    _apply_cuda_visible_devices(cfg)
    setup_torch_distributed_device()
    try:
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Stage08 SFT training now uses TRL SFTTrainer and requires transformers, datasets, peft, and trl. "
            "Install training dependencies first, for example: pip install -r requirements.txt"
        ) from exc

    model_name = str(cfg["model_name_or_path"])
    output_dir = root / str(cfg["output_dir"])
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=bool(cfg.get("trust_remote_code", True)))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {
        "trust_remote_code": bool(cfg.get("trust_remote_code", True)),
        "torch_dtype": str(cfg.get("torch_dtype", "auto")),
    }
    device_map = None if is_distributed() else _sft_training_device_map(cfg)
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    lora_cfg = cfg.get("lora") or {}
    peft_config = (
        LoraConfig(
            r=int(lora_cfg.get("r", 8)),
            lora_alpha=int(lora_cfg.get("alpha", 16)),
            lora_dropout=float(lora_cfg.get("dropout", 0.05)),
            bias=str(lora_cfg.get("bias", "none")),
            task_type="CAUSAL_LM",
            target_modules=list(lora_cfg.get("target_modules") or _default_lora_target_modules()),
        )
        if bool(lora_cfg.get("enabled", True))
        else None
    )

    data_files = {"train": str(train_path)}
    if eval_path and eval_path.exists():
        data_files["validation"] = str(eval_path)
    dataset = load_dataset("json", data_files=data_files)
    tokenized = _tokenize_dataset_with_assistant_loss_mask(
        dataset=dataset,
        tokenizer=tokenizer,
        max_seq_length=int(cfg.get("max_seq_length", 4096)),
    )
    args = _build_trl_sft_config(SFTConfig, cfg=cfg, output_dir=output_dir, has_eval="validation" in dataset)
    _move_model_to_training_device(model, args=args, device_map=device_map)
    trainer_kwargs = _filter_kwargs_for_callable(
        SFTTrainer,
        {
            "model": model,
            "args": args,
            "train_dataset": tokenized["train"],
            "eval_dataset": tokenized.get("validation"),
            "peft_config": peft_config,
            "processing_class": tokenizer,
            "tokenizer": tokenizer,
            "data_collator": DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                padding=True,
                label_pad_token_id=-100,
            ),
        },
    )
    trainer = SFTTrainer(**trainer_kwargs)
    trainer_device = getattr(getattr(trainer, "accelerator", None), "device", None) or getattr(trainer.args, "device", None)
    if trainer_device is not None:
        trainer.model.to(trainer_device)
    resume_checkpoint = latest_trainer_checkpoint(output_dir) if resume else None
    manifest["resume_from_checkpoint"] = str(resume_checkpoint or "")
    if resume_checkpoint:
        trainer.train(resume_from_checkpoint=str(resume_checkpoint))
    else:
        trainer.train()
    if is_main_process():
        trainer.model.save_pretrained(str(output_dir / "adapter"))
        tokenizer.save_pretrained(str(output_dir / "adapter"))
        manifest["adapter_dir"] = str(output_dir / "adapter")
    barrier()


def _default_lora_target_modules() -> list[str]:
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _apply_cuda_visible_devices(cfg: dict[str, Any]) -> None:
    if is_distributed():
        return
    value = cfg.get("cuda_visible_devices", "0")
    if value is None:
        return
    text = str(value).strip()
    if text.lower() in {"", "none", "null", "false", "all"}:
        return
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", text)


def _build_trl_sft_config(config_cls: Any, *, cfg: dict[str, Any], output_dir: Path, has_eval: bool) -> Any:
    kwargs = {
        "output_dir": str(output_dir),
        "max_steps": int(cfg.get("max_steps", 50)),
        "learning_rate": float(cfg.get("learning_rate", 2e-4)),
        "per_device_train_batch_size": int(cfg.get("per_device_train_batch_size", 1)),
        "per_device_eval_batch_size": int(cfg.get("per_device_eval_batch_size", cfg.get("per_device_train_batch_size", 1))),
        "gradient_accumulation_steps": int(cfg.get("gradient_accumulation_steps", 4)),
        "logging_steps": int(cfg.get("logging_steps", 5)),
        "save_steps": int(cfg.get("save_steps", 50)),
        "eval_steps": int(cfg.get("eval_steps", 50)),
        "save_total_limit": int(cfg.get("save_total_limit", 2)),
        "report_to": list(cfg.get("report_to", [])),
        "fp16": bool(cfg.get("fp16", False)),
        "bf16": bool(cfg.get("bf16", False)),
        "max_seq_length": int(cfg.get("max_seq_length", 4096)),
        "max_length": int(cfg.get("max_seq_length", cfg.get("max_length", 4096))),
        "dataset_text_field": "text",
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "remove_unused_columns": False,
        "packing": bool(cfg.get("packing", False)),
        "ddp_find_unused_parameters": bool(cfg.get("ddp_find_unused_parameters", False)),
        "gradient_checkpointing": bool(cfg.get("gradient_checkpointing", False)),
    }
    eval_strategy_key = _training_args_eval_strategy_key(config_cls)
    if eval_strategy_key:
        kwargs[eval_strategy_key] = "steps" if has_eval else "no"
    return config_cls(**_filter_kwargs_for_callable(config_cls, kwargs))


def _filter_kwargs_for_callable(target: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    parameters = inspect.signature(target).parameters
    if any(param.kind == param.VAR_KEYWORD for param in parameters.values()):
        return {key: value for key, value in kwargs.items() if value is not None}
    return {key: value for key, value in kwargs.items() if key in parameters and value is not None}


def _training_args_eval_strategy_key(training_args_cls: Any) -> str | None:
    parameters = inspect.signature(training_args_cls).parameters
    if not parameters:
        parameters = inspect.signature(training_args_cls.__init__).parameters
    if "evaluation_strategy" in parameters:
        return "evaluation_strategy"
    if "eval_strategy" in parameters:
        return "eval_strategy"
    return None


def _write_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
    if not is_main_process():
        return
    output_dir.joinpath("train_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
