#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "research" / "g1_grass" / "experiment_matrix.yaml"
COMMAND_LOG_NAME = "commands_train.jsonl"


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_seed_csv(value: str) -> list[int]:
    seeds = []
    for item in parse_csv(value):
        try:
            seeds.append(int(item))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid seed '{item}'. Seeds must be integers.") from exc
    if not seeds:
        raise argparse.ArgumentTypeError("--seeds must include at least one integer.")
    return seeds


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid integer '{value}'.") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return parsed


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"null", "Null", "NULL", "~"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [parse_scalar(item) for item in inner.split(",")]
    if value.startswith(("'", '"')) and value.endswith(("'", '"')):
        return ast.literal_eval(value)
    try:
        return int(value)
    except ValueError:
        return value


def load_simple_yaml(path: Path) -> dict[str, Any]:
    """Load the simple YAML subset used by the experiment matrix when PyYAML is unavailable."""
    config: dict[str, Any] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        raw_line = lines[index]
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            index += 1
            continue
        if raw_line.startswith(" "):
            raise ValueError(f"Unexpected indentation in {path}: {raw_line}")

        key, separator, value = stripped.partition(":")
        if not separator:
            raise ValueError(f"Invalid YAML line in {path}: {raw_line}")
        key = key.strip()
        value = value.strip()
        if value:
            config[key] = parse_scalar(value.split("#", 1)[0].strip())
            index += 1
            continue

        items: list[dict[str, Any]] = []
        index += 1
        current_item: dict[str, Any] | None = None
        while index < len(lines):
            nested_raw = lines[index]
            nested_stripped = nested_raw.strip()
            if not nested_stripped or nested_stripped.startswith("#"):
                index += 1
                continue
            indentation = len(nested_raw) - len(nested_raw.lstrip(" "))
            if indentation == 0:
                break
            if nested_stripped.startswith("- "):
                current_item = {}
                items.append(current_item)
                item_content = nested_stripped[2:].strip()
                if item_content:
                    item_key, item_separator, item_value = item_content.partition(":")
                    if not item_separator:
                        raise ValueError(f"Invalid YAML list item in {path}: {nested_raw}")
                    current_item[item_key.strip()] = parse_scalar(item_value.split("#", 1)[0].strip())
            elif current_item is not None:
                item_key, item_separator, item_value = nested_stripped.partition(":")
                if not item_separator:
                    raise ValueError(f"Invalid YAML mapping item in {path}: {nested_raw}")
                current_item[item_key.strip()] = parse_scalar(item_value.split("#", 1)[0].strip())
            else:
                raise ValueError(f"Invalid YAML block in {path}: {nested_raw}")
            index += 1
        config[key] = items
    return config


def load_config(path: Path) -> dict[str, Any]:
    if yaml is None:
        config = load_simple_yaml(path)
    else:
        with path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return config


def validate_config(config: dict[str, Any]) -> None:
    required_keys = {"experiment_name", "seeds", "train_steps", "methods", "output_root"}
    missing_keys = sorted(required_keys - set(config))
    if missing_keys:
        raise ValueError(f"Config is missing required key(s): {', '.join(missing_keys)}")

    if not isinstance(config["experiment_name"], str) or not config["experiment_name"]:
        raise ValueError("experiment_name must be a non-empty string.")

    if not isinstance(config["seeds"], list) or not all(isinstance(seed, int) for seed in config["seeds"]):
        raise ValueError("seeds must be a list of integers.")

    train_steps = config["train_steps"]
    if train_steps is not None and (not isinstance(train_steps, int) or train_steps <= 0):
        raise ValueError("train_steps must be null or a positive integer.")

    methods = config["methods"]
    if not isinstance(methods, list) or not methods:
        raise ValueError("methods must be a non-empty list.")
    seen_method_names = set()
    for method in methods:
        if not isinstance(method, dict):
            raise ValueError("Each method must be a mapping.")
        name = method.get("name")
        task = method.get("task")
        if not isinstance(name, str) or not name:
            raise ValueError("Each method must include a non-empty name.")
        if not isinstance(task, str) or not task:
            raise ValueError(f"Method '{name}' must include a non-empty task.")
        if name in seen_method_names:
            raise ValueError(f"Duplicate method name: {name}")
        seen_method_names.add(name)

    if not isinstance(config["output_root"], str) or not config["output_root"]:
        raise ValueError("output_root must be a non-empty string.")


def select_methods(config: dict[str, Any], requested_methods: list[str] | None) -> list[dict[str, str]]:
    methods_by_name = {method["name"]: method for method in config["methods"]}
    if requested_methods is None:
        return config["methods"]

    unknown_methods = [name for name in requested_methods if name not in methods_by_name]
    if unknown_methods:
        known = ", ".join(methods_by_name)
        unknown = ", ".join(unknown_methods)
        raise ValueError(f"Unknown method(s): {unknown}. Known methods: {known}")
    return [methods_by_name[name] for name in requested_methods]


def build_command(task: str, seed: int, train_steps: int | None) -> list[str]:
    command = [
        "python",
        "scripts/rsl_rl/train.py",
        "--headless",
        "--task",
        task,
        "--seed",
        str(seed),
    ]
    if train_steps is not None:
        command.extend(["--max_iterations", str(train_steps)])
    return command


def command_records(
    config: dict[str, Any],
    config_path: Path,
    methods: list[dict[str, str]],
    seeds: list[int],
    train_steps: int | None,
    execute: bool,
) -> list[dict[str, Any]]:
    created_at = datetime.now(timezone.utc).isoformat()
    try:
        config_display = str(config_path.relative_to(REPO_ROOT))
    except ValueError:
        config_display = str(config_path)
    records = []
    for method in methods:
        for seed in seeds:
            command = build_command(method["task"], seed, train_steps)
            records.append(
                {
                    "created_at": created_at,
                    "experiment_name": config["experiment_name"],
                    "config": config_display,
                    "method": method["name"],
                    "task": method["task"],
                    "seed": seed,
                    "train_steps": train_steps,
                    "execute": execute,
                    "command": shlex.join(command),
                }
            )
    return records


def write_command_log(output_root: Path, records: list[dict[str, Any]]) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    command_log = output_root / COMMAND_LOG_NAME
    with command_log.open("a", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    return command_log


def run_records(records: list[dict[str, Any]]) -> None:
    for record in records:
        print(record["command"], flush=True)
        subprocess.run(shlex.split(record["command"]), cwd=REPO_ROOT, check=True)


def resolve_output_root(config: dict[str, Any]) -> Path:
    output_root = Path(config["output_root"])
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the G1 grass experiment matrix.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to experiment matrix YAML config.")
    parser.add_argument("--methods", type=parse_csv, default=None, help="Comma-separated method names to run.")
    parser.add_argument("--seeds", type=parse_seed_csv, default=None, help="Comma-separated integer seeds to run.")
    parser.add_argument("--train-steps", type=positive_int, default=None, help="Override train.py --max_iterations.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Print commands and do not run training.")
    mode.add_argument("--execute", action="store_true", help="Run the generated training commands.")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    validate_config(config)

    methods = select_methods(config, args.methods)
    seeds = args.seeds if args.seeds is not None else config["seeds"]
    train_steps = args.train_steps if args.train_steps is not None else config["train_steps"]

    records = command_records(
        config=config,
        config_path=config_path,
        methods=methods,
        seeds=seeds,
        train_steps=train_steps,
        execute=args.execute,
    )
    write_command_log(resolve_output_root(config), records)

    if args.execute:
        run_records(records)
        return

    for record in records:
        print(record["command"])


if __name__ == "__main__":
    main()
