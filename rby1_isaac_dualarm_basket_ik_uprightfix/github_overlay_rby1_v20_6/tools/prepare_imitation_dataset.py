#!/usr/bin/env python3
"""Convert RB-Y1 v7 JSONL demonstrations into fixed-length BC sequences."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PHASES = ("teleop", "raise", "translate", "lift", "carry")


def feature(record: dict) -> np.ndarray:
    phase = np.zeros(len(PHASES), dtype=np.float32)
    if record.get("phase") in PHASES:
        phase[PHASES.index(record["phase"])] = 1.0
    values = (
        list(record["joint_pos"])
        + list(record["joint_vel"])
        + list(record["ee_pos"])
        + list(record["ee_quat_wxyz"])
        + list(record["target_pos"])
        + list(record["target_quat_wxyz"])
        + list(record["shoulder_target"])
        + [float(record.get("pinch", 0.0)), float(bool(record.get("attached", False)))]
        + phase.tolist()
    )
    return np.asarray(values, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--side", choices=["left", "right", "both"], default="right")
    parser.add_argument("--include-invalid", action="store_true")
    args = parser.parse_args()

    rows: list[dict] = []
    with args.input.expanduser().open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON on line {line_number}") from exc
            if args.side != "both" and row.get("side") != args.side:
                continue
            if not args.include_invalid and not row.get("pose_valid", False) and row.get("phase") not in ("lift", "carry"):
                continue
            rows.append(row)

    length = max(2, args.sequence_length)
    stride = max(1, args.stride)
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for start in range(0, len(rows) - length + 1, stride):
        window = rows[start : start + length]
        if len({row.get("side") for row in window}) != 1:
            continue
        observations.append(np.stack([feature(row) for row in window]))
        actions.append(np.asarray([row["joint_command"] for row in window], dtype=np.float32))

    if not observations:
        raise RuntimeError("No valid sequences found. Record more successful demonstrations.")
    obs = np.stack(observations)
    act = np.stack(actions)
    args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output.expanduser(),
        observations=obs,
        actions=act,
        phases=np.asarray(PHASES),
    )
    print(f"[OK] observations={obs.shape} actions={act.shape} -> {args.output.expanduser()}")


if __name__ == "__main__":
    main()
