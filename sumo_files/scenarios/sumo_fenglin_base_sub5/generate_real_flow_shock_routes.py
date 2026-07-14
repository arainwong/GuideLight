#!/usr/bin/env python3
"""Generate shock-case routes from real-flow XML files.

The script keeps the original base_*.flows.xml files untouched and writes a
separate route directory that can be passed to eval_model.py via --route-dir.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scale real flows during a shock window and generate SUMO routes."
    )
    parser.add_argument("--scenario-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--num-routes", type=int, default=50)
    parser.add_argument("--shock-begin", type=float, default=7 * 3600)
    parser.add_argument("--shock-end", type=float, default=8 * 3600)
    parser.add_argument("--multiplier", type=float, default=2.0)
    parser.add_argument("--allow-lower-multiplier", action="store_true",
                        help="Allow multipliers below 2.0, useful for regenerating baseline real-flow routes.")
    parser.add_argument("--jtrrouter", default="jtrrouter")
    parser.add_argument("--net-file", default="base_v2.net.xml")
    parser.add_argument("--turn-ratio-file", default="input_turns.turns.xml")
    parser.add_argument("--end-time", type=int, default=86400)
    return parser.parse_args()


def format_number(value: float) -> str:
    rounded = round(value)
    if math.isclose(value, rounded):
        return str(int(rounded))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def overlaps(begin: float, end: float, shock_begin: float, shock_end: float) -> bool:
    return begin < shock_end and end > shock_begin


def scale_flow_file(
    src: Path,
    dst: Path,
    multiplier: float,
    shock_begin: float,
    shock_end: float,
) -> tuple[int, float, float]:
    tree = ET.parse(src)
    root = tree.getroot()

    changed = 0
    before_total = 0.0
    after_total = 0.0
    number_entries = []

    for flow in root.iter("flow"):
        begin = float(flow.get("begin", "0"))
        end = float(flow.get("end", "0"))
        if not overlaps(begin, end, shock_begin, shock_end):
            continue

        if "number" in flow.attrib:
            before = float(flow.attrib["number"])
            after = before * multiplier
            number_entries.append((flow, before, after))
            continue
        elif "vehsPerHour" in flow.attrib:
            before = float(flow.attrib["vehsPerHour"])
            after = before * multiplier
            flow.set("vehsPerHour", format_number(after))
        elif "probability" in flow.attrib:
            before = float(flow.attrib["probability"])
            after = min(before * multiplier, 1.0)
            flow.set("probability", format_number(after))
        else:
            continue

        changed += 1
        before_total += before
        after_total += after

    if number_entries:
        targets = [target for _, _, target in number_entries]
        floors = [math.floor(target) for target in targets]
        desired_total = round_half_up(sum(targets))
        remainder = desired_total - sum(floors)
        order = sorted(
            range(len(targets)),
            key=lambda idx: targets[idx] - floors[idx],
            reverse=True,
        )
        increments = {idx for idx in order[:remainder]}

        for idx, (flow, before, target) in enumerate(number_entries):
            after = int(floors[idx] + (1 if idx in increments else 0))
            flow.set("number", str(after))
            changed += 1
            before_total += before
            after_total += after

    dst.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(dst, encoding="utf-8", xml_declaration=True)
    return changed, before_total, after_total


def run_jtrrouter(
    jtrrouter: str,
    scenario_dir: Path,
    route_file: Path,
    output_file: Path,
    net_file: Path,
    turn_ratio_file: Path,
    end_time: int,
) -> None:
    cmd = [
        jtrrouter,
        f"--route-files={route_file}",
        f"--turn-ratio-files={turn_ratio_file}",
        f"--net-file={net_file}",
        f"--output-file={output_file}",
        "--accept-all-destinations=true",
        "--no-internal-links=true",
        "--no-step-log=true",
        "-e",
        str(end_time),
    ]
    subprocess.run(cmd, cwd=scenario_dir, check=True)


def main() -> None:
    args = parse_args()
    scenario_dir = args.scenario_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else scenario_dir / "shock_2p0_h07_h08"
    )
    net_file = scenario_dir / args.net_file
    turn_ratio_file = scenario_dir / args.turn_ratio_file

    if args.multiplier < 2.0 and not args.allow_lower_multiplier:
        raise ValueError("--multiplier must be at least 2.0 unless --allow-lower-multiplier is set.")
    if args.shock_end <= args.shock_begin:
        raise ValueError("--shock-end must be greater than --shock-begin.")
    if not net_file.exists():
        raise FileNotFoundError(net_file)
    if not turn_ratio_file.exists():
        raise FileNotFoundError(turn_ratio_file)

    total_before = 0.0
    total_after = 0.0
    total_changed = 0

    last_index = args.start_index + args.num_routes - 1
    for index in range(args.start_index, last_index + 1):
        src_flow = scenario_dir / f"base_{index}.flows.xml"
        dst_flow = output_dir / f"base_{index}.flows.xml"
        dst_route = output_dir / f"base_{index}.rou.xml"
        if not src_flow.exists():
            raise FileNotFoundError(src_flow)

        changed, before, after = scale_flow_file(
            src_flow,
            dst_flow,
            args.multiplier,
            args.shock_begin,
            args.shock_end,
        )
        if changed == 0:
            raise RuntimeError(f"No flow entries overlapped the shock window in {src_flow}")

        run_jtrrouter(
            args.jtrrouter,
            scenario_dir,
            dst_flow,
            dst_route,
            net_file,
            turn_ratio_file,
            args.end_time,
        )

        total_changed += changed
        total_before += before
        total_after += after
        ratio = after / before if before else 0.0
        print(
            f"[done] base_{index}: changed={changed}, "
            f"shock_flow={before:.2f}->{after:.2f}, ratio={ratio:.2f}"
        )

    overall_ratio = total_after / total_before if total_before else 0.0
    print(f"[complete] output_dir={output_dir}")
    print(
        f"[summary] routes={args.num_routes}, changed_flows={total_changed}, "
        f"shock_flow={total_before:.2f}->{total_after:.2f}, ratio={overall_ratio:.2f}"
    )


if __name__ == "__main__":
    main()
