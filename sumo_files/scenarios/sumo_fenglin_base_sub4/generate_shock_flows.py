#!/usr/bin/env python3
"""Create 24-hour demand files for sudden traffic-volume shift evaluation.

This script lives in the isolated ``sumo_fenglin_base_sub4`` experiment
folder. By default it recovers hourly entrance demand read-only from the
released ``sumo_fenglin_base_sub1/base_INDEX.rou.xml`` baseline routes and
writes shock inputs in this folder. Use ``--baseline template`` only when
baseline route outputs are unavailable.
"""

import argparse
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET


DURATION = 86400
INTERVAL = 3600


def parse_indices(value):
    indices = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            if end < start:
                raise ValueError("an index range has its end before its start")
            indices.extend(range(start, end + 1))
        else:
            indices.append(int(part))
    if not indices or any(index < 1 for index in indices):
        raise ValueError("indices must be positive, for example 1 or 1-50")
    return sorted(set(indices))


def read_template(path):
    flows = list(ET.parse(str(path)).getroot().findall("flow"))
    if not flows:
        raise ValueError("{} contains no flow elements".format(path))
    try:
        return [(flow.attrib["from"], int(flow.attrib["number"])) for flow in flows]
    except KeyError as exc:
        raise ValueError("template flow is missing {}".format(exc))


def recover_counts(route_path, entrances, periods):
    counts = Counter()
    for _, element in ET.iterparse(str(route_path), events=("end",)):
        if element.tag == "vehicle":
            vehicle_id = element.attrib.get("id", "")
            try:
                counts[int(vehicle_id.split(".", 1)[0])] += 1
            except ValueError:
                raise ValueError("{} has an unsupported vehicle id: {}".format(route_path, vehicle_id))
        element.clear()
    limit = entrances * periods
    if any(flow_id >= limit for flow_id in counts):
        raise ValueError("{} contains flow ids outside 0-{}".format(route_path, limit - 1))
    return counts


def scaled(number, multiplier):
    return int((Decimal(number) * multiplier).quantize(Decimal("1"), ROUND_HALF_UP))


def write_flows(path, template, counts, periods, interval, shock_begin, shock_end, multiplier, source):
    with path.open("w", encoding="utf-8") as output:
        output.write("<routes>\n\n")
        output.write("    <!-- {} -->\n".format(source))
        output.write("    <!-- volume multiplier {} in [{}, {}) seconds -->\n\n".format(
            multiplier, shock_begin, shock_end))
        for period in range(periods):
            begin = period * interval
            end = begin + interval
            apply_shock = shock_begin <= begin and end <= shock_end
            for entrance, (edge, original_number) in enumerate(template):
                flow_id = period * len(template) + entrance
                number = original_number if counts is None else counts.get(flow_id, 0)
                if apply_shock:
                    number = scaled(number, multiplier)
                flow = ET.Element("flow", {"id": str(flow_id), "from": edge,
                                           "begin": str(begin), "end": str(end),
                                           "number": str(number)})
                output.write("    {}\n".format(ET.tostring(flow, encoding="unicode")))
        output.write("\n</routes>\n")


def generate_route(flow_path, route_path, scenario_dir, executable, duration):
    subprocess.run([executable,
                    "--route-files={}".format(flow_path),
                    "--turn-ratio-files={}".format(scenario_dir / "input_turns.turns.xml"),
                    "--net-file={}".format(scenario_dir / "base_v2.net.xml"),
                    "--output-file={}".format(route_path),
                    "--accept-all-destinations=true", "--no-internal-links=true",
                    "-e", str(duration)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indices", required=True, help="For example: 1,3,5-20")
    parser.add_argument("--output-dir", type=Path,
                        help="Destination directory; defaults to this sub4 experiment folder.")
    parser.add_argument("--baseline-route-dir", type=Path,
                        help="Directory containing original routes; defaults to sibling sub1.")
    parser.add_argument("--baseline", choices=("routes", "template"), default="routes")
    parser.add_argument("--multiplier", type=Decimal, default=Decimal("1.5"))
    parser.add_argument("--shock-begin", type=int, default=25200)
    parser.add_argument("--shock-end", type=int, default=28800)
    parser.add_argument("--duration", type=int, default=DURATION)
    parser.add_argument("--interval", type=int, default=INTERVAL)
    parser.add_argument("--generate-routes", action="store_true")
    parser.add_argument("--jtrrouter", default="jtrrouter")
    args = parser.parse_args()
    try:
        indices = parse_indices(args.indices)
    except ValueError as exc:
        parser.error(str(exc))
    source_dir = Path(__file__).resolve().parent
    baseline_route_dir = (args.baseline_route_dir or
                          (source_dir.parent / "sumo_fenglin_base_sub1")).resolve()
    output_dir = (args.output_dir or source_dir).resolve()
    if output_dir == baseline_route_dir:
        parser.error("--output-dir must differ from the baseline route directory")
    if args.duration <= 0 or args.interval <= 0 or args.duration % args.interval:
        parser.error("--duration must be a positive multiple of --interval")
    if (args.shock_begin < 0 or args.shock_end > args.duration or
            args.shock_end <= args.shock_begin or
            args.shock_begin % args.interval or args.shock_end % args.interval):
        parser.error("shock bounds must be aligned intervals within the duration")
    if args.multiplier <= 0:
        parser.error("--multiplier must be positive")
    if args.generate_routes and shutil.which(args.jtrrouter) is None:
        parser.error("{} was not found on PATH".format(args.jtrrouter))
    template = read_template(source_dir / "input_flows.flows.xml")
    periods = args.duration // args.interval
    output_dir.mkdir(parents=True, exist_ok=True)
    for index in indices:
        if args.baseline == "routes":
            route_path = baseline_route_dir / "base_{}.rou.xml".format(index)
            if not route_path.exists():
                parser.error("{} does not exist".format(route_path))
            counts = recover_counts(route_path, len(template), periods)
            source = "demand recovered from {}".format(route_path.name)
        else:
            counts = None
            source = "hourly demand repeated from input_flows.flows.xml"
        flow_path = output_dir / "base_{}.flows.xml".format(index)
        write_flows(flow_path, template, counts, periods, args.interval,
                    args.shock_begin, args.shock_end, args.multiplier, source)
        print("wrote {}".format(flow_path))
        if args.generate_routes:
            route_path = output_dir / "base_{}.rou.xml".format(index)
            generate_route(flow_path, route_path, source_dir, args.jtrrouter, args.duration)
            print("wrote {}".format(route_path))


if __name__ == "__main__":
    main()
