#!/usr/bin/env python3
"""Export 5-minute cohort statistics for traffic-shock route and policy evaluation."""

import argparse
import csv
import pickle
import warnings
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


HERE = Path(__file__).resolve().parent
HORIZON = 24 * 60 * 60
BIN_SECONDS = 5 * 60
BINS = HORIZON // BIN_SECONDS


def parse_indices(value):
    indices = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            begin, end = (int(item) for item in part.split("-", 1))
            if end < begin:
                raise ValueError("an index range has its end before its start")
            indices.extend(range(begin, end + 1))
        else:
            indices.append(int(part))
    if not indices or any(index < 1 for index in indices):
        raise ValueError("indices must be positive, for example 1-50")
    return sorted(set(indices))


def bin_times(index):
    begin = index * BIN_SECONDS
    end = begin + BIN_SECONDS
    return begin, end, begin / 3600, end / 3600


def analysis_bins(begin, end):
    if begin < 0 or end > HORIZON or end <= begin:
        raise ValueError("analysis window must be within the 24-hour horizon")
    if begin % BIN_SECONDS or end % BIN_SECONDS:
        raise ValueError("analysis window must align with 5-minute bins")
    return range(begin // BIN_SECONDS, end // BIN_SECONDS)


def cohort_label(indices):
    if len(indices) == 1:
        return "route_{}".format(indices[0])
    if indices == list(range(indices[0], indices[-1] + 1)):
        return "routes_{}_{}".format(indices[0], indices[-1])
    return "routes_{}".format("_".join(str(index) for index in indices))


def window_label(begin, end):
    if begin % 3600 == 0 and end % 3600 == 0:
        return "h{:02d}_h{:02d}".format(begin // 3600, end // 3600)
    return "s{}_s{}".format(begin, end)


def read_route_flow(route_path):
    counts = np.zeros(BINS, dtype=float)
    for _, element in ET.iterparse(str(route_path), events=("end",)):
        if element.tag == "vehicle":
            depart = float(element.attrib["depart"])
            if 0 <= depart < HORIZON:
                counts[int(depart // BIN_SECONDS)] += 1
        element.clear()
    return counts


def read_route_cohort(route_dir, indices):
    values = []
    for index in indices:
        path = route_dir / "base_{}.rou.xml".format(index)
        if not path.exists():
            raise FileNotFoundError(str(path))
        values.append(read_route_flow(path))
        print("read {}".format(path))
    return np.asarray(values)


def summary(matrix):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return {
            "count": np.sum(~np.isnan(matrix), axis=0),
            "total": np.nansum(matrix, axis=0),
            "mean": np.nanmean(matrix, axis=0),
            "std": np.nanstd(matrix, axis=0),
            "p25": np.nanpercentile(matrix, 25, axis=0),
            "p50": np.nanpercentile(matrix, 50, axis=0),
            "p75": np.nanpercentile(matrix, 75, axis=0),
            "min": np.nanmin(matrix, axis=0),
            "max": np.nanmax(matrix, axis=0),
        }

def plotted_limits(values, lower_bound=None):
    finite = np.asarray(values)[np.isfinite(values)]
    if not len(finite):
        raise ValueError("cannot determine plot limits from empty data")
    low = float(np.min(finite))
    high = float(np.max(finite))
    padding = max((high - low) * 0.05, 1.0)
    lower = low - padding
    if lower_bound is not None:
        lower = max(lower, lower_bound)
    return lower, high + padding



def write_flow_detail(path, condition_values, indices):
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["condition", "route_index", "bin_index", "time_begin_seconds",
                         "time_end_seconds", "time_begin_hour", "time_end_hour", "vehicles_5min"])
        for condition, matrix in condition_values.items():
            for route_pos, route_index in enumerate(indices):
                for bin_index, value in enumerate(matrix[route_pos]):
                    writer.writerow([condition, route_index, bin_index, *bin_times(bin_index), int(value)])


def write_flow_summary(path, condition_values):
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["condition", "bin_index", "time_begin_seconds", "time_end_seconds",
                         "time_begin_hour", "time_end_hour", "route_count", "total_vehicles_5min",
                         "mean_vehicles_5min", "std_vehicles_5min", "p25_vehicles_5min",
                         "p50_vehicles_5min", "p75_vehicles_5min", "min_vehicles_5min",
                         "max_vehicles_5min", "mean_vehicles_per_hour_equivalent"])
        for condition, matrix in condition_values.items():
            stats = summary(matrix)
            for index in range(BINS):
                writer.writerow([condition, index, *bin_times(index), int(stats["count"][index]),
                                 int(stats["total"][index]), stats["mean"][index], stats["std"][index],
                                 stats["p25"][index], stats["p50"][index], stats["p75"][index],
                                 stats["min"][index], stats["max"][index], stats["mean"][index] * 12])


def plot_cohort(path, condition_values, title, ylabel, shock_begin, shock_end,
                view_begin=0, view_end=HORIZON):
    times = np.arange(BINS + 1) * BIN_SECONDS / 3600
    colors = {"baseline": "#277da1", "shock": "#f94144"}
    labels = {"baseline": "Baseline routes (sub1)", "shock": "Shock routes (sub4)"}
    fig, ax = plt.subplots(figsize=(12, 4.8))
    for condition in ["baseline", "shock"]:
        stats = summary(condition_values[condition])
        mean = stats["mean"]
        std = stats["std"]
        ax.step(times, np.r_[mean, mean[-1]], where="post", label=labels[condition],
                color=colors[condition], linewidth=1.3)
        ax.fill_between(times, np.r_[np.maximum(mean - std, 0), max(mean[-1] - std[-1], 0)],
                        np.r_[mean + std, mean[-1] + std[-1]], step="post",
                        color=colors[condition], alpha=0.12)
    ax.axvspan(shock_begin / 3600, shock_end / 3600, color="#f8961e", alpha=0.15,
               label="Shock interval")
    tick_step = 1 if view_end - view_begin <= 8 * 3600 else 2
    ax.set(xlim=(view_begin / 3600, view_end / 3600),
           xticks=np.arange(view_begin // 3600, view_end // 3600 + 1, tick_step),
           xlabel="Time of day (hour)", ylabel=ylabel, title=title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_synchronization(path, cycle_values, flow_values, condition, flow_ylabel,
                         cycle_ylim=None, flow_ylim=None, title=None):
    times = np.arange(BINS) * BIN_SECONDS / 3600
    cycle_mean = summary(cycle_values)["mean"]
    flow_mean = summary(flow_values)["mean"]
    fig, ax_cycle = plt.subplots(figsize=(6.2, 4.0))
    ax_flow = ax_cycle.twinx()
    cycle_line, = ax_cycle.plot(times, cycle_mean, color="#1747ff", linewidth=1.8,
                                label="cycle time")
    flow_line, = ax_flow.plot(times, flow_mean, color="#239b3a", linewidth=1.8,
                              label="flow")
    ax_cycle.set(xlim=(0, 24), xticks=[0, 6, 12, 18, 24], xlabel="Local Time",
                 ylabel="Cycle Time (s)",
                 title=title or "{} case".format(condition.capitalize()))
    ax_cycle.set_xticklabels(["00:00", "06:00", "12:00", "18:00", "24:00"], fontsize=11)
    ax_cycle.title.set_fontsize(14)
    ax_cycle.xaxis.label.set_fontsize(12)
    ax_cycle.yaxis.label.set_fontsize(12)
    if cycle_ylim is not None:
        ax_cycle.set_ylim(cycle_ylim)
    ax_cycle.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax_cycle.tick_params(axis="y", colors=cycle_line.get_color(), labelsize=11)
    ax_cycle.tick_params(axis="x", labelsize=11)
    ax_cycle.yaxis.label.set_color(cycle_line.get_color())
    ax_flow.set_ylabel(flow_ylabel, color=flow_line.get_color(), fontsize=12)
    if flow_ylim is not None:
        ax_flow.set_ylim(flow_ylim)
    ax_flow.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax_flow.tick_params(axis="y", colors=flow_line.get_color(), labelsize=11)
    ax_cycle.grid(True, axis="y", alpha=0.25, linewidth=0.8)
    ax_cycle.legend([cycle_line, flow_line], ["cycle time", "flow"], loc="upper left",
                    frameon=True, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)

def run_flow(args):
    indices = parse_indices(args.indices)
    routes = cohort_label(indices)
    output_dir = (args.output_dir or (HERE / "analysis_plots" / "{}_5min".format(routes))).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    shock_dir = (args.shock_route_dir or HERE).resolve()
    baseline_dir = (args.baseline_route_dir or (HERE.parent / "sumo_fenglin_base_sub1")).resolve()
    values = {"shock": read_route_cohort(shock_dir, indices),
              "baseline": read_route_cohort(baseline_dir, indices)}
    detail_path = output_dir / "{}_traffic_flow_5min_by_route.csv".format(routes)
    summary_path = output_dir / "{}_traffic_flow_5min_summary.csv".format(routes)
    plot_path = output_dir / "{}_baseline_vs_shock_traffic_flow_5min.png".format(routes)
    write_flow_detail(detail_path, values, indices)
    write_flow_summary(summary_path, values)
    plot_cohort(plot_path, values, "Traffic Flow: Baseline vs Shock Routes (5-minute bins)",
                "Mean departing vehicles / route / 5 min", args.shock_begin, args.shock_end)
    for condition in ["baseline", "shock"]:
        print("{} routes: {}, total vehicles: {}".format(condition, len(indices),
                                                           int(np.sum(values[condition]))))
    for path in [detail_path, summary_path, plot_path]:
        print("wrote {}".format(path))


def policy_pickle(condition, trial):
    paths = sorted((HERE / "eval_logs" / condition / "trial_{}".format(trial)).glob("plt_*.pkl"))
    if not paths:
        raise FileNotFoundError("missing evaluation pkl for {} trial_{}".format(condition, trial))
    return paths[0]


def read_action_cycles(conditions, indices, value_key="cycle_time"):
    samples = []
    labels = []
    for trial, route_index in enumerate(indices):
        route_values = None
        junctions = None
        for condition in conditions:
            path = policy_pickle(condition, trial)
            with path.open("rb") as source:
                data = pickle.load(source)
            if "tl_cycle_flow" not in data:
                raise ValueError("{} lacks tl_cycle_flow; rerun evaluation with the current logger".format(path))
            segment_junctions = sorted(data["tl_cycle_flow"][0]["junctions"])
            if junctions is None:
                junctions = segment_junctions
                route_values = {junction: np.full(BINS, np.nan) for junction in junctions}
            elif segment_junctions != junctions:
                raise ValueError("{} contains different controlled junctions".format(path))
            for record in data["tl_cycle_flow"]:
                begin = int(record["begin"])
                end = int(record["end"])
                if begin % BIN_SECONDS or end % BIN_SECONDS:
                    raise ValueError("{} contains an interval not aligned to 5-minute bins".format(path))
                for bin_index in range(max(0, begin) // BIN_SECONDS, min(HORIZON, end) // BIN_SECONDS):
                    for junction in junctions:
                        value = record["junctions"][junction][value_key]
                        existing = route_values[junction][bin_index]
                        if not np.isnan(existing) and existing != value:
                            raise ValueError("{} overlaps another segment at bin {}".format(path, bin_index))
                        route_values[junction][bin_index] = value
            print("read {} for base_{}".format(path, route_index))
        for junction in junctions:
            samples.append(route_values[junction])
            labels.append((route_index, junction))
    return np.asarray(samples), labels


def write_cycle_detail(path, condition_values, condition_labels, bins):
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["condition", "route_index", "junction", "bin_index", "time_begin_seconds",
                         "time_end_seconds", "time_begin_hour", "time_end_hour", "policy_action_applied",
                         "action_generated_cycle_time_seconds"])
        for condition, matrix in condition_values.items():
            for sample_index, (route_index, junction) in enumerate(condition_labels[condition]):
                for bin_index in bins:
                    value = matrix[sample_index][bin_index]
                    writer.writerow([condition, route_index, junction, bin_index, *bin_times(bin_index),
                                     int(not np.isnan(value)), "" if np.isnan(value) else value])


def write_cycle_summary(path, condition_values, route_count, bins):
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["condition", "bin_index", "time_begin_seconds", "time_end_seconds",
                         "time_begin_hour", "time_end_hour", "route_count", "signal_sample_count",
                         "policy_action_applied", "mean_action_cycle_time_seconds",
                         "std_action_cycle_time_seconds", "p25_action_cycle_time_seconds",
                         "p50_action_cycle_time_seconds", "p75_action_cycle_time_seconds",
                         "min_action_cycle_time_seconds", "max_action_cycle_time_seconds"])
        for condition, matrix in condition_values.items():
            stats = summary(matrix)
            for index in bins:
                controlled = int(stats["count"][index] > 0)
                row_stats = ["", "", "", "", "", "", ""] if not controlled else [
                    stats["mean"][index], stats["std"][index], stats["p25"][index],
                    stats["p50"][index], stats["p75"][index], stats["min"][index],
                    stats["max"][index]]
                writer.writerow([condition, index, *bin_times(index), route_count,
                                 int(stats["count"][index]), controlled, *row_stats])


def run_cycle(args):
    indices = parse_indices(args.indices)
    bins = analysis_bins(args.window_begin, args.window_end)
    label = window_label(args.window_begin, args.window_end)
    routes = cohort_label(indices)
    output_dir = (args.output_dir or (HERE / "analysis_plots" / "{}_{}_5min".format(routes, label))).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    shock_conditions = [value.strip() for value in args.shock_conditions.split(",") if value.strip()]
    baseline_conditions = [value.strip() for value in args.baseline_conditions.split(",") if value.strip()]
    shock, shock_labels = read_action_cycles(shock_conditions, indices)
    baseline, baseline_labels = read_action_cycles(baseline_conditions, indices)
    values = {"shock": shock, "baseline": baseline}
    labels = {"shock": shock_labels, "baseline": baseline_labels}
    detail_path = output_dir / "{}_{}_policy_action_cycle_time_5min_by_route_signal.csv".format(routes, label)
    summary_path = output_dir / "{}_{}_policy_action_cycle_time_5min_summary.csv".format(routes, label)
    plot_path = output_dir / "{}_{}_policy_action_cycle_time_baseline_vs_shock_5min.png".format(routes, label)
    write_cycle_detail(detail_path, values, labels, bins)
    write_cycle_summary(summary_path, values, len(indices), bins)
    plot_cohort(plot_path, values,
                "Policy Action Cycle Time: Baseline vs Shock, {} (5-minute bins)".format(label),
                "Action-generated cycle time (seconds)", args.shock_begin, args.shock_end,
                args.window_begin, args.window_end)
    for path in [detail_path, summary_path, plot_path]:
        print("wrote {}".format(path))


def run_sync(args):
    indices = parse_indices(args.indices)
    label = window_label(args.window_begin, args.window_end)
    routes = cohort_label(indices)
    output_dir = (args.output_dir or (HERE / "analysis_plots" / "{}_{}_5min".format(routes, label))).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    shock_conditions = [value.strip() for value in args.shock_conditions.split(",") if value.strip()]
    baseline_conditions = [value.strip() for value in args.baseline_conditions.split(",") if value.strip()]
    shock_cycles, shock_labels = read_action_cycles(shock_conditions, indices)
    baseline_cycles, baseline_labels = read_action_cycles(baseline_conditions, indices)
    shock_flows, shock_flow_labels = read_action_cycles(shock_conditions, indices, "flow")
    baseline_flows, baseline_flow_labels = read_action_cycles(baseline_conditions, indices, "flow")
    labels = {"shock": shock_labels, "baseline": baseline_labels}
    if shock_labels != shock_flow_labels or baseline_labels != baseline_flow_labels:
        raise ValueError("cycle and flow records contain different route/junction samples")
    if sorted(set(shock_labels)) != sorted(set(baseline_labels)):
        raise ValueError("shock and baseline records contain different route/junction samples")
    cycles = {"shock": shock_cycles, "baseline": baseline_cycles}
    flows = {"shock": shock_flows, "baseline": baseline_flows}
    cycle_ylim = plotted_limits(summary(cycles["shock"])["mean"])
    flow_ylim = plotted_limits(summary(flows["shock"])["mean"], 0.0)
    for condition in ["baseline", "shock"]:
        path = output_dir / "{}_{}_{}_observed_flow_cycle_time_synchronization_5min.png".format(
            routes, label, condition)
        plot_synchronization(path, cycles[condition], flows[condition], condition,
                             "Observed Flow (vehicles / 15 min / signal)",
                             cycle_ylim, flow_ylim)
        print("wrote {}".format(path))
    junction_dir = output_dir / "by_intersection"
    junction_dir.mkdir(parents=True, exist_ok=True)
    junctions = sorted(set(junction for _, junction in shock_labels))
    for junction in junctions:
        junction_cycles = {}
        junction_flows = {}
        for condition in ["baseline", "shock"]:
            positions = [index for index, (_, item) in enumerate(labels[condition])
                         if item == junction]
            junction_cycles[condition] = cycles[condition][positions]
            junction_flows[condition] = flows[condition][positions]
        cycle_ylim = plotted_limits(summary(junction_cycles["shock"])["mean"])
        flow_ylim = plotted_limits(summary(junction_flows["shock"])["mean"], 0.0)
        for condition in ["baseline", "shock"]:
            path = junction_dir / "{}_{}_{}_{}_observed_flow_cycle_time_synchronization_5min.png".format(
                routes, label, junction, condition)
            plot_synchronization(path, junction_cycles[condition], junction_flows[condition],
                                 condition, "Observed Flow (vehicles / 15 min / signal)",
                                 cycle_ylim, flow_ylim,
                                 "{} - {} case".format(junction, condition.capitalize()))
            print("wrote {}".format(path))


def run_sync_demand(args):
    indices = parse_indices(args.indices)
    label = window_label(args.window_begin, args.window_end)
    routes = cohort_label(indices)
    output_dir = (args.output_dir or (HERE / "analysis_plots" / "{}_{}_5min".format(routes, label))).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    shock_conditions = [value.strip() for value in args.shock_conditions.split(",") if value.strip()]
    baseline_conditions = [value.strip() for value in args.baseline_conditions.split(",") if value.strip()]
    shock_cycles, shock_labels = read_action_cycles(shock_conditions, indices)
    baseline_cycles, baseline_labels = read_action_cycles(baseline_conditions, indices)
    if sorted(set(shock_labels)) != sorted(set(baseline_labels)):
        raise ValueError("shock and baseline records contain different route/junction samples")

    shock_dir = args.shock_route_dir.resolve()
    baseline_dir = args.baseline_route_dir.resolve()
    shock_flows = read_route_cohort(shock_dir, indices)
    baseline_flows = read_route_cohort(baseline_dir, indices)
    labels = {"shock": shock_labels, "baseline": baseline_labels}
    cycles = {"shock": shock_cycles, "baseline": baseline_cycles}
    flows = {"shock": shock_flows, "baseline": baseline_flows}

    cycle_ylim = plotted_limits(summary(cycles["shock"])["mean"])
    flow_ylim = plotted_limits(summary(flows["shock"])["mean"], 0.0)
    for condition in ["baseline", "shock"]:
        path = output_dir / "{}_{}_{}_demand_flow_cycle_time_synchronization_5min.png".format(
            routes, label, condition)
        plot_synchronization(path, cycles[condition], flows[condition], condition,
                             "Demand Flow (vehicles / 5 min / route)",
                             cycle_ylim, flow_ylim)
        print("wrote {}".format(path))

    junction_dir = output_dir / "by_intersection"
    junction_dir.mkdir(parents=True, exist_ok=True)
    junctions = sorted(set(junction for _, junction in shock_labels))
    for junction in junctions:
        junction_cycles = {}
        for condition in ["baseline", "shock"]:
            positions = [index for index, (_, item) in enumerate(labels[condition])
                         if item == junction]
            junction_cycles[condition] = cycles[condition][positions]
        cycle_ylim = plotted_limits(summary(junction_cycles["shock"])["mean"])
        flow_ylim = plotted_limits(summary(flows["shock"])["mean"], 0.0)
        for condition in ["baseline", "shock"]:
            path = junction_dir / "{}_{}_{}_{}_demand_flow_cycle_time_synchronization_5min.png".format(
                routes, label, junction, condition)
            plot_synchronization(path, junction_cycles[condition], flows[condition],
                                 condition, "Demand Flow (vehicles / 5 min / route)",
                                 cycle_ylim, flow_ylim,
                                 "{} - {} case".format(junction, condition.capitalize()))
            print("wrote {}".format(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    flow = subparsers.add_parser("flow", help="Aggregate route departure counts in 5-minute bins.")
    flow.add_argument("--indices", default="1-50")
    flow.add_argument("--shock-route-dir", type=Path)
    flow.add_argument("--baseline-route-dir", type=Path)
    flow.add_argument("--output-dir", type=Path)
    flow.add_argument("--shock-begin", type=int, default=25200)
    flow.add_argument("--shock-end", type=int, default=28800)
    flow.set_defaults(func=run_flow)
    cycle = subparsers.add_parser("cycle", help="Aggregate policy action cycle time in 5-minute bins.")
    cycle.add_argument("--indices", default="1-50")
    cycle.add_argument("--shock-conditions", "--shock-condition", dest="shock_conditions",
                       default="shock_routes_1_50_h05_h11",
                       help="Comma-separated evaluation conditions that together cover the analysis window.")
    cycle.add_argument("--baseline-conditions", "--baseline-condition", dest="baseline_conditions",
                       default="baseline_routes_1_50_h05_h11",
                       help="Comma-separated evaluation conditions that together cover the analysis window.")
    cycle.add_argument("--window-begin", type=int, default=18000)
    cycle.add_argument("--window-end", type=int, default=39600)
    cycle.add_argument("--output-dir", type=Path)
    cycle.add_argument("--shock-begin", type=int, default=25200)
    cycle.add_argument("--shock-end", type=int, default=28800)
    cycle.set_defaults(func=run_cycle)
    sync = subparsers.add_parser("sync", help="Plot cycle time and flow for each condition on twin axes.")
    sync.add_argument("--indices", default="1-50")
    sync.add_argument("--shock-conditions", "--shock-condition", dest="shock_conditions",
                      default="shock_routes_1_50_h05_h11")
    sync.add_argument("--baseline-conditions", "--baseline-condition", dest="baseline_conditions",
                      default="baseline_routes_1_50_h05_h11")
    sync.add_argument("--window-begin", type=int, default=0)
    sync.add_argument("--window-end", type=int, default=HORIZON)
    sync.add_argument("--output-dir", type=Path)
    sync.set_defaults(func=run_sync)
    sync_demand = subparsers.add_parser("sync-demand",
                                        help="Plot cycle time against route demand flow on twin axes.")
    sync_demand.add_argument("--indices", default="1-50")
    sync_demand.add_argument("--shock-conditions", "--shock-condition", dest="shock_conditions",
                             default="shock_routes_1_50_h05_h11")
    sync_demand.add_argument("--baseline-conditions", "--baseline-condition", dest="baseline_conditions",
                             default="baseline_routes_1_50_h05_h11")
    sync_demand.add_argument("--shock-route-dir", type=Path, required=True)
    sync_demand.add_argument("--baseline-route-dir", type=Path, required=True)
    sync_demand.add_argument("--window-begin", type=int, default=0)
    sync_demand.add_argument("--window-end", type=int, default=HORIZON)
    sync_demand.add_argument("--output-dir", type=Path)
    sync_demand.set_defaults(func=run_sync_demand)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
