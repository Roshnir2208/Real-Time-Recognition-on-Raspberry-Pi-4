
import json
import matplotlib.pyplot as plt
import numpy as np
import sys
from collections import defaultdict
import os

# --------------------------
# FUNCTIONS
# --------------------------

def ensure_output_folder():
    folder = "analysis_plots"
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder

def load_log(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
        if isinstance(data, dict):
            data = data.get("log", [])
        return data

def analyse_log(log):
    label_times = defaultdict(list)
    label_conf = defaultdict(list)
    fps_values = []

    for entry in log:
        if not isinstance(entry, dict):
            continue
        lbl = entry.get("label")
        t = entry.get("time")
        conf = entry.get("confidence")
        if lbl is not None and t is not None and conf is not None:
            label_times[lbl].append(float(t))
            label_conf[lbl].append(float(conf))
        if "fps" in entry:
            fps_values.append(float(entry["fps"]))

    return label_times, label_conf, fps_values

def plot_presence_timeline(label_times, duration, scenario, folder):
    labels = list(label_times.keys())
    fig, ax = plt.subplots(figsize=(10, 5))

    for idx, lbl in enumerate(labels):
        times = sorted(label_times[lbl])
        if not times:
            continue
        intervals = []
        start = times[0]
        for i in range(1, len(times)):
            if times[i] - times[i-1] > 1.0:
                intervals.append((start, times[i-1]))
                start = times[i]
        intervals.append((start, times[-1]))

        for (s, e) in intervals:
            ax.plot([s, e], [idx, idx], linewidth=6)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Presence Timeline - {scenario}")
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"timeline_{scenario}.png"))
    plt.close()

def plot_confidence(label_conf, scenario, folder):
    fig, ax = plt.subplots(figsize=(10, 5))
    for lbl, confs in label_conf.items():
        if not confs:
            continue
        times = np.arange(len(confs))
        ax.plot(times, confs, label=lbl)
    ax.set_xlabel("Detection Index")
    ax.set_ylabel("Confidence")
    ax.set_title(f"Confidence Over Time - {scenario}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"confidence_{scenario}.png"))
    plt.close()

def plot_avg_confidence(label_conf, scenario, folder):
    avg_conf = {lbl: np.mean(confs) for lbl, confs in label_conf.items() if confs}
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(avg_conf.keys(), avg_conf.values(), color='skyblue')
    ax.set_ylabel("Average Confidence")
    ax.set_title(f"Average Confidence per Label - {scenario}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"avg_confidence_{scenario}.png"))
    plt.close()

def plot_detection_count(label_conf, scenario, folder):
    counts = {lbl: len(confs) for lbl, confs in label_conf.items()}
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts.keys(), counts.values(), color='orange')
    ax.set_ylabel("Detection Count")
    ax.set_title(f"Detection Count per Label - {scenario}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"detection_count_{scenario}.png"))
    plt.close()


def quality_assessment(label_conf):
    issues = {}
    for lbl, confs in label_conf.items():
        low_conf = [c for c in confs if c < 0.4]
        if low_conf:
            issues[lbl] = len(low_conf)/len(confs)
    if not issues:
        print("No major quality issues detected. All detections are reasonably confident.")
    else:
        print("Quality Issues Detected:")
        for lbl, ratio in issues.items():
            print(f"  {lbl}: {ratio*100:.1f}% of detections are low confidence (<0.4)")



# --------------------------
# MAIN
# --------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 analyse_logs.py <log_file.json> [additional_log_files...]")
        sys.exit()

    log_files = sys.argv[1:]
    folder = ensure_output_folder()

    for log_file in log_files:
        if not os.path.exists(log_file):
            print(f"File not found: {log_file}")
            continue

        log = load_log(log_file)
        scenario = log_file.split(".")[0].replace("log_", "")

        label_times, label_conf, fps_values = analyse_log(log)

        plot_presence_timeline(label_times, duration=20, scenario=scenario, folder=folder)
        plot_confidence(label_conf, scenario, folder)
        plot_avg_confidence(label_conf, scenario, folder)
        plot_detection_count(label_conf, scenario, folder)
    
        quality_assessment(label_conf)
        print(f"Analysis completed for scenario: {scenario}. Plots saved in '{folder}'.")

    
