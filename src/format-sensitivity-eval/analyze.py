"""Compare free-form vs binary refusal rates across multiple runs.

Reports min / mean / max for each (task, harmfulness level) cell so run-to-run
variance is visible. Pulls all logs in logs/ matching the expected task names.
"""

from collections import defaultdict
from inspect_ai.log import read_eval_log, list_eval_logs


EXPECTED_TASKS = ["free_form_eval", "binary_eval"]
LEVELS = ["Low", "Medium", "High"]


def load_all_logs() -> dict[str, list[str]]:
    """Returns {task_name: [log_path, ...]} for all logs matching expected tasks."""
    all_logs = list_eval_logs("logs/")
    by_task: dict[str, list[tuple[float, str]]] = defaultdict(list)

    for log_info in all_logs:
        try:
            log = read_eval_log(log_info.name, header_only=True)
            task_name = log.eval.task
            if task_name in EXPECTED_TASKS:
                by_task[task_name].append((log_info.mtime, log_info.name))
        except Exception as e:
            print(f"⚠️  Skipping unreadable log {log_info.name}: {e}")

    # Sort each task's logs by mtime ascending
    return {
        task: [name for _, name in sorted(entries)] for task, entries in by_task.items()
    }


def refusal_rates_by_level(log_path: str) -> dict[str, tuple[int, int]]:
    """Returns {level: (refused, total)} for a single log."""
    log = read_eval_log(log_path)
    by_level: dict[str, list[float]] = defaultdict(list)
    for sample in log.samples:
        level = sample.metadata.get("harmfulness_level", "Unknown")
        score_obj = next(iter(sample.scores.values()))
        by_level[level].append(score_obj.value)
    return {level: (int(sum(vs)), len(vs)) for level, vs in by_level.items()}


def summarize_runs(task_logs: list[str]) -> dict[str, dict]:
    """Aggregates refusal rates across runs of the same task.
    Returns {level: {rates: [...], min, mean, max, n_runs}}.
    """
    per_level_rates: dict[str, list[float]] = defaultdict(list)

    for path in task_logs:
        per_run = refusal_rates_by_level(path)
        for level, (refused, total) in per_run.items():
            if total:
                per_level_rates[level].append(refused / total)

    summary = {}
    for level in LEVELS:
        rates = per_level_rates.get(level, [])
        if not rates:
            continue
        summary[level] = {
            "rates": rates,
            "min": min(rates),
            "mean": sum(rates) / len(rates),
            "max": max(rates),
            "n_runs": len(rates),
        }
    return summary


def fmt_pct(v: float) -> str:
    return f"{v * 100:>5.1f}%"


def fmt_cell(stats: dict | None) -> str:
    if stats is None:
        return "—"
    if stats["n_runs"] == 1:
        return fmt_pct(stats["mean"])
    return f"{fmt_pct(stats['mean'])} [{fmt_pct(stats['min']).strip()}–{fmt_pct(stats['max']).strip()}]"


def fmt_delta(ff: dict | None, bin_: dict | None) -> str:
    if ff is None or bin_ is None:
        return "—"
    delta_mean = (bin_["mean"] - ff["mean"]) * 100
    # Range of the delta is bounded by best/worst case combinations
    delta_min = (bin_["min"] - ff["max"]) * 100
    delta_max = (bin_["max"] - ff["min"]) * 100
    if ff["n_runs"] == 1 and bin_["n_runs"] == 1:
        return f"{delta_mean:+.1f}pp"
    return f"{delta_mean:+.1f}pp [{delta_min:+.1f} to {delta_max:+.1f}]"


def main():
    task_logs = load_all_logs()

    print("Logs found per task:")
    for task in EXPECTED_TASKS:
        n = len(task_logs.get(task, []))
        marker = "✓" if n > 0 else "⚠️"
        print(f"  {marker} {task}: {n} run(s)")
    print()

    missing = [t for t in EXPECTED_TASKS if t not in task_logs]
    if missing:
        print(f"Missing logs for: {missing}\n")

    summaries = {
        task: summarize_runs(task_logs[task])
        for task in EXPECTED_TASKS
        if task in task_logs
    }

    # Comparison table
    print(f"{'Level':<10} {'Free-form':<25} {'Binary':<25} {'Δ (Bin − FF)':<25}")
    print("-" * 85)
    for level in LEVELS:
        ff = summaries.get("free_form_eval", {}).get(level)
        bin_ = summaries.get("binary_eval", {}).get(level)
        print(
            f"{level:<10} "
            f"{fmt_cell(ff):<25} "
            f"{fmt_cell(bin_):<25} "
            f"{fmt_delta(ff, bin_):<25}"
        )
    print("-" * 85)

    # Overall row (across all prompts, across all runs)
    for task in EXPECTED_TASKS:
        if task not in task_logs:
            continue
        all_refused = 0
        all_n = 0
        for path in task_logs[task]:
            per_run = refusal_rates_by_level(path)
            for refused, total in per_run.values():
                all_refused += refused
                all_n += total
        n_runs = len(task_logs[task])
        avg = all_refused / all_n * 100 if all_n else 0
        print(
            f"  {task} (averaged across {n_runs} runs): {avg:.1f}% ({all_refused}/{all_n})"
        )


if __name__ == "__main__":
    main()
