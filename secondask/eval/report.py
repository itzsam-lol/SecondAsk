"""Result formatting.

Two rules here, both learned the hard way during this build.

**Never print a value-weighted rate on its own.** An early version of this
benchmark reported only "percent of rupees recovered". Invoices were 7% of the
items and 69% of the value, so the headline was decided by whether about twenty
items happened to land, and two agents that differed by nothing meaningful
looked like they differed by a factor of three. Every table here prints the
value rate and the item rate side by side, and the per-method breakdown is one
command away.

**Never print a recovery figure without its violation count.** An agent that
ignores the contact window and the capacity limits recovers more. That number is
not a result, and putting it in a column next to a compliant one without a
marker invites exactly the comparison it should not support.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from ..money import fmt, pct, safe_div

NOT_VALID_NOTE = "not a valid result: see the flag column"


def _flag(key: str, invalid: frozenset[str]) -> str:
    if key in invalid:
        return "!"
    return " "


def comparison_table(
    runs: dict[str, list[dict[str, Any]]],
    order: Sequence[str],
    invalid: frozenset[str],
    *,
    width: int = 0,
) -> str:
    from .harness import CONFIGS, aggregate

    rows: list[tuple[str, dict[str, Any]]] = []
    for key in order:
        if key not in runs or not runs[key]:
            continue
        rows.append((key, aggregate(runs[key])))
    if not rows:
        return "no runs"

    header = (
        f"{'':1s} {'agent':26s} {'recovered':>12s} {'of value':>9s} {'of items':>9s} "
        f"{'msgs':>6s} {'per msg':>10s} {'net':>12s} {'viol':>6s} {'opt-out':>8s}"
    )
    lines = [header, "-" * len(header)]

    for key, agg in rows:
        rupees_per_message = safe_div(agg["recovered_paise"], agg["messages_sent"])
        lines.append(
            f"{_flag(key, invalid):1s} {key:26s} "
            f"{fmt(agg['recovered_paise'], compact=True):>12s} "
            f"{pct(agg['recovered_paise'], agg['at_risk_paise']):>9s} "
            f"{pct(agg['items_recovered'], agg['items_total']):>9s} "
            f"{agg['messages_sent']:>6d} "
            f"{fmt(int(rupees_per_message), compact=True):>10s} "
            f"{fmt(agg['net_paise'], compact=True):>12s} "
            f"{agg['total_violations']:>6d} "
            f"{agg['opt_outs']:>8d}"
        )

    lines.append("")
    lines.append("! marks a figure that is not a legitimate result:")
    for key, _ in rows:
        if key in invalid:
            note = CONFIGS[key].note if key in CONFIGS else ""
            lines.append(f"    {key}: {note}")
    return "\n".join(lines)


def ablation_table(runs: dict[str, list[dict[str, Any]]], base_key: str = "secondask") -> str:
    """What each component contributes, measured by removing it."""
    from .harness import aggregate

    if base_key not in runs or not runs[base_key]:
        return "no base run to ablate against"
    base = aggregate(runs[base_key])

    ablations = [
        ("secondask_no_policy", "remove the Constitution"),
        ("secondask_no_underwriter", "remove expected value pricing"),
        ("secondask_no_llm", "remove language understanding"),
    ]
    header = (
        f"{'ablation':34s} {'recovered':>12s} {'delta':>10s} {'msgs':>7s} "
        f"{'viol':>7s} {'promises':>9s} {'disputes':>9s}"
    )
    lines = [
        f"{'baseline: ' + base_key:34s} {fmt(base['recovered_paise'], compact=True):>12s} "
        f"{'':>10s} {base['messages_sent']:>7d} {base['total_violations']:>7d} "
        f"{base['promises_captured']:>9d} {base['disputes_detected']:>9d}",
        "-" * len(header),
        header,
        "-" * len(header),
    ]
    for key, label in ablations:
        if key not in runs or not runs[key]:
            continue
        agg = aggregate(runs[key])
        delta = agg["recovered_paise"] - base["recovered_paise"]
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"{label:34s} {fmt(agg['recovered_paise'], compact=True):>12s} "
            f"{sign + fmt(delta, compact=True, symbol=False):>10s} "
            f"{agg['messages_sent']:>7d} {agg['total_violations']:>7d} "
            f"{agg['promises_captured']:>9d} {agg['disputes_detected']:>9d}"
        )
    return "\n".join(lines)


def violations_table(runs: dict[str, list[dict[str, Any]]], key: str) -> str:
    from .harness import aggregate

    if key not in runs or not runs[key]:
        return f"no runs for {key}"
    agg = aggregate(runs[key])
    violations = agg.get("violations", {})
    if not violations:
        return f"{key}: no rule was broken in any run"
    lines = [f"{key}: rules broken", "-" * 60]
    for rule, count in sorted(violations.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {rule:28s} {count:>8d}")
    lines.append(f"  {'actions that broke a rule':28s} {agg.get('violating_actions', 0):>8d}")
    return "\n".join(lines)


def denial_table(runs: dict[str, list[dict[str, Any]]], key: str) -> str:
    """What the gate refused, which is as informative as what it allowed."""
    from .harness import aggregate

    if key not in runs or not runs[key]:
        return f"no runs for {key}"
    agg = aggregate(runs[key])
    denials = agg.get("policy_denials", {})
    if not denials:
        return f"{key}: nothing was refused"
    lines = [f"{key}: actions the gate refused", "-" * 60]
    for rule, count in sorted(denials.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {rule:28s} {count:>8d}")
    return "\n".join(lines)


def calibration_table(report: dict[str, Any]) -> str:
    lines = [
        f"n={report['n']}  base rate={report['base_rate']:.4f}",
        f"AUC          {report['auc']:.4f}",
        f"Brier        {report['brier']:.5f}  (base rate model: {report['brier_baseline']:.5f})",
        f"Brier skill  {report['brier_skill']:+.4f}   (above 0 beats predicting the base rate)",
        f"ECE          {report['ece']:.4f}",
        f"max cal err  {report['max_calibration_error']:.4f}",
        "",
        f"{'predicted':>12s} {'observed':>12s} {'n':>8s}",
        "-" * 34,
    ]
    for b in report["bins"]:
        lines.append(f"{b['predicted']:>12.4f} {b['observed']:>12.4f} {b['count']:>8d}")
    return "\n".join(lines)


def method_breakdown(rows: list[tuple[str, int, int, int, int]]) -> str:
    """Per-method recovery. ``rows`` is (method, at_risk, recovered, n, n_recovered)."""
    header = f"{'method':16s} {'at risk':>11s} {'recovered':>11s} {'of value':>9s} {'items':>12s}"
    lines = [header, "-" * len(header)]
    for method, at_risk, recovered, n, n_rec in sorted(rows, key=lambda r: -r[1]):
        lines.append(
            f"{method:16s} {fmt(at_risk, compact=True):>11s} {fmt(recovered, compact=True):>11s} "
            f"{pct(recovered, at_risk):>9s} {f'{n_rec}/{n}':>12s}"
        )
    return "\n".join(lines)


def confidence_table(runs: dict[str, list[dict[str, Any]]], order: Sequence[str]) -> str:
    """Per-seed variability, and the paired comparison that actually matters.

    Two things are reported that a single aggregate cannot show.

    The first is how much a headline moves between seeds. A ratio quoted to two
    decimals from three runs implies a precision that is not there, and this
    table is the correction to that.

    The second is the paired comparison against the fixed-schedule baseline.
    Both agents saw the same worlds under common random numbers, so the
    difference is measured seed by seed and the interval is over those
    differences. Treating the two as independent samples would discard the
    pairing that the shared seeds exist to create, and would give an interval
    several times too wide.
    """
    from .stats import MIN_SEEDS_FOR_CI, bootstrap_ci, paired_bootstrap_ci, ratio_ci, sign_test_p

    baseline_key = "b1_fixed_schedule"
    present = [k for k in order if k in runs and runs[k]]
    if not present:
        return "no runs"
    n_seeds = len(runs[present[0]])

    lines = [f"per-seed variability and paired comparison ({n_seeds} seeds, 95% bootstrap)"]
    if n_seeds < MIN_SEEDS_FOR_CI:
        lines.append(
            f"  n={n_seeds} is below the {MIN_SEEDS_FOR_CI}-seed minimum, so intervals are"
            " withheld rather than printed as though they meant something."
        )
    lines.append("")
    header = (
        f"{'agent':26s} {'recovered/seed':>22s} {'vs b1 (paired delta)':>26s} {'sign p':>8s}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    base = [r["recovered_paise"] for r in runs[baseline_key]] if baseline_key in runs else []
    for key in present:
        values = [r["recovered_paise"] for r in runs[key]]
        own = bootstrap_ci(values, salt=f"ci:{key}")
        cell = own.render(lambda v: fmt(int(v), compact=True))
        delta_cell, p_cell = "", ""
        if base and key != baseline_key and len(values) == len(base):
            delta = paired_bootstrap_ci(values, base, salt=f"d:{key}")
            delta_cell = delta.render(lambda v: ("+" if v >= 0 else "") + fmt(int(v), compact=True))
            p_cell = f"{sign_test_p([a - b for a, b in zip(values, base)]):.3f}"
        lines.append(f"{key:26s} {cell:>22s} {delta_cell:>26s} {p_cell:>8s}")

    if base and "secondask" in runs and len(runs["secondask"]) == len(base):
        sa = [r["recovered_paise"] for r in runs["secondask"]]
        ratio = ratio_ci(sa, base, salt="ratio:secondask")
        msgs_sa = [r["messages_sent"] for r in runs["secondask"]]
        msgs_b1 = [r["messages_sent"] for r in runs[baseline_key]]
        msg_ratio = ratio_ci(msgs_sa, msgs_b1, salt="ratio:messages")
        lines.append("")
        lines.append("secondask against the fixed schedule:")
        lines.append(f"  money    x{ratio.render(lambda v: f'{v:.2f}')}")
        lines.append(f"  messages x{msg_ratio.render(lambda v: f'{v:.2f}')}")
        if not ratio.reliable:
            lines.append("  (point estimates only; run more seeds for an interval)")
    return "\n".join(lines)
