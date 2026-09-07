"""Evaluation harness.

Runs a set of agents against identical worlds and collects the comparison.

Every agent sees the same seeds, the same items, the same downtime calendar and,
through common random numbers, the same outcome draws for identical actions. The
only thing that varies is the decision. That is what makes the table a
comparison rather than a collection of anecdotes.

Agent configurations bundle three switches, which is how the ablations are
expressed without touching agent code:

    policy   the Constitution gate, open or closed
    llm      the model boundary, on or off
    agent    which decision policy is running

So ``secondask_no_policy`` is the identical agent with the identical underwriter
and only the gate open, and any violation it produces is attributable to the
gate and nothing else.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Optional

from ..agents.base import Agent
from ..agents.baselines import AggressiveAgent, DoNothingAgent, FixedScheduleAgent, LLMOnlyAgent
from ..agents.oracle import OracleAgent
from ..execute.razorpay_client import RazorpayClient
from ..llm.anthropic_client import build_backend
from ..llm.gateway import LLMGateway
from ..policy.engine import PolicyEngine
from ..policy.rules import DEFAULT_RULES
from ..runtime import RunResult, Runtime
from ..world.generator import World, generate_world


@dataclass(frozen=True)
class AgentConfig:
    key: str
    label: str
    factory: Callable[[], Agent]
    policy_enabled: bool = True
    llm_enabled: bool = True
    note: str = ""


def _secondask_factory(**kwargs: Any) -> Callable[[], Agent]:
    def make() -> Agent:
        from ..agents.secondask import SecondAskAgent

        return SecondAskAgent(**kwargs)

    return make


CONFIGS: dict[str, AgentConfig] = {
    "b0_do_nothing": AgentConfig(
        "b0_do_nothing", "B0 do nothing", DoNothingAgent,
        note="floor: money that arrives without any intervention",
    ),
    "b1_fixed_schedule": AgentConfig(
        "b1_fixed_schedule", "B1 fixed +1h/+24h/+72h", FixedScheduleAgent,
        note="what most dunning systems actually do",
    ),
    "b2_aggressive": AgentConfig(
        "b2_aggressive", "B2 aggressive, gated", AggressiveAgent,
        note="maximum persistence, still behind the policy gate",
    ),
    "b2_aggressive_ungated": AgentConfig(
        "b2_aggressive_ungated", "B2 aggressive, ungated", AggressiveAgent,
        policy_enabled=False,
        note="the same agent with the gate open, to count what the gate was stopping",
    ),
    "b3_llm_only_ungated": AgentConfig(
        "b3_llm_only_ungated", "B3 LLM loop, no gate, no pricing", LLMOnlyAgent,
        policy_enabled=False,
        note=(
            "the common agentic demo shape. Its recovery number is NOT a valid "
            "result: it is achieved by taking actions no deployed system is "
            "permitted to take, and the violation count is the measure of that"
        ),
    ),
    "b3_llm_only_gated": AgentConfig(
        "b3_llm_only_gated", "B3 LLM loop, gated, no pricing", LLMOnlyAgent,
        policy_enabled=True,
        note="the same LLM loop made compliant, which is the fair comparison",
    ),
    "oracle": AgentConfig(
        "oracle", "Oracle (reads latent state)", OracleAgent,
        note=(
            "upper bound, not an agent: it sees the true blocker and the exact "
            "resolution time. Gated like everything else, so it isolates the "
            "value of information"
        ),
    ),
    "secondask": AgentConfig(
        "secondask", "SecondAsk (full)", _secondask_factory(),
        note="policy gate + underwriter + scoped model",
    ),
    "secondask_no_policy": AgentConfig(
        "secondask_no_policy", "SecondAsk without the Constitution", _secondask_factory(),
        policy_enabled=False,
        note="ablation: identical agent, gate open",
    ),
    "secondask_no_underwriter": AgentConfig(
        "secondask_no_underwriter", "SecondAsk without the underwriter",
        _secondask_factory(use_underwriter=False),
        note="ablation: same gate and model, heuristic action choice",
    ),
    "secondask_no_llm": AgentConfig(
        "secondask_no_llm", "SecondAsk without the model", _secondask_factory(),
        llm_enabled=False,
        note="ablation: no reply parsing, no slot filling",
    ),
}

DEFAULT_SUITE = [
    "b0_do_nothing",
    "b1_fixed_schedule",
    "b2_aggressive",
    "b2_aggressive_ungated",
    "b3_llm_only_gated",
    "b3_llm_only_ungated",
    "secondask_no_policy",
    "secondask_no_underwriter",
    "secondask_no_llm",
    "secondask",
    "oracle",
]

# Agents whose recovery figure is not a legitimate result, either because they
# ignored the policy gate or because they read latent state. Reported, and
# labelled, so nobody quotes them as an achievement.
NOT_VALID_RESULTS = frozenset({"b2_aggressive_ungated", "b3_llm_only_ungated", "oracle"})


def run_agent(
    config: AgentConfig,
    world: World,
    *,
    razorpay_mode: str = "mock",
    gateway_failure_rate: float = 0.06,
    spend_cap_paise: int = 0,
    escalation_daily_cap: int = 0,
    use_real_llm: bool = False,
) -> RunResult:
    """Run one agent against one world.

    The world is regenerated by the caller for each agent rather than reused,
    because a run mutates item and customer state. Sharing a world between
    agents would let the first one's messages fatigue the customers the second
    one meets.
    """
    agent = config.factory()
    policy = PolicyEngine(list(DEFAULT_RULES), enabled=config.policy_enabled)
    client = RazorpayClient(
        mode=razorpay_mode,
        seed=world.seed,
        failure_rate=gateway_failure_rate,
    )
    backend = build_backend() if (use_real_llm and config.llm_enabled) else None
    llm = LLMGateway(backend=backend, enabled=config.llm_enabled)
    runtime = Runtime(
        world, agent, policy, client, llm,
        spend_cap_paise=spend_cap_paise,
        escalation_daily_cap=escalation_daily_cap,
    )
    result = runtime.run()
    result.agent_name = config.key
    return result


def run_suite(
    keys: Optional[list[str]] = None,
    *,
    seeds: tuple[int, ...] = (7,),
    n_items: int = 1000,
    horizon_days: int = 21,
    razorpay_mode: str = "mock",
    gateway_failure_rate: float = 0.06,
    use_real_llm: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
    keys = keys or DEFAULT_SUITE
    unknown = [k for k in keys if k not in CONFIGS]
    if unknown:
        raise ValueError(f"unknown agent keys: {unknown}. known: {sorted(CONFIGS)}")

    runs: dict[str, list[dict[str, Any]]] = {k: [] for k in keys}
    world_summaries: list[dict[str, Any]] = []

    for seed in seeds:
        reference = generate_world(seed=seed, n_items=n_items, horizon_days=horizon_days)
        world_summaries.append(reference.summary())
        for key in keys:
            if progress:
                progress(f"seed {seed}: {key}")
            # Fresh world per agent so state mutation cannot leak across runs.
            world = generate_world(seed=seed, n_items=n_items, horizon_days=horizon_days)
            result = run_agent(
                CONFIGS[key],
                world,
                razorpay_mode=razorpay_mode,
                gateway_failure_rate=gateway_failure_rate,
                use_real_llm=use_real_llm,
            )
            runs[key].append(result.to_dict())

    return {
        "config": {
            "seeds": list(seeds),
            "n_items": n_items,
            "horizon_days": horizon_days,
            "razorpay_mode": razorpay_mode,
            "gateway_failure_rate": gateway_failure_rate,
            "llm": "claude" if use_real_llm else "deterministic",
        },
        "worlds": world_summaries,
        "runs": runs,
    }


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Average a set of per-seed runs into one row.

    Sums for money and counts, means for rates. Reported alongside the number of
    seeds so a single-seed result is never mistaken for a replicated one.
    """
    if not runs:
        return {}
    total = {
        "seeds": len(runs),
        "at_risk_paise": sum(r["at_risk_paise"] for r in runs),
        "recovered_paise": sum(r["recovered_paise"] for r in runs),
        "cost_paise": sum(r["cost_paise"] for r in runs),
        "net_paise": sum(r["net_paise"] for r in runs),
        "items_total": sum(r["items_total"] for r in runs),
        "items_recovered": sum(r["items_recovered"] for r in runs),
        "items_partial": sum(r["items_partial"] for r in runs),
        "items_escalated": sum(r["items_escalated"] for r in runs),
        "messages_sent": sum(r["messages_sent"] for r in runs),
        "silent_retries": sum(r["silent_retries"] for r in runs),
        "wasted_retries": sum(r["wasted_retries"] for r in runs),
        "actions_executed": sum(r["actions_executed"] for r in runs),
        "opt_outs": sum(r["opt_outs"] for r in runs),
        "complaints": sum(r["complaints"] for r in runs),
        "churns": sum(r["churns"] for r in runs),
        "replies_received": sum(r["replies_received"] for r in runs),
        "promises_captured": sum(r["promises_captured"] for r in runs),
        "disputes_detected": sum(r["disputes_detected"] for r in runs),
        "total_violations": sum(r["total_violations"] for r in runs),
        "violating_actions": sum(r["violating_actions"] for r in runs),
        "delivery_failures": sum(r["delivery_failures"] for r in runs),
        "p95_decision_ms": max(r["p95_decision_ms"] for r in runs),
        "ledger_verified": all(r["ledger_verified"] for r in runs),
    }
    violations: dict[str, int] = {}
    denials: dict[str, int] = {}
    kinds: dict[str, int] = {}
    for run in runs:
        for rule, count in run["violations"].items():
            violations[rule] = violations.get(rule, 0) + count
        for rule, count in run["policy_denials"].items():
            denials[rule] = denials.get(rule, 0) + count
        for kind, count in run["actions_by_kind"].items():
            kinds[kind] = kinds.get(kind, 0) + count
    total["violations"] = dict(sorted(violations.items()))
    total["policy_denials"] = dict(sorted(denials.items()))
    total["actions_by_kind"] = dict(sorted(kinds.items()))
    return total
