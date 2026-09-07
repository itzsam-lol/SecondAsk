"""Command line interface.

    python -m secondask world        describe a generated batch
    python -m secondask train        fit the underwriter on training seeds
    python -m secondask calibrate    calibration on held-out seeds
    python -m secondask eval         the comparison table
    python -m secondask run          one agent, with a decision receipt
    python -m secondask ablate       what each component contributes
    python -m secondask injection    the prompt injection suite
    python -m secondask verify       check a ledger's hash chain
    python -m secondask sweep        sensitivity to the goodwill price
    python -m secondask serve        the dashboard

Everything is deterministic given a seed. Nothing needs credentials: without an
``ANTHROPIC_API_KEY`` the model layer runs its deterministic path, and without
Razorpay test keys the gateway runs a faithful mock. Both are stated in the
output so a result is never ambiguous about which path produced it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

from .money import fmt, force_ascii, pct, safe_div, setup_stdout


def _add_world_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seeds", type=str, default="", help="comma separated, overrides --seed")
    parser.add_argument("-n", "--items", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=21, help="days")


def _seeds(args: argparse.Namespace) -> tuple[int, ...]:
    if args.seeds:
        return tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())
    return (args.seed,)


def _progress(message: str) -> None:
    print(f"  {message}", flush=True)


def cmd_world(args: argparse.Namespace) -> int:
    from .eval.bounds import best_single_action
    from .world.generator import generate_world

    for seed in _seeds(args):
        world = generate_world(seed=seed, n_items=args.items, horizon_days=args.horizon)
        summary = world.summary()
        at_risk = summary.pop("at_risk_paise")
        print(json.dumps(summary, indent=2))
        print(f"at risk: {fmt(at_risk)}")
        if args.bound:
            bound = best_single_action(world)
            print()
            print(f"best single action bound : {fmt(bound['best_single_action_paise'], compact=True)} "
                  f"({bound['best_single_action_share']:.1%} of value)")
            print(f"structurally hard         : {fmt(bound['structurally_hard_paise'], compact=True)} "
                  f"({bound['structurally_hard_share']:.1%} of value, "
                  f"{bound['structurally_hard_items']} items whose best possible "
                  f"single action is under p={bound['structurally_hard_threshold']})")
            print(f"note: {bound['caveat']}")
        print()
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from .underwrite.model import assert_disjoint
    from .underwrite.training import DEFAULT_TRAIN_SEEDS, train

    train_seeds = tuple(int(s) for s in args.train_seeds.split(",")) if args.train_seeds else DEFAULT_TRAIN_SEEDS
    eval_seeds = tuple(int(s) for s in args.eval_seeds.split(",")) if args.eval_seeds else (7, 11, 13)
    # Refuses to proceed rather than warning. A model fitted on an evaluation
    # seed would make every reported number meaningless, and a warning printed
    # into a long training log is a warning nobody reads.
    assert_disjoint(train_seeds, eval_seeds)

    print(f"training on seeds {list(train_seeds)}, held out from {list(eval_seeds)}")
    started = time.time()
    underwriter = train(train_seeds, n_items=args.items, horizon_days=args.horizon, progress=_progress)
    underwriter.save(args.model)
    print(f"fitted {len(underwriter.models)} models on {underwriter.n_samples} samples "
          f"in {time.time() - started:.1f}s -> {args.model}")

    report = underwriter.report()
    print()
    for action, info in report["actions"].items():
        print(f"{action:22s} n={info['n_train']:6d} base={info['base_rate']:.3f}")
        for name, weight in info["top_coefficients"][:4]:
            print(f"      {name:34s} {weight:+.3f}")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    from .eval.report import calibration_table
    from .underwrite import calibration
    from .underwrite.model import Underwriter, assert_disjoint
    from .underwrite.training import collect_eval_samples

    underwriter = Underwriter.load(args.model)
    eval_seeds = tuple(int(s) for s in args.eval_seeds.split(",")) if args.eval_seeds else (301, 302)
    assert_disjoint(underwriter.train_seeds, eval_seeds)
    print(f"model trained on {underwriter.train_seeds}, calibrating on held-out {list(eval_seeds)}")

    samples = collect_eval_samples(eval_seeds, n_items=args.items, horizon_days=args.horizon)
    every_p: list[float] = []
    every_y: list[int] = []
    print()
    print(f"{'action':22s} {'n':>7s} {'base':>7s} {'AUC':>7s} {'Brier':>9s} {'skill':>8s} {'ECE':>7s}")
    print("-" * 70)
    for action, (X, y) in sorted(samples.items()):
        if not X:
            continue
        model = underwriter.models.get(action)
        if model is None:
            continue
        predictions = model.predict(X)
        report = calibration.evaluate(predictions, y)
        every_p += predictions
        every_y += list(y)
        print(f"{action:22s} {report.n:>7d} {report.base_rate:>7.3f} {report.auc:>7.3f} "
              f"{report.brier:>9.5f} {report.brier_skill:>+8.3f} {report.ece:>7.3f}")
    print()
    overall = calibration.evaluate(every_p, every_y)
    print(calibration_table(overall.to_dict()))
    if args.json:
        with open(args.json, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(overall.to_dict(), handle, indent=2)
        print(f"\nwrote {args.json}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .eval.harness import CONFIGS, DEFAULT_SUITE, NOT_VALID_RESULTS, run_suite
    from .eval.report import ablation_table, comparison_table, denial_table, violations_table

    keys = [k.strip() for k in args.agents.split(",")] if args.agents else DEFAULT_SUITE
    seeds = _seeds(args)
    print(f"seeds={list(seeds)} items={args.items} horizon={args.horizon}d "
          f"gateway={args.razorpay} llm={'claude' if args.real_llm else 'deterministic'}")
    started = time.time()
    results = run_suite(
        keys,
        seeds=seeds,
        n_items=args.items,
        horizon_days=args.horizon,
        razorpay_mode=args.razorpay,
        gateway_failure_rate=args.gateway_failure_rate,
        use_real_llm=args.real_llm,
        progress=_progress if args.verbose else None,
    )
    print(f"done in {time.time() - started:.0f}s\n")
    print(comparison_table(results["runs"], keys, NOT_VALID_RESULTS))
    print()
    print(ablation_table(results["runs"]))
    print()
    print(violations_table(results["runs"], "secondask"))
    print()
    print(denial_table(results["runs"], "secondask"))

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(results, handle, indent=1)
        print(f"\nwrote {args.json}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from .eval.harness import CONFIGS
    from .eval.report import method_breakdown
    from .execute.razorpay_client import RazorpayClient
    from .llm.anthropic_client import build_backend
    from .llm.gateway import LLMGateway
    from .policy.engine import PolicyEngine
    from .policy.rules import DEFAULT_RULES
    from .runtime import Runtime
    from .world.generator import generate_world

    if args.agent not in CONFIGS:
        print(f"unknown agent {args.agent!r}. known: {', '.join(sorted(CONFIGS))}", file=sys.stderr)
        return 2
    config = CONFIGS[args.agent]

    # Built here rather than through run_agent so the ledger is reachable and
    # --ledger can write the actual chain.
    world = generate_world(seed=args.seed, n_items=args.items, horizon_days=args.horizon)
    backend = build_backend() if (args.real_llm and config.llm_enabled) else None
    runtime = Runtime(
        world,
        config.factory(),
        PolicyEngine(list(DEFAULT_RULES), enabled=config.policy_enabled),
        RazorpayClient(mode=args.razorpay, seed=world.seed),
        LLMGateway(backend=backend, enabled=config.llm_enabled),
    )
    result = runtime.run()
    result.agent_name = args.agent
    data = result.to_dict()

    print(f"{args.agent}  seed={args.seed}  items={args.items}")
    print(f"  at risk       {fmt(data['at_risk_paise'])}")
    print(f"  recovered     {fmt(data['recovered_paise'])}  "
          f"({pct(data['recovered_paise'], data['at_risk_paise'])} of value, "
          f"{pct(data['items_recovered'], data['items_total'])} of items)")
    print(f"  cost          {fmt(data['cost_paise'])}")
    print(f"  net           {fmt(data['net_paise'])}")
    print(f"  messages      {data['messages_sent']}  "
          f"({fmt(int(safe_div(data['recovered_paise'], data['messages_sent'])))} per message)")
    print(f"  violations    {data['total_violations']}")
    print(f"  opt-outs      {data['opt_outs']}   complaints {data['complaints']}")
    print(f"  ledger        {data['ledger_entries']} entries, verified={data['ledger_verified']}")
    print(f"  chain head    {data['ledger_head'][:32]}...")
    print(f"  p95 decision  {data['p95_decision_ms']:.2f} ms")
    print(f"  gateway       {json.dumps(data['razorpay']['stats'])}")
    print(f"  breaker       {json.dumps(data['razorpay']['breaker'])}")
    print(f"  guards        {json.dumps(data['guard_hits'])}")
    print()

    rows = []
    by_method: dict[str, list[int]] = {}
    for item in world.items:
        bucket = by_method.setdefault(item.method.value, [0, 0, 0, 0])
        bucket[0] += item.amount_paise
        bucket[1] += item.recovered_paise
        bucket[2] += 1
        if item.recovered_paise > 0:
            bucket[3] += 1
    for method, (at_risk, recovered, n, n_rec) in by_method.items():
        rows.append((method, at_risk, recovered, n, n_rec))
    print(method_breakdown(rows))

    if args.ledger:
        directory = os.path.dirname(os.path.abspath(args.ledger))
        os.makedirs(directory, exist_ok=True)
        runtime.ledger.write_jsonl(args.ledger)
        print()
        print(f"wrote {len(runtime.ledger)} ledger entries to {args.ledger}")
        print(f"check it with:  python -m secondask verify {args.ledger}")
    return 0


def cmd_injection(args: argparse.Namespace) -> int:
    """Run the adversarial corpus through the real reply parser."""
    from datetime import datetime, timezone

    from .llm.anthropic_client import build_backend
    from .llm.gateway import LLMGateway
    from .llm.injection import INJECTIONS
    from .world.entities import ReplyIntent

    backend = build_backend() if args.real_llm else None
    gateway = LLMGateway(backend=backend)
    now = datetime(2026, 3, 10, 12, 0, tzinfo=timezone.utc)

    print(f"parser backend: {gateway.backend_name}")
    print(f"{len(INJECTIONS)} adversarial inputs\n")

    allowed = {intent.value for intent in ReplyIntent}
    escaped = 0
    for text, why in INJECTIONS:
        parsed = gateway.parse_reply(text, now)
        in_enum = parsed.intent.value in allowed
        # The pass criterion is structural, not behavioural. No intent in the
        # enum can settle, write off or re-price anything, so the only way an
        # injection could succeed is by producing something outside the enum.
        if not in_enum:
            escaped += 1
        marker = "ok " if in_enum else "ESCAPED"
        shown = text if len(text) <= 52 else text[:49] + "..."
        print(f"  {marker} {parsed.intent.value:16s} {shown!r}")
        if args.verbose:
            print(f"          why dangerous: {why}")

    print()
    print(f"escaped the enum: {escaped}")
    print("no enum member can settle an item, change an amount, or widen scope.")
    print("settlement is written only from a payment event; R-AMOUNT-BOUND ties every")
    print("money-moving action to the ledger balance, so a message cannot re-price one.")
    return 0 if escaped == 0 else 1


def cmd_verify(args: argparse.Namespace) -> int:
    from .ledger import Ledger

    ledger = Ledger.read_jsonl(args.path)
    ok, message = ledger.verify()
    print(f"{args.path}: {len(ledger)} entries")
    print(f"head: {ledger.head}")
    if ok:
        print("chain verified")
        return 0
    print(f"CHAIN BROKEN: {message}")
    return 1


def cmd_sweep(args: argparse.Namespace) -> int:
    """Sensitivity of the results to the goodwill price.

    A headline that only holds at one arbitrary constant is not a result, so the
    constant is swept and the whole curve is reported.
    """
    from .agents.secondask import SecondAskAgent
    from .eval.harness import AgentConfig, run_agent
    from .world.generator import generate_world

    prices = [int(p) for p in args.prices.split(",")]
    print(f"{'goodwill price':>15s} {'recovered':>12s} {'of value':>9s} {'msgs':>7s} "
          f"{'per msg':>10s} {'net':>12s} {'opt-out':>8s}")
    print("-" * 78)
    for price in prices:
        totals = {"rec": 0, "at_risk": 0, "msgs": 0, "net": 0, "opt": 0}
        for seed in _seeds(args):
            world = generate_world(seed=seed, n_items=args.items, horizon_days=args.horizon)
            config = AgentConfig(
                "sweep", "sweep",
                lambda p=price: SecondAskAgent(annoyance_price_paise=p),
            )
            data = run_agent(config, world).to_dict()
            totals["rec"] += data["recovered_paise"]
            totals["at_risk"] += data["at_risk_paise"]
            totals["msgs"] += data["messages_sent"]
            totals["net"] += data["net_paise"]
            totals["opt"] += data["opt_outs"]
        print(f"{price:>15d} {fmt(totals['rec'], compact=True):>12s} "
              f"{pct(totals['rec'], totals['at_risk']):>9s} {totals['msgs']:>7d} "
              f"{fmt(int(safe_div(totals['rec'], totals['msgs'])), compact=True):>10s} "
              f"{fmt(totals['net'], compact=True):>12s} {totals['opt']:>8d}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .server.app import serve

    serve(host=args.host, port=args.port, items=args.items, horizon=args.horizon)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="secondask", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ascii", action="store_true", help="use 'Rs.' instead of the rupee sign")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("world", help="describe a generated batch")
    _add_world_args(p)
    p.add_argument("--bound", action="store_true", help="also compute the single-action bound")
    p.set_defaults(func=cmd_world)

    p = sub.add_parser("train", help="fit the underwriter")
    _add_world_args(p)
    p.set_defaults(items=800)
    p.add_argument("--model", default=os.path.join("models", "underwriter.json"))
    p.add_argument("--train-seeds", default="")
    p.add_argument("--eval-seeds", default="")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("calibrate", help="calibration on held-out seeds")
    _add_world_args(p)
    p.set_defaults(items=400)
    p.add_argument("--model", default=os.path.join("models", "underwriter.json"))
    p.add_argument("--eval-seeds", default="")
    p.add_argument("--json", default="")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("eval", help="the comparison table")
    _add_world_args(p)
    p.add_argument("--agents", default="")
    p.add_argument("--razorpay", choices=["mock", "live_test"], default="mock")
    p.add_argument("--gateway-failure-rate", type=float, default=0.06)
    p.add_argument("--real-llm", action="store_true")
    p.add_argument("--json", default="")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("run", help="one agent in detail")
    _add_world_args(p)
    p.add_argument("--agent", default="secondask")
    p.add_argument("--razorpay", choices=["mock", "live_test"], default="mock")
    p.add_argument("--real-llm", action="store_true")
    p.add_argument("--ledger", default="")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("injection", help="prompt injection suite")
    p.add_argument("--real-llm", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_injection)

    p = sub.add_parser("verify", help="check a ledger hash chain")
    p.add_argument("path")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("sweep", help="sensitivity to the goodwill price")
    _add_world_args(p)
    p.add_argument("--prices", default="0,1000,2000,4000,8000,16000")
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("serve", help="run the dashboard")
    _add_world_args(p)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8420)
    p.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    setup_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "ascii", False):
        force_ascii()
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except BrokenPipeError:
        # Piping into `head` closes the pipe early. Not an error worth a
        # traceback.
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
