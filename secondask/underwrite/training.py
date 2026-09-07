"""Exploration and model fitting.

The underwriter cannot be trained on data from a good policy, because a good
policy only ever takes actions it already believes will work. It would never
retry a dead card, so it would never learn that retrying a dead card fails, so it
would have no basis for believing that in the first place. This is the standard
off-policy bootstrap problem and the standard answer applies: explore.

So training data comes from a **random policy**. For each item it picks legal
actions at random times spread across the horizon and records what happened. The
distribution is deliberately wide and deliberately stupid; it exists to cover the
action-by-timing space, not to collect money.

Two constraints are respected even during exploration, because violating them
would put samples in the training set that no deployed policy could ever
generate:

* rail legality, so no silent retries on one-off payments,
* channel availability, so no SMS to a customer with no phone.

The regulatory constraints are deliberately *not* applied here. Whether an action
would have been permitted at 3 AM is a policy question; whether it would have
worked is a physical one, and the model is estimating the second. Confusing the
two would make the model unable to price a legal 8 AM action because its only
evidence came from illegal 3 AM ones.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Callable, Optional

from ..rng import Stream
from ..world.entities import ACTION_CHANNEL, ActionKind, Channel, Customer, RiskItem
from ..world.generator import World, generate_world
from ..world.outcomes import execute_counterfactual
from . import features as F
from .model import MODELABLE_ACTIONS, Underwriter

DEFAULT_TRAIN_SEEDS = (101, 102, 103, 104, 105, 106)
MAX_EXPLORE_STEPS = 5


def legal_actions(item: RiskItem, customer: Customer) -> list[ActionKind]:
    """Physically possible actions, ignoring regulation.

    See the module docstring: the model estimates whether an action *works*, and
    the policy engine decides separately whether it is *allowed*.
    """
    out: list[ActionKind] = []
    for action in MODELABLE_ACTIONS:
        if action in (ActionKind.SILENT_RETRY, ActionKind.MANDATE_DEBIT):
            if not item.method.supports_silent_retry:
                continue
            if action == ActionKind.MANDATE_DEBIT and not item.method.is_mandate:
                continue
            if action == ActionKind.SILENT_RETRY and item.method.is_mandate:
                continue
            out.append(action)
            continue
        channel = ACTION_CHANNEL[action]
        if channel in (Channel.SMS, Channel.WHATSAPP, Channel.EMAIL, Channel.VOICE):
            if not customer.channel_available(channel):
                continue
        out.append(action)
    return out


def collect_samples(
    seeds: tuple[int, ...] = DEFAULT_TRAIN_SEEDS,
    *,
    n_items: int = 800,
    horizon_days: int = 21,
    progress: Optional[Callable[[str], None]] = None,
) -> dict[str, tuple[list[list[float]], list[int]]]:
    """Run the exploration policy and return per-action training sets."""
    samples: dict[str, tuple[list[list[float]], list[int]]] = {
        action.value: ([], []) for action in MODELABLE_ACTIONS
    }

    for seed in seeds:
        if progress:
            progress(f"exploring seed {seed}")
        world = generate_world(seed=seed, n_items=n_items, horizon_days=horizon_days)
        stream = Stream(seed, "explore")

        for item in world.items:
            customer = world.customers[item.customer_id]
            bank = world.bank_of(item)
            now = item.failed_at

            for step in range(MAX_EXPLORE_STEPS):
                options = legal_actions(item, customer)
                if not options:
                    break
                action = stream.choice(options)

                # Spread offsets so timing effects are actually observed. The
                # long tail is what teaches the model about payday: without
                # samples a week or two out, "wait for the 1st" is invisible.
                offset_hours = stream.weighted(
                    [
                        (stream.uniform(0.2, 4.0), 0.28),
                        (stream.uniform(4.0, 24.0), 0.24),
                        (stream.uniform(24.0, 96.0), 0.24),
                        (stream.uniform(96.0, 24.0 * 14), 0.24),
                    ]
                )
                now = now + timedelta(hours=offset_hours)
                if now >= world.end:
                    break

                if action == ActionKind.MANDATE_DEBIT:
                    item.prenotified_at = now - timedelta(hours=26)

                vector = F.extract(item, customer, now, world.downtime, bank)
                outcome = execute_counterfactual(world, item, customer, action, now)

                X, y = samples[action.value]
                X.append(vector)
                y.append(1 if outcome.success else 0)

                # Advance state so later samples reflect a realistic history of
                # attempts and accumulated annoyance rather than a fresh item.
                item.attempts += 1
                if action.is_contact:
                    item.contacts += 1
                    item.last_contact_at = now
                customer.annoyance += outcome.annoyance_delta
                if outcome.opted_out is not None:
                    customer.opted_out_channels.add(outcome.opted_out)
                if outcome.success:
                    break

    return samples


def train(
    seeds: tuple[int, ...] = DEFAULT_TRAIN_SEEDS,
    *,
    n_items: int = 800,
    horizon_days: int = 21,
    progress: Optional[Callable[[str], None]] = None,
) -> Underwriter:
    samples = collect_samples(seeds, n_items=n_items, horizon_days=horizon_days, progress=progress)
    if progress:
        counts = ", ".join(f"{k}={len(v[0])}" for k, v in sorted(samples.items()))
        progress(f"fitting on {counts}")
    underwriter = Underwriter(train_seeds=list(seeds))
    underwriter.fit(samples, seed=seeds[0] if seeds else 0)
    return underwriter


def collect_eval_samples(
    seeds: tuple[int, ...],
    *,
    n_items: int = 400,
    horizon_days: int = 21,
) -> dict[str, tuple[list[list[float]], list[int]]]:
    """Held-out samples for calibration, drawn the same way from other seeds."""
    return collect_samples(seeds, n_items=n_items, horizon_days=horizon_days)
