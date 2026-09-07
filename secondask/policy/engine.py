"""The Constitution: hard constraints, checked before anything happens.

Every action any agent wants to take is a ``ProposedAction`` and must pass
``PolicyEngine.evaluate`` before an executor will touch it. There is no bypass,
including for actions the language model proposes. That is the structural answer
to "how do you stop the model doing something it shouldn't": the model does not
have the capability in the first place.

Two properties are deliberate.

**Fail closed.** If a rule raises an exception, whether from missing data, a bad
type or a plain bug, the engine records a denial rather than letting the action through. An
availability bug should cost a recovery attempt, never a regulatory breach. Rules
are also evaluated *exhaustively* rather than short-circuiting on the first
denial, so the audit trail shows every reason an action was refused rather than
just the first one encountered.

**Deferral, not just refusal.** A rule that fails on timing returns the earliest
instant the action *would* be permitted. This turns the 8 PM contact attempt into
an 8 AM contact attempt instead of a dropped one, which is the difference between
a compliant system and a compliant system that also collects money.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

from ..clock import iso
from ..world.entities import ActionKind, Channel, Customer, RiskItem


@dataclass
class ProposedAction:
    """A request to do something. Not yet permission to do it."""

    item_id: str
    customer_id: str
    kind: ActionKind
    channel: Channel
    scheduled_at: datetime
    amount_paise: int
    idempotency_key: str
    template_id: Optional[str] = None
    slots: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    p_recover: float = 0.0
    expected_value_paise: int = 0
    proposed_by: str = "agent"

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "customer_id": self.customer_id,
            "kind": self.kind.value,
            "channel": self.channel.value,
            "scheduled_at": iso(self.scheduled_at),
            "amount_paise": self.amount_paise,
            "idempotency_key": self.idempotency_key,
            "template_id": self.template_id,
            "rationale": self.rationale,
            "p_recover": round(self.p_recover, 4),
            "expected_value_paise": self.expected_value_paise,
            "proposed_by": self.proposed_by,
        }


@dataclass
class RuleVerdict:
    rule_id: str
    allowed: bool
    reason: str = ""
    retry_at: Optional[datetime] = None
    citation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "allowed": self.allowed,
            "reason": self.reason,
            "retry_at": iso(self.retry_at) if self.retry_at else None,
            "citation": self.citation,
        }


@dataclass
class PolicyDecision:
    allowed: bool
    checked: list[str]
    denials: list[RuleVerdict]
    earliest_allowed_at: Optional[datetime] = None

    @property
    def denied_rules(self) -> list[str]:
        return [d.rule_id for d in self.denials]

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "checked": self.checked,
            "denials": [d.to_dict() for d in self.denials],
            "earliest_allowed_at": iso(self.earliest_allowed_at) if self.earliest_allowed_at else None,
        }


@dataclass
class ContactRecord:
    at: datetime
    channel: Channel
    kind: ActionKind


@dataclass
class PolicyContext:
    """Everything a rule may look at.

    Assembled by the runtime, never by the agent. An agent cannot construct a
    context that makes its own action look permissible.
    """

    item: RiskItem
    customer: Customer
    now: datetime
    contacts_24h: int = 0
    contacts_7d: int = 0
    last_contact_at: Optional[datetime] = None
    used_idempotency_keys: frozenset[str] = frozenset()
    spend_paise: int = 0
    spend_cap_paise: int = 0
    consent_promotional: bool = False
    dnd_registered: bool = False
    merchant_name: str = "Merchant"


Rule = Callable[[ProposedAction, PolicyContext], RuleVerdict]


class PolicyEngine:
    def __init__(self, rules: list[Rule], *, enabled: bool = True) -> None:
        """``enabled=False`` produces the no-policy ablation.

        The ablation runs the identical agent with the identical underwriter and
        differs only in that this gate is open, so any violation it produces is
        attributable to the absence of the Constitution and nothing else.
        """
        self._rules = rules
        self.enabled = enabled

    @property
    def rule_ids(self) -> list[str]:
        return [getattr(r, "rule_id", r.__name__) for r in self._rules]

    def evaluate(self, action: ProposedAction, ctx: PolicyContext) -> PolicyDecision:
        checked: list[str] = []
        denials: list[RuleVerdict] = []
        retry_candidates: list[datetime] = []

        for rule in self._rules:
            rule_id = getattr(rule, "rule_id", rule.__name__)
            checked.append(rule_id)
            try:
                verdict = rule(action, ctx)
            except Exception as exc:  # noqa: BLE001, fail closed deliberately
                denials.append(
                    RuleVerdict(
                        rule_id=rule_id,
                        allowed=False,
                        reason=f"rule raised {type(exc).__name__}: {exc}; failing closed",
                    )
                )
                continue
            if not verdict.allowed:
                denials.append(verdict)
                if verdict.retry_at is not None:
                    retry_candidates.append(verdict.retry_at)

        if not self.enabled:
            # The ablation still *records* what would have been denied, so the
            # violation count is measured rather than assumed. This is the whole
            # point of running it.
            return PolicyDecision(allowed=True, checked=checked, denials=denials)

        allowed = not denials
        # Only offer a deferred time if timing was the *sole* obstacle. If some
        # other rule denies outright, rescheduling would not help and suggesting
        # a time would be misleading.
        earliest = None
        if denials and len(retry_candidates) == len(denials):
            earliest = max(retry_candidates)
        return PolicyDecision(allowed=allowed, checked=checked, denials=denials, earliest_allowed_at=earliest)
