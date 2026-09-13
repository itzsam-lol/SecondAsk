"""Reply parser accuracy against known ground truth.

The project claims a language model earns its place on exactly one task: reading
free-text customer replies that are Hinglish, transliterated Hindi, typos and
emoji, where "no regex holds". Until now that claim was argued rather than
measured, because every benchmark ran the deterministic parser.

This measures it directly, and it is the right experiment for two reasons.

**It isolates the variable.** Running the whole agent loop against a real model
buries the parser's contribution under the planner, the gate and the simulator's
noise, and a small effect needs more seeds than a rate-limited project can
afford. Here the parser is the only thing being tested.

**It is cheap.** A few hundred calls rather than tens of thousands, which matters
when the constraint is a per-minute quota rather than money.

Ground truth comes from the world's own reply corpus, where each message was
generated *from* a known latent intent. That is a real label, not a human
annotation to be argued with, and it is the same corpus the benchmark uses.

What is reported is per-intent precision and recall, not just accuracy. A parser
that gets 85% right by never predicting ``opt_out`` is worse than useless, and
accuracy alone hides that.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from ..llm.gateway import LLMGateway
from ..world.entities import ReplyIntent
from ..world.outcomes import _REPLY_BANK


@dataclass
class IntentScore:
    intent: str
    support: int = 0
    predicted: int = 0
    correct: int = 0

    @property
    def precision(self) -> float:
        return self.correct / self.predicted if self.predicted else 0.0

    @property
    def recall(self) -> float:
        return self.correct / self.support if self.support else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "support": self.support,
            "predicted": self.predicted,
            "correct": self.correct,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


@dataclass
class ParserReport:
    backend: str
    n: int = 0
    correct: int = 0
    per_intent: dict[str, IntentScore] = field(default_factory=dict)
    confusions: Counter = field(default_factory=Counter)
    errors: int = 0
    fell_back: int = 0
    seconds: float = 0.0
    # message -> was it classified correctly. Keyed by text so two parsers can be
    # compared case by case, which is what a paired test needs.
    outcomes: dict[str, bool] = field(default_factory=dict)
    hard_total: int = 0
    hard_correct: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else 0.0

    @property
    def macro_f1(self) -> float:
        """Unweighted mean F1.

        Macro rather than micro on purpose. The corpus is unbalanced, and a micro
        average lets a parser look good by handling the common intents while
        missing every ``hardship`` and ``wrong_number``, which are exactly the
        ones with consequences.
        """
        scores = [s.f1 for s in self.per_intent.values() if s.support]
        return sum(scores) / len(scores) if scores else 0.0

    # Intents where a miss is a compliance or conduct failure rather than a
    # missed rupee. Tracked separately because they are the reason the parser
    # exists at all.
    CRITICAL = ("opt_out", "dispute", "hardship", "wrong_number")

    @property
    def hard_accuracy(self) -> float:
        """Accuracy on cases written to defeat a keyword matcher."""
        return self.hard_correct / self.hard_total if self.hard_total else 0.0

    @property
    def critical_recall(self) -> float:
        support = sum(self.per_intent[i].support for i in self.CRITICAL if i in self.per_intent)
        correct = sum(self.per_intent[i].correct for i in self.CRITICAL if i in self.per_intent)
        return correct / support if support else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "n": self.n,
            "accuracy": round(self.accuracy, 4),
            "macro_f1": round(self.macro_f1, 4),
            "critical_recall": round(self.critical_recall, 4),
            "hard_accuracy": round(self.hard_accuracy, 4),
            "hard_n": self.hard_total,
            "errors": self.errors,
            "fell_back": self.fell_back,
            "seconds": round(self.seconds, 1),
            "per_intent": [s.to_dict() for s in sorted(self.per_intent.values(), key=lambda s: s.intent)],
            "top_confusions": [
                {"true": t, "predicted": p, "n": n} for (t, p), n in self.confusions.most_common(8)
            ],
        }


def corpus(repeats: int = 1, *, source: str = "eval") -> list[tuple[str, ReplyIntent]]:
    """Labelled replies to score against.

    ``source="eval"`` uses the held-out evaluation corpus, which is the default
    and the right one: the deterministic parser's patterns were written against
    the world corpus, so scoring on that measures memorisation rather than
    generalisation.

    ``source="world"`` uses the simulator's own corpus, and ``source="both"``
    concatenates them.

    ``repeats`` is for variance, not volume. At temperature zero a model should
    give the same answer twice, and a repeat that disagrees with itself is worth
    knowing about. It does not create independent samples and must not be used to
    inflate n for a significance claim.
    """
    from .reply_corpus import EVAL_REPLIES

    base: list[tuple[str, ReplyIntent]] = []
    if source in ("eval", "both"):
        base.extend(EVAL_REPLIES)
    if source in ("world", "both"):
        for intent, messages in _REPLY_BANK.items():
            for text in messages:
                base.append((text, intent))
    if source not in ("eval", "world", "both"):
        raise ValueError(f"unknown corpus source {source!r}")

    out: list[tuple[str, ReplyIntent]] = []
    for _ in range(max(1, repeats)):
        out.extend(base)
    return out


def hard_cases() -> list[tuple[str, ReplyIntent]]:
    from .reply_corpus import HARD_CASES

    return list(HARD_CASES)


def mcnemar(a: "ParserReport", b: "ParserReport") -> dict[str, Any]:
    """Exact two-sided sign test over the disagreements between two parsers.

    The paired test is the right one: both parsers saw identical inputs, so what
    matters is the cases where they differ, not their marginal accuracies. On 39
    messages an earlier run produced four disagreements, all favouring the model,
    which is p=0.125. Unambiguous in direction and not significant, and reporting
    only "100% versus 89.7%" would have hidden that entirely.
    """
    import math

    only_a = sum(1 for k, v in a.outcomes.items() if v and not b.outcomes.get(k, False))
    only_b = sum(1 for k, v in b.outcomes.items() if v and not a.outcomes.get(k, False))
    n = only_a + only_b
    if n == 0:
        return {"discordant": 0, "favours_a": 0, "favours_b": 0, "p": 1.0}
    k = min(only_a, only_b)
    p = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2**n))
    return {
        "discordant": n,
        "favours_a": only_a,
        "favours_b": only_b,
        "p": round(p, 4),
        "significant_at_05": p < 0.05,
    }


_HARD_TEXTS: set[str] = set()


def score(
    gateway: LLMGateway,
    now: datetime,
    samples: Optional[list[tuple[str, ReplyIntent]]] = None,
    *,
    progress: Optional[Any] = None,
) -> ParserReport:
    import time

    samples = samples if samples is not None else corpus()
    if not _HARD_TEXTS:
        _HARD_TEXTS.update(text for text, _ in hard_cases())
    report = ParserReport(backend=gateway.backend_name)
    started = time.time()

    for index, (text, truth) in enumerate(samples):
        parsed = gateway.parse_reply(text, now)
        predicted = parsed.intent.value
        expected = truth.value

        report.n += 1
        report.per_intent.setdefault(expected, IntentScore(expected)).support += 1
        report.per_intent.setdefault(predicted, IntentScore(predicted)).predicted += 1

        # A multi-intent parse counts as correct if the true intent is present
        # anywhere. The precedence table decides which one governs, and that is
        # a separate, deliberate decision: a message that is both a promise and a
        # stop should be led by the stop, and marking that wrong would penalise
        # the parser for being right about safety.
        hit = predicted == expected or truth in parsed.intents
        report.outcomes[text] = hit
        if text in _HARD_TEXTS:
            report.hard_total += 1
            if hit:
                report.hard_correct += 1
        if hit:
            report.correct += 1
            report.per_intent[expected].correct += 1
        else:
            report.confusions[(expected, predicted)] += 1

        if "fallback" in parsed.source or "backend_down" in parsed.source:
            report.fell_back += 1
        if progress and index % 10 == 0:
            progress(f"{index + 1}/{len(samples)}")

    report.errors = gateway.stats.errors
    report.seconds = time.time() - started
    return report


def render(reports: list[ParserReport]) -> str:
    """Side-by-side comparison table."""
    lines = []
    header = (
        f"{'parser':28s} {'n':>5s} {'accuracy':>9s} {'macro F1':>9s} "
        f"{'critical':>9s} {'hard':>9s} {'errors':>7s} {'time':>7s}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for r in reports:
        lines.append(
            f"{r.backend:28s} {r.n:>5d} {r.accuracy:>8.1%} {r.macro_f1:>9.3f} "
            f"{r.critical_recall:>8.1%} {r.hard_accuracy:>8.1%} {r.errors:>7d} {r.seconds:>6.0f}s"
        )
    lines.append("")
    lines.append("critical: recall on opt_out, dispute, hardship and wrong_number, where a")
    lines.append("miss is a conduct failure rather than a missed rupee.")
    lines.append("hard: cases written specifically to defeat a keyword matcher.")

    if len(reports) == 2:
        test = mcnemar(reports[1], reports[0])
        lines.append("")
        lines.append(
            f"paired comparison: {test['discordant']} disagreements, "
            f"{test['favours_a']} favour {reports[1].backend}, "
            f"{test['favours_b']} favour {reports[0].backend}"
        )
        lines.append(
            f"exact two-sided sign test p={test['p']}"
            + ("  (significant)" if test.get('significant_at_05') else "  (not significant)")
        )

    intents = sorted({i for r in reports for i, s in r.per_intent.items() if s.support})
    lines.append("")
    head = f"{'intent':26s} {'support':>8s}" + "".join(f"{r.backend[:16]:>18s}" for r in reports)
    lines.append(head)
    lines.append("-" * len(head))
    for intent in intents:
        support = next((r.per_intent[intent].support for r in reports if intent in r.per_intent), 0)
        row = f"{intent:26s} {support:>8d}"
        for r in reports:
            s = r.per_intent.get(intent)
            row += f"{(f'{s.recall:.0%} recall' if s and s.support else '-'):>18s}"
        lines.append(row)
    return "\n".join(lines)
