"""Parser accuracy benchmark.

The scoring code decides a published claim, so it gets tested like anything else
that decides a published claim.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from secondask.eval.parser_bench import ParserReport, corpus, hard_cases, mcnemar, render, score
from secondask.eval.reply_corpus import EVAL_REPLIES, HARD_CASES
from secondask.llm.gateway import LLMGateway
from secondask.world.entities import ReplyIntent

NOW = datetime(2026, 3, 10, 12, tzinfo=timezone.utc)


class CorpusTest(unittest.TestCase):
    def test_every_label_is_a_real_intent(self):
        for text, intent in EVAL_REPLIES:
            self.assertIsInstance(intent, ReplyIntent)
            self.assertTrue(text.strip(), "empty message in the corpus")

    def test_no_duplicate_messages(self):
        texts = [t for t, _ in EVAL_REPLIES]
        self.assertEqual(len(texts), len(set(texts)), "a duplicate inflates n without adding evidence")

    def test_every_intent_has_support(self):
        """A corpus missing an intent silently excuses a parser from handling it."""
        covered = {i.value for _, i in EVAL_REPLIES}
        for intent in ReplyIntent:
            if intent in (ReplyIntent.NONE,):
                continue
            self.assertIn(intent.value, covered, f"{intent.value} has no labelled examples")

    def test_hard_cases_are_drawn_from_the_corpus(self):
        lookup = dict(EVAL_REPLIES)
        for text, intent in HARD_CASES:
            self.assertIn(text, lookup, "a hard case must also be a scored case")
            self.assertEqual(lookup[text], intent, "hard case label disagrees with the corpus")

    def test_eval_corpus_is_separate_from_the_world(self):
        """Changing the eval set must not change any benchmark number."""
        world = {t for t, _ in corpus(source="world")}
        evaluation = {t for t, _ in corpus(source="eval")}
        overlap = world & evaluation
        self.assertEqual(overlap, set(), f"shared messages would couple the two: {list(overlap)[:3]}")

    def test_sources(self):
        self.assertEqual(len(corpus(source="both")),
                         len(corpus(source="eval")) + len(corpus(source="world")))
        with self.assertRaises(ValueError):
            corpus(source="nonsense")

    def test_repeats_multiply(self):
        self.assertEqual(len(corpus(repeats=3)), 3 * len(corpus(repeats=1)))


class ScoringTest(unittest.TestCase):
    def test_deterministic_parser_scores(self):
        report = score(LLMGateway(backend=None), NOW, corpus())
        self.assertEqual(report.n, len(EVAL_REPLIES))
        self.assertGreater(report.accuracy, 0.0)
        self.assertLessEqual(report.accuracy, 1.0)
        self.assertEqual(report.hard_total, len(HARD_CASES))

    def test_macro_f1_is_not_dominated_by_the_common_class(self):
        """A parser that only ever says the majority intent must score badly."""
        report = ParserReport(backend="stubborn")
        from secondask.eval.parser_bench import IntentScore

        report.n = 100
        report.correct = 60
        report.per_intent["promise_to_pay"] = IntentScore("promise_to_pay", support=60, predicted=100, correct=60)
        report.per_intent["opt_out"] = IntentScore("opt_out", support=40, predicted=0, correct=0)
        self.assertAlmostEqual(report.accuracy, 0.60)
        self.assertLess(report.macro_f1, 0.40, "macro F1 must punish ignoring a whole class")

    def test_critical_recall_covers_the_conduct_intents(self):
        self.assertEqual(
            set(ParserReport.CRITICAL),
            {"opt_out", "dispute", "hardship", "wrong_number"},
        )

    def test_empty_report_does_not_divide_by_zero(self):
        report = ParserReport(backend="none")
        self.assertEqual(report.accuracy, 0.0)
        self.assertEqual(report.macro_f1, 0.0)
        self.assertEqual(report.critical_recall, 0.0)
        self.assertEqual(report.hard_accuracy, 0.0)

    def test_render_handles_one_and_two_parsers(self):
        one = score(LLMGateway(backend=None), NOW, corpus()[:12])
        self.assertIn("parser", render([one]))
        self.assertNotIn("paired comparison", render([one]))
        self.assertIn("paired comparison", render([one, one]))


class PairedTestTest(unittest.TestCase):
    def build(self, outcomes: dict[str, bool]) -> ParserReport:
        report = ParserReport(backend="x")
        report.outcomes = dict(outcomes)
        return report

    def test_no_disagreement_is_p_one(self):
        a = self.build({"m1": True, "m2": False})
        self.assertEqual(mcnemar(a, self.build({"m1": True, "m2": False}))["p"], 1.0)

    def test_all_disagreements_one_way_are_significant_when_numerous(self):
        outcomes_a = {f"m{i}": True for i in range(12)}
        outcomes_b = {f"m{i}": False for i in range(12)}
        result = mcnemar(self.build(outcomes_a), self.build(outcomes_b))
        self.assertEqual(result["discordant"], 12)
        self.assertEqual(result["favours_a"], 12)
        self.assertTrue(result["significant_at_05"])

    def test_four_one_way_disagreements_are_not_significant(self):
        """The exact case that made the first 39-message run unpublishable."""
        a = self.build({f"m{i}": True for i in range(4)})
        b = self.build({f"m{i}": False for i in range(4)})
        result = mcnemar(a, b)
        self.assertEqual(result["discordant"], 4)
        self.assertAlmostEqual(result["p"], 0.125, places=3)
        self.assertFalse(result["significant_at_05"])

    def test_it_is_paired_not_marginal(self):
        """Equal accuracy on different cases is still a real disagreement."""
        a = self.build({"m1": True, "m2": False})
        b = self.build({"m1": False, "m2": True})
        result = mcnemar(a, b)
        self.assertEqual(result["discordant"], 2)
        self.assertEqual(result["favours_a"], 1)
        self.assertEqual(result["favours_b"], 1)


if __name__ == "__main__":
    unittest.main()
