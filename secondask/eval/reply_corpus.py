"""Held-out evaluation corpus for the reply parser.

Deliberately **separate** from ``world.outcomes._REPLY_BANK``.

The world's corpus is part of the simulator: changing it changes every benchmark
number. This one exists only to measure the parser, so it can grow without
invalidating anything, and growing it is the point. An earlier version of this
benchmark ran on the world's 39 messages and found the model beating the regex
100% to 89.7%. Four disagreements, all favouring the model, which is an exact
two-sided sign test p of 0.125. Directionally clear, statistically nothing. The
corpus was the bottleneck, not the model.

It is also a genuine held-out set: the deterministic parser's patterns were
written against the world corpus, so measuring it here is measuring
generalisation rather than memorisation. That is the fair comparison, and it is
the one that flatters the regex least.

Composition is deliberate:

* **Hinglish and transliterated Hindi**, because that is what an Indian
  collections inbox actually contains and it is the case a keyword list handles
  worst.
* **Typos, missing spaces, SMS compression**, since these arrive from phone
  keyboards.
* **Ambiguity and multi-intent**, where the label is the intent that should
  *govern* under the precedence table, not merely one that is present.
* **Near-misses**, phrased to look like a different intent than they are. These
  are where a keyword matcher fails and they are over-represented on purpose,
  because an evaluation set of easy cases measures nothing.

No message here also appears in the world corpus. That is enforced by
``test_eval_corpus_is_separate_from_the_world``, which caught six collisions the
first time it ran: those six cases were not held out at all, since the
deterministic parser's patterns were written against them.
"""

from __future__ import annotations

from ..world.entities import ReplyIntent

I = ReplyIntent

# (message, governing intent)
EVAL_REPLIES: list[tuple[str, ReplyIntent]] = [
    # -- opt out -----------------------------------------------------------
    ("stop", I.OPT_OUT),
    ("STOP SENDING ME THIS", I.OPT_OUT),
    ("please remove my number from your list", I.OPT_OUT),
    ("mujhe aur message mat bhejo", I.OPT_OUT),
    ("dobara call mat karna", I.OPT_OUT),
    ("unsubscribe me", I.OPT_OUT),
    ("bandh karo ye sab", I.OPT_OUT),
    ("i will pay but stop calling me every day", I.OPT_OUT),
    ("kal pay kar dunga, par ab message band karo", I.OPT_OUT),
    ("do not contact me on this number again", I.OPT_OUT),

    # -- dispute -----------------------------------------------------------
    ("i did not order anything from you", I.DISPUTE),
    ("ye order maine kiya hi nahi", I.DISPUTE),
    ("the product never arrived, why should i pay", I.DISPUTE),
    ("maine cancel kar diya tha phir bhi charge kar rahe ho", I.DISPUTE),
    ("this is a fraud charge, i am going to my bank", I.DISPUTE),
    ("service kaam hi nahi ki, paisa kyun du", I.DISPUTE),
    ("wrong amount charged, i owe less than this", I.DISPUTE),
    ("i am raising a chargeback on this", I.DISPUTE),
    ("galat bill bheja hai aapne", I.DISPUTE),

    # -- hardship ----------------------------------------------------------
    ("i lost my job, please give me some time", I.HARDSHIP),
    ("papa hospital me hai, abhi kuch nahi kar sakta", I.HARDSHIP),
    ("medical emergency chal rahi hai ghar me", I.HARDSHIP),
    ("my father passed away last week", I.HARDSHIP),
    ("naukri chali gayi, koi income nahi hai abhi", I.HARDSHIP),
    ("i am in the hospital right now", I.HARDSHIP),

    # -- wrong number ------------------------------------------------------
    ("wrong number", I.WRONG_NUMBER),
    ("ye number mera nahi hai", I.WRONG_NUMBER),
    ("you have the wrong person, i don't know this company", I.WRONG_NUMBER),
    ("kisi aur ka number laga diya hai aapne", I.WRONG_NUMBER),
    ("i think you meant to message someone else", I.WRONG_NUMBER),

    # -- partial payment promise -------------------------------------------
    ("i can pay half now and half next month", I.PARTIAL_PAYMENT_PROMISE),
    ("abhi 500 de deta hoon, baaki baad me", I.PARTIAL_PAYMENT_PROMISE),
    ("can i pay Rs 2000 now and the rest on the 10th", I.PARTIAL_PAYMENT_PROMISE),
    ("kist me kar sakta hoon kya", I.PARTIAL_PAYMENT_PROMISE),
    ("only part payment possible this month", I.PARTIAL_PAYMENT_PROMISE),
    ("aadha aaj aadha agle hafte", I.PARTIAL_PAYMENT_PROMISE),
    ("i can manage 1500 out of it right now", I.PARTIAL_PAYMENT_PROMISE),
    ("thoda thoda karke chuka dunga", I.PARTIAL_PAYMENT_PROMISE),
    ("installment me pay karne ka option hai?", I.PARTIAL_PAYMENT_PROMISE),
    ("Rs 800 abhi bhej raha hoon, baaki salary ke baad", I.PARTIAL_PAYMENT_PROMISE),

    # -- promise to pay ----------------------------------------------------
    ("will pay tomorrow", I.PROMISE_TO_PAY),
    ("kal kar dunga pakka", I.PROMISE_TO_PAY),
    ("salary aate hi pay kar dunga", I.PROMISE_TO_PAY),
    ("give me time till the 5th", I.PROMISE_TO_PAY),
    ("teen din me ho jayega bhai", I.PROMISE_TO_PAY),
    ("i will clear it by month end", I.PROMISE_TO_PAY),
    ("agle hafte tak clear kar dunga", I.PROMISE_TO_PAY),
    ("pakka is baar, 15 tarikh tak", I.PROMISE_TO_PAY),
    ("ok will do it by friday", I.PROMISE_TO_PAY),
    ("thoda time do, kar dunga", I.PROMISE_TO_PAY),
    ("paisa aate hi settle kar dunga", I.PROMISE_TO_PAY),
    ("sure, processing it this week", I.PROMISE_TO_PAY),

    # -- already paid ------------------------------------------------------
    ("already paid", I.ALREADY_PAID),
    ("maine to kal hi kar diya tha", I.ALREADY_PAID),
    ("payment ho chuka hai, check karo", I.ALREADY_PAID),
    ("i paid this via upi yesterday, ref 88213", I.ALREADY_PAID),
    ("transaction successful dikha raha hai mujhe", I.ALREADY_PAID),
    ("bhej diya hai paisa, screenshot chahiye kya", I.ALREADY_PAID),
    ("this was settled last month", I.ALREADY_PAID),
    ("amount already debited from my account", I.ALREADY_PAID),

    # -- needs help --------------------------------------------------------
    ("the link is not opening", I.NEEDS_HELP),
    ("payment ka link open hi nahi ho raha", I.NEEDS_HELP),
    ("my card expired, how do i update it", I.NEEDS_HELP),
    ("naya card kaise add karun", I.NEEDS_HELP),
    ("payment page pe error aa raha hai", I.NEEDS_HELP),
    ("otp nahi aa raha hai", I.NEEDS_HELP),
    ("how do i pay by upi instead", I.NEEDS_HELP),
    ("it says transaction failed every time", I.NEEDS_HELP),
    ("mujhe samajh nahi aa raha kaise karun", I.NEEDS_HELP),
    ("can you send the link again, i deleted it", I.NEEDS_HELP),

    # -- unintelligible ----------------------------------------------------
    ("kk", I.UNINTELLIGIBLE),
    ("???", I.UNINTELLIGIBLE),
    ("....", I.UNINTELLIGIBLE),
    ("hmmm", I.UNINTELLIGIBLE),
    ("asdfgh", I.UNINTELLIGIBLE),
    ("\U0001f644", I.UNINTELLIGIBLE),
    ("ok", I.UNINTELLIGIBLE),
    (".", I.UNINTELLIGIBLE),
]

# Cases written specifically to defeat a keyword matcher. Kept as a named subset
# so the report can show how each parser does on the hard tail rather than only
# in aggregate, since an average over easy cases hides exactly this.
HARD_CASES: list[tuple[str, ReplyIntent]] = [
    # "pay" appears, but the governing intent is a stop.
    ("i will pay but stop calling me every day", I.OPT_OUT),
    ("kal pay kar dunga, par ab message band karo", I.OPT_OUT),
    # "paid" appears, but it is a dispute about the amount.
    ("wrong amount charged, i owe less than this", I.DISPUTE),
    # Reads like a promise, is actually hardship.
    ("naukri chali gayi, koi income nahi hai abhi", I.HARDSHIP),
    # Reads like a payment intent, is a wrong number.
    ("you have the wrong person, i don't know this company", I.WRONG_NUMBER),
    # Partial, with no explicit "half" or digit.
    ("thoda thoda karke chuka dunga", I.PARTIAL_PAYMENT_PROMISE),
    ("kist me kar sakta hoon kya", I.PARTIAL_PAYMENT_PROMISE),
    # Bare affirmative that means nothing actionable.
    ("ok", I.UNINTELLIGIBLE),
    # Help request phrased as a complaint.
    ("it says transaction failed every time", I.NEEDS_HELP),
    ("otp nahi aa raha hai", I.NEEDS_HELP),
]


def counts() -> dict[str, int]:
    out: dict[str, int] = {}
    for _, intent in EVAL_REPLIES:
        out[intent.value] = out.get(intent.value, 0) + 1
    return dict(sorted(out.items()))
