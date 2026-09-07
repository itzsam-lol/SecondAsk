# How the numbers are measured

The hardest part of this project was not the agent. It was making "money
recovered" mean something.

## The problem

In production you cannot measure recovery honestly without a holdout. You see
the branch you took and never the branch you didn't, so "we recovered 19%" has
no denominator. Recovered compared to what? Some of those customers would have
paid anyway. Some of the ones you gave up on would have paid if you had waited
two days.

Running a real holdout means deliberately not chasing a slice of real money for
several weeks, which no merchant will agree to for a student project. So the
measurement has to come from somewhere else.

## The approach

A simulator where the counterfactual is known by construction.

Every item carries a latent `Blocker`: the true reason the money did not move.
The world knows it. The agent never sees it. Given the blocker, the simulator
can answer the question production cannot: *if this action were taken at this
time, would the money arrive?*

The seven blockers respond to genuinely different interventions, which is what
makes this a decision problem rather than a scheduling one:

| blocker | what actually works | what is wasted |
|---|---|---|
| `TRANSIENT_INFRA` | a free silent retry once the outage closes | messaging the customer about somebody else's outage |
| `LIQUIDITY` | waiting for the salary credit | every attempt before payday |
| `INSTRUMENT_DEAD` | asking for a new instrument | every retry, at probability exactly zero |
| `AUTH_FRICTION` | a fast nudge while intent is warm | anything slow |
| `INTENT_LOST` | an incentive, early | repetition |
| `DISPUTE` | a human | automated contact, which produces complaints |
| `UNREACHABLE` | nothing | everything |

## What keeps it honest

**The agent cannot see latent state.** Features are built from an explicit
allow-list of observable fields. `tests/test_integrity.py::NoLatentLeakageTest`
proves it destructively: it builds a world, computes every feature vector,
corrupts every latent field on every item and customer, recomputes, and asserts
the vectors are byte-identical. A simulator-trained model that peeks produces
beautiful numbers and is worth nothing, and the failure is invisible unless
something checks for it.

**Observables are deliberately aliased.** Blockers emit error signatures
probabilistically, and the distributions overlap. `payment_timed_out` comes from
both infrastructure and authentication failures. Six percent of dead instruments
report `insufficient_funds`. So no single field identifies a cause, and the
model has to combine the error signature with method, downtime and calendar
context to beat the base rate.

**Common random numbers.** Outcome draws are keyed on
`(seed, item, action_kind, hour_bucket)` rather than drawn from a sequential
stream. Two agents taking the same action on the same item in the same hour get
the same coin. Differences between agents are attributable to decisions rather
than to one of them having consumed the stream differently. This is why the
deltas are readable at n=1000 instead of needing tens of thousands of
replications.

CRN shares the *draw*, not the *probability*. Two agents can take the same
action at the same time and still get different outcomes, because their prior
behaviour left the customer in a different state. The coin is shared; the bias
is not.

**Retries share one draw; contacts do not.** A retry asks a question about an
account: is there money, is the mandate alive, is the issuer up. That has one
answer at a given moment, so every retry taken while the account is in the same
state gets the same answer. Modelling retries as independent coins was an early
bug and it made persistence pay: eight retries at p=0.7 succeed almost surely,
so the agent learned to hammer rather than to time. Contacts stay independent,
damped by fatigue, because people genuinely do ignore the first message and act
on the second.

**Training and evaluation seeds are disjoint, and it is enforced.**
`assert_disjoint` raises rather than warns. The underwriter is fitted on seeds
101 to 106; every reported figure comes from 7, 11 and 13. Calibration is
measured on 301 and 302.

**The underwriter is never handed the simulator's parameters.** It is fitted on
outcomes from a random exploration policy, exactly as it would be in production
from historical attempts. It learns that retrying a dead card fails by retrying
dead cards and observing failures.

## Why exploration data

A good policy only takes actions it already believes will work. It would never
retry a dead card, so it would never learn that retrying a dead card fails, so
it would have no basis for that belief in the first place. Standard off-policy
bootstrap problem, standard answer: explore.

Training data comes from a random policy taking legal actions at random times
across the horizon. The distribution is deliberately wide and deliberately
stupid. It exists to cover the action-by-timing space, not to collect money.

Two constraints are respected during exploration, because violating them would
put samples in the training set that no deployed policy could generate: rail
legality, and channel availability. The regulatory constraints are deliberately
*not* applied. Whether an action would have been permitted at 3 AM is a policy
question; whether it would have worked is a physical one, and the model
estimates the second. Confusing them would leave the model unable to price a
legal 8 AM action because its only evidence came from illegal 3 AM ones.

## Calibration, not just discrimination

The underwriter multiplies its probability by a rupee amount and compares the
result against the cost of a message and against other items competing for the
same contact budget. A model that ranks perfectly but reports 0.9 whenever it
means 0.3 would order actions correctly and price every one of them at triple
its worth.

So calibration is reported, on held-out seeds, with reliability bins:

```
python -m secondask calibrate
```

## Reading the results table

Two rules, both learned by getting them wrong first.

**Never quote the value-weighted rate alone.** An early version of this
benchmark reported only percent of rupees recovered. Invoices were 7% of items
and 69% of value, so the headline was decided by whether about twenty items
happened to land, and two agents that differed by nothing meaningful looked like
they differed by a factor of three. Every table prints value and item rates side
by side, plus a per-method breakdown.

**Never quote a recovery figure without its violation count.** An agent that
ignores the contact window and the capacity limits recovers more. That number is
not a result. Runs that are not legitimate results are marked `!` in the table
and named underneath.

## What the reference numbers are

`python -m secondask world --bound` reports the best single action per item,
maximised over every action and every candidate time against a customer with no
accumulated annoyance.

This is an upper bound on **one attempt per item**. It is not a bound on the
problem: agents get several attempts, and independent contact draws compound. It
ignores contact caps, human review capacity, the goodwill budget and the fact
that several items share a customer. Exceeding it is possible and is not a bug.

What it is good for is scale. It says how much of the money in a batch is
reachable at all, as opposed to structurally lost to dead instruments,
unreachable customers and liquidity that never arrives.

`oracle_greedy` reads latent state and plays greedily. It was originally
included as a ceiling and turned out not to be one: under the same contact caps
it recovers less than SecondAsk does. That is a real result about greedy play,
so it is reported under an honest name rather than quietly dropped.

## Known limitations

Stated plainly, because a methodology section that lists none is not describing
a real system.

1. **The world is synthetic.** Parameters are anchored to published aggregates
   and were fixed before any agent existed, but they are not fitted to a real
   failure log. The mechanisms are right; the exact magnitudes are not
   independently validated.

2. **`b0_do_nothing` recovers zero.** There is no organic self-recovery path, so
   "recovered" means agent-attributable recovery. This is stated rather than
   modelled because adding a uniform organic rate would shift every agent
   equally without changing any comparison.

3. **The goodwill price is a judgment.** 4,000 paise per unit of annoyance is
   derived in `planner.py` from opt-out and churn probabilities against a
   conservative customer contribution, and it is an estimate. The sensitivity is
   reported rather than hidden: `python -m secondask sweep` runs the whole curve.

4. **Human escalation capacity is assumed**, at one ops person per 150 open
   items working about 25 cases a day. It matters a lot: without a cap, expected
   value planning correctly concludes that everything should go to a person.

5. **Single-batch horizon.** Items that would recover on day 30 are counted as
   losses at day 21. Every agent is charged this equally.

6. **The language model path is not exercised by default.** Without an API key
   the deterministic parser runs. Numbers produced with a real model are labelled
   as such; the reproducible defaults are not.
