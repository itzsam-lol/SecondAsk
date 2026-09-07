# Architecture

## The shape of it

```
   payment.failed          downtime feed          calendar
   webhook                 (bank outages)         (salary cycle)
        |                        |                     |
        +------------+-----------+---------------------+
                     |
                     v
        +------------------------+
        |     THE UNDERWRITER    |   one calibrated logistic regression
        |     (no LLM)           |   per action kind. answers:
        |                        |   p(recover | action, time, observables)
        +------------------------+
                     |
                     v
        +------------------------+
        |     THE PLANNER        |   prices every (action, time) pair:
        |     (no LLM)           |   EV = p x outstanding x discount
        |                        |        - channel cost - goodwill
        |                        |   picks the max, or STOPS if none clears 0
        +------------------------+
                     |
                     v  proposes
        +------------------------+
        |     THE CONSTITUTION   |   19 rules, checked exhaustively,
        |     (no LLM)           |   fails closed. RBI contact hours,
        |                        |   TRAI DLT, e-mandate 24h notice,
        |                        |   frequency caps, stopping rules,
        |                        |   amount binding, idempotency
        +------------------------+
                     |
              allowed|denied -> deferred to the earliest legal time,
                     |          or the agent proposes something else
                     v
        +------------------------+
        |     THE EXECUTOR       |   Razorpay test-mode payment link,
        |                        |   then deliver, then record
        +------------------------+
                     |
                     v
        +------------------------+
        |     THE LEDGER         |   append-only, hash-chained.
        |                        |   one 64-char commitment per run
        +------------------------+

        the LLM sits off to the side and touches three things only:
          - parse an inbound reply into a closed enum
          - fill declared slots inside a registered template
          - narrate a failure cluster for a human to read
        it proposes. it never executes.
```

## Why the model is the smallest part

The judging criterion that shaped this design is "the right tool in the right
place, and where you chose not to use one".

A payment recovery agent has three kinds of work in it.

**Arithmetic**, like expected value and budget allocation. A language model is
worse at this than a calculator, non-deterministic, and unauditable. It is a
linear model and a sort.

**Rules**, like contact hours, consent, and the pre-debit notice. These must be
*enforced*, not *requested*. A prompt that says "never contact outside 8 AM to
7 PM" is a preference. A precondition that refuses the action is a guarantee.
The difference shows up in the results table as the gap between `secondask`
(0 violations) and `secondask_no_policy` (thousands).

**Language**, like reading "salary aane ke baad kar dunga, 3 tarikh tak" and
extracting a promise for the 3rd. No regex holds here. The input is Hinglish,
transliterated Hindi, typos and emoji. This is where the model earns its place,
and it is a small fraction of the system.

## The security boundary

The agent reads free text written by people who owe money and do not want to
pay. Some of them will work out that a model is reading it.

The defence is structural rather than behavioural:

- `ReplyIntent` is a closed enum with **no settlement member**. There is no
  `MARK_AS_PAID`. `ALREADY_PAID` records a *claim* and pauses contact; it does
  not change money state.
- Settlement is written only from a payment event, never from a message.
- `R-AMOUNT-BOUND` ties every money-moving action to the ledger balance, so an
  amount cannot come from anywhere else.
- Model output is schema-validated. Unknown keys are not read, so an injected
  `"write_off": true` is simply invisible.
- Messages are DLT templates with typed slots. There is no free-text send path.
- PII is redacted before anything crosses the model boundary, in the gateway
  rather than at each call site, because a control each caller must remember is
  a control that will be forgotten.

`python -m secondask injection` runs 15 adversarial inputs through the real
parser. The pass criterion is not "the model resisted". It is that no input can
produce anything outside the enum, and nothing in the enum can move money.

## Module map

| module | responsibility | LLM? |
|---|---|---|
| `money.py` | integer paise, Indian formatting | no |
| `clock.py` | virtual clock, fixed +05:30 IST | no |
| `rng.py` | seeded streams and common random numbers | no |
| `ledger.py` | append-only hash chain | no |
| `world/entities.py` | domain types, observable/latent split | no |
| `world/generator.py` | batch generation, blocker priors, aliasing | no |
| `world/outcomes.py` | the counterfactual response model | no |
| `world/downtime.py` | bank outage feed | no |
| `policy/engine.py` | exhaustive, fail-closed gate | no |
| `policy/rules.py` | 19 rules, each with a citation | no |
| `policy/templates.py` | DLT templates with typed slots | no |
| `underwrite/features.py` | observable features only | no |
| `underwrite/logreg.py` | logistic regression, pure Python | no |
| `underwrite/model.py` | per-action models, seed discipline | no |
| `underwrite/planner.py` | expected value, goodwill pricing, stopping | no |
| `underwrite/calibration.py` | Brier, ECE, reliability, AUC | no |
| `llm/gateway.py` | the model boundary, three tasks | **yes** |
| `llm/redact.py` | PII redaction | no |
| `execute/executor.py` | perform the action | no |
| `execute/razorpay_client.py` | test-mode API, idempotency | no |
| `execute/circuit.py` | breaker and backoff | no |
| `agents/secondask.py` | sequencing, mandate notices, escalation | no |
| `runtime.py` | the event loop, state application | no |

## Design decisions worth arguing with

**Integer paise everywhere.** No float touches money. A float rupee amount
accumulates representation error across the tens of thousands of operations an
evaluation performs, and a recovery agent whose reported total drifts from its
ledger is worthless.

**Fixed `+05:30` rather than `zoneinfo`.** India has observed no daylight saving
since 1945, so a fixed offset is exactly correct for every timestamp this system
will see, and it removes a dependency on system tzdata that is absent on a stock
Windows install.

**Per-action models rather than one model with an action feature.** The effects
that matter are almost entirely interactions between action and state. A retry
is worth zero on a dead card and 0.78 after an outage closes. A linear model
with an action one-hot cannot express that at all.

**A separate model per action also means "no evidence" is expressible.** An
action the exploration policy never sampled returns exactly 0.0, not a small
positive. This mattered: an earlier version floored it at 0.0005, which on a
one lakh rupee item is fifty rupees of expected value for a free action, so the
planner kept proposing silent retries on one-off rails and burning its proposal
budget on actions the gate would always refuse.

**The policy engine evaluates every rule rather than short-circuiting.** The
audit trail should say every reason an action was refused, not the first one hit.

**Deferral, not just refusal.** A rule that fails on timing returns the earliest
instant the action would be permitted. That turns an 8 PM contact attempt into
an 8 AM one instead of a dropped one, which is the difference between a
compliant system and a compliant system that also collects money.

**A refused action does not kill the item.** The agent is told why and gets to
propose something else, up to four proposals per visit. Without this, an agent
that opened with one illegal action would be terminated on the spot while an
agent that opened legally ran to completion, and the comparison table would be
measuring opening moves rather than policies.

**Human escalation is capacity-constrained, not just priced.** Escalation is
genuinely the most effective action for a disputed or high-value case, so a
planner that prices it without a capacity limit chooses it for the entire batch.
An early version produced 630 escalations for 300 items. Pricing scarce capacity
as though it were elastic is how an optimiser produces a plan nobody can execute.

**The goodwill budget scales with the amount, and is hard-capped.** A flat
budget is wrong in both directions: too tolerant over a 200 rupee wallet top-up,
far too quick to give up on a 1.5 lakh receivable. The cap is an ethical line
rather than an economic one. Past a point the answer to an unresponsive customer
is a human conversation or a write-off, not more messages, however large the
invoice.
