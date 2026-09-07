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
        |     THE CONSTITUTION   |   22 rules, checked exhaustively,
        |     (no LLM)           |   fails closed. RBI contact hours,
        |                        |   TRAI DLT, e-mandate 24h notice,
        |                        |   frequency caps, stopping rules,
        |                        |   amount binding, idempotency,
        |                        |   holidays, escalation capacity
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
| `policy/rules.py` | 22 rules, each with a citation | no |
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
| `async_runtime.py` | concurrent I/O, deterministic application order | no |
| `policy/holidays.py` | national and festival calendar | no |
| `execute/webhooks.py` | HMAC verification, dedupe, freshness | no |
| `execute/async_client.py` | bounded-concurrency gateway and channel I/O | no |
| `server/api.py` | ingestion service, transport-independent | no |
| `server/fastapi_app.py` | FastAPI transport (optional) | no |

## Things added in the production pass

### Dispatch is staggered, not scheduled on the hour

Every action deferred out of the forbidden contact window used to return exactly
`08:00:00`. Overnight failures accumulate for eleven hours and then fire in the
same second: the gateway sees a spike two orders of magnitude above steady state,
the SMS provider rate-limits, the breaker opens, and a compliant system has
manufactured its own outage.

`deferred_start` spreads each item across the first two hours of the window using
a stable hash of the item id. Measured uniform to within 25% across twelve
ten-minute buckets, deterministic on replay, and independent of arrival order or
queue depth, so a replayed run schedules identically.

### Webhooks are verified before they are parsed

`WebhookVerifier` takes raw bytes. The three mistakes that make a signature check
decorative are each tested:

- verifying re-serialised JSON rather than the bytes that were signed,
- comparing digests with `==` instead of `hmac.compare_digest`,
- treating a valid signature as proof of freshness, when a captured payload stays
  validly signed forever. Event ids are deduplicated; a replay returns 409.

### Online learning, and why it is opt-in

The underwriter can fold live outcomes in with a single SGD step per observation,
plus a diagonal LinUCB-style uncertainty bonus for optimism under uncertainty.
The diagonal approximation ignores correlation between features and so understates
uncertainty, which is the right direction to be wrong in for something that spends
money on exploration: it explores less than full LinUCB would, never more.

Standardisation statistics stay **frozen** at their batch values. Letting mean and
scale drift while the weights are expressed in terms of them silently rescales
every existing coefficient, which presents as a model that slowly forgets things
nobody changed.

It is off by default. Online updates mutate the model mid-run, so the benchmark
would no longer be measuring a fixed policy.

### Concurrency that cannot change the answer

`AsyncRuntime` is structured as *plan sequentially, execute concurrently, apply
in planning order*. Applying results as they complete would make state
transitions depend on network timing, and two runs of the same batch would
produce different ledgers. Verified identical at concurrency 1 and 24.

The synchronous `Runtime` remains the path every reported number comes from.
There is no real I/O in the simulation to overlap, so async there would add
risk and buy nothing.

### The claim that is not an amount

`PARTIAL_PAYMENT_PROMISE` carries `claimed_partial_paise`: a figure that arrived
inside a message written by somebody who owes money. It is recorded on the item
under a name that forces anyone reaching for it to notice what it is, and it is
never passed to `ProposedAction.amount_paise`. `R-AMOUNT-BOUND` would refuse it
anyway, and `test_llm_multi_intent` does exactly that adversarially: routes the
parsed claim straight into an action and asserts the gate refuses it.

Intent precedence is a safety ordering, not a parsing convenience. "I'll pay half
next week but stop messaging me" carries three intents and the one that governs
is the stop.

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
