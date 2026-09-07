# Compliance rules as code

Every rule in `secondask/policy/rules.py` is a pure function that either allows
an action or refuses it with a reason and a citation. This file maps each rule
to the obligation it implements.

**This is engineering, not legal advice.** The point is that each constraint is
machine-checked, traceable to a stated source, and tested at its boundaries. It
is not that this file is a legal opinion. Anyone deploying something like this
should have the rule set reviewed by someone qualified, and several of these
would need tightening against the current text of the relevant circular.

## The rules

| rule | what it enforces | source |
|---|---|---|
| `R-RBI-HOURS` | no customer contact outside 08:00 to 19:00 IST, on any channel | RBI Fair Practices Code and recovery agent conduct norms. Covers calls, SMS and instant messaging, not only voice. |
| `R-VOICE-WINDOW` | voice calls restricted to 10:00 to 18:00 | our own control, stricter than the regulatory minimum |
| `R-HOLIDAY-WINDOW` | no automated outreach on a national holiday or major festival | internal control, aligned with RBI fair practice expectations |
| `R-COOLDOWN` | minimum 6 hours between contacts to one customer | internal control |
| `R-FREQ-24H` | at most 2 contacts per customer per 24 hours | TRAI TCCCPA preference framework, RBI conduct norms |
| `R-FREQ-7D` | at most 4 contacts per customer per 7 days | as above |
| `R-MAX-ATTEMPTS` | at most 6 attempts per item | internal control |
| `R-OPTOUT` | an opt-out is absolute and per customer, not per item | DPDP Act 2023 consent withdrawal; TRAI preference registration |
| `R-CHANNEL-EXISTS` | do not attempt a channel with no address | internal control |
| `R-DLT-CONSENT` | promotional content needs consent and DND scrubbing; transactional does not | TRAI TCCCPA |
| `R-DLT-TEMPLATE` | every message must render from a registered template | TRAI TCCCPA: commercial messages must match a DLT-registered template against a registered header |
| `R-EMANDATE-PRENOTIFY` | a pre-debit notice at least 24 hours before a recurring debit | RBI e-mandate framework |
| `R-SILENT-RETRY-RAIL` | only mandate rails may be debited without the payer present | consent boundary, see below |
| `R-STOP-SETTLED` | stop on settlement or any terminal state | internal control |
| `R-STOP-DISPUTE` | a disputed amount goes to a human, not an automated sequence | RBI Fair Practices Code |
| `R-STOP-HARDSHIP` | declared hardship suspends automated collection | RBI Fair Practices Code, forbearance |
| `R-HONOUR-PROMISE` | do not chase before a promised date | internal control |
| `R-AMOUNT-BOUND` | a money-moving action's amount must equal the ledger balance | internal control, and the security boundary |
| `R-IDEMPOTENCY` | every side-effecting action carries a key usable once | internal control |
| `R-SPEND-CAP` | a hard ceiling on recovery spend | internal control |
| `R-ESCALATE-ONCE` | an item handed to a human leaves the automated loop | internal control |
| `R-ESCALATION-CAPACITY` | human review is a fixed daily capacity | internal control |

## Five that are worth explaining

### `R-SILENT-RETRY-RAIL` is a consent boundary, not a technical detail

In India there is no standing authorisation to re-charge a one-off UPI or card
payment. A customer who abandoned a checkout has not authorised anyone to charge
them later, and RBI's tokenisation and additional-factor rules mean a saved card
cannot simply be re-run. Only a registered mandate carries that authority.

So "retry the payment" is two different actions depending on the rail. On a
mandate it is a debit. On a one-off payment it is *sending the customer a link*,
which is a contact, and must be priced and gated as one.

Systems that model retry uniformly across rails are describing an action they
are not entitled to take. The baselines in this repo deliberately attempt it,
and the denial count for this rule measures how often the naive design would
have charged somebody it had no mandate to charge.

### `R-RBI-HOURS` applies to messaging, not only calls

The intuitive reading of "recovery agents may not call outside 8 AM to 7 PM" is
that it constrains phone calls. It does not. The restriction covers the channel
of contact, and an automated SMS at 10 PM is a contact. This is the single
easiest rule for an automated system to break without anybody noticing, because
nobody's phone rang and no human made a decision.

Silent retries are exempt, correctly: no human is contacted.

The window is implemented as the half-open interval `[08:00, 19:00)`, so
18:59:59 is permitted and 19:00:00 is not. All four boundary instants are pinned
in `tests/test_policy.py`.

### `R-HOLIDAY-WINDOW` is a judgment call, not a statute

Nothing in Indian law forbids a dunning SMS on Diwali morning. It is a bad idea
anyway: it produces a complaint rather than a payment, and it is the sort of
thing that ends up in a screenshot. Expected value does not price reputational
damage, so this is a constraint rather than a cost term.

Two carve-outs. Silent retries are exempt, because nobody is contacted and a
mandate that would have succeeded should not be delayed a day for a reason the
customer will never observe. Human escalation is exempt, because a person
deciding to call is a judgment this rule has no business overriding.

The calendar has two tiers with different reliability, and they are kept apart
deliberately. The three gazetted national holidays plus Christmas are fixed
Gregorian dates, computed, and correct for any year. Lunar festivals (Holi, the
two Eids, Dussehra, Diwali, Guru Nanak Jayanti) move annually and are
**tabulated per year, currently 2025 to 2027**. There is no approximate lunar
calculation, because presenting the output of one as a compliance control would
be worse than a table with a known expiry. `/health` reports whether the current
year is covered, so a stale table surfaces rather than silently stopping matching.

Regional holidays (Pongal, Onam, Bihu, Gudi Padwa) are deliberately absent. They
would need a state mapping per customer, which this system does not have, and
guessing is worse than the visible gap.

### Dispatch staggering is a compliance control too

`R-RBI-HOURS` and `R-VOICE-WINDOW` do not defer to the start of the window. They
defer to a per-item offset spread over the first two hours of it.

The reason is operational rather than legal, and it matters more than it looks.
Returning `08:00:00` to every deferred action means eleven hours of overnight
failures fire in the same second. The SMS provider rate-limits, the gateway sees
a spike two orders of magnitude above steady state, and the compliant behaviour
has manufactured its own outage. A regulator who asked why ten thousand messages
left in one second would not be reassured that they were all inside the window.

### `R-AMOUNT-BOUND` is the load-bearing security control

Everything else in the injection defence is a layer. This is the floor.

Any action that moves money must carry an amount exactly equal to the item's
outstanding balance, which is read from the ledger. It is not supplied by the
agent, not by the planner, and not by any model. Even if every other defence
failed, and something proposed a debit for a different figure, it dies here.

It also rejects non-positive amounts, which would otherwise be a route to a
negative-value transfer through a collection channel.

## What is deliberately not implemented

Stated so the gaps are visible rather than assumed covered:

- **Actual DLT registration.** Template IDs are structurally correct
  placeholders. A real deployment registers them with a DLT operator.
- **Real DND scrubbing.** The DND flag is modelled from customer history rather
  than queried against the national preference register.
- **Recording retention.** RBI requires recovery call recordings to be retained.
  There is no voice channel here, only a simulated one.
- **Female borrower protections.** RBI requires that only female recovery agents
  contact female borrowers, within specified hours. Not modelled, because the
  world has no gender attribute and inventing one to demonstrate a rule would be
  worse than leaving the gap visible.
- **Cross-border and multi-currency.** INR only throughout.
- **Lunar festival dates beyond 2027.** The table has a known expiry and reports
  its own coverage rather than failing quietly.
- **Grievance redressal routing.** Complaints are counted, not routed.

## Where the constraint actually lives

The rules above are enforced at the gate. Two further protections are structural
rather than procedural, and are worth separating because they cannot be
misconfigured:

1. There is **no free-text send path**. A message is a template ID plus typed
   slots. Slot values are length-limited and reject control and structural
   characters, so a slot cannot forge an extra line or open another placeholder.

2. There is **no settlement intent** in the reply parser's output type. A
   customer message cannot mark anything paid, because the enum has no member
   that means that. Settlement is written only from a payment event, and that
   event must pass HMAC verification, event-id deduplication and an allow-list
   of events permitted to settle.

3. A **customer-stated amount is never an action amount**. The
   `PARTIAL_PAYMENT_PROMISE` intent carries `claimed_partial_paise`, which is
   recorded on the item as a claim and read by nothing that computes an action.
   `R-AMOUNT-BOUND` would refuse it regardless, and `test_llm_multi_intent`
   asserts that adversarially: it routes a parsed claim straight into an action
   and checks the gate refuses it.
