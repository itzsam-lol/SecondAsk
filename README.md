# SecondAsk

A revenue recovery agent that prices every retry and every reminder before making
it, instead of running a fixed schedule.

Built for the Razorpay AI Buildathon, Track 03 (AI Revenue Recovery).

## The idea

When a payment fails, most systems run the same sequence at everyone: retry at
+1h, +24h, +72h, send a reminder alongside each. That sequence is wrong in both
directions at once. It retries cards that expired three months ago, where the
probability of success is exactly zero no matter how many times you try. And it
gives up on customers whose salary lands on the 1st, where waiting two days would
have worked.

SecondAsk treats a contact as a priced action rather than a scheduled one. For
each failure it asks: what is actually blocking this money, what would unblock
it, what is that attempt worth, and am I allowed to make it right now.

## Status

Work in progress. See ARCHITECTURE.md once it lands.

## Running it

Python 3.10+. No third party dependencies for the core.

```
python -m secondask --help
```
