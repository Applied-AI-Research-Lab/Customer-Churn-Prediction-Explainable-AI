# A9 Validation Spot-Check — Annotator Instructions

A quick follow-up to the faithfulness audit you already completed — thank
you again for that work. We tested a fix for the fabrication problem you
found (a one-sentence prompt instruction telling the model only to cite the
numbered SHAP reasons, not the raw behavioral data shown elsewhere). An
automated check suggests it works well. This is a small, fast human check
to confirm that before we rely on it.

## What's different from last time

Much simpler and faster: **one question per text, not five.** This is 25
*new* customers (not the same 50 as before), each with **four** texts to
judge instead of two — still only the one question, so total effort is
similar to or less than last time.

## The question

For each of Text A / B / C / D:

`fabricated_A` / `_B` / `_C` / `_D` —

- **`yes`** — the text cites a specific factor as important/decisive that is
  **not** among the top-5 SHAP reasons shown above it (e.g. mentions cart
  abandonment, login frequency, social engagement, tenure, etc. as evidence,
  when that factor isn't in the numbered list).
- **`no`** — everything cited as evidence is among the top-5 reasons shown
  (or the text doesn't cite anything outside that list).

That's it — just this one call per text. As before, you're not told which
AI model wrote which text, or which version (original vs. fixed) it is —
judge each one independently.

## Notes column
Optional, a few words if something's ambiguous.

## Estimated time

25 customers x 4 texts = 100 quick judgments, single yes/no each. Should be
well under an hour, likely 30-45 minutes.

Same as before: work independently from the other coder until both sheets
are submitted, and don't try to guess which model/version you're reading.
