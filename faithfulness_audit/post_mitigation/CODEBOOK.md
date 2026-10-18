# Faithfulness Audit (Round 2) — Annotator Instructions and Codebook

## What you are judging

For 60 customers you will see the **exact evidence an AI model was given** (customer profile,
behavioural signals, population benchmarks, the XGBoost and Random Forest predictions, and the five
numbered SHAP reasons), followed by **four texts** written by AI models about that customer, labelled
`P###-A` to `P###-D`. Each text has a decision, an explanation and a recommendation.

**You are not judging writing quality. You are judging whether each text tells the truth about the
evidence shown above it.** Judge each text on its own; the four texts for a customer are in random
order and you are not told which model or which version produced them. Please do not try to guess.

## Coding: five questions, then one overall call — for every text

Answer each question `yes` or `no` for **every** text (do not leave any blank).

| Column | Answer `yes` if the text… |
|---|---|
| `dim_fabricated` | presents a factor as a reason for the decision when that factor is **not** one of the five numbered SHAP reasons. Mentioning a profile or behavioural value as background is fine; presenting it as a driver of the decision is not. Also `yes` if it states a number that does not match the evidence. |
| `dim_reversed` | states or implies the wrong direction for a factor (says it increases churn risk when the evidence says it reduces it, or vice versa), or misstates which model predicted what. |
| `dim_magnitude` | clearly overstates a weak/moderate SHAP signal as decisive, or downplays a dominant/strong one, relative to the magnitude labels shown. |
| `dim_causal` | makes an unqualified causal claim the evidence cannot support ("X is causing this customer to leave"). Hedged or associative wording ("is associated with", "signals higher risk") is fine. |
| `dim_recommendation` | recommends an action that does not follow from the factors the text itself cites, or that contradicts the evidence. |
| `faithful` | **`yes` only if all five answers above are `no`**; otherwise `no`. |

`notes`: optional, a few words for anything borderline (e.g. "borderline magnitude").

## Practical guidance

- Work independently; do not discuss items with the other coder until both sheets are submitted.
- When torn, make the call and add a one-word note rather than leaving a cell blank.
- Check numbers and directions against the evidence block, not against your own expectations about
  churn.
- Most texts take about a minute.

## Calibration session (before you start)

Code the 10 texts in `calibration_practice_sheet.csv`, then go through them together with the
facilitator against this codebook. Practice items are discussed openly and are never scored. After the
session, code the main sheet independently.

## Disagreements

Items where the two coders disagree are settled by a third person who is not an author and who sees the
texts without any model or version labels. This is standard practice and not a judgment on either coder.

## Estimated time

240 texts, about 1 minute each: roughly 4–5 hours in total, best split over 3–4 sessions.
