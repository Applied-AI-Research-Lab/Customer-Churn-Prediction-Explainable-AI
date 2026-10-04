# Faithfulness Audit — Annotator Instructions and Codebook

Addresses R3.8 ("mentioning a feature is lexical consistency, not faithfulness")
and the human-in-the-loop half of R2.2 (annotator protocol + inter-rater
reliability).

## What you are judging

For 50 customers, two AI models (Qwen3.5-4B and Gemma3-4B) each read the same
evidence about a customer — their profile, behavioural data, and a ranked list
of the top reasons a churn-prediction model flagged them as likely to churn or
stay (the "SHAP reasons") — and each wrote a short explanation and
recommendation. **Your job is not to judge whether the explanation is
well-written. Your job is to judge whether it is telling the truth about the
evidence it was given.**

A model can write something that sounds completely reasonable while getting
the underlying evidence wrong — citing a factor that was never actually
important, reversing whether a factor increases or decreases risk, or
recommending an action that doesn't follow from anything in the evidence. This
audit exists to catch exactly that.

You will see two texts per customer, labelled **Text A** and **Text B** — you
are not told which AI model wrote which, and the assignment is randomised
per-customer, so please judge each one independently without trying to guess
which model it is.

## The coding process — two steps, but step 2 is usually skipped

**Step 1 (always do this): one holistic call per text.**

`faithful_A` / `faithful_B` — read the text against the SHAP evidence shown
above it, and answer:

- **`yes`** — the explanation and recommendation are consistent with the
  evidence: it doesn't invent factors that aren't there, doesn't reverse
  which way a factor points, doesn't wildly overstate/understate how strong
  a factor is, doesn't make unwarranted causal claims, and the
  recommendation follows from what's discussed.
- **`no`** — something above is wrong. Any one problem is enough for `no`.

For most texts this should be a quick, confident call — you're checking "does
this basically match the evidence," not proofreading every sentence. **Most
texts should take well under a minute.**

**Step 2 (only if you answered `no`): tag which kind of problem it was.**

`failure_tags_A` / `failure_tags_B` — a comma-separated list from:

| Tag | Meaning |
|---|---|
| `fabricated` | Cites a factor as important that isn't in the top-5 SHAP reasons shown |
| `reversed` | Gets the direction backwards (calls a protective factor a risk factor, or vice versa) |
| `magnitude` | Wildly overstates a weak/moderate signal as decisive, or downplays a dominant one |
| `causal` | Makes an unqualified causal claim ("X is causing this") the evidence doesn't support |
| `recommendation` | The recommended action doesn't follow from the cited evidence |

You can tag more than one if more than one thing is wrong (e.g.
`reversed,recommendation`). **If `faithful_A`/`faithful_B` = yes, leave the
tag column blank** — no need to double-check yourself.

### Notes column
Optional. A few words is plenty, for anything borderline you want flagged for
the adjudication step.

## Practical guidance

- **Work independently** — please don't discuss specific items with the other
  coder until both sheets are submitted.
- **When genuinely torn, make the call and add a one-word note** (e.g.
  "borderline") rather than leaving it blank.
- **Don't try to identify which model wrote which text** — it shouldn't
  change your judgment, and guessing can bias it.
- You don't need to fact-check the customer's real-world situation, only
  whether the text is faithful to the evidence shown in this sheet.

## What happens with disagreements

Every item both of you code will be compared. Where you disagree, the
corresponding author reviews that specific item and makes the adjudicating
call — this is standard practice and not a judgment on either coder.

## Estimated time

50 customers x 2 texts = 100 quick yes/no calls, with the tag step needed
only for the (typically minority of) texts flagged `no`. We'd estimate
**roughly 1.5-2.5 hours total**, and it's fine to split across a couple of
short sessions.
