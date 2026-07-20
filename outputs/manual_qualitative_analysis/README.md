# Manual Qualitative Analysis — Proof of Work

This folder contains the **raw, unedited console output** generated while performing
the manual/qualitative content-analysis checks referenced in the paper
(`Paper/template.tex`, Section 3.5.3 "Qualitative Content Analysis") and reported in
Tables 14–15 of the paper draft (explanation quality patterns and recommendation
quality patterns).

These files serve as **verifiable evidence** that the 200-sample-per-model manual
inspection, the 10-full-transcript manual reading, and the 5 disagreement-case manual
reading described in the paper were actually executed against the real model outputs
in `outputs/llm_predictions/test_llm_predictions.csv`, using a fixed random seed
(`random_state=42`) for reproducibility.

## Files

| File | Content | Corresponds to |
|---|---|---|
| `01_explanations_200sample_analysis.txt` | Automated + manually-coded analysis of 200 randomly sampled `qwen_explanation` / `gemma_explanation` texts per model (400 total). Covers structural patterns, hedging language, SHAP/benchmark citation rates, reasoning-depth indicators, content themes (feature citation frequency), and 5 representative full-text samples per model. | Table 14 (`tab:qual_explanations`) in `Paper/template.tex`; Section 3.5.3 methodology |
| `02_recommendations_200sample_analysis.txt` | Same 200-sample-per-model methodology applied to `qwen_recommendation` / `gemma_recommendation` texts. Covers action verbs, urgency markers, intervention types, channels, conditional language, churn-vs-retain differentiation, and 5 representative full-text samples per model. | Table 15 (`tab:qual_recommendations`) in `Paper/template.tex`; Section 3.5.3 methodology |
| `03_manual_inspection_10_transcripts_plus_disagreements.txt` | Full, unabridged manual reading of 10 complete explanation+recommendation transcripts per model (5 churn-decision, 5 retain-decision cases) plus 5 head-to-head Qwen-vs-Gemma disagreement cases, each showing both models' full explanation and recommendation side by side. This is the deepest level of manual verification, used to validate the automated counts in the other two files and to surface the qualitative "narrative arc" vs "sequential listing" and "benefit of the doubt" findings discussed in Sections 4.9–4.10 and the Discussion of the paper. | "Manual inspection" paragraphs in Tables 14 and 15 discussion; Discussion Section (research question 3) |

## Methodology (reproducible)

1. Load `outputs/llm_predictions/test_llm_predictions.csv` (7,501 rows, both LLM
   decisions/explanations/recommendations already generated).
2. `df.sample(200, random_state=42)` — draw one fixed, reproducible 200-row sample
   per model (same seed for both Qwen and Gemma sampling calls, applied to the same
   underlying test set).
3. Run keyword/regex-based automated counters (hedging language, SHAP magnitude
   citations, benchmark references, action verbs, etc.) over the sampled rows.
4. Manually read a diverse sub-selection (5 churn + 5 retain per model from the
   10-full-transcript file, plus 5 head-to-head disagreement cases) to validate that
   the automated counts reflect genuine patterns in the text, not keyword-matching
   artefacts.
5. All percentages and counts reported in the paper's Tables 14 and 15 are computed
   directly from this sampling procedure; the numbers in this folder are the original
   console output from which those tables were transcribed.

## Data provenance

- Source data: `outputs/llm_predictions/test_llm_predictions.csv`
- Models: Qwen3.5-4B (`unsloth/Qwen3.5-4B`) and Gemma3-4B-IT (`unsloth/gemma-3-4b-it`),
  both run in zero-shot mode as described in `Paper/template.tex` Section 3.4.
- Random seed: 42 (fixed, reproducible)
- Sample size: 200 explanations + 200 recommendations per model (400 + 400 = 800
  generated texts analysed manually/semi-automatically in total)
- Deep-read subsample: 10 full transcripts per model + 5 disagreement cases (30
  complete explanation/recommendation pairs read in full by the authors)
