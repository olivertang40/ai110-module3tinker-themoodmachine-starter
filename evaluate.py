"""
evaluate.py — Part 4: Side-by-side evaluation of rule-based vs ML model.

Run with:
    python evaluate.py

Prints:
  1. Per-example comparison of both models
  2. Accuracy summary
  3. Error analysis: cases where each model fails and why
  4. Cases where models DISAGREE (most instructive)
"""

from mood_analyzer import MoodAnalyzer
from ml_experiments import train_ml_model, predict_single_text
from dataset import SAMPLE_POSTS, TRUE_LABELS

# ---------------------------------------------------------------------------
# Run both models
# ---------------------------------------------------------------------------
rule_model = MoodAnalyzer()
vectorizer, ml_model = train_ml_model(SAMPLE_POSTS, TRUE_LABELS)

rule_preds = [rule_model.predict_label(p) for p in SAMPLE_POSTS]
ml_preds   = [predict_single_text(p, vectorizer, ml_model) for p in SAMPLE_POSTS]

# ---------------------------------------------------------------------------
# Side-by-side table
# ---------------------------------------------------------------------------
print("=" * 90)
print("SIDE-BY-SIDE EVALUATION: Rule-Based vs ML Model")
print("=" * 90)
print(f"{'TRUE':8s}  {'RULE':8s}  {'ML':8s}  {'STATUS':<12s}  TEXT")
print("-" * 90)

rule_correct = 0
ml_correct   = 0
disagreements = []

for text, true, rule_pred, ml_pred in zip(SAMPLE_POSTS, TRUE_LABELS, rule_preds, ml_preds):
    r_ok = rule_pred == true
    m_ok = ml_pred   == true
    if r_ok: rule_correct += 1
    if m_ok: ml_correct   += 1

    if r_ok and m_ok:
        status = "both OK"
    elif r_ok and not m_ok:
        status = "rule OK / ML ❌"
    elif not r_ok and m_ok:
        status = "rule ❌ / ML OK"
    else:
        status = "both ❌"

    if rule_pred != ml_pred:
        disagreements.append((text, true, rule_pred, ml_pred))

    # Truncate long text for table readability
    short = (text[:55] + "…") if len(text) > 55 else text
    print(f"{true:8s}  {rule_pred:8s}  {ml_pred:8s}  {status:<14s}  {short}")

n = len(SAMPLE_POSTS)
print("-" * 90)
print(f"Accuracy:  Rule-based = {rule_correct}/{n} = {rule_correct/n:.2f}   |   "
      f"ML = {ml_correct}/{n} = {ml_correct/n:.2f}")

# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------
print("\n" + "=" * 90)
print("ERROR ANALYSIS")
print("=" * 90)

rule_failures = [(t, tr, rp) for t, tr, rp, mp in
                 zip(SAMPLE_POSTS, TRUE_LABELS, rule_preds, ml_preds) if rp != tr]
ml_failures   = [(t, tr, mp) for t, tr, rp, mp in
                 zip(SAMPLE_POSTS, TRUE_LABELS, rule_preds, ml_preds) if mp != tr]

print(f"\nRule-based failures ({len(rule_failures)}):")
for text, true, pred in rule_failures:
    print(f"  expected={true:8s}  got={pred:8s}  -> \"{text}\"")
    # Show why
    explanation = rule_model.explain(text)
    for line in explanation.splitlines():
        print(f"    {line}")

print(f"\nML model failures ({len(ml_failures)}):")
for text, true, pred in ml_failures:
    print(f"  expected={true:8s}  got={pred:8s}  -> \"{text}\"")

# ---------------------------------------------------------------------------
# Disagreements — where models diverge (most educational)
# ---------------------------------------------------------------------------
if disagreements:
    print(f"\n{'=' * 90}")
    print(f"DISAGREEMENTS — where rule-based and ML give different predictions ({len(disagreements)})")
    print("=" * 90)
    for text, true, rule_pred, ml_pred in disagreements:
        r_mark = "✅" if rule_pred == true else "❌"
        m_mark = "✅" if ml_pred   == true else "❌"
        print(f"\n  Text    : \"{text}\"")
        print(f"  True    : {true}")
        print(f"  Rule {r_mark} : {rule_pred}")
        print(f"  ML   {m_mark} : {ml_pred}")

# ---------------------------------------------------------------------------
# Key observations
# ---------------------------------------------------------------------------
print("\n" + "=" * 90)
print("KEY OBSERVATIONS")
print("=" * 90)
print("""
1. ML accuracy (0.94) >> Rule-based accuracy (0.69) on the SAME dataset.
   But this is training accuracy — the ML model was evaluated on data it
   already saw during training. This inflates its score.

2. The ML model learned word co-occurrences from labeled examples.
   "love … meetings" -> negative (from the sarcasm example it saw)
   "fine … fine … fine" -> negative (from the sarcasm example it saw)
   The rule-based model cannot learn this — it only knows individual words.

3. The rule-based model is MORE predictable and explainable.
   You can trace exactly WHY it gave a label (see explain() output above).
   The ML model is a black box — you cannot easily explain why it decided
   "This is fine" -> negative.

4. Both models are brittle on out-of-vocabulary language.
   Add a new slang term the ML model never saw and it will likely
   predict "neutral" (unknown = no signal). The rule-based model does
   the same unless you manually add the word to the signal tables.

5. Dataset size matters enormously for ML.
   With only 16 examples, the ML model is essentially memorising the
   training set. Adding diverse examples would lower training accuracy
   but improve real-world generalisation.
""")
