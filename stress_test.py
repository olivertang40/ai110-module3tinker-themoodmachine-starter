"""
stress_test.py — Part 3: Stress-test the rule-based MoodAnalyzer.

Run with:
    python stress_test.py

Each "breaker" sentence is designed to expose a specific failure mode.
The script prints:
  - the sentence
  - the model's prediction vs. the expected label
  - the explain() breakdown so we can see exactly WHY it went wrong
"""

from mood_analyzer import MoodAnalyzer

analyzer = MoodAnalyzer()

# ---------------------------------------------------------------------------
# Breaker sentences grouped by failure type
# Format: (sentence, expected_label, failure_category)
# ---------------------------------------------------------------------------
BREAKERS = [
    # --- 1. Sarcasm: positive words, genuinely negative intent -----------
    ("I absolutely love sitting in 2-hour meetings",    "negative", "sarcasm"),
    ("Oh great, another Monday",                        "negative", "sarcasm"),
    ("wow can't wait to do this homework all weekend",  "negative", "sarcasm"),
    ("so fun getting rained on with no umbrella",       "negative", "sarcasm"),

    # --- 2. Slang not in word list ----------------------------------------
    ("this movie is totally mid",                       "negative", "slang"),
    ("that concert was straight fire 🔥",               "positive", "slang"),
    ("I'm dead 💀 that was so funny",                   "positive", "slang"),
    ("no shot I'm staying up all night for this",       "negative", "slang"),

    # --- 3. Negation edge cases -------------------------------------------
    ("I am not sad at all",                             "positive", "negation"),
    ("nothing about today was bad",                     "positive", "negation"),
    ("I never feel stressed about exams",               "positive", "negation"),
    ("not gonna lie I actually had fun",                "positive", "negation"),

    # --- 4. Emoji-only or emoji-dominant signals --------------------------
    ("😊😊😊",                                          "positive", "emoji"),
    ("😢😢",                                            "negative", "emoji"),
    ("I'm fine 🙂",                                     "negative", "emoji"),   # sarcasm emoji
    ("today was 💀",                                    "negative", "emoji"),

    # --- 5. Ambiguous / mixed ---------------------------------------------
    ("I'm exhausted but honestly grateful",             "mixed",    "mixed"),
    ("crying laughing at how bad this went 😂",         "mixed",    "mixed"),
    ("it hurts but I know it'll be worth it",           "mixed",    "mixed"),
]

# ---------------------------------------------------------------------------
# Run the tests
# ---------------------------------------------------------------------------
print("=" * 70)
print("STRESS TEST — Rule-Based MoodAnalyzer")
print("=" * 70)

failures = []
passes   = []

for sentence, expected, category in BREAKERS:
    predicted = analyzer.predict_label(sentence)
    ok = predicted == expected
    status = "✅ PASS" if ok else "❌ FAIL"

    print(f"\n[{category.upper()}] {status}")
    print(f"  Text     : {sentence}")
    print(f"  Expected : {expected}   |   Predicted : {predicted}")
    print(f"  Debug    :")
    for line in analyzer.explain(sentence).splitlines():
        print(f"    {line}")

    if ok:
        passes.append(sentence)
    else:
        failures.append((sentence, expected, predicted, category))

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print(f"RESULTS: {len(passes)} passed / {len(BREAKERS)} total")
print("=" * 70)

if failures:
    print("\n📋 Failures by category:")
    by_cat: dict = {}
    for sent, exp, pred, cat in failures:
        by_cat.setdefault(cat, []).append((sent, exp, pred))
    for cat, items in by_cat.items():
        print(f"\n  [{cat}]")
        for sent, exp, pred in items:
            print(f"    expected={exp:8s}  got={pred:8s}  → \"{sent}\"")

print("\n💡 Key failure patterns to fix:")
print("  1. Sarcasm   — model trusts 'love/fun/great' at face value")
print("  2. Slang     — 'mid', 'fire', 'dead' not in any word list")
print("  3. Negation  — double negation ('not sad') sometimes misfires")
print("  4. Emoji     — 🙂 used sarcastically still scores neutral")
