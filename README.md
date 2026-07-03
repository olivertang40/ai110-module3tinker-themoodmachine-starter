# The Mood Machine

The Mood Machine is a simple text classifier that begins with a rule based approach and can optionally be extended with a small machine learning model. It tries to guess whether a short piece of text sounds **positive**, **negative**, **neutral**, or even **mixed** based on patterns in your data.

This lab gives you hands on experience with how basic systems work, where they break, and how different modeling choices affect fairness and accuracy. You will edit code, add data, run experiments, and write a short model card reflection.

---

## Repo Structure

```plaintext
├── dataset.py          # Expanded word lists and 16 labeled posts (slang, emoji, sarcasm)
├── mood_analyzer.py    # Rule-based classifier with negation, emoji, and slang support
├── main.py             # Runs the rule-based model and interactive demo
├── ml_experiments.py   # Tiny ML classifier using scikit-learn (Logistic Regression)
├── stress_test.py      # (New) Test suite for edge cases: sarcasm, slang, negation, emoji
├── evaluate.py         # (New) Side-by-side comparison of rule-based vs ML models
├── PART4_ANALYSIS.md   # (New) Detailed evaluation findings and failure analysis
├── model_card.md       # Template to document model behavior and limitations
└── requirements.txt    # Dependencies: scikit-learn, matplotlib, ipykernel
```

---

## Getting Started

1. Open this folder in VS Code.
2. Create and activate a virtual environment:

    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS / Linux
    source venv/bin/activate
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Run the rule-based model:

    ```bash
    python main.py
    ```

5. Run the stress test suite to see edge case failures:

    ```bash
    python stress_test.py
    ```

6. Run the side-by-side evaluation (rule-based vs ML):

    ```bash
    python evaluate.py
    ```

7. Train and evaluate the ML model:

    ```bash
    python ml_experiments.py
    ```

---

## What Was Completed

This implementation includes:

### Part 1: Dataset Construction
- ✅ Expanded `SAMPLE_POSTS` from 6 → 16 examples
- ✅ Added diverse language: slang, emoji, sarcasm, mixed feelings, negation
- ✅ All posts labeled with matching `TRUE_LABELS`

### Part 2: Rule-Based Brain
- ✅ Implemented `preprocess()` with:
  - Punctuation removal
  - Unicode emoji tokenization (😊😊😊 → ['😊', '😊', '😊'])
  - Repeated character normalization ("soooo" → "soo")
  - Text emoji preservation (":)", ":(")
- ✅ Implemented `score_text()` with:
  - Positive/negative word matching
  - Emoji signal tables (+2 / -2 for strong signals)
  - Slang signal tables ("lowkey", "brutal", "mid", "fire")
  - Negation handling with 3-token look-ahead window
  - Negation blockers for idiomatic phrases ("can't stop", "no reason", "don't even know")
- ✅ Implemented `predict_label()` with:
  - "mixed" detection when both positive AND negative signals present
  - Threshold-based classification for positive/negative/neutral
- ✅ Enhanced `explain()` to trace negation flips and matched tokens

### Part 3: Stress Testing & Model Breaking
- ✅ Created `stress_test.py` with 19 edge cases:
  - Sarcasm (4 cases)
  - Slang (4 cases)
  - Negation edge cases (4 cases)
  - Emoji-only or emoji-dominant (4 cases)
  - Ambiguous/mixed (3 cases)
- ✅ Fixed negation blocker bug: "can't stop smiling" now correctly positive
- ✅ Documented failure patterns and fundamental limitations

### Part 4: System Evaluation
- ✅ Created `evaluate.py` for side-by-side rule-based vs ML comparison
- ✅ Trained ML model (Logistic Regression + CountVectorizer)
- ✅ Documented accuracy results:
  - Rule-based: **12/16 = 0.75**
  - ML (training accuracy): **15/16 = 0.94**
- ✅ Analyzed 6 disagreement cases between models
- ✅ Created `PART4_ANALYSIS.md` with detailed findings:
  - Why rule-based fails (sarcasm, idioms, context)
  - Why ML wins (learns patterns) and loses (memorization, black box)
  - Dataset size impact on each approach

### Part 5: Model Card (Next)
- ⏳ To be completed in `model_card.md`

---

## Current Performance

| Model Type | Accuracy | Details |
|------------|----------|---------|
| **Rule-Based** | 0.75 (12/16) | Fully explainable, handles negation and slang, fails on sarcasm and idioms |
| **ML (Logistic Regression)** | 0.94 (15/16) | Learns patterns from examples, black box, overfitting on small dataset |
| **Stress Test** | 11/19 pass | Exposes sarcasm (0/4), slang edge cases, negation idioms |

---

## Key Learnings

1. **Rule-based systems are transparent but brittle**
   - Every decision is traceable via `explain()`
   - Cannot handle sarcasm, context, or world knowledge
   - New language requires manual rule updates

2. **ML systems learn patterns but require data**
   - Learned sarcasm from labeled examples ("I absolutely love sitting in 2-hour meetings")
   - With only 16 examples, it's memorizing not generalizing
   - Black box — cannot explain WHY a prediction was made

3. **No model is perfect**
   - Rule-based: deterministic, predictable failures
   - ML: probabilistic, unpredictable on unseen data
   - Hybrid approaches may be the best real-world solution

4. **Evaluation reveals hidden bugs**
   - "not bad, not great" incorrectly flagged as mixed (double negation artifact)
   - "don't even know" incorrectly flipped emoji (idiomatic phrase not blocked)
   - Testing is the only way to discover these issues

---

## What You Will Do

During this lab you will:

- ✅ Implement the missing parts of the rule-based `MoodAnalyzer`.
- ✅ Add new positive and negative words.
- ✅ Expand the dataset with more posts, including slang, emojis, sarcasm, or mixed emotions.
- ✅ Observe unusual or incorrect predictions and think about why they happen.
- ✅ Train a tiny machine learning model and compare its behavior to your rule-based system.
- ⏳ Complete the model card with your findings about data, behavior, limitations, and improvements.

The goal is to help you reason about how models behave, how data shapes them, and why even small design choices matter.

---

## Tips

- Start with preprocessing before updating scoring rules. ✅ Done
- When debugging, print tokens, scores, or intermediate choices. ✅ Use `explain()`
- Ask an AI assistant to help create edge case posts or unusual wording. ✅ Used for brainstorming
- Try examples that mislead or confuse your model. Failure cases teach you the most. ✅ See `stress_test.py`
- Use `evaluate.py` to compare rule-based and ML models side-by-side
- Read `PART4_ANALYSIS.md` for detailed reasoning about failures and tradeoffs
