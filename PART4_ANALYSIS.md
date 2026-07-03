# Part 4: System Evaluation — Analysis & Findings

## Quick Facts

| Model | Accuracy | Correct | Total |
|-------|----------|---------|-------|
| Rule-based | 0.75 | 12/16 | 16 |
| ML (Logistic Regression) | 0.94 | 15/16 | 16 |

---

## 1. Which examples does my model get right, and which does it get wrong?

### ✅ Rule-based successes (12/16)
The rule-based model correctly handles:
- **Clear sentiment**: `"I love this class so much"` → positive
- **Simple negation**: `"I am not happy about this"` → negative
- **Emoji + word combos**: `"just got promoted omg I'm so happy 😭😭"` → positive
- **Mixed signals**: `"Feeling tired but kind of hopeful"` → mixed
- **Slang we added**: `"lowkey obsessed with this song rn 🎶"` → positive

### ❌ Rule-based failures (4/16)

| Sentence | Expected | Predicted | Why it failed |
|----------|----------|-----------|---------------|
| `"I'm fine. totally fine. everything is fine 🙂"` | negative | neutral | **Sarcasm** — "fine" isn't in our dictionaries, emoji 🙂 not in negative signals. The model cannot detect sarcasm without understanding context. |
| `"not bad, not great, just kinda meh"` | neutral | mixed | **Double negation creates false signals** — "not bad" → flips to +1, "not great" → flips to -1, triggering "mixed" when there's actually no real sentiment. |
| `"I absolutely love sitting in 2-hour meetings"` | negative | positive | **Pure sarcasm** — the word "love" is positive in isolation. Rule systems cannot detect irony. This is a fundamental limitation. |
| `"everyone left and honestly? relieved"` | mixed | positive | **Missing context** — "relieved" is positive, but the sentence implies both relief AND loneliness (everyone left). The model doesn't see "left" as negative. |

---

## 2. Why do some failures happen?

### Fundamental Limits of Rule-Based Systems

1. **Sarcasm requires world knowledge**
   - `"I absolutely love sitting in 2-hour meetings"` uses positive words to express negative sentiment
   - Detecting this requires knowing that 2-hour meetings are unpleasant — information not in word lists
   - Rule systems only see individual words, not their relationship to reality

2. **Context-dependent words**
   - `"fine"` can mean "good" (positive) or "I'm not actually fine" (sarcasm/negative)
   - `"relieved"` is positive, but `"everyone left"` adds loneliness
   - Rules cannot track multi-sentence context or tone changes

3. **Idiomatic phrases create false signals**
   - `"not bad"` is idiomatic for "okay/acceptable" (neutral), but our negation logic flips "bad" into +1
   - We can't hardcode all idioms — language is too rich

4. **The vocabulary problem**
   - Every new slang term requires manual addition
   - Emojis have multiple meanings (😭 = sad OR happy-crying)
   - Regional/cultural variations break rules

---

## 3. How do rule-based and ML systems perform differently on the same dataset?

### ML Model Advantages (0.94 accuracy)

| Capability | Example | Why ML wins |
|------------|---------|-------------|
| **Learns sarcasm from examples** | `"I absolutely love sitting in 2-hour meetings"` → negative | Saw this exact pattern in training, learned the co-occurrence of "love" + "meetings" + "hours" → negative |
| **Learns idioms** | `"I'm fine. totally fine."` → negative | Repetition of "fine" appeared in training data labeled as sarcasm |
| **Word combinations** | `"This is fine"` → neutral in isolation, but ML learned the phrase from context | Pattern matching on token sequences, not just individual words |

### ML Model Disadvantages

| Problem | Why it matters |
|---------|----------------|
| **Memorization** | With only 16 training examples, it's memorizing, not generalizing. Add a new sentence and it will likely fail. |
| **Black box** | Cannot explain WHY it chose a label. Rule-based model shows exact logic with `explain()`. |
| **Requires labeled data** | Need hundreds of examples for real generalization. Rule-based needs 0 labeled data. |
| **Data bias** | If training data has more sarcastic "This is fine" than genuine ones, it will always predict negative. |

### Where they DISAGREE (6 cases)

These disagreements are the **most educational** — they reveal each model's blind spots:

1. **`"This is fine"`** — Rule: neutral ✅ | ML: negative ❌
   - ML saw sarcastic examples and now over-predicts negativity on neutral phrases
   - Rule model correctly gives neutral for lack of sentiment words

2. **`"I absolutely love sitting in 2-hour meetings"`** — Rule: positive ❌ | ML: negative ✅
   - ML learned the sarcasm pattern from training data
   - Rule model cannot detect sarcasm at all

3. **`"this is so overwhelming I don't even know 😰"`** — Rule: neutral ❌ (before fix) → negative ✅ (after fix) | ML: negative ✅
   - Before adding "know" to blockers, "don't" flipped the emoji
   - ML correctly learned "overwhelming" is strongly negative

---

## 4. Dataset size matters

### Experiment: What happens when we add MORE data?

If we added 100 more labeled posts:
- **Rule-based accuracy would DROP** — more edge cases = more failures unless we add more rules
- **ML accuracy would IMPROVE (on unseen data)** — more examples = better generalization

Current ML accuracy (0.94) is inflated because we're testing on the training set.
With more data and a proper train/test split, ML would likely drop to 0.70-0.80 initially, then improve as we add more examples.

Rule-based accuracy is **deterministic** — it won't improve unless we manually add rules.

---

## 5. Key Takeaways (Answer to Checkpoint)

**What I learned from evaluation:**

1. **Rule-based models are transparent but brittle**
   - I can trace every decision with `explain()`
   - But they fail on sarcasm, idioms, and context — limitations I cannot fix with more rules

2. **ML models are powerful but require data**
   - They learn patterns rule systems cannot encode (sarcasm, co-occurrences)
   - But with 16 examples, they're memorizing, not learning
   - They're black boxes — harder to debug when wrong

3. **No model is perfect**
   - Rule-based: 0.75 accuracy, fully explainable
   - ML: 0.94 accuracy, unexplainable, likely overfitting

4. **The right choice depends on your needs**
   - Need transparency? Use rules
   - Need accuracy at scale? Use ML
   - Need both? Combine them (hybrid)

5. **Evaluation reveals hidden assumptions**
   - I thought "not bad" would work with negation — it doesn't (idiom)
   - I thought 😰 was clearly negative — but "don't" flipped it
   - Testing is the only way to find these bugs

---

## Next Step: Model Card (Part 5)

Now that I understand both models' strengths and weaknesses, I can document them formally in a model card for users.
