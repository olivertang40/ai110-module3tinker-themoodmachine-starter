# Model Card: Mood Machine

This model card documents the Mood Machine project, which implements and compares **two** versions of a short-text mood classifier:

1. A **rule-based model** in `mood_analyzer.py`
2. A **machine learning model** in `ml_experiments.py` (Logistic Regression)

Both were built, stress-tested, and evaluated on the same labeled dataset. This document covers how each works, what data shaped them, where they succeed, where they fail, and why those failures matter.

---

## 1. Model Overview

**Model type:**
Both models were built and compared.
- Rule-based: hand-crafted scoring logic with word lists, negation handling, and emoji signals.
- ML: Logistic Regression trained on bag-of-words features extracted from the same labeled posts.

**Intended purpose:**
Classify short text messages (social media-style posts, ~5–20 words) into one of four mood labels:
- `positive` — the overall tone is upbeat, happy, or enthusiastic
- `negative` — the overall tone is sad, stressed, angry, or distressed
- `neutral` — no clear emotional signal
- `mixed` — contains both positive and negative signals simultaneously

**How it works (brief):**

*Rule-based:*
Each post is preprocessed into tokens, then scored by matching tokens against curated word lists and signal tables. Negation words ("not", "can't", "never") flip the sign of nearby sentiment words within a 3-token window, with blocker words absorbing negation when it would incorrectly propagate (e.g., "can't stop smiling"). The final integer score maps to a label. If both positive AND negative tokens are present, the label is "mixed" regardless of the total score.

*ML model:*
Texts are converted into bag-of-words vectors using `CountVectorizer`. A `LogisticRegression` classifier is trained on the resulting vectors and their true labels. The model learns which word patterns correlate with which labels purely from the training data — no rules are manually written.

---

## 2. Data

**Dataset description:**
`SAMPLE_POSTS` contains **16 labeled posts** total:
- 6 original starter posts (simple, clear-sentiment examples)
- 10 new posts added during Part 1

**Labeling process:**
Each post was assigned a label based on the intended real-world meaning of the sentence, not just its surface words. This means:
- `"I absolutely love sitting in 2-hour meetings"` → labeled `negative` (sarcasm, despite "love")
- `"just got promoted omg I'm so happy 😭😭"` → labeled `positive` (😭 = happy-crying here)
- `"everyone left and honestly? relieved"` → labeled `mixed` (relief + implied loneliness)

Some posts were genuinely hard to label and would likely produce disagreement among human raters:

| Post | Why it's ambiguous |
|------|-------------------|
| `"I'm fine. totally fine. everything is fine 🙂"` | Sarcasm vs. genuine reassurance — depends on context and speaker |
| `"everyone left and honestly? relieved"` | Could be positive (finally alone) or negative (abandoned) |
| `"not bad, not great, just kinda meh"` | Double negation + slang; "neutral" or "slightly negative"? |
| `"feeling 💀 after that exam but proud I finished"` | Pain + pride — "mixed" is defensible but "negative" would also be reasonable |

**Important characteristics:**
- Contains **slang**: "lowkey", "no cap", "meh", "mid", "omg", "rn"
- Contains **emoji signals**: 🎶 😭 🙂 😁 💀 😰 🔥
- Contains **sarcasm**: 2 clear cases ("love sitting in meetings", "I'm fine. totally fine.")
- Contains **negation**: "not happy", "can't stop", "don't even know"
- Contains **mixed sentiment**: 3 posts explicitly labeled "mixed"
- Language is primarily **informal American English internet slang** (Gen Z register)

**Possible issues:**
- **Very small dataset (16 examples)** — far too few for robust ML generalization
- **Label imbalance**: positive=5, negative=6, neutral=2, mixed=3 — the model will perform worse on underrepresented classes
- **Single annotator** — all labels reflect one person's interpretation; no inter-rater agreement check
- **Cultural/demographic skew** — slang like "lowkey", "no cap", "mid", "fire" is specific to a particular language community; the model will fail on posts from other dialects or cultural contexts
- **No truly neutral examples** — "This is fine" is arguably the only real neutral post; others labeled neutral are debatable

---

## 3. How the Rule-Based Model Works

**Scoring rules:**

1. **Preprocessing pipeline** (`preprocess()`):
   - Lowercase and strip whitespace
   - Pad Unicode emoji with spaces so they tokenize separately: `"😊😊😊"` → `['😊', '😊', '😊']`
   - Preserve text emoji tokens (`:)`, `:(`, `:-)`  ) before punctuation stripping
   - Strip leading/trailing punctuation from word tokens (preserving apostrophes in contractions)
   - Normalize repeated characters: `"soooo"` → `"soo"`, `"noooo"` → `"noo"`

2. **Token scoring** (`_token_score()`):

   | Signal type | Examples | Score |
   |-------------|----------|-------|
   | Strong positive (emoji/slang) | 😊, 😁, 🔥, "fire", "obsessed", "lowkey", "proud" | +2 |
   | Positive word list | "happy", "love", "great", "amazing", "grateful" | +1 |
   | No signal | "today", "the", "just" | 0 |
   | Negative word list | "sad", "bad", "tired", "stressed", "brutal" | -1 |
   | Strong negative (emoji/slang) | 😢, 💀, "mid", "hurts", "overwhelming" | -2 |

3. **Negation handling** (`score_text()`):
   - Negation words trigger a 3-token look-ahead window: `{"not", "never", "no", "can't", "don't", ...}`
   - The **first sentiment-bearing token** within the window gets its score flipped
   - **Negation blockers** absorb the negation without flipping the next sentiment word:
     - Verbal blockers: `"stop"`, `"quit"`, `"help"` → prevents `"can't stop smiling"` → negative
     - Idiom blockers: `"reason"`, `"idea"`, `"clue"` → prevents `"for no reason 😁"` → negative
     - Filler blockers: `"even"`, `"know"`, `"care"` → prevents `"don't even know 😰"` → positive

4. **Label assignment** (`predict_label()`):
   - If both positive AND negative tokens are present → `"mixed"` (regardless of net score)
   - Otherwise: score > 0 → `"positive"`, score < 0 → `"negative"`, score == 0 → `"neutral"`

**Strengths of the rule-based approach:**
- Fully transparent: `explain()` shows every matched token, score, and negation flip
- Handles simple negation well: `"I am not happy"` → correctly negative
- Handles slang and emoji once manually added to signal tables
- Deterministic — same input always produces same output
- Works with zero labeled training data

**Weaknesses of the rule-based approach:**
- **Cannot detect sarcasm at all.** `"I absolutely love sitting in 2-hour meetings"` → predicted `positive` because "love" (+1) is the only signal. The model has no way to know that liking meetings is unusual.
- **Idioms create false signals.** `"not bad, not great"` → negation flips "bad" to +1 and "great" to -1, producing "mixed" instead of "neutral". The phrase is an idiom for "mediocre", not a true mixture of positive and negative feelings.
- **Vocabulary bottleneck.** Any word not in the word lists or signal tables scores 0. New slang, rare words, or domain-specific language are invisible to the model.
- **Context blindness.** `"everyone left and honestly? relieved"` → only sees "relieved" (+2), misses the implied loneliness from "everyone left".
- **Every fix can break something else.** Adding negation blockers fixed "can't stop smiling" but required careful tuning to avoid breaking "not sad at all".

---

## 4. How the ML Model Works

**Features used:**
Bag-of-words using scikit-learn's `CountVectorizer`. Each post becomes a vector of word counts across the entire vocabulary. No preprocessing beyond what CountVectorizer applies by default (lowercase, basic tokenization).

**Training data:**
Trained and evaluated on the same 16 examples in `SAMPLE_POSTS` / `TRUE_LABELS`.

**Training behavior:**
The model was evaluated on its own training data (no held-out test set). This means the accuracy number (0.94) is inflated — it reflects memorization, not generalization.

Observed behavior when examining individual cases:
- The ML model learned `"love … meetings"` → negative from the sarcasm example in the training set
- It learned `"fine … fine … fine"` → negative from the repeated-word sarcasm example
- It misclassified `"This is fine"` → negative because it associated the word "fine" with sarcasm from the training data — a spurious correlation caused by small dataset size

**Strengths of the ML model:**
- Learned sarcasm patterns from labeled examples without being told about sarcasm explicitly
- Correctly classified 5 sentences that the rule-based model failed on
- Learns word co-occurrence patterns (e.g., "love" + "meetings" → negative) that rule systems cannot encode
- Scales: adding more labeled data improves performance without rewriting rules

**Weaknesses of the ML model:**
- **Overfitting.** With only 16 training examples, the model memorizes rather than generalizes. Real-world accuracy would be much lower on unseen posts.
- **Black box.** There is no way to ask the model "why did you predict negative here?" The rule-based `explain()` has no equivalent for the ML model.
- **Label sensitivity.** Changing a single label in the training data can flip predictions on multiple other posts. The model is extremely sensitive to small dataset changes.
- **Vocabulary dependence.** Like the rule-based model, it cannot handle words it has never seen. A post using entirely new slang would likely be classified as "neutral" (no matching features).
- **No "mixed" awareness by design.** The ML model predicts "mixed" only if that label appeared often enough in training. With just 3 mixed examples, it sometimes defaults to the nearest dominant class.

---

## 5. Evaluation

**How the models were evaluated:**
Both models were evaluated on the full 16-example `SAMPLE_POSTS` dataset using `evaluate.py`. This is **in-sample evaluation** (training = test for ML), so the ML accuracy is optimistic.

| Model | Accuracy | Correct / Total |
|-------|----------|----------------|
| Rule-based | **0.75** | 12 / 16 |
| ML (Logistic Regression) | **0.94** | 15 / 16 |

A stress test (`stress_test.py`) evaluated 19 adversarial edge cases designed to expose specific failure modes:

| Category | Cases | Rule-Based Pass | Notes |
|----------|-------|----------------|-------|
| Sarcasm | 4 | 0/4 | Fundamental limit — model cannot detect irony |
| Slang | 4 | 2/4 | "fire 🔥", "mid" work; "dead 💀 = funny" fails |
| Negation | 4 | 3/4 | Fixed window works; "nothing" not in negation list |
| Emoji-only | 4 | 3/4 | Unicode tokenization works; 🙂 sarcasm impossible |
| Mixed | 3 | 3/3 | "mixed" detection via dual-signal logic works |
| **Total** | **19** | **11/19 (58%)** | |

**Examples of correct predictions (rule-based):**

1. `"I am not happy about this"` → `negative` ✅
   - `"not"` triggers negation window, flips `"happy"` (+1) to -1
   - Score = -1 → negative

2. `"Feeling tired but kind of hopeful"` → `mixed` ✅
   - `"tired"` (-1) and `"hopeful"` (+1) both detected
   - Both polarities present → mixed

3. `"lowkey obsessed with this song rn 🎶"` → `positive` ✅
   - `"obsessed"` hits POSITIVE_SIGNALS (+2)
   - Score = +2 → positive

**Examples of incorrect predictions (rule-based):**

1. `"I absolutely love sitting in 2-hour meetings"` → predicted `positive`, true `negative`
   - `"love"` scores +1; no negative words present
   - **Root cause**: sarcasm requires understanding that meetings are unpleasant — world knowledge the model doesn't have. No rule change can fix this without a lookup table for "known-bad activities".

2. `"I'm fine. totally fine. everything is fine 🙂"` → predicted `neutral`, true `negative`
   - `"fine"` is not in any word list; 🙂 is not in NEGATIVE_SIGNALS
   - **Root cause**: sarcasm via repetition and tone. The model has no concept of repeated reassurance being suspicious. Would require detecting the pattern "X. X. X." explicitly.

3. `"not bad, not great, just kinda meh"` → predicted `mixed`, true `neutral`
   - `"not"` flips `"bad"` to +1; `"not"` flips `"great"` to -1 → both polarities → mixed
   - **Root cause**: "not bad" is an idiom meaning "acceptable", not literally the opposite of bad. A rule system cannot distinguish literal negation from idiomatic negation without a full idiom dictionary.

**Where ML and rule-based disagree (6 cases):**

| Post | Rule | ML | True | Winner |
|------|----- |----|------|--------|
| `"This is fine"` | neutral | negative | neutral | Rule ✅ |
| `"I'm fine. totally fine."` | neutral | negative | negative | ML ✅ |
| `"not bad, not great, just kinda meh"` | mixed | neutral | neutral | ML ✅ |
| `"I absolutely love sitting in 2-hour meetings"` | positive | negative | negative | ML ✅ |
| `"everyone left and honestly? relieved"` | positive | mixed | mixed | ML ✅ |
| `"this is so overwhelming I don't even know 😰"` | neutral→negative* | negative | negative | Both ✅* |

*After the negation blocker fix.

---

## 6. Limitations

**1. Sarcasm is undetectable by rules.**
The rule-based model has no mechanism for detecting irony. `"I absolutely love sitting in 2-hour meetings"` is classified positive because "love" (+1) is the only matched token. Detecting sarcasm requires either: (a) a large training set of sarcasm examples for ML, (b) a dedicated sarcasm detector, or (c) contextual cues like tone, punctuation patterns, and discourse history that a single-sentence classifier cannot access.

**2. The dataset is too small for meaningful ML generalization.**
With 16 training examples, the ML model is memorizing the training set (0.94 in-sample accuracy). On genuinely new posts, accuracy would likely drop to 0.50–0.65. A minimum of 100–200 labeled examples per class would be needed for basic generalization.

**3. The vocabulary is frozen at development time.**
Both models fail silently on unknown words. A post written entirely in new slang ("that was bussin fr no cap") would score 0 by rule-based (unknown words = 0) and be misclassified by ML (unseen vocabulary = zero-vector).

**4. Language is culturally and demographically specific.**
The dataset was constructed using informal Gen Z American English internet language. The model is likely to fail on:
- Formal or academic language (no signal words match)
- Non-English words or code-switching ("tan cansado pero grateful")
- British slang ("gutted", "chuffed", "knackered")
- Older demographic language patterns

**5. Emoji meanings are context-dependent.**
😭 was added to POSITIVE_SIGNALS because in the dataset it appeared in `"just got promoted... 😭😭"` (happy-crying). But 😭 is also used for genuine sadness. The model cannot distinguish based on context.

**6. Labels reflect one annotator's interpretation.**
Every TRUE_LABEL reflects a single person's reading of a post. Human labelers routinely disagree on sarcasm, mixed emotions, and culturally specific phrases. No inter-rater reliability was measured.

**7. No held-out test set.**
All evaluation was done on training data. We have no honest estimate of how either model would perform on posts it has never seen.

---

## 7. Ethical Considerations

**Misclassifying distress signals.**
If the Mood Machine were deployed in a real application (e.g., social media content moderation, student wellness monitoring), false negatives on distressed posts could have serious consequences. `"I'm fine. totally fine."` — a post that could indicate someone is not okay — is classified as `neutral`. Any system making safety decisions must not rely on a model with this failure mode.

**Sarcasm misclassification creates false positives.**
A post like `"absolutely loving this situation"` (sarcastic despair) would be classified `positive`. In a monitoring context, this would suppress alerts for someone who genuinely needs support.

**Cultural and linguistic bias.**
The model was built on one person's informal English. It will systematically perform worse for speakers of other dialects, languages, or registers. Deploying this to a multilingual or multicultural user base would produce unfair outcomes — certain communities would be more likely to be misclassified.

**Privacy considerations.**
Mood analysis on personal messages is intrinsically sensitive. Even a research prototype should not be applied to real user messages without explicit consent and clear disclosure of the model's limitations and error rate.

**The "neutral" catch-all is misleading.**
`neutral` in this model means "no matching keywords found" — it does not mean the post has no emotional content. Communicating this distinction to end users is important; a `neutral` prediction should never be interpreted as "this person is fine".

---

## 8. Ideas for Improvement

**Short-term (achievable with current architecture):**
- Add a real train/test split (e.g., 80/20) to get honest accuracy estimates for the ML model
- Add 50+ labeled posts to reduce ML overfitting
- Expand the NEGATIVE_SIGNALS table to include more slang used for frustration/exhaustion
- Add `"nothing"`, `"nobody"`, `"nowhere"` to `NEGATION_WORDS` to handle `"nothing was bad"` → positive

**Medium-term (architectural changes):**
- Replace `CountVectorizer` with `TfidfVectorizer` to downweight common words and improve feature quality
- Use cross-validation instead of single train/test split for more reliable accuracy estimates
- Add a simple rule-based sarcasm detector for known patterns: `"I absolutely love [thing]"`, `"so fun to [activity]"`
- Improve emoji handling: build a lookup table mapping emoji to sentiment based on a larger emoji sentiment dataset

**Long-term (different model class):**
- Use a pre-trained sentence embedding model (e.g., `sentence-transformers`) as features instead of bag-of-words — embeddings capture word context, not just word presence
- Fine-tune a small pre-trained language model (e.g., DistilBERT) on a larger mood/sentiment dataset
- Add a confidence score so the model can express uncertainty: "I'm 60% sure this is negative" is more honest than a hard label

**Process improvements:**
- Collect labels from multiple annotators and measure inter-rater agreement (Cohen's kappa)
- Build a diverse dataset that includes formal English, British slang, non-native speaker patterns, and code-switching
- Create separate test sets for each failure category: sarcasm test set, slang test set, negation test set

---

*Model card completed after Parts 1–4 of the Mood Machine lab.*
*Rule-based final accuracy: 0.75 (12/16). ML in-sample accuracy: 0.94 (15/16).*
