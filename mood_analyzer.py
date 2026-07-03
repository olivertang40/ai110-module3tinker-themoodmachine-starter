# mood_analyzer.py
"""
Rule-based mood analyzer for short text snippets.

Data flow:
  raw text
      |
      v
  preprocess()  ->  List[str] tokens
      |
      v
  score_text()  ->  int score   (calls preprocess internally)
      |
      v
  predict_label() ->  str label  (calls score_text internally)

The explain() helper runs the same logic as score_text() and returns
a human-readable breakdown — useful for debugging.
"""

import re
from typing import List, Optional, Tuple

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS


# ---------------------------------------------------------------------------
# Emoji / slang sentiment tables
# ---------------------------------------------------------------------------

# Positive emoji & slang  -> +2 (strong signal)
POSITIVE_SIGNALS = {
    # emoji
    "😊", "😄", "😁", "🥰", "❤️", "💕", "🎉", "✨", "🙌", "👏",
    "😂", "🤣", "🔥",
    # text emoji
    ":)", ":-)", ":d", "=)", "<3",
    # slang
    "obsessed", "lowkey", "highkey", "slay", "goated", "fire",
    "blessed", "hyped", "pumped", "winning", "proud", "smiling",
    "promoted", "relieved", "grateful", "worth",
}

# Negative emoji & slang  -> -2 (strong signal)
NEGATIVE_SIGNALS = {
    # emoji  (😭 removed — too ambiguous, used for both joy and sadness)
    "😢", "😞", "😤", "😠", "😡", "💀", "😰", "😨", "🤮",
    # text emoji
    ":(", ":-(", ":/",
    # slang
    "brutal", "trash", "awful", "horrible", "rip",
    "overwhelming", "overwhelmed", "exhausted",
    "mid",       # mediocre/bad
    "hurts",     # "it hurts but…"
    "crying",    # standalone "crying" often negative
}

# Negation words: flip the sentiment of the next sentiment-bearing token
# within a look-ahead window (see score_text).
NEGATION_WORDS = {
    "not", "never", "no", "neither", "nor",
    "can't", "cant", "won't", "wont", "don't", "dont",
    "didn't", "didnt", "isn't", "isnt", "wasn't", "wasnt",
    "couldn't", "couldnt", "wouldn't", "wouldnt",
}

# Negation blockers: these words absorb a negation without passing it on.
# Example: "can't stop smiling" — "stop" absorbs the negation from "can't"
# so "smiling" is NOT flipped.
# Example: "for no reason 😁" — "reason" absorbs "no" so the emoji is not flipped.
# Without this, the window would skip these words and flip the next sentiment
# word incorrectly.
NEGATION_BLOCKERS = {
    # verbal blockers — the negation targets the verb, not what follows
    "stop", "quit", "help", "think", "believe", "imagine",
    "tell", "say", "make", "let", "get",
    # idiomatic noun phrases — "no reason", "no idea", "no clue"
    # these mean "without cause", not a negation of sentiment
    "reason", "idea", "clue", "doubt", "wonder", "question",
}

# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MoodAnalyzer:
    """
    A rule-based mood classifier.

    Scoring improvements over the bare starter:
      1. Punctuation stripping so "happy," and "happy" match the same word.
      2. Repeated-character normalisation: "soooo" -> "soo" (catches typos).
      3. Negation handling: "not happy" flips the word's sentiment.
      4. Emoji / slang signals with stronger weight (+/-2) so short posts
         that use no dictionary words still get a score.
      5. predict_label uses a ±2 threshold to emit "mixed" for posts that
         contain both positive AND negative signals.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens.

        Steps:
          1. Lowercase and strip surrounding whitespace.
          2. Insert spaces around Unicode emoji so they become standalone tokens.
          3. Isolate text-emoji tokens (":)", etc.) before stripping punctuation.
          4. Strip punctuation from word tokens (but NOT from emoji tokens).
          5. Normalise repeated characters: "soooo" -> "soo".
          6. Drop empty strings.

        >>> MoodAnalyzer().preprocess("I'm SO happy!!! :)")
        ["i'm", 'so', 'happy', ':)']
        >>> MoodAnalyzer().preprocess("😊😊😊")
        ['😊', '😊', '😊']
        """
        # Step 1 — lowercase
        text = text.strip().lower()

        # Step 2 — pad Unicode emoji with spaces so they split cleanly.
        # Matches any character in the emoji Unicode ranges.
        text = re.sub(
            r"([\U00010000-\U0010ffff"   # supplementary planes (most emoji)
            r"\U00002600-\U000027BF"     # misc symbols
            r"\U0001F300-\U0001FAFF"     # emoticons / transport / misc
            r"\U00002702-\U000027B0"
            r"])",
            r" \1 ",
            text,
        )

        # Step 3 — split on whitespace
        raw_tokens = text.split()

        tokens: List[str] = []
        for tok in raw_tokens:
            # Keep text-emoji tokens intact
            if tok in {":)", ":-)", ":d", "=)", "<3", ":(", ":-(", ":/"}:
                tokens.append(tok)
                continue

            # Step 4 — strip leading/trailing punctuation from word tokens
            tok = re.sub(r"^[^a-z0-9'\U00010000-\U0010ffff]+|[^a-z0-9'\U00010000-\U0010ffff]+$", "", tok)

            # Step 5 — normalise runs of 3+ identical chars -> 2
            tok = re.sub(r"(.)\1{2,}", r"\1\1", tok)

            if tok:
                tokens.append(tok)

        return tokens

    # ------------------------------------------------------------------
    # Scoring logic
    # ------------------------------------------------------------------

    def _token_score(self, token: str) -> Tuple[int, str]:
        """
        Return (raw_score, category) for a single token.

        category is one of: "positive", "negative", "neutral"
        raw_score is +2 for strong signals, +1/-1 for word lists, 0 otherwise.
        """
        if token in POSITIVE_SIGNALS:
            return 2, "positive"
        if token in NEGATIVE_SIGNALS:
            return -2, "negative"
        if token in self.positive_words:
            return 1, "positive"
        if token in self.negative_words:
            return -1, "negative"
        return 0, "neutral"

    def score_text(self, text: str) -> int:
        """
        Compute a numeric mood score for the given text.

          +2  strong positive signal (emoji / slang)
          +1  positive word from the word list
           0  neutral / unknown word
          -1  negative word from the word list
          -2  strong negative signal (emoji / slang)

        Negation window: a negation word ("not", "never", "no", …) flips
        the next sentiment-bearing token within a window of up to 3 tokens.
        Non-sentiment tokens (score == 0) consume the window but do NOT
        use up the negation — we keep looking forward until we hit a word
        that actually has a score, or the window closes.

        Returns the sum of all token scores.
        """
        tokens = self.preprocess(text)
        score = 0
        negate_remaining = 0   # how many more tokens to still check for negation

        for token in tokens:
            if token in NEGATION_WORDS:
                negate_remaining = 3   # look forward up to 3 tokens
                continue

            # A blocker word (stop, quit, …) absorbs the negation entirely
            # so it doesn't propagate to the next sentiment word.
            if negate_remaining > 0 and token in NEGATION_BLOCKERS:
                negate_remaining = 0
                continue

            word_score, _ = self._token_score(token)

            if negate_remaining > 0:
                negate_remaining -= 1
                if word_score != 0:
                    word_score = -word_score
                    negate_remaining = 0   # negation consumed

            score += word_score

        return score

    # ------------------------------------------------------------------
    # Label prediction
    # ------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Map a numeric score to a mood label.

        Thresholds chosen to match the label vocabulary in TRUE_LABELS:

          score >= 2   -> "positive"   (clear positive signal)
          score <= -2  -> "negative"   (clear negative signal)
          score == 1   -> "positive"   (mild but real positive lean)
          score == -1  -> "negative"   (mild but real negative lean)
          score == 0   -> "neutral"    (no signal either way)

        "mixed" is returned when the text contains BOTH positive AND
        negative tokens (regardless of total score), reflecting posts
        like "feeling awful but kind of proud".
        """
        tokens = self.preprocess(text)

        pos_count = 0
        neg_count = 0
        for token in tokens:
            _, category = self._token_score(token)
            if category == "positive":
                pos_count += 1
            elif category == "negative":
                neg_count += 1

        # Both polarities present -> mixed
        if pos_count > 0 and neg_count > 0:
            return "mixed"

        score = self.score_text(text)
        if score > 0:
            return "positive"
        if score < 0:
            return "negative"
        return "neutral"

    # ------------------------------------------------------------------
    # Explanation (debugging helper)
    # ------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a human-readable breakdown of the scoring decision.

        Example:
          Score = 2 | label = positive
          positive tokens : ['love', 'great']
          negative tokens : []
          negation flips  : ['not happy -> -1']
        """
        tokens = self.preprocess(text)

        positive_hits: List[str] = []
        negative_hits: List[str] = []
        negation_flips: List[str] = []
        score = 0
        negate_remaining = 0
        pending_negation_token = ""

        for token in tokens:
            if token in NEGATION_WORDS:
                negate_remaining = 3
                pending_negation_token = token
                continue

            if negate_remaining > 0 and token in NEGATION_BLOCKERS:
                negate_remaining = 0
                continue

            word_score, category = self._token_score(token)

            if negate_remaining > 0:
                negate_remaining -= 1
                if word_score != 0:
                    flip_desc = f"{pending_negation_token} … {token} -> {-word_score:+d}"
                    negation_flips.append(flip_desc)
                    word_score = -word_score
                    negate_remaining = 0

            score += word_score

            if word_score > 0:
                positive_hits.append(token)
            elif word_score < 0:
                negative_hits.append(token)

        label = self.predict_label(text)

        lines = [
            f'Score = {score:+d} | label = "{label}"',
            f"  positive tokens : {positive_hits if positive_hits else []}",
            f"  negative tokens : {negative_hits if negative_hits else []}",
        ]
        if negation_flips:
            lines.append(f"  negation flips  : {negation_flips}")

        return "\n".join(lines)
