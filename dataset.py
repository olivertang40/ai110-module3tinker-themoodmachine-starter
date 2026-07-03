"""
Shared data for the Mood Machine lab.

This file defines:
  - POSITIVE_WORDS: starter list of positive words
  - NEGATIVE_WORDS: starter list of negative words
  - SAMPLE_POSTS: short example posts for evaluation and training
  - TRUE_LABELS: human labels for each post in SAMPLE_POSTS
"""

# ---------------------------------------------------------------------
# Starter word lists
# ---------------------------------------------------------------------

POSITIVE_WORDS = [
    # original starter
    "happy", "great", "good", "love", "excited",
    "awesome", "fun", "chill", "relaxed", "amazing",
    # extended
    "wonderful", "fantastic", "joyful", "grateful", "thankful",
    "proud", "hopeful", "smile", "smiling", "confident",
    "motivated", "inspired", "peaceful", "content", "thrilled",
    "delighted", "relieved", "blessed", "energized", "pumped",
]

NEGATIVE_WORDS = [
    # original starter
    "sad", "bad", "terrible", "awful", "angry",
    "upset", "tired", "stressed", "hate", "boring",
    # extended
    "miserable", "depressed", "frustrated", "anxious", "lonely",
    "exhausted", "overwhelmed", "hopeless", "drained", "hurt",
    "worried", "scared", "nervous", "disappointed", "gloomy",
    "brutal", "horrible", "painful", "annoyed", "dread",
    "overwhelming",
]

# ---------------------------------------------------------------------
# Starter labeled dataset
# ---------------------------------------------------------------------

# Short example posts written as if they were social media updates or messages.
SAMPLE_POSTS = [
    "I love this class so much",
    "Today was a terrible day",
    "Feeling tired but kind of hopeful",
    "This is fine",
    "So excited for the weekend",
    "I am not happy about this",
]

# Human labels for each post above.
# Allowed labels in the starter:
#   - "positive"
#   - "negative"
#   - "neutral"
#   - "mixed"
TRUE_LABELS = [
    "positive",  # "I love this class so much"
    "negative",  # "Today was a terrible day"
    "mixed",     # "Feeling tired but kind of hopeful"
    "neutral",   # "This is fine"
    "positive",  # "So excited for the weekend"
    "negative",  # "I am not happy about this"
]

# --- New posts added for Part 1 ---
# Each entry below has a matching label. Styles covered:
#   slang, emojis, sarcasm, negation, ambiguous/mixed feelings.

SAMPLE_POSTS += [
    "lowkey obsessed with this song rn 🎶",          # slang + emoji
    "no cap this week has been absolutely brutal",    # slang + negative
    "I'm fine. totally fine. everything is fine 🙂", # sarcasm / masked distress
    "just got promoted omg I'm so happy 😭😭",        # emoji ambiguity (crying = joy)
    "not bad, not great, just kinda meh",             # negation + neutral/mixed
    "can't stop smiling today for no reason 😁",     # positive
    "I absolutely love sitting in 2-hour meetings",  # sarcasm -> negative
    "feeling 💀 after that exam but proud I finished", # emoji slang + mixed
    "everyone left and honestly? relieved",           # ambiguous — could be pos or neg
    "this is so overwhelming I don't even know 😰",  # negative with emoji
]

TRUE_LABELS += [
    "positive",  # lowkey obsessed with this song
    "negative",  # no cap this week has been brutal
    "negative",  # I'm fine. totally fine. (sarcasm)
    "positive",  # just got promoted (crying emojis = happy here)
    "neutral",   # not bad, not great, meh
    "positive",  # can't stop smiling
    "negative",  # love sitting in 2-hour meetings (sarcasm)
    "mixed",     # feeling 💀 after exam but proud
    "mixed",     # everyone left and honestly relieved (debatable)
    "negative",  # so overwhelming
]

# Sanity check — will raise immediately if lists drift out of sync.
assert len(SAMPLE_POSTS) == len(TRUE_LABELS), (
    f"Length mismatch: {len(SAMPLE_POSTS)} posts vs {len(TRUE_LABELS)} labels"
)
