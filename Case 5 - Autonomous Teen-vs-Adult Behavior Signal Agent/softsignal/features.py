"""Allowed model inputs (Tier 1, step 1). Everything else in the CSV is off limits."""
SEED = 42
N_TEST = 900
TARGET = "label_teen"
ID_COL = "blogger_id"

ACTIVITY_COLS = [
    "pct_active_school_hours",
    "pct_active_evening",
    "pct_active_late_night",
    "weekend_weekday_session_ratio",
    "sessions_per_day",
    "avg_session_minutes",
    "share_short_video_views",
    "share_news_views",
    "night_notification_open_rate",
]
TEXT_COLS = [
    "avg_word_len",
    "first_person_rate",
    "school_token_rate",
    "birthday_token_rate",
    "exclaim_rate",
    "slang_emoji_rate",
    "keyword_teen_flag",
]
FEATURE_COLS = ACTIVITY_COLS + TEXT_COLS

# Typed age, the label, self-declared profile fields and account-level proxies.
# birthday_token_rate is allowed on purpose: it comes from post text, not the typed birthday.
# gender and job are kept in the CSV for fairness monitoring only.
FORBIDDEN = {
    "age",
    TARGET,
    "is_teen",
    "gender",
    "job",
    "account_age_days",
    "friend_count",
}

assert len(FEATURE_COLS) == 16 and len(set(FEATURE_COLS)) == 16
assert not FORBIDDEN & set(FEATURE_COLS), "a forbidden column is in the feature allow-list"
