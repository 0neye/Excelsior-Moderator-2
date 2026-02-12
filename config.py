import logging
import os
import sys
from logging import Logger

from dotenv import load_dotenv

load_dotenv()

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# The text or forum channels to allow
excelsior = [
    546229904488923145,
    1101149194498089051,
    546327169014431746,
    1240185912525324300,
    546907635149045775,
    546947839008440330,
]
CHANNEL_ALLOW_LIST = excelsior + [1077332289190625300]

# Rating system
RATING_CHANNEL_ID = 1439846071991013418
RATING_SYSTEM_LOG_FILE = "rating_system_log.json"

MODERATOR_ROLES = ["Sentinel (mod)", "Custodian (admin)"]
ADMIN_ROLE = "Custodian (admin)"
RANK_ROLES = ["Adept", "Expert", "Paragon"]
OTHER_ROLES = ["Architect", "Artisan", "Visionary"]

# The role for people who don't care about harsh feedback
WAIVER_ROLE_NAME = "Criticism Pass"
# Toggle waiver-target persistence behavior in moderation:
# - True keeps waived-target flagged rows in DB and suppresses Discord action
# - False drops waived-target candidates early (legacy behavior)
SAVE_WAIVER_FILTERED_FLAGS = True

DB_FILE = "excelsior.db"

# How many messages to wait for before sending them to the llm for moderation
MESSAGES_PER_CHECK = 30

# How many messages of history to send to the llm for analysis
HISTORY_PER_CHECK = 40

# If there are new messages in a channel that haven't been checked, but not enough to trigger the above, check anyway after this time
# Resets after a new message, and doesn't trigger if all messages in channel have already been checked
SECS_BETWEEN_AUTO_CHECKS = 240
# Minimum number of new messages required before starting the idle moderation timer
# This helps avoid expensive checks for one-off chatter in low-activity channels
NEW_MESSAGES_BEFORE_TIMER_START = 3

# Backoff settings for failed moderation runs to prevent rapid retries.
MODERATION_FAILURE_BACKOFF_BASE = 30
MODERATION_FAILURE_BACKOFF_MAX = 600

# The number of new ratings before the classifier model is retrained
NEW_RATINGS_BEFORE_RETRAIN = 20
# Exclude waived-target records from the public `/rate` candidate pool
EXCLUDE_WAIVER_FILTERED_FROM_RATE_POOL = True
# Exclude waived-target records from coverage/stat denominators
EXCLUDE_WAIVER_FILTERED_FROM_COVERAGE = True
# Exclude waived-target records from continuous training datasets
EXCLUDE_WAIVER_FILTERED_FROM_TRAINING = True

# Feature extraction mode for continuous training:
# - "existing_only": Only use messages with pre-extracted features (faster, no LLM cost)
# - "extract_on_demand": Extract features for new messages via LLM when training (more complete)
CONTINUOUS_TRAINING_FEATURE_MODE = "existing_only"

# Category collapsing for continuous training:
# - If True, collapses 5-category ratings to binary "flag"/"no-flag" for training
# - Mapping: "unsolicited" and "unconstructive" -> "flag"; "NA" and "no-flag" -> "no-flag"
# - "ambiguous" handling controlled by CONTINUOUS_TRAINING_COLLAPSE_AMBIGUOUS
CONTINUOUS_TRAINING_COLLAPSE_CATEGORIES = True

# If collapsing is enabled, whether to map "ambiguous" to "no-flag" instead of keeping it separate
CONTINUOUS_TRAINING_COLLAPSE_AMBIGUOUS = True


# The guidelines for constructive criticism
GUIDELINES = """
__**Giving Feedback**__

To put this post in two sentences:

Make sure your feedback is __consented__.
Be __positive__ with that feedback.

When giving feedback, above all else, you want to be respectful. It's easy to point out a bunch of flaws in someone's build, especially if you're an experienced player, but that's not your goal unless that's what that player is specifically requesting. Please be mindful of this...pretty much always. Only give feedback on role requests, classroom posts, module posts, ship posts, idea posts, etc, in the way that is being requested, if at all. At the time of writing, this has been a huge issue for a while. Adhering to this rule will help solve that.

It's easy to say "your X is wrong" or "your layout is unoptimal", but these are all negative phrases. Feedback is the most useful to the most people when phrased positively. "I like the way you did "X" is a good optimistic comment you will always find yourself able to make about something. Include at least a few positive takeaways in your feedback as well as your suggested improvements - it shows that you recognize that you're talking to a person and not just grading their homework.

Additionally, try to focus on the *how* and *why* instead of just the *what*. Telling someone their crew management is bad doesn't help them improve. Instead phrase things like "instead of doing X, you could do Y which would be better because Z".
"""


LOG_CHANNEL_ID = 1333899222541406310

REACTION_EMOJI = "👁️"


DEFAULT_LLM_PROVIDER = "cerebras"
DEFAULT_LLM_MODEL = "gpt-oss-120b"


# Logging config
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
_LOGGING_CONFIGURED = False


def configure_logging() -> None:
    """
    Configure the root logger with a console handler.

    Uses LOG_LEVEL env override if provided. Idempotent to avoid duplicate
    handler stacks on reloads.
    """
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    logging.basicConfig(
        level=LOG_LEVEL,
        format=LOG_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    _LOGGING_CONFIGURED = True


def get_logger(name: str) -> Logger:
    """Return a module-specific logger after ensuring global config is set."""
    configure_logging()
    return logging.getLogger(name)
