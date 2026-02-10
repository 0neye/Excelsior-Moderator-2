# Excelsior Moderator 2 - Technical Documentation

This document provides in-depth technical information about the bot's architecture, implementation, and development workflow.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Major Components](#major-components)
3. [Data Flow](#data-flow)
4. [Major Workflows](#major-workflows)
5. [Feature Extraction](#feature-extraction)
6. [Database Schema](#database-schema)
7. [Configuration Reference](#configuration-reference)
8. [Installation & Setup](#installation--setup)
9. [Development Guide](#development-guide)
10. [Performance Considerations](#performance-considerations)
11. [Troubleshooting](#troubleshooting)

## Architecture Overview

Excelsior Moderator 2 uses a two-stage ML pipeline:

1. **Feature Extraction**: LLMs (Cerebras, OpenRouter, or Gemini) analyze message history to extract semantic features like tone harshness, positive framing, actionability, and context appropriateness.
2. **Classification**: A LightGBM classifier predicts whether messages should be flagged based on extracted features combined with user statistics.

The bot implements a continuous learning loop where human ratings are used to retrain the model automatically every N ratings (default: 20).

## Major Components

### 1. Bot Core (`bot.py`)

The main Discord bot implementation using py-cord.

**Key Classes**:
- **`ExcelsiorBot`**: Custom bot class that manages:
  - Message store (in-memory deque per channel)
  - Database sessions
  - Moderation scheduling
  - Feature extraction and prediction workflow

**Message Store**: In-memory deque-based history buffer per channel (see `history.py`)
- Max size: `max(HISTORY_PER_CHECK, MESSAGES_PER_CHECK)` (default: 90)
- Separate deque per channel/thread
- Automatic eviction of oldest messages

**Channel Scheduling**: Automatic moderation triggers based on:
- Message count threshold (`MESSAGES_PER_CHECK`)
- Idle timer (`SECS_BETWEEN_AUTO_CHECKS`)
- Manual trigger via `/check` command

### 2. Database Layer

**`database.py`**: SQLAlchemy ORM models for all data structures

**Core Models**:
- `FlaggedMessage`: Stores flagged messages with content and context
- `FlaggedMessageRating`: Human ratings for continuous learning
- `MessageFeatures`: Extracted LLM features (both bootstrapped and runtime)
- `UserStats`: User activity metrics (message/character counts, join time)
- `UserCoOccurrence`: Tracks user interaction frequency for familiarity scores
- `FeatureExtractionRun`: Versioning for batch feature extraction runs
- `LogChannelRatingPost`: Maps log channel messages to flagged messages
- `DictionaryEntry`: Terminology definitions

**`db_config.py`**: Database configuration and session management

### 3. Machine Learning (`ml.py`)

LightGBM classifier implementation with abstract base class pattern.

**Features**:
- 18 total features (15 LLM-based + 3 database-derived)
- Abstract base class `ModerationClassifier`
- Concrete `LightGBMClassifier` implementation
- Model serialization/deserialization with joblib
- Balanced class weights for imbalanced data

**Model Parameters**:
- 200 estimators
- Max depth: 6
- Learning rate: 0.1
- Binary or multi-class objective

### 4. Training Module (`training.py`)

Continuous training system that automatically retrains on new ratings.

**Modes**:
- **`existing_only`** (default): Load features from database (no LLM cost)
- **`extract_on_demand`**: Extract only missing features via LLM (more complete, higher cost)
  - Loads rated message records from the database
  - Reuses stored context when available
  - Fetches context from Discord API by `channel_id` + `message_id` when context is missing

**Category Collapsing** (configurable):
- **Binary mode** (default): flag/no-flag (recommended for <2000 ratings)
  - `"flag"`: "unsolicited" + "unconstructive"
  - `"no-flag"`: "NA" + "no-flag" + "ambiguous" (if enabled)
- **Multi-class mode**: All 5 categories (requires substantial balanced data)

**Trigger**: Automatically after `NEW_RATINGS_BEFORE_RETRAIN` new ratings (default: 20)

### 5. LLM Integration (`llms.py`)

Extracts semantic features from message history using structured JSON output.

**Supported Providers**:
- **Cerebras**: Fastest, cheapest (~$0.10 per million tokens)
- **OpenRouter**: OpenAI-compatible API, various models
- **Gemini**: Google's models

**Features Extracted**: 15 LLM-based features per candidate message (see [Feature Extraction](#feature-extraction))

**Implementation Details**:
- Structured JSON schema for reliable output
- Rate limiting and timeout handling
- Retry logic for transient failures
- Augmentation with 3 database-derived stat features

### 6. User Statistics (`user_stats.py`)

Tracks user behavior to compute contextual features.

**Metrics Tracked**:
- **Seniority scores**: Based on days since join, message count, and character count
- **Familiarity scores**: Co-occurrence statistics between message authors and targets

**Update Strategy**:
- Real-time updates on every message in tracked channels
- Co-occurrence windows: Sliding 30-message window
- Bootstrapping: Can backfill statistics from Discord history

**Formulas**:
```python
# Seniority scores: author/target activity ratio, then normalized (0-1)
msg_ratio = author_msg_count / max(target_msg_count, 1)
char_ratio = author_char_count / max(target_char_count, 1)
seniority_messages = msg_ratio / (msg_ratio + 1.0)   # _normalize_ratio
seniority_characters = char_ratio / (char_ratio + 1.0)

# Familiarity score (co-occurrence in sliding windows)
familiarity_score_stat = co_occurrence_count / (co_occurrence_count + 10)
```

### 7. Cogs (Discord Extensions)

**`cogs/events.py`**: Message/reaction listeners
- Updates message store and user stats
- Handles message edits and deletions
- Maintains co-occurrence statistics

**`cogs/rating.py`**: Rating system
- `/rate` command for public rating
- `/view_score` for personal statistics
- Reaction handlers for log channel ratings
- Leaderboard management
- Automatic retraining triggers

**`cogs/restricted.py`**: Admin/moderator commands
- `/check` for manual moderation (moderator role)
- `/retrain` for forced model retraining (admin role only)
- Moderator and admin permission checks

**`cogs/public.py`**: Public utility commands
- `/ping` for latency check
- `/info` for bot information

### 8. Bootstrapping (`bootstrapping.py`)

Full training pipeline for initial model development.

**Pipeline Steps**:
1. Load ratings from `rating_system_log.json`
2. Fetch Discord context around each rated message
3. Bootstrap user stats from Discord history
4. Extract features via LLM (multiple runs for robustness)
5. Train LightGBM classifier
6. Evaluate with cross-validation and confusion matrices

**Interactive REPL Menu**:
```
1. Run full pipeline (load -> fetch -> extract -> train -> eval)
2. Load data only (from rating_system_log.json)
3. Fetch Discord context (requires bot connection)
4. Extract LLM features (from stored context)
5. Train model (LightGBM)
6. Evaluate model (on held-out test set)
7. Show current state
8. List/Load features from database
9. Collect user statistics from Discord (channels & threads)
10. Exit
```

**State Management**:
- In-memory state during REPL session
- Features saved to database during extraction
- Option 8 loads features from database into state

**Statistics Bootstrapping**:
- Scans up to 10,000 messages per tracked channel
- Takes ~1 hour for large servers
- Only needs to run once

## Data Flow

```
Discord Message
    ↓
[events.py] on_message listener
    ↓
Message Store + User Stats Update
    ↓
Channel Scheduler checks thresholds
    ↓
[bot.py] moderate_channel()
    ↓
[llms.py] Extract features for candidates
    ↓
[ml.py] LightGBM classifier predicts
    ↓
[bot.py] Flag messages + Post to log channel
    ↓
[database.py] Save FlaggedMessage + MessageFeatures
    ↓
    ├─→ Moderator reacts in log channel (private)
    │       ↓
    │   [rating.py] Create FlaggedMessageRating
    │
    └─→ User uses /rate in rating channel (public)
            ↓
        [rating.py] Create FlaggedMessageRating
    ↓
[rating.py] Counter reaches threshold
    ↓
[training.py] Auto-retrain model with all ratings
    ↓
New model used on next moderation run
```

## Major Workflows

### 1. Channel Check Scheduling

**Components**: `bot.py` (`ExcelsiorBot` class), `cogs/events.py`

**Process**:
1. Each tracked channel/thread gets a `ChannelModerationState` with counters and timers
2. `on_message` event increments `messages_since_check` and starts idle timer
3. `_moderation_scheduler` runs in background for each channel:
   - Waits for either `MESSAGES_PER_CHECK` messages or `SECS_BETWEEN_AUTO_CHECKS` seconds
   - Triggers `moderate_channel()` when threshold met
   - Resets counters after successful moderation

**Trigger Conditions**:
- **Message count**: After N new messages since last check
- **Idle timer**: After M seconds of inactivity with new unchecked messages
- **Manual**: Via `/check` command

### 2. Message Flagging Logic

**Components**: `bot.py` (`moderate_channel`), `llms.py` (`get_candidate_features`)

**Process**:
1. **Get candidates**: LLM analyzes last `HISTORY_PER_CHECK` messages (default: 40) and identifies potential feedback candidates
2. **Filter candidates**:
   - Remove messages with `discusses_ellie > 0.2` (bot-related noise)
   - Remove messages without Discord IDs
   - Remove messages that target users with the "Criticism Pass" role
   - Remove messages too close to channel start (insufficient context)
3. **Extract features**: LLM provides 15 semantic features per candidate
4. **Augment with stats**: Add 3 database-derived features (seniority, familiarity)
5. **Predict**: LightGBM classifier predicts "flag" or "no-flag"
6. **Flag actions**:
   - Add 👁️ reaction to original message
   - Post to log channel with rating buttons
   - Save to `FlaggedMessage` table
   - Save runtime features to `MessageFeatures` (with `extraction_run_id=NULL`)

### 3. Model Retraining

**Components**: `training.py`, `cogs/rating.py`

**Trigger**: Automatically after N new ratings (configured via `NEW_RATINGS_BEFORE_RETRAIN`)

**Process**:
1. Load all rated messages from `FlaggedMessageRating` table
2. Load features based on mode:
   - **`existing_only`**: Load from `MessageFeatures` table (no LLM cost)
   - **`extract_on_demand`**: Extract missing features via LLM using DB-backed rated messages; missing context is fetched from Discord when possible
3. Prepare training data:
   - Build feature matrix X with 18 features
   - Build label vector y with rating categories
   - Apply category collapsing if enabled
4. Train LightGBM classifier with balanced class weights
5. Save model to `models/lightgbm_model.joblib`
6. Model is immediately used by bot (loaded on next moderation run)

**Feature Sources** (priority order):
1. Runtime features (no `extraction_run_id`)
2. Most recent extraction run features

### 4. Database Updates

**Real-time (Runtime)**:
- **Message Store**: In-memory deque, updated on every message/edit/delete event
- **User Stats**: Updated on every message in tracked channels
- **Co-occurrence**: Updated on every message with 30-message sliding window
- **Flagged Messages**: Created when model predicts "flag"
- **Runtime Features**: Saved alongside flagged messages (`extraction_run_id=NULL`)

**Batch (Bootstrapping)**:
- **Feature Extraction Runs**: Created for each bootstrapping feature extraction
- **Message Features**: Bulk inserted with `extraction_run_id` for versioning
- Supports multiple runs per message for stochastic feature extraction

**Rating System**:
- **Log Channel Posts**: Created when message is flagged
- **Ratings**: Created when moderator reacts with rating emoji
- Ratings trigger continuous training after threshold reached

### 5. Public Rating System

**Overview**: Community-driven rating system for training data collection

**Components**:
1. **Rating Channel Setup**: Maintains public rating channel with:
   - Pinned instructions
   - Live leaderboard
   - Interactive `/rate` command

2. **Message Selection**: `/rate` command intelligently selects messages:
   - Prioritizes messages with fewer ratings
   - Ensures each user sees different messages
   - Random selection among equal-priority candidates

3. **Rating Interface**:
   - **Log channel**: Moderators react with 1️⃣ No Flag | 2️⃣ Ambiguous | 3️⃣ Unconstructive | 4️⃣ Unsolicited | 5️⃣ N/A
   - **Public `/rate`**: Interactive button view with labels (No Flag, Ambiguous, Unconstructive, Unsolicited, Not Applicable)

4. **Session Management**:
   - 5-minute timeout per rating session
   - Each user can only rate each message once
   - Users can update their rating by rating again

**Dual Rating System**:
- **Public Rating Channel**: Community members use `/rate` command (button interface)
- **Private Log Channel**: Moderators rate via emoji reactions (1️⃣-5️⃣)

Both systems save to the same database and contribute to training.

## Feature Extraction

### LLM Features (15 total)

Extracted by LLM from message history context:

1. **`discusses_ellie`**: Whether message discusses the bot (0-1, used as filter)
2. **`familiarity_score`**: LLM's assessment of author-target familiarity (0-1)
3. **`tone_harshness_score`**: Harshness of tone (0-1)
4. **`positive_framing_score`**: Degree of positive framing (0-1)
5. **`includes_positive_takeaways`**: Presence of positive comments (0-1)
6. **`explains_why_score`**: Explanation quality (0-1)
7. **`actionable_suggestion_score`**: Actionability of suggestions (0-1)
8. **`context_is_feedback_appropriate`**: Feedback solicitation context (0-1)
9. **`target_uncomfortableness_score`**: Target's likely discomfort (0-1)
10. **`is_part_of_discussion`**: Message is part of ongoing discussion (0-1)
11. **`criticism_directed_at_image`**: Criticism targets uploaded image (0-1)
12. **`criticism_directed_at_statement`**: Criticism targets specific statement (0-1)
13. **`criticism_directed_at_generality`**: Criticism targets general behavior (0-1)
14. **`reciprocity_score`**: Reciprocal feedback relationship (0-1)
15. **`solicited_score`**: Degree to which feedback was solicited (0-1)

### Stat Features (3 total)

Computed from database statistics:

1. **`seniority_score_messages`**: Author/target message count ratio, normalized
   ```python
   ratio = author_msg_count / max(target_msg_count, 1)
   score = ratio / (ratio + 1.0)
   ```

2. **`seniority_score_characters`**: Author/target character count ratio, normalized
   ```python
   ratio = author_char_count / max(target_char_count, 1)
   score = ratio / (ratio + 1.0)
   ```

3. **`familiarity_score_stat`**: Co-occurrence count between author and target
   ```python
   score = co_occurrence_count / (co_occurrence_count + 10)
   ```

### Extraction Process

1. Format message history as numbered list with author names and content
2. Send to LLM with structured JSON schema
3. LLM identifies candidate messages by relative ID and extracts features
4. Look up author IDs from message history
5. Query database for seniority and familiarity statistics
6. Compute normalized stat features
7. Return combined 18-feature vectors

## Database Schema

### File Structure

```
excelsior.db (SQLite database)
```

### Core Tables

#### `flagged_messages`

Messages flagged by the bot.

**Columns**:
- `id` (Integer, PK): Auto-increment primary key
- `message_id` (BigInteger, Unique): Discord message ID
- `guild_id` (BigInteger): Discord guild ID
- `channel_id` (BigInteger): Discord channel ID
- `author_id` (BigInteger): Discord author ID
- `author_display_name` (String): Author's display name
- `author_username` (String): Author's username
- `content` (Text): Message content
- `context_message_ids` (JSON): List of message IDs used as context
- `context_messages` (JSON): Full serialized context messages (list of dicts)
- `timestamp` (DateTime): Original message timestamp
- `flagged_at` (DateTime): When message was flagged
- `target_user_id` (BigInteger, Nullable): Target user if message was directed at someone
- `target_display_name` (String, Nullable): Target user display name
- `target_username` (String, Nullable): Target user username

**Indexes**: `message_id` (unique)

#### `flagged_message_ratings`

Human ratings for flagged messages.

**Columns**:
- `id` (Integer, PK): Auto-increment primary key
- `rating_id` (String, Unique): Unique identifier for the rating
- `flagged_message_id` (BigInteger, FK): References `flagged_messages.message_id`
- `rater_user_id` (BigInteger): Discord user ID of rater
- `category` (Enum): RatingCategory (no-flag, ambiguous, unconstructive, unsolicited, NA)
- `target_user_id` (BigInteger, Nullable): Target attribution from rater
- `target_display_name` (String, Nullable): Target display name
- `target_username` (String, Nullable): Target username
- `started_at` (DateTime): Rating session start
- `completed_at` (DateTime, Nullable): Rating completion timestamp

**Indexes**: `rating_id` (unique)

#### `message_features`

Extracted LLM and stat features.

**Columns**:
- `id` (Integer, PK): Auto-increment primary key
- `message_id` (BigInteger, FK): References `flagged_messages.message_id`
- `extraction_run_id` (Integer, FK, Nullable): References `feature_extraction_runs.id` (NULL for runtime)
- `run_index` (Integer): Run number for this message (0-indexed)
- `features` (JSON): Dictionary of 18 features
- `target_username` (String, Nullable): Target username for stat feature refresh
- `created_at` (DateTime): Extraction timestamp

**Indexes**: `(extraction_run_id, message_id, run_index)` (unique)

#### `user_stats`

User activity statistics.

**Columns**:
- `id` (Integer, PK): Auto-increment primary key
- `user_id` (BigInteger, Unique): Discord user ID
- `username` (String): Username
- `display_name` (String, Nullable): Display name
- `message_count` (Integer): Total messages sent
- `character_count` (Integer): Total characters sent
- `join_timestamp` (DateTime, Nullable): When user joined server

**Indexes**: `user_id` (unique)

#### `user_co_occurrences`

User interaction frequency for familiarity scores.

**Columns**:
- `id` (Integer, PK): Auto-increment primary key
- `user_a_id` (BigInteger): First user ID (smaller numeric ID)
- `user_b_id` (BigInteger): Second user ID (larger numeric ID)
- `co_occurrence_count` (Integer): Number of co-occurrences in sliding windows

**Indexes**: `(user_a_id, user_b_id)` (unique)

**Note**: User pairs are always ordered (user_a_id < user_b_id) to avoid duplicates.

### Supporting Tables

#### `feature_extraction_runs`

Groups batch feature extractions for versioning.

**Columns**:
- `id` (Integer, PK): Auto-increment primary key
- `name` (String, Nullable): Human-readable name
- `provider` (String): LLM provider used
- `model_name` (String, Nullable): Model name
- `runs_per_message` (Integer): Number of extraction runs per message
- `message_count` (Integer): Number of messages with features
- `created_at` (DateTime): Run timestamp

#### `log_channel_rating_posts`

Maps log channel messages to flagged messages (enables reaction-based rating).

**Columns**:
- `id` (Integer, PK): Auto-increment primary key
- `bot_message_id` (BigInteger, Unique): Discord message ID in log channel
- `flagged_message_id` (BigInteger, FK): References `flagged_messages.message_id`
- `posted_at` (DateTime): Post timestamp

#### `dictionary_entries`

Community-specific terminology definitions.

**Columns**:
- `id` (Integer, PK): Auto-increment primary key
- `term` (String, Unique): Term or abbreviation
- `short_description` (String): Brief explanation
- `long_description` (Text, Nullable): Longer explanation

## Configuration Reference

All configuration is in `config.py`. Environment variables are loaded from `.env` using `python-dotenv`.

### Discord Settings

```python
DISCORD_BOT_TOKEN         # Bot token from Discord Developer Portal (required)
CHANNEL_ALLOW_LIST        # List of channel IDs to monitor, includes forum channels (required)
LOG_CHANNEL_ID            # Private mod channel for flagged message reviews (required)
RATING_CHANNEL_ID         # Public channel for rating system (required)
RATING_SYSTEM_LOG_FILE    # Filename for historical ratings (default: "rating_system_log.json")
WAIVER_ROLE_NAME          # Role name that exempts users from flagging (default: "Criticism Pass")
MODERATOR_ROLES           # List of moderator role names for restricted commands
```

**Channel Setup**:
- `LOG_CHANNEL_ID`: Private, moderator-only channel where flagged messages are posted for quick review
- `RATING_CHANNEL_ID`: Public/semi-public channel where community members help rate flagged messages
- Both channels can be used simultaneously for maximum coverage

### Moderation Settings

```python
MESSAGES_PER_CHECK        # Messages before auto-check (default: 30)
HISTORY_PER_CHECK         # Messages of history sent to LLM (default: 40)
SECS_BETWEEN_AUTO_CHECKS  # Idle timeout before auto-check (default: 240 seconds = 4 minutes)
REACTION_EMOJI            # Emoji added to flagged messages (default: 👁️)
```

**Tuning Guidelines**:
- Lower `MESSAGES_PER_CHECK` for more frequent checks (higher LLM costs)
- Higher `HISTORY_PER_CHECK` for more context (higher LLM costs, better accuracy)
- Lower `SECS_BETWEEN_AUTO_CHECKS` for faster response in slow channels

### LLM Settings

```python
DEFAULT_LLM_PROVIDER      # "cerebras", "openrouter", or "gemini" (default: "cerebras")
DEFAULT_LLM_MODEL         # Model name (default: "gpt-oss-120b")
CEREBRAS_API_KEY          # API key for Cerebras (from .env)
OPENROUTER_API_KEY        # API key for OpenRouter (from .env)
GEMINI_API_KEY            # API key for Google Gemini (from .env)
```

Only the API key for the active `DEFAULT_LLM_PROVIDER` is required at runtime.

**Provider Recommendations**:
- **Cerebras**: Best cost/performance for production (~$0.10/M tokens)
- **OpenRouter**: Good for testing different models (GPT-4, Claude, etc.)
- **Gemini**: Google's models, competitive pricing

### Training Settings

```python
NEW_RATINGS_BEFORE_RETRAIN                   # Ratings before auto-retrain (default: 20)
CONTINUOUS_TRAINING_FEATURE_MODE             # "existing_only" or "extract_on_demand"
CONTINUOUS_TRAINING_COLLAPSE_CATEGORIES      # Collapse to binary flag/no-flag (default: True)
CONTINUOUS_TRAINING_COLLAPSE_AMBIGUOUS       # Map ambiguous to no-flag (default: True)
```

**Training Modes**:
- **`existing_only`**: Fast, uses pre-extracted features from database (recommended)
- **`extract_on_demand`**: Extracts only missing features via LLM (expensive when many are missing)
  - Uses `flagged_messages` rows as source of truth
  - Falls back to Discord API context fetch when a row has empty/missing context

**Category Collapsing**:
- **Binary mode** (recommended): Simpler classification, works with <2000 ratings
- **Multi-class mode**: All 5 categories, requires substantial balanced training data

### Database Settings

```python
DB_FILE                   # SQLite database filename (default: "excelsior.db")
```

### Guidelines

```python
GUIDELINES                # Community guidelines text for LLM context
```

This text is included in LLM prompts to help the model understand community standards.

## Installation & Setup

### Prerequisites

- **Python 3.10+**
- **Discord bot token** with required intents:
  - Message Content Intent (required)
  - Server Members Intent (required)
  - Message/Reaction/Guild intents

### 1. Clone Repository

```bash
git clone <repository-url>
cd excelsior-moderator-2
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key Dependencies**:
- `py-cord`: Discord bot framework
- `lightgbm`: Gradient boosting classifier
- `sqlalchemy`: Database ORM
- `cerebras_cloud_sdk`: Cerebras LLM client
- `openai`: OpenRouter/OpenAI LLM client
- `google-genai`: Google Gemini client
- `scikit-learn`: ML utilities
- `numpy`, `pandas`: Data processing
- `matplotlib`, `seaborn`: Visualization
- `python-dotenv`: Environment variable management

### 3. Configure Environment

Create `.env` file in project root:

```bash
# Discord
DISCORD_BOT_TOKEN=your_discord_bot_token

# LLM API Keys (get at least one)
CEREBRAS_API_KEY=your_cerebras_key
OPENROUTER_API_KEY=your_openrouter_key
GEMINI_API_KEY=your_gemini_key
```

### 4. Configure Bot Settings

Edit `config.py`:

```python
# Set your channel IDs (get by enabling Developer Mode in Discord)
CHANNEL_ALLOW_LIST = [
    123456789012345678,  # channel-1
    234567890123456789,  # channel-2
]

LOG_CHANNEL_ID = 345678901234567890      # Private mod channel
RATING_CHANNEL_ID = 456789012345678901   # Public rating channel

# Set your moderator roles
MODERATOR_ROLES = [
    "Moderator",
    "Admin",
]

# Adjust thresholds if needed (defaults are usually fine)
MESSAGES_PER_CHECK = 30
HISTORY_PER_CHECK = 40
SECS_BETWEEN_AUTO_CHECKS = 240

# Choose LLM provider
DEFAULT_LLM_PROVIDER = "cerebras"  # or "openrouter" or "gemini"
DEFAULT_LLM_MODEL = "gpt-oss-120b"  # or other model
```

### 5. Bootstrap Initial Model

**If you have existing rated data** (e.g., from previous bot version):

```bash
python bootstrapping.py
```

In the REPL menu:
1. Select option `1` to run the full pipeline
2. Follow prompts to configure LLM provider and settings
3. Wait for feature extraction and training to complete (may take 10-60 minutes depending on data size)

The pipeline will:
- Load ratings from `data/rating_system_log.json`
- Fetch Discord context for each rated message
- Bootstrap user statistics from Discord history
- Extract features via LLM
- Train and evaluate the model
- Save model to `models/lightgbm_model.joblib`

**If starting fresh**:

Currently requires existing rated data. You'll need to:
1. Run the bot without a model (it won't flag anything)
2. Manually flag some messages using another method
3. Have moderators rate them
4. Once you have 50-100 ratings, run continuous training

### 6. Run the Bot

```bash
python bot.py
```

The bot will:
- Initialize database tables
- Load the trained model (if exists)
- Backfill message history for tracked channels (up to `max(HISTORY_PER_CHECK, MESSAGES_PER_CHECK)` per channel)
- Start moderation schedulers for each tracked channel
- Begin monitoring for new messages

**First Run Checklist**:
- ✅ Bot appears online in Discord
- ✅ Bot responds to `/ping`
- ✅ Console shows "Logged in as [BotName]"
- ✅ Console shows "Backfilled X messages for channel Y"
- ✅ No error messages in console

### 7. Test the Bot

1. **Test commands**:
   ```
   /ping         # Should respond with latency
   /info         # Should show bot information
   ```

2. **Test manual moderation** (requires moderator role):
   ```
   /check        # Should analyze recent messages in current channel
   ```

3. **Test rating system**:
   - Go to rating channel
   - Type `/rate`
   - Rate a flagged message
   - Check that rating is saved (use `/view_score`)

4. **Test automatic moderation**:
   - Send 30+ messages in a tracked channel
   - Bot should automatically run moderation check
   - Check console logs for feature extraction and prediction

## Development Guide

### Project Structure

```
excelsior-moderator-2/
├── bot.py                    # Main bot entry point
├── bootstrapping.py          # Initial training pipeline with REPL
├── config.py                 # Configuration and environment variables
├── database.py               # SQLAlchemy ORM models
├── db_config.py              # Database setup and session management
├── history.py                # In-memory message store
├── llms.py                   # LLM feature extraction
├── ml.py                     # LightGBM classifier
├── training.py               # Continuous training module
├── user_stats.py             # User statistics tracking
├── utils.py                  # Utility functions
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (not in git)
├── .gitignore                # Git ignore patterns
│
├── cogs/                     # Discord bot extensions
│   ├── events.py            # Message/reaction listeners
│   ├── rating.py            # Rating system
│   ├── restricted.py        # Admin commands
│   └── public.py            # Public commands
│
├── data/                     # Data files (not in git)
│   ├── rating_system_log.json      # Historical ratings for bootstrapping
│   ├── flagged_messages.json       # Cached flagged messages
│   └── rating_metadata.json        # Rating system state
│
├── models/                   # Trained models (not in git)
│   └── lightgbm_model.joblib       # Current production model
│
└── excelsior.db             # SQLite database (not in git)
```

### Adding New Features

To add a new feature to the model:

#### 1. Update LLM Schema (`llms.py`)

Add to `CANDIDATE_FEATURES_SCHEMA` under `properties.candidates.items.properties.features.properties`:

```python
# Inside the features object properties
"new_feature_name": {"type": "number"}
```

Add to the `required` list in the features object if the LLM must always provide it.
Update the system prompt in `extract_features_from_formatted_history` to instruct the LLM to extract the new feature.

#### 2. Update Feature Names (`ml.py`)

Add to `FEATURE_NAMES` list:

```python
FEATURE_NAMES = [
    # ... existing features ...
    "new_feature_name",
]
```

**Order matters!** Feature order must match the order used during training.

#### 3. Update Stats (if database-derived)

If the feature is computed from database statistics (not LLM-extracted):

1. Modify `user_stats.py` to track necessary data and add a getter (e.g., `get_new_stat()`)
2. Add computation logic inside `_augment_stat_features()` in `llms.py` (within `extract_features_from_formatted_history`)

Example:
```python
# Inside the loop over candidates in _augment_stat_features
if author_id is not None and target_id is not None:
    new_stat = get_new_stat(author_id, target_id, session)
    features["new_feature_name"] = new_stat
```

#### 4. Retrain Model

Re-run bootstrapping or continuous training with the new feature:

```bash
python bootstrapping.py
# Select option 5 (Train Model) or option 1 (Run Full Pipeline)
```

**Important**: You must re-extract features if adding a new LLM feature. Existing features in the database won't include the new field.

### Testing Changes

#### 1. Test Feature Extraction

```bash
python bootstrapping.py
```

Select option `4` (Extract Features) to test feature extraction with the new schema. Check console output for:
- Successful LLM responses
- Correct feature values in JSON output
- No missing or malformed features

#### 2. Test Model Training

```bash
python bootstrapping.py
```

Select option `5` (Train Model) to train with new features. Check:
- Feature matrix has correct number of columns
- No NaN values in features
- Model trains without errors
- Evaluation metrics (accuracy, precision, recall)

#### 3. Test Runtime Prediction

1. Start the bot:
   ```bash
   python bot.py
   ```

2. Trigger manual moderation in a test channel:
   ```
   /check
   ```

3. Check console logs for:
   - Feature extraction with new feature
   - Model prediction
   - Any errors or warnings

4. Verify flagged messages have the new feature in database:
   ```sql
   SELECT features FROM message_features WHERE extraction_run_id IS NULL ORDER BY created_at DESC LIMIT 1;
   ```

### Debugging

#### Enable Debug Logging

Set environment variable:
```bash
LOG_LEVEL=DEBUG
```

Or add to `.env` file:
```
LOG_LEVEL=DEBUG
```

This will show:
- Full LLM prompts and responses
- Feature extraction details
- Model prediction scores
- Database queries

#### Inspect Database

Use SQLite CLI:
```bash
sqlite3 excelsior.db
```

Useful queries:
```sql
-- View recent flagged messages
SELECT message_id, author_display_name, content, flagged_at 
FROM flagged_messages 
ORDER BY flagged_at DESC 
LIMIT 10;

-- View ratings for a message
SELECT r.category, r.rater_user_id, r.started_at
FROM flagged_message_ratings r
JOIN flagged_messages m ON r.flagged_message_id = m.message_id
WHERE m.message_id = 'DISCORD_MESSAGE_ID';

-- View runtime features
SELECT features 
FROM message_features 
WHERE extraction_run_id IS NULL 
ORDER BY created_at DESC 
LIMIT 1;

-- View user stats
SELECT user_id, message_count, character_count 
FROM user_stats 
ORDER BY message_count DESC 
LIMIT 10;

-- View co-occurrence counts
SELECT user_a_id, user_b_id, co_occurrence_count 
FROM user_co_occurrences 
ORDER BY co_occurrence_count DESC 
LIMIT 10;
```

#### Monitor LLM Calls

LLM extraction logs include:
- Full formatted message history sent to LLM
- LLM response (JSON)
- Parsing errors (if any)
- Feature values extracted

Check console output when `LOG_LEVEL=DEBUG`.

#### Inspect Runtime Features

Runtime features are saved to `message_features` table with `extraction_run_id=NULL`:

```python
from db_config import SessionLocal
from database import MessageFeatures

session = SessionLocal()
latest = session.query(MessageFeatures).filter_by(extraction_run_id=None).order_by(MessageFeatures.created_at.desc()).first()
print(latest.features)
```

### Code Style Guidelines

- Use descriptive variable and function names
- Add docstrings to all non-trivial functions
- Include type hints where helpful
- Comment complex logic
- Use list comprehensions and iterators where appropriate (but don't overdo it)

## Performance Considerations

### Message Store

- **In-memory deque** with max size = `HISTORY_PER_CHECK + MESSAGES_PER_CHECK` (default: 90)
- **Separate deque per channel/thread**
- **Automatic eviction** of oldest messages when capacity reached
- **Memory usage**: ~40–100 KB per channel (depends on `max(HISTORY_PER_CHECK, MESSAGES_PER_CHECK)`)

**Scaling**:
- 10 channels = ~1 MB
- 100 channels = ~10 MB
- Not a concern for most deployments

### LLM Costs

Feature extraction runs on every moderation check:
- **Frequency**: Every 30 messages or 4 minutes (whichever comes first)
- **Context size**: `HISTORY_PER_CHECK` messages (default 40) * ~100 tokens/message = ~4K tokens input
- **Output size**: ~1K tokens (JSON features)
- **Total**: ~7K tokens per check

**Cost Estimates** (with Cerebras at ~$0.10/M tokens):
- 1 check = ~$0.0007
- 100 checks/day = ~$0.07/day = ~$2/month
- 1000 checks/day = ~$0.70/day = ~$21/month

**Optimization Tips**:
- Use Cerebras for lowest cost
- Reduce `HISTORY_PER_CHECK` if context isn't critical
- Increase `MESSAGES_PER_CHECK` to reduce check frequency
- Use `existing_only` training mode to avoid re-extracting features

### Database

**SQLite Performance**:
- Sufficient for small-medium servers (<10,000 users)
- Fast indexed queries for user stats and co-occurrence lookups
- Single-file database (easy backup and migration)

**Bottlenecks**:
- Concurrent writes (SQLite limitation)
- Not an issue for bot's write patterns (sequential)

**Scaling to PostgreSQL**:
For large deployments (>10,000 users), consider PostgreSQL:
1. Update `db_config.py` to use PostgreSQL connection string
2. Install `psycopg2` driver
3. No code changes needed (SQLAlchemy handles it)

### Scheduling

- **One background task per tracked channel/thread**
- **Idle timer prevents unnecessary checks** when channels are inactive
- **Manual moderation bypasses schedulers** for immediate response

**Scaling**:
- 10 channels = 10 background tasks (minimal overhead)
- 100 channels = 100 background tasks (still fine, ~1 KB per task)
- Discord.py/py-cord handles asyncio efficiently

### Memory Usage Estimates

Component estimates for typical deployment:
- **Bot process**: ~50 MB base
- **Message stores** (10 channels): ~1 MB
- **Database connections**: ~5 MB
- **Model in memory**: ~10 MB
- **Total**: ~70-100 MB

## Troubleshooting

### Bot doesn't flag any messages

**Possible causes**:
1. **No model file**: Check that `models/lightgbm_model.joblib` exists
   - **Solution**: Run bootstrapping to create initial model
2. **Invalid LLM API key**: Check `.env` file and API key validity
   - **Solution**: Verify API key, check account balance
3. **LLM extraction failing**: Check console logs for errors
   - **Solution**: Enable debug logging, check LLM provider status
4. **Model always predicts "no-flag"**: Model may be undertrained
   - **Solution**: Collect more ratings, ensure class balance

**Debug steps**:
1. Enable debug logging: `LOG_LEVEL=DEBUG` in `.env`
2. Trigger manual check: `/check` in a channel
3. Check console for:
   - "Extracted features for N candidates"
   - "Model predictions: ..."
   - Any errors or exceptions
4. Verify model exists:
   ```bash
   ls -lh models/lightgbm_model.joblib
   ```

### Model always flags / never flags

**Possible causes**:
1. **Undertrained model**: Need more diverse training data
2. **Imbalanced training data**: Too many of one category
3. **Model overfitting**: Training on too few examples

**Solutions**:
1. **Collect more ratings**: Aim for at least 50-100 across all categories
2. **Check class balance**:
   - View leaderboard in rating channel (category breakdown)
   - Or query database:
```sql
SELECT category, COUNT(*) 
FROM flagged_message_ratings 
GROUP BY category;
```
3. **Retrain with balanced data**: Ensure reasonable distribution across categories
4. **Use binary mode**: If multi-class, switch to binary (flag/no-flag) in `config.py`

**Debug steps**:
1. Check model predictions in console logs (debug mode)
2. Look for prediction scores close to 0.0 or 1.0 (sign of overfitting)
3. Evaluate model with cross-validation (bootstrapping option 6)

### User stats not updating

**Possible causes**:
1. **`on_message` event not firing**: Check debug logs
2. **Channels not in `CHANNEL_ALLOW_LIST`**: Verify config
3. **Database write errors**: Check console for exceptions

**Solutions**:
1. **Enable debug logging**: Check if `on_message` fires for messages in tracked channels
2. **Verify channel IDs**: Check `CHANNEL_ALLOW_LIST` in `config.py`
3. **Bootstrap from history**: Run bootstrapping option 9 to backfill user stats:
   ```bash
   python bootstrapping.py
   # Select option 9: Collect user statistics from Discord
   ```
4. **Check database permissions**: Ensure `excelsior.db` is writable

**Debug steps**:
1. Send a message in tracked channel
2. Check console for "Updated user stats for user_id"
3. Query database:
   ```sql
   SELECT * FROM user_stats WHERE user_id = 'USER_ID';
   ```

### Rating system not working

#### For log channel reactions:

**Possible causes**:
1. **Wrong `LOG_CHANNEL_ID`**: Channel ID mismatch
2. **Missing moderator role**: Rater doesn't have required role
3. **Bot lacks permissions**: Can't add reactions

**Solutions**:
1. **Verify `LOG_CHANNEL_ID`**: Check `config.py`, enable Developer Mode in Discord to copy correct ID
2. **Check moderator roles**: Ensure rater has role from `MODERATOR_ROLES` list
3. **Check bot permissions**: Bot needs "Add Reactions" permission in log channel
4. **Check console logs**: Look for rating-related errors

#### For public `/rate` command:

**Possible causes**:
1. **Wrong `RATING_CHANNEL_ID`**: Channel ID mismatch
2. **No flagged messages**: Database empty
3. **Bot lacks permissions**: Can't send messages with buttons
4. **Command only works in rating channel**: User trying in wrong channel

**Solutions**:
1. **Verify `RATING_CHANNEL_ID`**: Check `config.py`
2. **Check database**: Ensure flagged messages exist:
   ```sql
   SELECT COUNT(*) FROM flagged_messages;
   ```
3. **Check bot permissions**: Bot needs "Send Messages", "Use Application Commands" in rating channel
4. **Try in correct channel**: `/rate` only works in the configured rating channel

**Debug steps**:
1. Enable debug logging
2. Use `/rate` command
3. Check console for:
   - "User requested rating"
   - "Selected message: ..."
   - Any errors
4. Query database:
   ```sql
   SELECT * FROM flagged_message_ratings WHERE rater_user_id = 'USER_ID' ORDER BY started_at DESC LIMIT 10;
   ```

### Out of memory

**Possible causes**:
1. **Too many tracked channels**: Each channel has separate message store
2. **`HISTORY_PER_CHECK` too high**: Large message buffers
3. **Memory leak**: Accumulating state over time

**Solutions**:
1. **Reduce `HISTORY_PER_CHECK`**: Lower context size (e.g., 30 instead of 40)
2. **Reduce tracked channels**: Remove unnecessary channels from `CHANNEL_ALLOW_LIST`
3. **Restart bot periodically**: Use process manager (systemd, PM2) to restart daily
4. **Monitor memory usage**: Use `htop` or `ps` to track memory over time

**Debug steps**:
1. Check memory usage:
   ```bash
   ps aux | grep bot.py
   ```
2. Profile memory:
   ```python
   import tracemalloc
   tracemalloc.start()
   # ... run bot ...
   snapshot = tracemalloc.take_snapshot()
   top_stats = snapshot.statistics('lineno')
   for stat in top_stats[:10]:
       print(stat)
   ```

### Continuous training fails

**Possible causes**:
1. **Not enough ratings**: Need minimum data for training
2. **Missing features**: Rated messages without extracted features
3. **Feature extraction errors**: LLM failures

**Solutions**:
1. **Collect more ratings**: Need at least 20 ratings (10 per class for binary)
2. **Use `extract_on_demand` mode**: Re-extract missing features
   - Set `CONTINUOUS_TRAINING_FEATURE_MODE = "extract_on_demand"` in `config.py`
   - Requires valid Discord access for rows without stored context
3. **Check console logs**: Look for extraction or training errors
4. **Manually trigger retraining**: Use `/retrain` command to see errors

**Debug steps**:
1. Check how many ratings exist:
   ```sql
   SELECT COUNT(*) FROM flagged_message_ratings;
   ```
2. Check how many have features:
   ```sql
   SELECT COUNT(DISTINCT m.message_id)
   FROM flagged_messages m
   JOIN message_features f ON m.message_id = f.message_id;
   ```
3. Enable debug logging and run `/retrain`

### LLM extraction errors

**Possible causes**:
1. **Invalid API key**: Expired or incorrect
2. **Rate limiting**: Too many requests
3. **Malformed response**: LLM didn't follow JSON schema
4. **Network issues**: Timeout or connection error

**Solutions**:
1. **Verify API key**: Check `.env` file, test key with provider's dashboard
2. **Retry logic**: Bot has built-in retries, check if transient
3. **Switch provider**: Try different LLM provider (Cerebras → OpenRouter)
4. **Check network**: Verify internet connection, proxy settings

**Debug steps**:
1. Enable debug logging
2. Check full LLM prompt and response in console
3. Verify JSON schema compliance
4. Test API key separately:
   ```python
   from llms import extract_features_from_formatted_history
   # Test extraction with sample data
   ```

### Database corruption

**Possible causes**:
1. **Unexpected shutdown**: Power loss, force kill
2. **Disk full**: No space for writes
3. **File system errors**: Disk corruption

**Solutions**:
1. **Restore from backup**: If you have one
2. **Repair database**:
   ```bash
   sqlite3 excelsior.db "PRAGMA integrity_check;"
   ```
3. **Rebuild from Discord history**: Re-run bootstrapping

**Prevention**:
- Regular backups: `cp excelsior.db excelsior.db.backup`
- Use systemd or PM2 for graceful shutdowns
- Monitor disk space

---

## Additional Resources

- **py-cord Documentation**: https://docs.pycord.dev/
- **LightGBM Documentation**: https://lightgbm.readthedocs.io/
- **SQLAlchemy Documentation**: https://docs.sqlalchemy.org/
- **Discord Developer Portal**: https://discord.com/developers/applications

## Support

For issues or questions:
1. Check this documentation
2. Enable debug logging and review console output
3. Inspect database with SQLite CLI
4. Review code comments in relevant modules

---

**Document Version**: 1.0  
**Last Updated**: 2026-02-09
