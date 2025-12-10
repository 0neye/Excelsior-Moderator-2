# Excelsior Moderator 2

A Discord moderation bot that uses machine learning and LLM feature extraction to automatically flag potentially unconstructive or unsolicited feedback in community channels. The bot learns from moderator ratings to continuously improve its accuracy.

## Overview

Excelsior Moderator 2 monitors Discord channels for messages that may violate community guidelines around constructive criticism. It uses a two-stage approach:

1. **Feature Extraction**: LLMs (Cerebras, OpenRouter, or Gemini) analyze message history to extract semantic features like tone harshness, positive framing, actionability, and context appropriateness.
2. **Classification**: A LightGBM classifier predicts whether messages should be flagged based on extracted features combined with user statistics.

When the bot flags a message, it:
- Adds a reaction emoji (👁️) to the flagged message
- Posts to a private log channel where moderators can rate the decision
- Stores the flagged message with context in the database

Moderators rate flagged messages into categories (no-flag, ambiguous, unconstructive, unsolicited, N/A), and the bot uses these ratings to retrain its model every 20 new ratings, creating a continuous improvement loop.

## Architecture

### Major Components

#### 1. **Bot Core (`bot.py`)**
The main Discord bot implementation using py-cord. Contains:
- **`ExcelsiorBot`**: Custom bot class that manages the message store, database sessions, and moderation scheduling
- **Message Store**: In-memory deque-based history buffer per channel (see `history.py`)
- **Channel Scheduling**: Automatic moderation triggers based on message count or idle timers
- **Moderation Workflow**: Coordinates feature extraction, model prediction, and message flagging

#### 2. **Database Layer**
- **`database.py`**: SQLAlchemy ORM models for all data structures
  - `FlaggedMessage`: Stores flagged messages with content and context
  - `FlaggedMessageRating`: Human ratings for continuous learning
  - `MessageFeatures`: Extracted LLM features (both bootstrapped and runtime)
  - `UserStats`: User activity metrics (message/character counts, join time)
  - `UserCoOccurrence`: Tracks user interaction frequency for familiarity scores
  - `FeatureExtractionRun`: Versioning for batch feature extraction runs
  - `LogChannelRatingPost`: Maps log channel messages to flagged messages
  - `DictionaryEntry`: Terminology definitions
- **`db_config.py`**: Database configuration and session management

#### 3. **Machine Learning**
- **`ml.py`**: LightGBM classifier implementation
  - Defines 18 features including tone scores, context appropriateness, seniority, and familiarity
  - Abstract base class `ModerationClassifier` with concrete `LightGBMClassifier`
  - Model serialization/deserialization with joblib
- **`training.py`**: Continuous training module
  - Loads rated messages and features from database
  - Supports two modes: `existing_only` (fast) and `extract_on_demand` (complete)
  - Retrains model when sufficient new ratings are collected
  - Automatically triggered by rating system

#### 4. **LLM Integration (`llms.py`)**
Extracts semantic features from message history using structured JSON output:
- Supports three providers: Cerebras, OpenRouter (OpenAI-compatible), Gemini
- Extracts 15 LLM-based features per candidate message
- Augments with 3 database-derived stat features (seniority and familiarity)
- Uses structured JSON schema for reliable output
- Implements rate limiting and timeout handling

#### 5. **User Statistics (`user_stats.py`)**
Tracks user behavior to compute contextual features:
- **Seniority scores**: Based on days since join, message count, and character count
- **Familiarity scores**: Co-occurrence statistics between message authors and targets
- **Real-time updates**: Incremented on every message in tracked channels
- **Co-occurrence windows**: Sliding 30-message window for interaction tracking
- **Bootstrapping**: Can backfill statistics from Discord history

#### 6. **Cogs** (Discord extensions in `cogs/`)
- **`events.py`**: Message/reaction listeners that update message store and user stats
- **`rating.py`**: Rating system commands (`/rate`, `/view_score`) and reaction handlers
- **`restricted.py`**: Admin commands for manual moderation and retraining
- **`public.py`**: Public utility commands (`/ping`, `/info`)

#### 7. **Bootstrapping (`bootstrapping.py`)**
Full training pipeline for initial model development:
- Load rated messages from `rating_system_log.json`
- Fetch Discord context around each rated message
- Extract features via LLM with multiple runs for robustness
- Train and evaluate models with cross-validation
- Interactive REPL menu for running partial or full pipelines
- Saves state between runs to avoid re-extracting features

### Data Flow

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

**How it works**:
- Each tracked channel/thread gets a `ChannelModerationState` with counters and timers
- `on_message` event increments `messages_since_check` and starts idle timer
- `_moderation_scheduler` runs in background for each channel:
  - Waits for either `MESSAGES_PER_CHECK` messages (default: 30) or `SECS_BETWEEN_AUTO_CHECKS` seconds (default: 240)
  - Triggers `moderate_channel()` when threshold met
  - Resets counters after successful moderation

**Trigger conditions**:
- **Message count**: After N new messages since last check
- **Idle timer**: After M seconds of inactivity with new unchecked messages
- **Manual**: Via `/check` command

### 2. Message Flagging Logic

**Components**: `bot.py` (`moderate_channel`), `llms.py` (`get_candidate_features`)

**How it works**:
1. **Get candidates**: LLM analyzes last `HISTORY_PER_CHECK` messages (default: 60) and identifies potential feedback candidates
2. **Filter candidates**:
   - Remove messages with `discusses_ellie > 0.2` (bot-related noise)
   - Remove messages without Discord IDs
   - Remove messages from users with the "Criticism Pass" role
   - Remove messages too close to channel start (insufficient context)
3. **Extract features**: LLM provides 15 semantic features per candidate
4. **Augment with stats**: Add 3 database-derived features (seniority, familiarity)
5. **Predict**: LightGBM classifier predicts "flag" or "no-flag"
6. **Flag actions**:
   - Add 👁️ reaction to original message
   - Post to log channel with rating buttons
   - Save to `FlaggedMessage` table
   - Save runtime features to `MessageFeatures` (with `extraction_run_id=NULL`)

### 3. Feature Extraction Logic

**Components**: `llms.py` (`extract_features_from_formatted_history`), `user_stats.py`

**LLM Features** (15 total):
- `discusses_ellie`: Whether message discusses the bot (filter)
- `familiarity_score`: LLM's assessment of author-target familiarity
- `tone_harshness_score`: Harshness of tone (0-1)
- `positive_framing_score`: Degree of positive framing (0-1)
- `includes_positive_takeaways`: Presence of positive comments (0-1)
- `explains_why_score`: Explanation quality (0-1)
- `actionable_suggestion_score`: Actionability of suggestions (0-1)
- `context_is_feedback_appropriate`: Feedback solicitation context (0-1)
- `target_uncomfortableness_score`: Target's likely discomfort (0-1)
- `is_part_of_discussion`: Message is part of ongoing discussion (0-1)
- `criticism_directed_at_image`: Criticism targets uploaded image (0-1)
- `criticism_directed_at_statement`: Criticism targets specific statement (0-1)
- `criticism_directed_at_generality`: Criticism targets general behavior (0-1)
- `reciprocity_score`: Reciprocal feedback relationship (0-1)
- `solicited_score`: Degree to which feedback was solicited (0-1)

**Stat Features** (3 total):
- `seniority_score_messages`: Normalized by message count in server
- `seniority_score_characters`: Normalized by character count in server
- `familiarity_score_stat`: Co-occurrence count between author and target, normalized

**Process**:
1. Format message history as numbered list with author names and content
2. Send to LLM with structured JSON schema
3. LLM identifies candidate messages by relative ID and extracts features
4. Look up author IDs from message history
5. Query database for seniority (join date, message/char counts) and familiarity (co-occurrence)
6. Compute normalized stat features using sigmoid-like functions
7. Return combined feature vectors

### 4. Stat Features Calculation

**Components**: `user_stats.py`

**Data Sources**:
- **`UserStats` table**: Tracks per-user message count, character count, and join timestamp
- **`UserCoOccurrence` table**: Tracks co-occurrence counts between user pairs

**Seniority Scores**:
```python

# Based on message activity
message_score = message_count / (message_count + 100)
character_score = character_count / (character_count + 5000)

# Combined into seniority_score_messages and seniority_score_characters
```

**Familiarity Score**:
```python
# Based on co-occurrence count in rolling 30-message windows
familiarity = co_occurrence_count / (co_occurrence_count + 10)
```

**Real-time Updates**:
- `on_message` event calls `update_user_stats_from_message()` to increment counts
- `update_co_occurrences_from_window()` processes last 30 messages to update pairs
- All updates committed to database immediately

### 5. Database Updates

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
- Ratings trigger continuous training after `NEW_RATINGS_BEFORE_RETRAIN` (default: 20)

### 6. Model Retraining

**Components**: `training.py`, `cogs/rating.py`

**Trigger**: Automatically after N new ratings (configured in `config.py`)

**Process**:
1. Load all rated messages from `FlaggedMessageRating` table
2. Load features based on mode:
   - **`existing_only`** (default): Load from `MessageFeatures` table (no LLM cost)
   - **`extract_on_demand`**: Extract missing features via LLM
3. Prepare training data:
   - Build feature matrix X with 18 features
   - Build label vector y with rating categories
4. Train LightGBM classifier with balanced class weights
5. Save model to `models/lightgbm_model.joblib`
6. Model is immediately used by bot (loaded on next moderation run)

**Feature Sources** (priority order):
1. Runtime features (no `extraction_run_id`)
2. Most recent extraction run features

**Model Configuration**:
- 200 estimators, max depth 6, learning rate 0.1
- Balanced class weights to handle imbalanced categories
- Multi-class objective for 5-category classification

### 7. Bootstrapping

**Components**: `bootstrapping.py`

**Purpose**: Initial model training from historical rated data

**Full Pipeline**:
1. **Load Ratings**: Read `rating_system_log.json` with human-rated flagged messages
2. **Fetch Context**: Download Discord context messages around each rated message
3. **Bootstrap Stats**: Backfill user statistics from Discord history
4. **Extract Features**: Run LLM feature extraction with configurable runs per message
5. **Train Model**: Train LightGBM classifier on extracted features
6. **Evaluate**: Cross-validation and confusion matrix analysis

**Interactive REPL**:
```
1) Load Rating Data
2) Fetch Discord Context
3) Bootstrap User Stats
4) Extract Features (LLM)
5) Train Model
6) Evaluate Model
7) Run Full Pipeline
8) Save State
9) Load State
0) Exit
```

**State Management**:
- Saves progress to `bootstrap_state.json` to avoid re-running expensive steps
- Can resume from any point in pipeline
- Supports multiple LLM providers and models

**Statistics Bootstrapping**:
- Scans up to 10,000 messages per tracked channel
- Builds `UserStats` with join dates, message counts, character counts
- Builds `UserCoOccurrence` with 30-message sliding window co-occurrence counts
- Takes ~1 hour for large servers but only needs to run once

## Configuration

All configuration is in `config.py`:

```python
# Discord
DISCORD_BOT_TOKEN        # Bot token from Discord Developer Portal
CHANNEL_ALLOW_LIST       # List of channel IDs to monitor (includes forum channels)
LOG_CHANNEL_ID           # Private channel for flagged message reviews
RATING_CHANNEL_ID        # Public channel for rating system instructions
WAIVER_ROLE_NAME         # Role name that exempts users from flagging ("Criticism Pass")

# Moderation
MESSAGES_PER_CHECK       # Messages before auto-check (default: 30)
HISTORY_PER_CHECK        # Messages of history sent to LLM (default: 60)
SECS_BETWEEN_AUTO_CHECKS # Idle timeout before auto-check (default: 240)
REACTION_EMOJI           # Emoji added to flagged messages (default: 👁️)

# LLM
DEFAULT_LLM_PROVIDER     # "cerebras", "openrouter", or "gemini" (default: "cerebras")
DEFAULT_LLM_MODEL        # Model name (default: "gpt-oss-120b")
CEREBRAS_API_KEY         # API key for Cerebras
OPENROUTER_API_KEY       # API key for OpenRouter
GEMINI_API_KEY           # API key for Google Gemini

# Training
NEW_RATINGS_BEFORE_RETRAIN           # Ratings before auto-retrain (default: 20)
CONTINUOUS_TRAINING_FEATURE_MODE     # "existing_only" or "extract_on_demand"

# Database
DB_FILE                  # SQLite database filename (default: "excelsior.db")

# Guidelines
GUIDELINES               # Community guidelines text for context
```

Environment variables are loaded from `.env` file using `python-dotenv`.

## Installation & Setup

### Prerequisites
- Python 3.10+
- Discord bot token with required intents:
  - Message Content Intent (required)
  - Server Members Intent (required)
  - Message/Reaction/Guild intents

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key dependencies**:
- `py-cord`: Discord bot framework
- `lightgbm`: Gradient boosting classifier
- `sqlalchemy`: Database ORM
- `cerebras_cloud_sdk`, `openai`, `google-genai`: LLM clients
- `scikit-learn`: ML utilities
- `numpy`, `pandas`, `matplotlib`, `seaborn`: Data analysis

### 2. Configure Environment

Create `.env` file:
```bash
DISCORD_BOT_TOKEN=your_discord_bot_token
CEREBRAS_API_KEY=your_cerebras_key
OPENROUTER_API_KEY=your_openrouter_key
GEMINI_API_KEY=your_gemini_key
```

### 3. Configure Bot Settings

Edit `config.py`:
- Set `CHANNEL_ALLOW_LIST` to your channel IDs (channels to monitor)
- Set `LOG_CHANNEL_ID` to your private mod channel (for moderator ratings)
- Set `RATING_CHANNEL_ID` to your public rating channel (for community participation)
- Set `MODERATOR_ROLES` to your server's moderator role names
- Adjust thresholds and LLM provider as needed

**Channel Setup Notes**:
- `LOG_CHANNEL_ID`: Should be private, moderator-only channel where flagged messages are posted for quick review
- `RATING_CHANNEL_ID`: Should be public or semi-public channel where community members can help rate flagged messages
- Both channels can be used simultaneously for maximum rating coverage

### 4. Bootstrap Initial Model

**If you have existing rated data**:
```bash
python bootstrapping.py
```
Follow the REPL menu to run the full pipeline (option 7).

**If starting fresh**:
Currently not supported.

### 5. Run the Bot

```bash
python bot.py
```

The bot will:
- Initialize database tables
- Backfill message history for tracked channels (up to 90 messages per channel)
- Start moderation schedulers for each tracked channel
- Begin monitoring for new messages

## Bot Commands

### Public Commands
**General** (`cogs/public.py`):
- `/ping` - Check the bot's latency
- `/info` - Get information about the bot

**Rating System** (`cogs/rating.py`):
- `/rate` - Get a random flagged message to rate (helps train the model, only works in rating channel)
- `/view_score` - View your personal rating statistics

**Note**: The leaderboard is automatically updated in the rating channel and doesn't have a dedicated command.

### Moderator Commands (`cogs/restricted.py`)
These commands require moderator roles configured in `MODERATOR_ROLES`:
- `/check` - Manually trigger moderation for the current channel or thread
- `/mod_ping` - Test moderator permissions
- `/mod_info` - View moderator role information

### Admin Commands (`cogs/restricted.py`)
These commands require the admin role (`Custodian (admin)`):
- `/retrain` - Manually trigger model retraining (bypasses rating count threshold)

### Not Yet Implemented
The following functionality exists in code but not as slash commands:
- Bootstrap user statistics - Run via `user_stats.py` module or add command
- Refresh stat features - Can be implemented as admin command if needed

## Public Rating System

The bot includes a community-driven rating system that allows users to help train the moderation model by rating flagged messages.

### Overview

**Purpose**: Enable community members to provide feedback on the bot's flagging decisions, creating training data for continuous model improvement.

**Location**: Dedicated rating channel (configured via `RATING_CHANNEL_ID`)

**Access**: Open to all community members with a certain role (not restricted to moderators)

### How It Works

1. **Rating Channel Setup**: The bot maintains a public rating channel with:
   - Pinned instructions explaining the rating categories
   - Live leaderboard showing top contributors
   - Interactive `/rate` command for users to participate

2. **Getting Messages to Rate**: Users invoke `/rate` in the rating channel
   - Bot selects a random flagged message that the user hasn't rated yet
   - Priority given to messages with fewer existing ratings (ensures broad coverage)
   - Displays message content, author, context link, and rating buttons

3. **Rating Interface**: Interactive button view with 5 categories:
   - **✅ No Flag** (Green): Message is fine, shouldn't have been flagged
   - **❓ Ambiguous** (Gray): Unclear whether message violates guidelines
   - **⚠️ Unconstructive** (Red): Message is harsh/negative without constructive elements
   - **📢 Unsolicited** (Red): Feedback given without being requested
   - **🚫 N/A** (Blue): Message doesn't fit any category or was deleted

4. **Rating Submission**: 
   - User clicks a category button
   - Rating saved to database immediately
   - Leaderboard updates automatically
   - After 20 new ratings, model retrains automatically (configurable)

5. **Session Management**:
   - 5-minute timeout per rating session
   - Each user can only rate each message once
   - Users can update their rating by rating the same message again

### Rating Categories Explained

**No Flag**: Use when the bot incorrectly flagged a message that:
- Contains constructive feedback following guidelines
- Is part of a friendly discussion
- Was consensual/solicited
- Has positive framing and actionable suggestions

**Ambiguous**: Use when it's genuinely unclear, such as:
- Borderline tone (could be interpreted multiple ways)
- Missing context to make proper judgment
- Technical criticism that may or may not be appropriate
- Mixed signals (some positive, some negative)

**Unconstructive**: Use when feedback violates guidelines by:
- Being harsh or negative without positive elements
- Lacking explanation of *why* or *how* to improve
- Focusing only on problems without suggestions
- Not recognizing the human on the receiving end

**Unsolicited**: Use when feedback was given without being requested:
- Critique on casual sharing (not asking for feedback)
- Advice given to someone who didn't ask
- Corrections in inappropriate contexts
- "Well actually" responses

**NA**: Use when the message is outside the bounds of the feedback guidelines:
- General discussion
- Meta-discussion
- Criticism at generalities, the game, etc.

### Leaderboard

The leaderboard tracks contribution statistics:
- **Rank**: Position among all raters
- **Total Ratings**: Number of messages rated
- **Coverage**: Percentage of all flagged messages rated
- **Category Breakdown**: Distribution across rating categories

The leaderboard is automatically updated and displayed in the rating channel after each rating submission (top 10 contributors). Individual users can view their personal statistics using `/view_score`.

### Rating Quality

**Dual Rating System**: The bot supports two rating interfaces:
1. **Public Rating Channel**: Community members use `/rate` command
2. **Private Log Channel**: Moderators can rate via emoji reactions (faster for active mods)

Both systems save to the same database and contribute to training data.

**Rating Coverage Strategy**: The `/rate` command intelligently selects messages:
- Prioritizes messages with fewer ratings (avoid redundancy)
- Ensures each user sees different messages
- Balances coverage across all flagged messages
- Random selection among equal-priority candidates

### Continuous Training Integration

Ratings directly feed the model improvement loop:
1. User submits rating via button or moderator reacts in log channel
2. Rating saved to `flagged_message_ratings` table
3. Rating counter increments (`new_ratings_since_retrain`)
4. When counter reaches `NEW_RATINGS_BEFORE_RETRAIN` (default: 20):
   - Automatic retraining triggered in background
   - Features loaded from database (no LLM cost)
   - New model trained on all ratings
   - Model saved and used immediately on next moderation run
   - Counter resets to 0

**Benefits**:
- Model adapts to community norms over time
- Captures edge cases and evolving guidelines
- Democratizes moderation decisions
- Creates engagement around community standards

### Privacy & Moderation

- Rated messages remain in the database even after Discord deletion
- Rating data is anonymous in logs (only user ID stored)
- Moderators can see individual ratings in the database
- Users cannot see others' ratings (prevents bias)
- No public display of "who rated what"

## File Structure

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

## Database Schema

### Core Tables

**`flagged_messages`**: Messages flagged by the bot
- Includes author info, content, timestamps, context messages
- Unique constraint on `message_id`

**`flagged_message_ratings`**: Human ratings for flagged messages
- Links to `flagged_messages` via `flagged_message_id`
- Stores category (no-flag, ambiguous, unconstructive, unsolicited, N/A)
- Tracks rater and timestamps

**`message_features`**: Extracted LLM features
- Links to `flagged_messages` via `message_id`
- Links to `feature_extraction_runs` via `extraction_run_id` (NULL for runtime)
- Stores feature dict as JSON
- Supports multiple runs per message (`run_index`)

**`user_stats`**: User activity statistics
- Tracks message count, character count, join timestamp
- Updated in real-time on every message

**`user_co_occurrences`**: User interaction frequency
- Stores co-occurrence counts for user pairs
- Updated via 30-message sliding window
- Unique constraint on ordered pair (user_a_id < user_b_id)

### Supporting Tables

**`feature_extraction_runs`**: Groups batch feature extractions
- Tracks LLM provider, model, runs per message
- Used for versioning and reproducibility

**`log_channel_rating_posts`**: Maps log messages to flagged messages
- Enables reaction-based rating in log channel

**`dictionary_entries`**: Terminology definitions
- Stores community-specific terms and abbreviations

## Development Notes

### Adding New Features

To add a new feature to the model:

1. **Update LLM Schema** (`llms.py`):
   - Add to `CANDIDATE_FEATURES_SCHEMA`
   - Update prompt to extract the new feature

2. **Update Feature Names** (`ml.py`):
   - Add to `FEATURE_NAMES` list

3. **Update Stats** (if database-derived):
   - Modify `user_stats.py` to compute new stat
   - Add to `_augment_stat_features()` in `llms.py`

4. **Retrain Model**:
   - Re-run bootstrapping or continuous training with new feature

### Testing Changes

1. **Test feature extraction**:
   ```bash
   python bootstrapping.py
   # Select option 4 to extract features with new schema
   ```

2. **Test model training**:
   ```bash
   python bootstrapping.py
   # Select option 5 to train with new features
   ```

3. **Test runtime prediction**:
   - Run bot and trigger manual moderation: `/check`
   - Check logs for feature extraction and prediction

### Debugging

- **Enable debug logging**: Set `LOG_LEVEL=DEBUG` in `.env`
- **Check database**: Use `sqlite3 excelsior.db` to inspect tables
- **Monitor LLM calls**: LLM extraction logs full prompts and responses
- **Feature inspection**: Runtime features are saved to `message_features` table

## Performance Considerations

### Message Store
- In-memory deque with max size = `HISTORY_PER_CHECK + MESSAGES_PER_CHECK` (default: 90 messages)
- Separate deque per channel/thread
- Automatic eviction of oldest messages when capacity reached

### LLM Costs
- Feature extraction runs on every moderation check (every 30 messages or 4 minutes)
- Cerebras is recommended for low cost (~$0.10 per million tokens)
- Use `extract_on_demand` training mode sparingly (re-extracts all features)

### Database
- SQLite is sufficient for small-medium servers (<10,000 users)
- User stats and co-occurrence updates are fast (indexed queries)
- Consider PostgreSQL for large deployments (>10,000 users)

### Scheduling
- One background task per tracked channel/thread
- Idle timer prevents unnecessary checks when channels are inactive
- Manual moderation bypasses schedulers for immediate response

## Troubleshooting

### Bot doesn't flag any messages
- Check that model exists: `models/lightgbm_model.joblib`
- Run bootstrapping to create initial model
- Check LLM API keys are valid
- Enable debug logging to see feature extraction

### Model always flags/never flags
- Model may be undertrained (need more ratings)
- Check class balance in ratings (view leaderboard in rating channel or check database)
- Retrain with at least 50-100 ratings across all categories

### User stats not updating
- Check that `on_message` event is firing (enable debug logs)
- Verify channels are in `CHANNEL_ALLOW_LIST`
- Run bootstrapping script to backfill from history (see `user_stats.py` module)

### Rating system not working
- **For log channel reactions**: 
  - Verify `LOG_CHANNEL_ID` is set correctly
  - Check moderator has required role from `MODERATOR_ROLES`
  - Ensure bot has permission to add reactions in log channel
- **For public `/rate` command**:
  - Verify `RATING_CHANNEL_ID` is set correctly
  - Command only works in the configured rating channel
  - Check that flagged messages exist in database to rate
  - Ensure bot has permission to send messages with buttons
- Check `flagged_message_ratings` table for saved ratings

### Out of memory
- Reduce `HISTORY_PER_CHECK` to decrease message store size
- Reduce number of tracked channels
- Restart bot periodically to clear accumulated state

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

## Credits

Developed for the Excelsior community to maintain constructive and welcoming feedback culture.
