# Discord Message Classification - Moderation Helper

The overall goal of this project is to create a discord bot that can accurately flag unconstructive/unsolicited feedback in gaming discussions.
The pipeline will be as follows:

Discord messages in channel + member attribute profiles + growing dictionary of lingo definitions -> context packet.
Context packet -> LLM -> answers to a list of specific categorical and numerical questions about aspects of the situation, for each candidate message that could violate feedback guidelines.
Answers to questions + potential minimal other stats -> logistic regression or small MLP for categorical prediction of FLAG | NO_FLAG | AMBIGUOUS.

For data we already have: we have a small list of ~40 flagged messages that have been rated by human moderators as either being correctly flagged or incorrectly flagged. We also have a large list of unrated message flags which could potentially be rated after we implement the ability for mods to answer with "ambiguous".

## Packages and APIs
We'll use all the packages in requirements.txt. Note that we will be using pycord v3, not discord.py.
For our LLM provider, we will use Cerebras through their SDK.
For the DB we will use sqlalchemy, and for embeddings we'll use chromadb.


## General plan

The bot will get a list of channel IDs to watch. It will use the raw on_message event to check for new messages and add them to the DB and in-memory history for that channel/thread in channel. It will also keep track of a counter that determines the number of messages since it last called the "check" function to run the workflow, as well as the timestamp of the last message sent in that channel. If the time goes over a certain amount specified in the config file, it will trigger a check. It will also trigger a check if the number of messages since the last check hits the number specified in config.

Config will be a yaml file loaded at the beginning of the program running. It will include stuff like the number of messages per channel since the last check before running one, the amount of time since the last message in a channel before running one, the channels to monitor, the number of messages in the message history to keep in memory and use for context, the DB path, etc.

## Architecture and worker model

- Background worker runs the "check" workflow off the gateway thread using an async in-memory queue (Redis later)
- Per-channel/thread state keeps `messages_since_last_check`, `last_message_at`, `last_check_at`, `last_processed_message_id`
- Coalesce by channel: only one pending job per channel; new triggers update the pending job's parameters
- Idempotency via `run_id` and `flag_status_version` so retries/reruns never double-react or double-log
- Persist `last_processed_message_id` per channel/thread to survive restarts

## Trigger policy and debounce

- Run a check when:
  - `messages_since_last_check >= config.message_count_threshold`
  - OR `now - last_message_at >= config.idle_seconds_threshold`
  - AND `now - last_check_at >= config.cooldown_seconds`
- Edits are debounced 2–5 seconds to batch rapid edits; deletions always trigger a rerun
- On startup, backfill per tracked channel/thread up to the last stored `message_id` and initialize state

**For the "check" workflow:**
First we call the necessary functions to create a context packet. We will check our DB for all the users present in the history of the current conversation and pull their attributes (stuff like sarcasm_rate, banter_tolerance, false_positive_history) to include in the context. Next we'll pull relevant dictionary entries based on keyword matches and potentially chroma embedding distance and include that in the context. Of course finally we'll format the message history itself as more context.

In total the context packet will contain:
- Feedback guidelines from server rules
- Message history (from in-memory cache of current channel/thread)
- Participant user attributes
- Dictionary with detected lingo/terms

Next we'll use this context packet to ask a series of very specific questions about the message history. First, we'll ask it to identify potential candidate messages, then reason about the questions for each one, and finally output json with all the answers for all candidates. We will use a reasoning model for this and a single API call. Some examples of questions we might ask include: 
- is_direct_address (true/false)
- target_user (username)
- sarcasm_marker_present (explicit|implicit|none)
- target_objection_present (true/false)
- power_gap (old_timer→newcomer|peer|unknown)
- preliminary_flag_percent (float; likelyhood of being flagged)
- unknown_terms (list of strings)
etc.

These will be included in the instructions portion of the LLM system prompt.

## LLM I/O contract (strict JSON)

- Require a strict JSON response conforming to this schema; validate and auto-repair on parse failure

```json
{
  "type": "object",
  "properties": {
    "candidates": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "message_id": { "type": "string" },
          "is_direct_address": { "type": "boolean" },
          "target_user_anon": { "type": "string", "nullable": true },
          "sarcasm_marker_present": { "type": "string", "enum": ["explicit", "implicit", "none"] },
          "target_objection_present": { "type": "boolean" },
          "power_gap": { "type": "string", "enum": ["old_timer_to_newcomer", "peer", "unknown"] },
          "preliminary_flag_percent": { "type": "number" },
          "unknown_terms": { "type": "array", "items": { "type": "string" } }
        },
        "required": [
          "message_id",
          "is_direct_address",
          "sarcasm_marker_present",
          "target_objection_present",
          "power_gap",
          "preliminary_flag_percent",
          "unknown_terms"
        ]
      }
    }
  },
  "required": ["candidates"]
}
```

- Context bounds: cap history to `config.max_history_messages`; summarize earlier turns if exceeded
- Injection safety: delimit user content; never treat message text as instructions

Finally, with the answers to these questions + the data from user profiles, we'll use a logistic regression or small MLP to predict the category of each candidate message being flagged; either (flagged, not_flagged, ambiguous).

Candidate messages that are considered ambiguous will be sent to a mod channel for human confirmation before flagging. Ones in the flagged category get flagged, and the rest not flagged.
Flagged messages are sent as logs in the mod channel, and the original message gets a reaction emoji added to it publicly. Ambiguous messages only are reacted to publicly once they are manually confirmed by a human moderator.

## Modeling and thresholds

- Start with logistic regression baseline; add a small MLP later as needed
- Calibrate probabilities (isotonic preferred or Platt) and version calibration in `model_runs`
- Decision band configured per server:
  - Flag if p >= `thresholds.t_high`
  - No-flag if p <= `thresholds.t_low`
  - Ambiguous otherwise

**Continuous training and DB updates:**
Once messages are flagged and sent to a mod channel in the discord server, human moderators can rate whether they think the flag was correct, incorrect, or ambiguous. This includes ones predicted as ambiguous and manually confirmed by a human mod.
Once this happens, it is added to a dataset in the DB with the number of human ratings in each category and the model prediction.
Then the mlp model or logistic regression is retrained on this new dataset. Train/test split is possible too, if it would be beneficial.

## Database schema (initial)

```sql
-- messages
CREATE TABLE messages (
  message_id TEXT PRIMARY KEY,
  channel_id TEXT NOT NULL,
  author_discord_id TEXT NOT NULL,
  created_at TIMESTAMP NOT NULL,
  edited_at TIMESTAMP,
  content TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  is_deleted BOOLEAN NOT NULL DEFAULT FALSE
);
CREATE INDEX idx_messages_channel_time ON messages(channel_id, created_at);

-- users
CREATE TABLE users (
  discord_id TEXT PRIMARY KEY,
  anon_id INTEGER UNIQUE NOT NULL,
  first_seen TIMESTAMP NOT NULL,
  last_seen TIMESTAMP NOT NULL
);

-- user_attributes
CREATE TABLE user_attributes (
  discord_id TEXT NOT NULL,
  name TEXT NOT NULL,
  value REAL NOT NULL,
  weight REAL NOT NULL,
  updated_at TIMESTAMP NOT NULL,
  PRIMARY KEY (discord_id, name),
  FOREIGN KEY (discord_id) REFERENCES users(discord_id)
);

-- lingo_terms
CREATE TABLE lingo_terms (
  term_id INTEGER PRIMARY KEY AUTOINCREMENT,
  term TEXT UNIQUE NOT NULL,
  definition TEXT NOT NULL,
  confidence REAL NOT NULL,
  updated_at TIMESTAMP NOT NULL
);

-- lingo_questions
CREATE TABLE lingo_questions (
  question_id INTEGER PRIMARY KEY AUTOINCREMENT,
  term TEXT NOT NULL,
  source_message_id TEXT NOT NULL,
  resolved BOOLEAN NOT NULL DEFAULT FALSE,
  resolver_discord_id TEXT,
  resolved_at TIMESTAMP,
  FOREIGN KEY (source_message_id) REFERENCES messages(message_id)
);

-- model_runs
CREATE TABLE model_runs (
  run_id TEXT PRIMARY KEY,
  started_at TIMESTAMP NOT NULL,
  finished_at TIMESTAMP,
  model_name TEXT NOT NULL,
  prompt_hash TEXT NOT NULL,
  context_digest TEXT NOT NULL,
  status TEXT NOT NULL
);

-- flags
CREATE TABLE flags (
  flag_id TEXT PRIMARY KEY,
  message_id TEXT NOT NULL,
  run_id TEXT NOT NULL,
  prediction TEXT NOT NULL, -- flag | no_flag | ambiguous
  score REAL NOT NULL,
  flag_status TEXT NOT NULL, -- active | cleared
  created_at TIMESTAMP NOT NULL,
  FOREIGN KEY (message_id) REFERENCES messages(message_id),
  FOREIGN KEY (run_id) REFERENCES model_runs(run_id)
);
CREATE INDEX idx_flags_message ON flags(message_id);
CREATE INDEX idx_flags_status ON flags(flag_status);

-- moderator_reviews
CREATE TABLE moderator_reviews (
  review_id TEXT PRIMARY KEY,
  flag_id TEXT NOT NULL,
  reviewer_discord_id TEXT NOT NULL,
  label TEXT NOT NULL, -- correct | incorrect | ambiguous
  notes TEXT,
  created_at TIMESTAMP NOT NULL,
  FOREIGN KEY (flag_id) REFERENCES flags(flag_id)
);
```

## Config (typed, env overrides)

```yaml
discord_token: ${DISCORD_TOKEN}
mod_channel_id: "123456789012345678"
channels_to_monitor:
  - "111111111111111111"
  - "222222222222222222"

message_count_threshold: 12
idle_seconds_threshold: 45
cooldown_seconds: 20
max_history_messages: 60

model:
  provider: "cerebras"
  name: "gpt-oss-120b"
  temperature: 0.2
  max_tokens: 6000
  timeout_seconds: 30
  retries: 2

thresholds:
  t_low: 0.35
  t_high: 0.7

database_url: "sqlite:///./excelsior.db"
reaction_emoji: "🛑"
```

- Validate at startup with pydantic; fail fast if secrets or required fields are missing

**Hyperparameter tuning:**
...

**Dictionary and user profiles:**
During the questioning of the reasoning LLM, the unknown_terms list is as it says, terms not present in those retrieved from the dictionary that the model is not familiar with. These terms will be sent as a question (one message per term) to the mod channel, where a moderator or moderators can reply to them with answers, and another LLM API call will reconcile the definitions of the terms in the DB to create a coherent description.

For user profiles/attributes, we wait until human mods rate a message flag. Then we look at the associated message history and use another LLM API call with the message history and flag ratings by human mods to determine its estimates for the person's attributes based on this one interaction. We keep a list of all interactions the user is present in and take the average of all attribute profiles, with a decay for older ones.

**Server feedback guidelines:**
The feedback guidelines of the discord server are the base of the bot, being included in the context packet for the LLM. This includes what qualifies as unconstructive and unsolicited feedback in general terms. To narrow down the exact line, that's what the continuous training is for.

**Handling message edits and deletions:**
We will keep messages in the DB and in-memory history up to date by handling message delete and edit events. If a message is edited or deleted after it exists the current in-memory history cache of the channel, the event is ignored. If it is edited or deleted while in the current in-memory history cache, whether before or after a check workflow has ran, we will make the changes to the DB and cache.

For edits, compute normalized Levenshtein distance (distance / max(1, len(old))) and rerun the check if it exceeds `config.edit_rerun_threshold` (e.g., 0.25). Debounce edits by 2–5 seconds to avoid thrashing. For deletions we always rerun it.

If we end up rerunning a check workflow and a message that was previously flagged was not flagged this time, we remove the reaction and delete the mod channel log message, update `flags.flag_status = 'cleared'`, and remove it from the flagged message dataset if present. This is the case regardless of whether the deleted message was the flagged one or a different one.

**On startup/restart:**
Bot will backfill messages in all tracked channels and threads up to the last message already in the DB. No messages in the DB will be edited.

**Anonymizing users:**
To comply with Discord's TOS, usernames will be anonymized and replaced with 'USER_[id_num]' identifiers, and all mentions of their aliases will be replaced with that identifier in message text when spotted. To save tokens, we'll not use the full Discord user id for this, but a new ID starting from 1 for each user. This mapping (from Discord user ID, username, and history of display names) will be a table in the DB. We may even include an option for lingo clarification questions for mods to specify that as an alias for a user, and/or a command to manually add a user alias.
Assign `anon_id` on first sight and store in `users`. Replace `@mentions` and known aliases with `USER_[anon_id]` before LLM calls.

**Rate limits and reliability:**
Use retries with exponential backoff and jitter (2–3 attempts) for LLM calls; Discord API calls are not an issue due to most things running on events

**Other stuff:**
Discord intents: message content
Will be hosted on a linux Google Cloud vm instance.


## Moderator UX
Mods will mainly interact through the special mod channel, where flagged message logs, ambiguous confirmation messages, and lingo questions are sent. For commands, they can manually trigger a check on any channel, manually set a new lingo item in the dictionary, and run an evaluation of the model on the evaluation/test dataset.

Flag cards will include buttons for Accept, Reject, and Mark ambiguous, plus a link to open the original message. Permissions will restrict commands to moderators, and responses will be ephemeral where appropriate to avoid channel noise.

## Testing plan

- Unit tests: anonymization, JSON schema validation, trigger logic, edit rerun decision
- Integration tests: sandbox guild end-to-end check, rate-limit handling
- Offline evaluation: holdout set with confusion matrix, precision/recall, PR AUC, and moderator load metrics

## MVP build order

1) Per-channel cache + triggers + background queue
2) DB schema and CRUD for messages, flags, reviews, and lingo
3) Logistic regression baseline + calibration + thresholds
4) Strict JSON LLM step for candidate extraction and features
5) Moderator UX actions and reconciliation, then robust edit/delete handling

### Unknowns/not finalized
- Whether to use chroma DB and embeddings at all
- Whether to use a train/test split and how much
- What LLM to use (tentatively gpt-oss-120b)
- Specific config and DB schema