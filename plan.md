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

Finally, with the answers to these questions + the data from user profiles, we'll use a logistic regression or small MLP to predict the category of each candidate message being flagged; either (flagged, not_flagged, ambiguous).

Candidate messages that are considered ambiguous will be sent to a mod channel for human confirmation before flagging. Ones in the flagged category get flagged, and the rest not flagged.
Flagged messages are sent as logs in the mod channel, and the original message gets a reaction emoji added to it publicly. Ambiguous messages only are reacted to publicly once they are manually confirmed by a human moderator.

**Continuous training and DB updates:**
Once messages are flagged and sent to a mod channel in the discord server, human moderators can rate whether they think the flag was correct, incorrect, or ambiguous. This includes ones predicted as ambiguous and manually confirmed by a human mod.
Once this happens, it is added to a dataset in the DB with the number of human ratings in each category and the model prediction.
Then the mlp model or logistic regression is retrained on this new dataset. Train/test split is possible too, if it would be beneficial.

**Dictionary and user profiles:**
During the questioning of the reasoning LLM, the unknown_terms list is as it says, terms not present in those retrieved from the dictionary that the model is not familiar with. These terms will be sent as a question (one message per term) to the mod channel, where a moderator or moderators can reply to them with answers, and another LLM API call will reconcile the definitions of the terms in the DB to create a coherent description.

For user profiles/attributes, we wait until human mods rate a message flag. Then we look at the associated message history and use another LLM API call with the message history and flag ratings by human mods to determine its estimates for the person's attributes based on this one interaction. We keep a list of all interactions the user is present in and take the average of all attribute profiles, with a decay for older ones.

**Server feedback guidelines:**
The feedback guidelines of the discord server are the base of the bot, being included in the context packet for the LLM. This includes what qualifies as unconstructive and unsolicited feedback in general terms. To narrow down the exact line, that's what the continuous training is for.

**Handling message edits and deletions:**
We will keep messages in the DB and in-memory history up to date by handling message delete and edit events. If a message is edited or deleted after it exists the current in-memory history cache of the channel, the event is ignored. If it is edited or deleted while in the current in-memory history cache, whether before or after a check workflow has ran, we will make the changes to the DB and cache. If a check workflow has already run, we will use the Levenshtein distance and a cuttoff to determine whether to rerun the check workflow again. For deletions we always rerun it.

If we end up rerunning a check workflow and a message that was previously flagged was not flagged this time, we remove the reaction and delete the mod channel log message, as well as remove it from the flagged message dataset if present. This is the case regardless of whether the deleted message was the flagged one or a different one.

**On startup/restart:**
Bot will backfill messages in all tracked channels and threads up to the last message already in the DB. No messages in the DB will be edited.

**Anonymizing users:**
To comply with Discord's TOS, usernames will be anonymized and replaced with 'USER_[id_num]' identifiers, and all mentions of their aliases will be replaced with that identifier in message text when spotted. To save tokens, we'll not use the full Discord user id for this, but a new ID starting from 1 for each user. This mapping (from Discord user ID, username, and history of display names) will be a table in the DB. We may even include an option for lingo clarification questions for mods to specify that as an alias for a user, and/or a command to manually add a user alias.

**Other stuff:**
Discord intents: message content
Will be hosted on a linux Google Cloud vm instance.


## Moderator UX
Mods will mainly interact through the special mod channel, where flagged message logs, ambiguous confirmation messages, and lingo questions are sent. For commands, they can manually trigger a check on any channel, manually set a new lingo item in the dictionary, and run an evaluation of the model on the evaluation/test dataset.

### Unknowns/not finalized
- Whether to use chroma DB and embeddings at all
- Whether to use a train/test split and how much
- What LLM to use (tentatively gpt-oss-120b)
- Specific config and DB schema