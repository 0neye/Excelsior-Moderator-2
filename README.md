# Excelsior Moderator 2 (AKA New Ellie)

A Discord bot that helps maintain constructive feedback culture by automatically identifying messages that may not follow community guidelines. The bot learns from community ratings to continuously improve its accuracy.

## What Does It Do?

The bot monitors Discord channels and watches for feedback that might be:
- **Unconstructive**: Harsh or negative without helpful suggestions
- **Unsolicited**: Given when nobody asked for it

When the bot thinks a message might violate guidelines, it:
- Adds a 👁️ reaction to the message
- Posts it to a review channel where moderators can confirm or correct the decision
- Learns from these ratings to get better over time

The more ratings the bot receives, the smarter it becomes at understanding the community's specific norms.

<img width="1086" height="362" alt="image" src="https://github.com/user-attachments/assets/128cea1d-bc2a-4322-bfa4-a7e687ce2a62" />

## How to Use the Bot

### For Community Members

**Help Train the Bot**: Use the `/rate` command in the rating channel to review flagged messages and help the bot learn. Your ratings directly improve the bot's accuracy!

**View Your Stats**: Use `/view_score` to see your contribution statistics and leaderboard position.

### For Moderators

**Review Flags**: Check the log channel where the bot posts flagged messages. React with rating emojis to confirm or correct the bot's decisions.

**Manual Check**: Use `/check` in any channel to manually trigger a review of recent messages.

**Retrain Model**: Use `/retrain` to immediately update the bot's model with new ratings (usually happens automatically).

### For Everyone

**Basic Commands**:
- `/ping` - Check if the bot is online
- `/info` - Get information about the bot

## Quick Setup

**NOTE:** Requires existing rating data in specific format. From-zero bootstrap currently not supported.

For detailed technical information, see [TECHNICAL.md](TECHNICAL.md).

## Rating System

Help the bot learn by rating flagged messages! Your input directly improves accuracy.

### How to Rate

1. Go to the rating channel
2. Type `/rate` to get a random flagged message
3. Click one of the five rating buttons:
   - **✅ No Flag** - Message is fine, bot made a mistake
   - **❓ Ambiguous** - Hard to tell if it violates guidelines
   - **⚠️ Unconstructive** - Harsh feedback without helpful suggestions
   - **📢 Unsolicited** - Feedback given when nobody asked
   - **🚫 N/A** - Doesn't fit any category

### Why Rate?

- **Every 20 ratings**, the bot automatically retrains and gets smarter
- Help the bot understand your community's specific norms
- See your contribution on the leaderboard
- Make moderation more fair and accurate

Moderators can also rate quickly by reacting to flagged messages in the private log channel.

## Credits

Developed for the Excelsior community to maintain constructive and welcoming feedback culture.

0neye + Opus 4.5
