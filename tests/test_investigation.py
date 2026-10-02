import importlib
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from test_moderation_state import ModerationTestCase


class InvestigationLimitTests(ModerationTestCase):
    async def test_count_25_with_long_ids_keeps_every_record_inspectable(self):
        import config
        with patch.object(config, "DB_FILE", ":memory:"):
            restricted = importlib.import_module("cogs.restricted")
        guild_id = 9000000000000000000
        channel_id = 8999999999999999999
        author_id = 8999999999999999998
        timestamp = datetime.now(timezone.utc)
        message_ids = [8999999999999999900 + index for index in range(25)]
        with self.sessions() as session:
            for index, message_id in enumerate(message_ids):
                session.add(self.module.FlaggedMessage(
                    message_id=message_id, channel_id=channel_id, guild_id=guild_id,
                    author_id=author_id, author_username="Alice", content="test content",
                    timestamp=timestamp, flagged_at=timestamp + timedelta(seconds=index),
                    waiver_filtered=True, was_acted_upon=False,
                ))
            session.commit()
        cog = restricted.Restricted(self.bot)
        context = SimpleNamespace(
            guild=SimpleNamespace(id=guild_id), author=SimpleNamespace(id=author_id), respond=AsyncMock(),
        )
        await restricted.Restricted.mod_flag_investigate.callback(
            cog, context, count=25, include_waiver_filtered=True,
        )
        context.respond.assert_awaited_once()
        sent = context.respond.call_args.kwargs
        view, embed = sent["view"], sent["embed"]
        self.addCleanup(view.stop)
        self.assertLessEqual(len(embed.description), 4096)
        self.assertLessEqual(len(embed), 6000)
        self.assertEqual(len(view.recent_records), 25)
        selector = next(child for child in view.children if isinstance(child, restricted.ModFlagRecordSelect))
        self.assertEqual(len(selector.options), 25)
        self.assertEqual({option.value for option in selector.options}, {str(message_id) for message_id in message_ids})
        for message_id in message_ids:
            self.assertIn(str(message_id), embed.description)
        self.assertEqual(embed.description.count("waiver filtered"), 25)

        # Exercise the real selector and Open Selected callbacks for every accepted record.
        for option in selector.options:
            interaction = SimpleNamespace(
                data={"values": [option.value]}, response=SimpleNamespace(edit_message=AsyncMock()),
            )
            selector.refresh_state(interaction)
            await selector.callback(interaction)
            selected_embed = interaction.response.edit_message.call_args.kwargs["embed"]
            self.assertLessEqual(len(selected_embed.description), 4096)
            self.assertLessEqual(len(selected_embed), 6000)
            open_button = next(child for child in view.children if getattr(child, "custom_id", None) == "mod_flag_open_selected")
            await open_button.callback(interaction)
            overview = interaction.response.edit_message.call_args.kwargs["embed"]
            self.assertIn(f"/{guild_id}/{channel_id}/{option.value}", overview.description)
            self.assertLessEqual(len(overview), 6000)
            self.assertIn("**Moderator Log Delivered:** No", overview.description)
            view.mode = "list"
