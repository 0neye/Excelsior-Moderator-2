import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

from test_moderation_state import ModerationTestCase


class ReconnectHistoryTests(ModerationTestCase):
    def message(self, message_id, content="history"):
        return SimpleNamespace(id=message_id, channel=self.channel, content=content)

    async def test_repeated_backfill_is_unique_and_chronological(self):
        messages = [self.message(2), self.message(1)]

        async def history(limit):
            for message in messages[:limit]:
                yield message

        self.channel.history = Mock(side_effect=history)
        for _ in range(2):
            self.assertEqual(await self.module.backfill_channel_history(
                self.channel, self.bot.message_store, 40,
            ), 2)
        store = self.bot.message_store
        self.assertEqual([message.id for message in store.get_whole_history(self.channel.id)], [1, 2])
        self.assertEqual(store.get_oldest_message(self.channel.id).id, 1)
        self.assertEqual(store.get_most_recent_message(self.channel.id).id, 2)

    async def test_arrival_during_reconnect_remains_after_backfilled_messages(self):
        entered, release = asyncio.Event(), asyncio.Event()
        live = self.message(3, "new arrival")

        async def history(limit):
            yield self.message(2)
            entered.set()
            await release.wait()
            yield self.message(1)

        self.channel.history = Mock(side_effect=history)
        backfill = self.start(self.module.backfill_channel_history(
            self.channel, self.bot.message_store, 40,
        ))
        await asyncio.wait_for(entered.wait(), 2)
        self.bot.message_store.add_message(live)
        release.set()
        await asyncio.wait_for(backfill, 2)
        self.assertEqual(
            [message.id for message in self.bot.message_store.get_whole_history(self.channel.id)],
            [1, 2, 3],
        )
        self.assertIs(self.bot.message_store.get_most_recent_message(self.channel.id), live)

    async def test_capacity_retains_newest_unique_messages_and_cached_live_copy(self):
        store = self.bot.message_store
        store._max_size = 3
        live = self.message(4, "live copy")
        store.add_message(live)
        store.add_message(self.message(5))

        async def history(limit):
            for message_id in (4, 3, 2, 1):
                yield self.message(message_id, "backfill copy")

        self.channel.history = Mock(side_effect=history)
        for _ in range(2):
            await self.module.backfill_channel_history(self.channel, store, 4)
        self.assertEqual([message.id for message in store.get_whole_history(self.channel.id)], [3, 4, 5])
        self.assertIs(store.get_message_by_id(4, self.channel.id), live)
        self.assertEqual(store.get_most_recent_message(self.channel.id).id, 5)
