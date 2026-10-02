import asyncio
import importlib.util
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class ModerationStateTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Configure memory storage before importing the startup module.
        import config
        with patch.object(config, "DB_FILE", ":memory:"):
            spec = importlib.util.spec_from_file_location("_moderation_state_bot", ROOT / "bot.py")
            self.module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = self.module
            self.addCleanup(sys.modules.pop, spec.name, None)
            spec.loader.exec_module(self.module)

        self.bot = self.module.ExcelsiorBot(intents=self.module.intents)
        self.channel = Mock(spec=self.module.discord.TextChannel)
        self.channel.id = 123
        self.channel.name = "test-channel"
        self.channel.guild = SimpleNamespace(id=456, roles=[])
        self.state = self.bot._get_or_create_channel_state(self.channel.id)
        self.enterContext(patch.object(self.module, "is_tracked_channel", return_value=True))
        self.enterContext(patch.object(self.bot, "_ensure_scheduler_task"))
        self.enterContext(patch.object(self.bot, "get_channel", return_value=self.channel))
        self.enterContext(patch.object(self.module, "MESSAGES_PER_CHECK", 30))
        self.enterContext(patch.object(self.module, "NEW_MESSAGES_BEFORE_TIMER_START", 3))
        self.tasks = []
        self.addAsyncCleanup(self.stop_tasks)

    async def stop_tasks(self):
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)

    def start(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.tasks.append(task)
        return task

    async def add_messages(self, first, last):
        for message_id in range(first, last + 1):
            message = SimpleNamespace(
                id=message_id, channel=self.channel,
                author=SimpleNamespace(id=789, name="Alice", display_name="Alice"),
                content="test message", created_at=datetime.now(timezone.utc),
                edited_at=None, reference=None, attachments=[], reactions=[],
            )
            self.bot.message_store.add_message(message)
            await self.bot.notify_moderation_on_message(message)

    async def test_arrivals_during_extraction_get_a_later_idle_run(self):
        await self.add_messages(1, 30)
        entered, release, later_run = asyncio.Event(), asyncio.Event(), asyncio.Event()
        snapshots = []

        async def extract(store, channel_id, **kwargs):
            snapshots.append([message.id for message in store.get_whole_history(channel_id)])
            if len(snapshots) == 1:
                entered.set()
                await release.wait()
            else:
                later_run.set()
            return []

        self.enterContext(patch.object(self.module, "get_candidate_features", side_effect=extract))
        first = self.start(self.bot.run_moderation_now(self.channel))
        await asyncio.wait_for(entered.wait(), 2)
        await self.add_messages(31, 33)
        arrival_timer = self.state.idle_timer_started_at
        release.set()
        self.assertTrue((await asyncio.wait_for(first, 2)).success)
        self.assertEqual(self.state.messages_since_check, 3)
        self.assertEqual(self.state.last_checked_message_id, 30)
        self.assertEqual(self.state.most_recent_message_id, 33)
        self.assertTrue(self.state.has_new_message_since_check)
        self.assertEqual(self.state.idle_timer_started_at, arrival_timer)
        self.assertEqual(snapshots, [list(range(1, 31))])
        self.assertFalse(self.bot._should_moderate(self.state))

        self.state.idle_timer_started_at -= timedelta(seconds=self.module.SECS_BETWEEN_AUTO_CHECKS + 1)
        self.start(self.bot._moderation_scheduler(self.channel.id))
        await asyncio.wait_for(later_run.wait(), 2)
        self.assertEqual(snapshots[-1][-3:], [31, 32, 33])
        self.assertEqual(self.state.messages_since_check, 0)
        self.assertEqual(self.state.last_checked_message_id, 33)
        self.assertIsNone(self.state.idle_timer_started_at)

    async def test_small_arrival_batch_starts_timer_at_its_third_message(self):
        await self.add_messages(1, 30)
        entered, release = asyncio.Event(), asyncio.Event()

        async def extract(*args, **kwargs):
            entered.set()
            await release.wait()
            return []

        self.enterContext(patch.object(self.module, "get_candidate_features", side_effect=extract))
        first = self.start(self.bot.run_moderation_now(self.channel))
        await asyncio.wait_for(entered.wait(), 2)
        await self.add_messages(31, 32)
        release.set()
        await asyncio.wait_for(first, 2)
        self.assertEqual(self.state.messages_since_check, 2)
        self.assertFalse(self.state.has_new_message_since_check)
        self.assertIsNone(self.state.idle_timer_started_at)
        await self.add_messages(33, 33)
        self.assertTrue(self.state.has_new_message_since_check)
        self.assertIsNotNone(self.state.idle_timer_started_at)

    async def test_failed_or_cancelled_extraction_retains_the_entire_batch(self):
        for outcome in ("result", "exception", "cancel"):
            with self.subTest(outcome=outcome):
                self.state = self.module.ChannelModerationState(channel_id=self.channel.id)
                self.bot.channel_states[self.channel.id] = self.state
                await self.add_messages(1, 30)
                original_timer = self.state.idle_timer_started_at
                entered, release = asyncio.Event(), asyncio.Event()

                async def moderate(channel):
                    entered.set()
                    await release.wait()
                    if outcome == "exception":
                        raise RuntimeError("synthetic failure")
                    return self.module.ModerationResult(success=False, reason="synthetic failure")

                with patch.object(self.bot, "moderate_channel", side_effect=moderate):
                    first = self.start(self.bot.run_moderation_now(self.channel))
                    await asyncio.wait_for(entered.wait(), 2)
                    await self.add_messages(31, 33)
                    if outcome == "cancel":
                        first.cancel()
                        with self.assertRaises(asyncio.CancelledError):
                            await first
                    else:
                        release.set()
                        if outcome == "exception":
                            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                                await asyncio.wait_for(first, 2)
                        else:
                            self.assertFalse((await asyncio.wait_for(first, 2)).success)
                self.assertEqual(self.state.messages_since_check, 33)
                self.assertIsNone(self.state.last_checked_message_id)
                self.assertEqual(self.state.idle_timer_started_at, original_timer)
                self.assertTrue(self.state.has_new_message_since_check)
                self.assertEqual(self.state.in_flight_message_count, 0)
                self.assertFalse(self.state.moderation_lock.locked())
                self.assertEqual(self.state.consecutive_failures, int(outcome != "cancel"))

    async def test_failed_small_batch_retains_timer_started_by_new_arrival(self):
        await self.add_messages(1, 2)
        entered, release = asyncio.Event(), asyncio.Event()

        async def moderate(channel):
            entered.set()
            await release.wait()
            return self.module.ModerationResult(success=False, reason="synthetic failure")

        with patch.object(self.bot, "moderate_channel", side_effect=moderate):
            first = self.start(self.bot.run_moderation_now(self.channel))
            await asyncio.wait_for(entered.wait(), 2)
            await self.add_messages(3, 3)
            arrival_timer = self.state.idle_timer_started_at
            release.set()
            self.assertFalse((await asyncio.wait_for(first, 2)).success)
        self.assertEqual(self.state.messages_since_check, 3)
        self.assertTrue(self.state.has_new_message_since_check)
        self.assertIsNotNone(arrival_timer)
        self.assertEqual(self.state.idle_timer_started_at, arrival_timer)

    async def assert_serialized_actions(self, scheduler_first):
        await self.add_messages(1, 30)
        engine = create_engine("sqlite:///:memory:")
        self.addCleanup(engine.dispose)
        from db_config import Base
        Base.metadata.create_all(engine)
        sessions = sessionmaker(bind=engine)
        self.bot.get_db_session = sessions
        entered, release = asyncio.Event(), asyncio.Event()
        candidate = {
            "discord_message_id": 30, "relative_message_index": 30,
            "target_user_id": 999, "target_username": "Bob",
            "features": {"discusses_ellie": 0},
        }
        classifier = SimpleNamespace(feature_names=["discusses_ellie"], predict=lambda _: ["flag"])
        self.enterContext(patch.object(self.module, "Path", return_value=SimpleNamespace(exists=lambda: True)))
        self.enterContext(patch.object(self.module, "load_classifier", return_value=classifier))

        async def extract(*args, **kwargs):
            if extractor.await_count > 1:
                with sessions() as session:
                    self.assertEqual(session.query(self.module.FlaggedMessage).count(), 1)
                self.assertEqual(self.state.last_checked_message_id, 30)
            return [candidate]

        extractor = self.enterContext(patch.object(self.module, "get_candidate_features", side_effect=extract))

        async def flag(*args, **kwargs):
            entered.set()
            await release.wait()
            return 1000

        action = self.enterContext(patch.object(self.bot, "flag_message", side_effect=flag))
        if scheduler_first:
            self.start(self.bot._moderation_scheduler(self.channel.id))
            await asyncio.wait_for(entered.wait(), 2)
            manual = self.start(self.bot.run_moderation_now(self.channel))
        else:
            manual = self.start(self.bot.run_moderation_now(self.channel))
            await asyncio.wait_for(entered.wait(), 2)
            self.start(self.bot._moderation_scheduler(self.channel.id))
        await asyncio.sleep(0)
        self.assertEqual(extractor.await_count, 1)
        self.assertFalse(manual.done())
        release.set()
        self.assertTrue((await asyncio.wait_for(manual, 2)).success)
        await asyncio.sleep(0)
        self.assertEqual(extractor.await_count, 2 if scheduler_first else 1)
        self.assertEqual(action.await_count, 1)
        with sessions() as session:
            self.assertEqual(session.query(self.module.FlaggedMessage).count(), 1)
            self.assertEqual(session.query(self.module.LogChannelRatingPost).count(), 1)
            self.assertEqual(session.query(self.module.MessageFeatures).count(), 1)
        self.assertEqual(self.state.messages_since_check, 0)
        self.assertEqual(self.state.last_checked_message_id, 30)

    async def test_manual_waits_for_scheduler_actions_and_commit(self):
        await self.assert_serialized_actions(scheduler_first=True)

    async def test_scheduler_rechecks_work_after_manual_actions_and_commit(self):
        await self.assert_serialized_actions(scheduler_first=False)
