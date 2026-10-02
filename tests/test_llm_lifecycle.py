"""Regression tests for provider-lock ownership during timed-out and cancelled calls."""

import asyncio
import threading
import unittest
from unittest.mock import patch

import llms


class ProviderLockLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        llms.LLM_PROVIDER_LOCKS.clear()

    async def _wait_for_thread_event(self, event):
        await asyncio.wait_for(asyncio.to_thread(event.wait), timeout=1)

    async def test_timeout_keeps_provider_lock_until_worker_returns(self):
        worker_started = threading.Event()
        worker_finished = threading.Event()
        release_worker = threading.Event()
        second_started = threading.Event()

        def blocking_call():
            worker_started.set()
            release_worker.wait(timeout=1)
            worker_finished.set()
            return "first"

        with patch("llms.LLM_TIMEOUT_SECONDS", 0.01):
            first = asyncio.create_task(llms._run_llm_call("test", blocking_call))
            await self._wait_for_thread_event(worker_started)
            with self.assertRaises(asyncio.TimeoutError):
                await first

        def second_call():
            self.assertTrue(worker_finished.is_set())
            second_started.set()
            return "second"

        second = asyncio.create_task(llms._run_llm_call("test", second_call))
        await asyncio.sleep(0.05)
        self.assertFalse(second_started.is_set())
        release_worker.set()
        self.assertEqual(await asyncio.wait_for(second, timeout=1), "second")

    async def test_cancellation_keeps_provider_lock_until_worker_returns(self):
        worker_started = threading.Event()
        worker_finished = threading.Event()
        release_worker = threading.Event()
        second_started = threading.Event()

        def blocking_call():
            worker_started.set()
            release_worker.wait(timeout=1)
            worker_finished.set()
            return "first"

        first = asyncio.create_task(llms._run_llm_call("test", blocking_call))
        await self._wait_for_thread_event(worker_started)
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first

        def second_call():
            self.assertTrue(worker_finished.is_set())
            second_started.set()
            return "second"

        second = asyncio.create_task(llms._run_llm_call("test", second_call))
        await asyncio.sleep(0.05)
        self.assertFalse(second_started.is_set())
        release_worker.set()
        self.assertEqual(await asyncio.wait_for(second, timeout=1), "second")

    async def test_timed_out_worker_exception_releases_the_lock(self):
        worker_started = threading.Event()
        release_worker = threading.Event()
        second_started = threading.Event()

        def failing_call():
            worker_started.set()
            release_worker.wait(timeout=1)
            raise RuntimeError("worker failed after timeout")

        with patch("llms.LLM_TIMEOUT_SECONDS", 0.01):
            first = asyncio.create_task(llms._run_llm_call("test", failing_call))
            await self._wait_for_thread_event(worker_started)
            with self.assertRaises(asyncio.TimeoutError):
                await first

        second = asyncio.create_task(llms._run_llm_call("test", lambda: second_started.set() or "second"))
        await asyncio.sleep(0.05)
        self.assertFalse(second_started.is_set())
        release_worker.set()
        self.assertEqual(await asyncio.wait_for(second, timeout=1), "second")
