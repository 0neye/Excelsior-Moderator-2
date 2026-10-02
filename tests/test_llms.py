import asyncio
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llms import extract_features_from_formatted_history  # noqa: E402


class CerebrasFallbackTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.candidate = {
            "message_id": "1",
            "target_username": "Nobody",
            "features": {
                "discusses_ellie": 0,
                "familiarity_score": 0,
                "tone_harshness_score": 0,
                "positive_framing_score": 0,
                "includes_positive_takeaways": 0,
                "explains_why_score": 0,
                "actionable_suggestion_score": 0,
                "context_is_feedback_appropriate": 0,
                "target_uncomfortableness_score": 0,
                "is_part_of_discussion": 0,
                "criticism_directed_at_image": 0,
                "criticism_directed_at_statement": 0,
                "criticism_directed_at_generality": 0,
                "reciprocity_score": 0,
                "solicited_score": 0,
            },
        }
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=json.dumps({"candidates": [self.candidate]})
        ))])
        self.cerebras = Mock()
        self.cerebras.chat.completions.create.return_value = response
        self.enterContext(patch("llms.OPENROUTER_API_KEY", "test-openrouter-key"))
        self.enterContext(patch("llms._get_cerebras_client", return_value=self.cerebras))
        self.openrouter = self.enterContext(patch(
            "llms._build_openrouter_request", return_value=response
        ))

        async def run_call(_provider, call):
            return call()

        self.run_call = self.enterContext(patch("llms._run_llm_call", side_effect=run_call))

    async def extract(self):
        return await extract_features_from_formatted_history(
            formatted_message_history="[2026-04-27 12:00] (1) Alice: hello",
            channel_name="general",
            provider="cerebras",
            model="gpt-oss-120b",
        )

    async def test_request_errors_fall_back_to_openrouter(self):
        for error in (
            RuntimeError("404 model endpoint not found"),
            RuntimeError("429 rate limited"),
            RuntimeError("connection failed"),
            TimeoutError("request timed out"),
        ):
            with self.subTest(error=str(error)):
                self.cerebras.chat.completions.create.side_effect = error
                self.openrouter.reset_mock()
                self.run_call.reset_mock()
                candidates = await self.extract()
                self.assertEqual(candidates[0]["message_id"], "1")
                self.assertEqual(candidates[0]["target_username"], "Nobody")
                self.assertEqual(candidates[0]["features"], {
                    **self.candidate["features"],
                    "seniority_score_messages": 0.0,
                    "seniority_score_characters": 0.0,
                    "familiarity_score_stat": 0.0,
                })
                self.assertEqual(
                    [call.args[0] for call in self.run_call.await_args_list],
                    ["cerebras", "openrouter"],
                )
                self.openrouter.assert_called_once()
                self.assertTrue(self.openrouter.call_args.kwargs["prefer_cerebras_route"])
                self.assertEqual(self.openrouter.call_args.kwargs["model"], "gpt-oss-120b")

    async def test_missing_key_preserves_cerebras_error(self):
        error = RuntimeError("Cerebras unavailable")
        self.cerebras.chat.completions.create.side_effect = error
        with patch("llms.OPENROUTER_API_KEY", None):
            with self.assertRaises(RuntimeError) as caught:
                await self.extract()
        self.assertIs(caught.exception, error)
        self.openrouter.assert_not_called()

    async def test_openrouter_failure_is_propagated(self):
        self.cerebras.chat.completions.create.side_effect = RuntimeError("Cerebras unavailable")
        error = RuntimeError("OpenRouter unavailable")
        self.openrouter.side_effect = error
        with self.assertRaises(RuntimeError) as caught:
            await self.extract()
        self.assertIs(caught.exception, error)
        self.openrouter.assert_called_once()

    async def test_success_does_not_fall_back(self):
        candidates = await self.extract()
        self.assertEqual(candidates[0]["message_id"], "1")
        self.openrouter.assert_not_called()

    async def test_cancellation_does_not_fall_back(self):
        self.cerebras.chat.completions.create.side_effect = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await self.extract()
        self.openrouter.assert_not_called()
