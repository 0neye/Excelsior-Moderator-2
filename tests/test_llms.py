import json
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llms import extract_features_from_formatted_history  # noqa: E402


class CerebrasFallbackTests(unittest.IsolatedAsyncioTestCase):
    """Regression tests for direct Cerebras -> OpenRouter fallback behavior."""

    async def test_cerebras_404_falls_back_to_openrouter(self):
        """Any direct Cerebras error, including 404, should retry through OpenRouter."""

        class CerebrasNotFoundError(Exception):
            status_code = 404

        fake_cerebras_client = Mock()
        fake_cerebras_client.chat.completions.create.side_effect = CerebrasNotFoundError(
            "404 model endpoint not found"
        )
        fake_openrouter_response = {
            "model": "openai/gpt-oss-120b",
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "candidates": [
                                    {
                                        "message_id": 1,
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
                                ]
                            }
                        )
                    }
                }
            ],
        }

        async def run_call(_provider, call):
            return call()

        with (
            patch("llms.OPENROUTER_API_KEY", "test-openrouter-key"),
            patch("llms._get_cerebras_client", return_value=fake_cerebras_client),
            patch("llms._run_llm_call", side_effect=run_call),
            patch("llms._build_openrouter_request", return_value=fake_openrouter_response) as build_openrouter,
        ):
            candidates = await extract_features_from_formatted_history(
                formatted_message_history="[2026-04-27 12:00] (1) Alice: hello",
                channel_name="general",
                provider="cerebras",
                model="gpt-oss-120b",
            )

        self.assertEqual(candidates[0]["llm_provider"], "openrouter")
        build_openrouter.assert_called_once()
        _, request_kwargs = build_openrouter.call_args
        self.assertTrue(request_kwargs["prefer_cerebras_route"])
