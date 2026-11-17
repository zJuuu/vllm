# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.entrypoints.generate.base.protocol import (
    DeltaMessage,
    ExtractedToolCallInformation,
)
from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParser
from vllm.tool_parsers.tool_strict_level import ToolStrictLevel

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike


class GptOssToolParser(ToolParser):
    """Stub tool parser for gpt-oss/harmony models.

    All output parsing is handled by HarmonyParser. This stub exists as a
    capability declaration via HarmonyParser.tool_parser_cls.
    """

    structural_tag_model = "harmony"
    # GPT-OSS often breaks argument schemas it can't see in the rendered tool
    # signature, and closes nested JSON unevenly; without a grammar, calls to
    # nested agent tools come back unparsable or schema-violating.
    default_tool_strict_level = ToolStrictLevel.PARAMETER

    def __init__(self, tokenizer: "TokenizerLike", tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)

    def extract_tool_calls(
        self, model_output, request, **kwargs
    ) -> ExtractedToolCallInformation:
        raise NotImplementedError(
            "GptOssToolParser is a stub. Use HarmonyParser for tool parsing."
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request,
    ) -> DeltaMessage | None:
        raise NotImplementedError(
            "GptOssToolParser is a stub. Use HarmonyParser for tool parsing."
        )
