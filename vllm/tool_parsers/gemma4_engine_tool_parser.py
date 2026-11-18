# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

from openai.types.responses import ToolChoiceCustom, ToolChoiceFunction
from xgrammar import StructuralTag

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.parser.engine.registered_adapters import Gemma4ParserToolAdapter
from vllm.tool_parsers.tool_strict_level import ToolStrictLevel


def _resolve_tokens(node: Any, vocab: dict[str, int]) -> Any:
    """Replace token strings in a dumped structural tag with their ids, so
    the tag compiles without a tokenizer (request validation does)."""
    if isinstance(node, list):
        return [_resolve_tokens(item, vocab) for item in node]
    if not isinstance(node, dict):
        return node
    out = {key: _resolve_tokens(value, vocab) for key, value in node.items()}
    if out.get("type") == "token" and isinstance(out["token"], str):
        out["token"] = vocab[out["token"]]
    for key in ("trigger_tokens", "exclude_tokens"):
        if key in out:
            out[key] = [vocab[t] if isinstance(t, str) else t for t in out[key]]
    return out


class Gemma4EngineToolParser(Gemma4ParserToolAdapter):  # type: ignore[valid-type, misc]
    supports_required_and_named = False
    structural_tag_model = "gemma4"
    # Constrain the call envelope of every request with tools: otherwise
    # response_format with non-strict tools leaves no way to call a tool.
    default_tool_strict_level = ToolStrictLevel.FUNCTION

    def get_structural_tag(
        self,
        request: ChatCompletionRequest | ResponsesRequest,
        *,
        reasoning: bool = False,
        strict_level: ToolStrictLevel = ToolStrictLevel.AUTO,
    ) -> StructuralTag | None:
        tag = super().get_structural_tag(
            request, reasoning=reasoning, strict_level=strict_level
        )
        if tag is None:
            return None
        return StructuralTag.model_validate(
            _resolve_tokens(tag.model_dump(), self.vocab)
        )

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        """Skip structured-output JSON for required/named tool choice.

        Gemma4 emits its native `<|tool_call>call:...` syntax, which the
        parser extracts directly. The base `ToolParser.adjust_request` would
        set `structured_outputs` for required/named and force JSON via guided
        decoding, conflicting with that native syntax (it leaks as content and
        crashes EngineCore under speculative decoding). Skip it so the model
        emits its native format (mirrors the GLM4 parser).
        """
        if request.tools:
            tc = request.tool_choice
            if tc == "required" or isinstance(
                tc,
                (
                    ChatCompletionNamedToolChoiceParam,
                    ToolChoiceFunction,
                    ToolChoiceCustom,
                ),
            ):
                request.skip_special_tokens = False
                return request
        return super().adjust_request(request)
