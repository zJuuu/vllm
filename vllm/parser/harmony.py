# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import NamedTuple

from openai.types.responses import FunctionTool
from openai.types.responses.tool import Tool as ResponsesTool
from openai_harmony import (
    HarmonyError,
    Message,
    Role,
    StreamableParser,
    StreamState,
)
from xgrammar import StructuralTag
from xgrammar.openai_tool_call_schema import BuiltinToolParam, FunctionToolParam
from xgrammar.structural_tag import (
    AnyTextFormat,
    ConstStringFormat,
    Format,
    JSONSchemaFormat,
    OptionalFormat,
    OrFormat,
    SequenceFormat,
    TagFormat,
)

from vllm.entrypoints.generate.base.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    FunctionCall,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.entrypoints.openai.parser.harmony_utils import (
    extract_function_from_recipient,
    forbids_tool_calls,
    get_encoding,
    get_streamable_parser_for_assistant,
    is_function_recipient,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.logger import init_logger
from vllm.parser.abstract_parser import DelegatingParser, structured_outputs_to_format
from vllm.reasoning.gptoss_reasoning_parser import GptOssReasoningParser
from vllm.renderers.chat_utils import make_tool_call_id
from vllm.sampling_params import StructuredOutputsParams
from vllm.tool_parsers.gptoss_tool_parser import GptOssToolParser
from vllm.tool_parsers.structural_tag_registry import (
    SimplifiedToolChoice,
    get_function_parameters,
    get_model_structural_tag,
    register_vllm_structural_tag,
    resolve_tool_strictness,
)
from vllm.tool_parsers.tool_strict_level import ToolStrictLevel

logger = init_logger(__name__)


class _SegmentType(Enum):
    TOOL = auto()
    REASONING = auto()
    CONTENT = auto()
    IGNORE = auto()

    @staticmethod
    def from_channel_and_recipient(
        channel: str | None, recipient: str | None
    ) -> _SegmentType:
        if recipient and is_function_recipient(recipient):
            return _SegmentType.TOOL
        if channel == "analysis":
            return _SegmentType.REASONING
        if channel == "final" or (channel == "commentary" and recipient is None):
            return _SegmentType.CONTENT
        return _SegmentType.IGNORE


class Segment(NamedTuple):
    channel: str | None
    recipient: str | None
    delta: str
    completed_message: Message | None = None


@dataclass(slots=True)
class ChunkResult:
    segments: list[Segment]
    reasoning_token_count: int


class HarmonyParser(DelegatingParser):
    def __init__(self, tokenizer, tools=None, *args, **kwargs):
        super().__init__(tokenizer, tools, *args, **kwargs)

        if self.reasoning_parser and not isinstance(
            self.reasoning_parser, GptOssReasoningParser
        ):
            raise ValueError(
                "Harmony requires GptOssReasoningParser, "
                f"got {self.reasoning_parser.__class__.__name__}."
            )

        if self.tool_parser and not isinstance(self.tool_parser, GptOssToolParser):
            raise ValueError(
                "Harmony requires GptOssToolParser, "
                f"got {self.tool_parser.__class__.__name__}."
            )

        self._parser: StreamableParser | None = None
        self._next_tool_call_index = 0
        self._num_processed_messages = 0

        self._num_counted_tokens = 0
        self._num_reasoning_tokens = 0

        # For error recovery
        self._current_message_tokens: list[int] = []

        # Final-channel text held back while it could begin a stop string
        self._final_pending = ""

    @property
    def _harmony_parser(self) -> StreamableParser:
        """Lazily initializes the Harmony parser."""
        if self._parser is None:
            self._parser = get_streamable_parser_for_assistant()
        return self._parser

    def _poll_completed_message(self) -> Message | None:
        messages = self._harmony_parser.messages
        if len(messages) <= self._num_processed_messages:
            return None
        msg = messages[self._num_processed_messages]
        msg.recipient = self._normalize_recipient(msg.recipient)
        self._num_processed_messages += 1
        return msg

    def flush(self) -> list[Segment]:
        segments: list[Segment] = []
        try:
            self._harmony_parser.process_eos()
            msg = self._poll_completed_message()
        except HarmonyError:
            msg = None
            header_tokens = get_encoding().encode(
                "<|start|><|channel|><|constrain|><|message|>", allowed_special="all"
            )
            # A Harmony header cut off by max_tokens or a stop string has no
            # content; only output without header tokens is recovered as text.
            if not any(
                token_id in header_tokens for token_id in self._current_message_tokens
            ):
                logger.warning(
                    "Harmony parser ended in a non-terminal state; returning the "
                    "recovered raw output."
                )

                final_channel = "final"
                text = self.model_tokenizer.decode(
                    self._current_message_tokens, skip_special_tokens=True
                )
                segments.append(
                    Segment(
                        channel=final_channel,
                        recipient=None,
                        delta=text,
                        completed_message=None,
                    )
                )
                msg = Message.from_role_and_content(Role.ASSISTANT, text).with_channel(
                    final_channel
                )

        # Reset to the initial assistant-parser state for the next turn.
        self._parser = None
        self._num_processed_messages = 0
        self._current_message_tokens.clear()

        if msg is None:
            return segments

        segments.append(
            Segment(
                channel=msg.channel,
                recipient=msg.recipient,
                delta="",
                completed_message=msg,
            )
        )
        return segments

    def parse(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
        enable_auto_tools: bool = False,
        model_output_token_ids: Sequence[int] = (),
    ) -> tuple[str | None, str | None, list[FunctionCall] | None]:
        """Parse Harmony output from token IDs.

        Tool calls are always extracted regardless of `enable_auto_tools`.
        Callers must decide whether to surface them.
        """
        result = self.process_chunk(model_output_token_ids)
        flushed_segments = self.flush()
        if flushed_segments:
            result.segments.extend(flushed_segments)

        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        tool_calls: list[FunctionCall] = []

        for segment in result.segments:
            msg = segment.completed_message
            if msg is None:
                continue
            if msg.author.role != "assistant" or not msg.content:
                continue
            text = msg.content[0].text
            segment_type = _SegmentType.from_channel_and_recipient(
                msg.channel, msg.recipient
            )
            match segment_type:
                case _SegmentType.REASONING if self.reasoning_parser and text:
                    reasoning_parts.append(text)
                case _SegmentType.CONTENT if text:
                    if msg.channel == "final":
                        text = self._stop_final_content(text, request, finished=True)
                    if text:
                        content_parts.append(text)
                case _SegmentType.TOOL if self.tool_parser:
                    recipient = msg.recipient
                    content_type = msg.content_type
                    assert recipient is not None
                    if content_type is not None and "json" not in content_type:
                        arguments = text
                    else:
                        try:
                            arguments = json.dumps(json.loads(text))
                        except json.JSONDecodeError:
                            arguments = text
                    tool_calls.append(
                        FunctionCall(
                            name=extract_function_from_recipient(recipient),
                            arguments=arguments,
                        )
                    )

        reasoning = "\n".join(reasoning_parts) or None
        content = "\n".join(content_parts) or None
        return reasoning, content, tool_calls or None

    def parse_delta(
        self,
        delta_text: str,
        delta_token_ids: list[int],
        request: ChatCompletionRequest | ResponsesRequest,
        prompt_token_ids: list[int] | None = None,
        *,
        finished: bool,
    ) -> DeltaMessage | None:
        prev_recipient = self._normalize_recipient(
            self._harmony_parser.current_recipient
        )
        result = self.process_chunk(delta_token_ids)
        if finished:
            flushed_segments = self.flush()
            if flushed_segments:
                result.segments.extend(flushed_segments)
        combined_content = ""
        combined_reasoning = ""
        tool_messages: list[DeltaToolCall] = []

        for segment in result.segments:
            if segment.completed_message is not None:
                prev_recipient = None
                continue

            segment_type = _SegmentType.from_channel_and_recipient(
                segment.channel, segment.recipient
            )
            match segment_type:
                case _SegmentType.REASONING if self.reasoning_parser:
                    combined_reasoning += segment.delta
                case _SegmentType.CONTENT:
                    delta = segment.delta
                    if segment.channel == "final":
                        delta = self._stop_final_content(delta, request, finished=False)
                    combined_content += delta
                case _SegmentType.TOOL if self.tool_parser:
                    assert segment.recipient is not None
                    if prev_recipient != segment.recipient:
                        tool_name = extract_function_from_recipient(segment.recipient)
                        tool_messages.append(
                            DeltaToolCall(
                                # HarmonyParser does not use _stream_state;
                                # "random" tool_call_id_type is always used
                                id=make_tool_call_id(),
                                type="function",
                                function=DeltaFunctionCall(
                                    name=tool_name,
                                    arguments=segment.delta,
                                ),
                                index=self._next_tool_call_index,
                            )
                        )
                        self._next_tool_call_index += 1
                        prev_recipient = segment.recipient
                    elif segment.delta:
                        idx = self._next_tool_call_index - 1
                        if tool_messages:
                            tool_msg = tool_messages[-1]
                            assert tool_msg.index == idx
                            fn = tool_msg.function
                            assert fn is not None and fn.arguments is not None
                            fn.arguments += segment.delta
                        else:
                            tool_messages.append(
                                DeltaToolCall(
                                    index=idx,
                                    function=DeltaFunctionCall(arguments=segment.delta),
                                )
                            )

        if finished:
            self._next_tool_call_index = 0
            combined_content += self._stop_final_content("", request, finished=True)

        if not combined_content and not combined_reasoning and not tool_messages:
            return None

        delta_message = DeltaMessage()
        if combined_content:
            delta_message.content = combined_content
        if combined_reasoning:
            delta_message.reasoning = combined_reasoning
        if tool_messages:
            delta_message.tool_calls = tool_messages

        # Suppress reasoning deltas if not requested
        if delta_message and not request.include_reasoning:
            delta_message.reasoning = None

            # If only reasoning was in the message (no content, no tool_calls)
            # skip emitting entirely
            if not delta_message.content and not delta_message.tool_calls:
                return None

        return delta_message

    def _stop_final_content(
        self,
        text: str,
        request: ChatCompletionRequest | ResponsesRequest,
        finished: bool,
    ) -> str:
        """Apply the client's stop strings to final-channel text.

        Text that could still begin a stop string is held back until the next
        chunk or the end of the output.
        """
        stop = (
            request._content_stop if isinstance(request, ChatCompletionRequest) else []
        )
        if not stop:
            return text
        if self.matched_stop is not None:
            return ""
        pending = self._final_pending + text
        matches = [(pending.find(s), s) for s in stop if s in pending]
        if matches:
            # The stop string that completes first, as the engine picks it.
            _, index, self.matched_stop = min(
                (index + len(s), index, s) for index, s in matches
            )
            self._final_pending = ""
            if request.include_stop_str_in_output:
                index += len(self.matched_stop)
            return pending[:index]
        held = 0 if finished else max(len(s) for s in stop) - 1
        cut = max(len(pending) - held, 0)
        self._final_pending = pending[cut:]
        return pending[:cut]

    def process_chunk(self, token_ids: Sequence[int]) -> ChunkResult:
        if not token_ids:
            return ChunkResult(segments=[], reasoning_token_count=0)

        segments: list[Segment] = []
        reasoning_token_count = 0
        for token_id in self._close_message_before_header(token_ids):
            try:
                self._harmony_parser.process(token_id)
            except HarmonyError as e:
                if not self._resync_after_error(e, token_id):
                    continue
            channel = self._harmony_parser.current_channel
            recipient = self._normalize_recipient(
                self._harmony_parser.current_recipient
            )
            delta = self._harmony_parser.last_content_delta or ""
            completed_message = self._poll_completed_message()

            if completed_message is not None:
                self._current_message_tokens.clear()
            else:
                self._current_message_tokens.append(token_id)

            if self._is_reasoning_token(token_id, channel, recipient):
                reasoning_token_count += 1

            segments.append(
                Segment(
                    channel=channel,
                    recipient=recipient,
                    delta=delta,
                    completed_message=completed_message,
                )
            )

            # TODO: Optionally merge and suppress empty Segments

        self._num_counted_tokens += len(token_ids)
        self._num_reasoning_tokens += reasoning_token_count
        return ChunkResult(
            segments=segments,
            reasoning_token_count=reasoning_token_count,
        )

    def _close_message_before_header(self, token_ids: Sequence[int]) -> Iterator[int]:
        """Yield `token_ids`, ending the current message before a `<|channel|>`.

        A message that quotes "<|end|><|start|>assistant" as text satisfies a
        structural tag's message end, and the model then opens the next header
        with real tokens. The Harmony parser would keep that header and the
        tool call after it as text of the quoting message. A recipient quoted
        after "assistant" belongs to the new header.
        """
        encoding = get_encoding()
        channel = encoding.encode("<|channel|>", allowed_special="all")[0]
        for token_id in token_ids:
            if (
                token_id == channel
                and self._harmony_parser.state == StreamState.CONTENT
            ):
                quoted = _QUOTED_HEADER.search(self._harmony_parser.current_content)
                yield from encoding.encode("<|end|><|start|>", allowed_special="all")
                yield from encoding.encode(
                    "assistant" + (quoted.group(1) or "" if quoted else "")
                )
            yield token_id

    def _resync_after_error(self, error: HarmonyError, token_id: int) -> bool:
        """Drop a malformed message and resume at the last `<|start|>` it holds.

        A malformed header keeps absorbing tokens, including the `<|start|>`
        and header of a well-formed message that follows it, until
        `<|message|>` fails to parse. Without a restart from that `<|start|>`
        the following tool call or answer would be dropped as well.

        Returns whether `token_id` was consumed by the restarted parser.
        """
        tokens = [*self._current_message_tokens, token_id]
        self._current_message_tokens.clear()
        if len(tokens) == 1:
            return False
        logger.warning("Skipping malformed Harmony message: %s", error)

        start = get_encoding().encode("<|start|>", allowed_special="all")[0]
        if start not in tokens:
            return False
        restart = len(tokens) - 1 - tokens[::-1].index(start)
        self._parser = StreamableParser(get_encoding(), role=None)
        self._num_processed_messages = 0
        try:
            for token in tokens[restart:]:
                self._parser.process(token)
        except HarmonyError:
            self._parser = StreamableParser(get_encoding(), role=None)
            return False
        self._current_message_tokens.extend(tokens[restart:-1])
        return True

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        if len(token_ids) == self._num_counted_tokens:
            return self._num_reasoning_tokens

        parser = get_streamable_parser_for_assistant()
        count = 0
        for token_id in token_ids:
            try:
                parser.process(token_id)
            except HarmonyError:
                continue
            recipient = self._normalize_recipient(parser.current_recipient)
            if self._is_reasoning_token(token_id, parser.current_channel, recipient):
                count += 1
        return count

    @staticmethod
    def _is_reasoning_token(
        token_id: int, channel: str | None, recipient: str | None
    ) -> bool:
        is_reasoning_channel = channel == "analysis" or (
            channel == "commentary" and recipient is not None
        )
        return is_reasoning_channel and not get_encoding().is_special_token(token_id)

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        if isinstance(request, ChatCompletionRequest) and request.stop:
            stop = [request.stop] if isinstance(request.stop, str) else request.stop
            # The engine matches stop strings in reasoning, call headers and
            # arguments too; apply them to the final answer only. An empty one
            # is left for the engine to reject.
            if "" not in stop:
                request._content_stop = list(stop)
                request.stop = []
        if (
            request.tools
            and self.tool_parser is not None
            and self.tool_strict_level is None
        ):
            # GPT-OSS closes nested JSON unevenly and breaks schemas without a
            # grammar, so a tool without `strict` is treated as `strict: true`.
            request.tools = [_strict_unless_disabled(tool) for tool in request.tools]
        add_tool_calls = bool(
            self.tool_parser is not None
            and request.tools
            and resolve_tool_strictness(
                request.tools, "auto", self.tool_strict_level or ToolStrictLevel.AUTO
            )
            is None
        )
        request = _adjust_output_format(request, add_tool_calls)
        return super().adjust_request(request)

    @staticmethod
    def _normalize_recipient(recipient: str | None) -> str | None:
        """Remove constrained formats misparsed into recipients by older Harmony."""
        if recipient is None:
            return None

        constrain_index = recipient.find("<|constrain|>")
        if constrain_index == -1:
            return recipient
        return recipient[:constrain_index].rstrip() or None


_JSON_CONSTRAINS = [" json", " <|constrain|>json"]
# Harmomy's stop tokens are <|return|>, <|call|>, <|endoftext|>
# They are represented as "" since xgrammar disallows stop tokens while
# under constraints, leading to bad or infinite generations.
_END_TAG = ["<|end|>", ""]
_FINAL_BEGIN = "<|channel|>final{constrain}<|message|>"
_TOOL_CALL_CHANNELS = [
    "<|channel|>commentary",
    "<|channel|>analysis",
    "<|channel|>final",
]
_FUNCTION_CALL_BEGINS = [
    " to=functions.{name}{channel}{constrain}<|message|>",
    "{channel} to=functions.{name}{constrain}<|message|>",
]
_ANY_CONTENT = AnyTextFormat()
_QUOTED_HEADER = re.compile(r"<\|start\|>assistant( to=\S+)?\s*$")


def _assemble_tag(
    allow_analysis: bool, allow_commentary: bool, content: Format
) -> StructuralTag:
    tags = []
    if allow_analysis:
        analysis_tag = OptionalFormat(
            content=SequenceFormat(
                elements=[
                    TagFormat(
                        begin="<|channel|>analysis<|message|>",
                        content=_ANY_CONTENT,
                        end="<|end|>",
                    ),
                    ConstStringFormat(value="<|start|>assistant"),
                ]
            )
        )
        tags.append(analysis_tag)

    if allow_commentary:
        commentary_tag = OptionalFormat(
            content=SequenceFormat(
                elements=[
                    TagFormat(
                        begin="<|channel|>commentary<|message|>",
                        content=_ANY_CONTENT,
                        end="<|end|>",
                    ),
                    ConstStringFormat(value="<|start|>assistant"),
                ]
            )
        )
        tags.append(commentary_tag)

    tags.append(content)

    return StructuralTag(format=SequenceFormat(elements=tags))


@register_vllm_structural_tag("harmony", builtin_tools=True)
def get_harmony_structural_tag(
    tools: list[FunctionToolParam],
    builtin_tools: list[BuiltinToolParam],
    tool_choice: SimplifiedToolChoice,
    reasoning: bool,
    token_suffix: str = "",
) -> StructuralTag:
    # reasoning always enabled for Harmony
    del reasoning, token_suffix

    if builtin_tools:
        # Fallback for built-in tools
        tags = [
            TagFormat(
                begin="to=",
                content=AnyTextFormat(excludes=["<|start|>"]),
                end=_END_TAG,
            )
        ]
        tags.extend(
            TagFormat(
                begin=channel + " to=",
                content=AnyTextFormat(excludes=["<|start|>", "<|channel|>"]),
                end=_END_TAG,
            )
            for channel in _TOOL_CALL_CHANNELS
        )
    else:
        tags = [
            TagFormat(
                begin=pattern.format(
                    name=tool.function.name, channel=channel, constrain=constrain
                ),
                content=JSONSchemaFormat(
                    json_schema=get_function_parameters(tool.function)
                ),
                end=_END_TAG,
            )
            for tool in tools
            for pattern in _FUNCTION_CALL_BEGINS
            for constrain in _JSON_CONSTRAINS
            for channel in _TOOL_CALL_CHANNELS
        ]

    if tool_choice == "auto":
        tags.extend(
            TagFormat(
                begin=_FINAL_BEGIN.format(constrain=constrain),
                content=_ANY_CONTENT,
                end=_END_TAG,
            )
            for constrain in _JSON_CONSTRAINS + [""]
        )

    return _assemble_tag(
        allow_analysis=True, allow_commentary=True, content=OrFormat(elements=tags)
    )


def _strict_unless_disabled(
    tool: ChatCompletionToolsParam | ResponsesTool,
) -> ChatCompletionToolsParam | ResponsesTool:
    if isinstance(tool, ChatCompletionToolsParam) and tool.function.strict is None:
        return tool.model_copy(
            update={"function": tool.function.model_copy(update={"strict": True})}
        )
    if isinstance(tool, FunctionTool) and tool.strict is None:
        return tool.model_copy(update={"strict": True})
    return tool


def _adjust_output_format(
    request: ChatCompletionRequest | ResponsesRequest,
    add_tool_calls: bool,
) -> ChatCompletionRequest | ResponsesRequest:
    """Canonicalize request constraints into a reasoning-aware StructuralTag."""
    params = request.extract_structured_outputs()
    if params is None:
        if not forbids_tool_calls(request):
            return request
        # GPT-OSS still calls a tool it needs, and the call would be dropped,
        # leaving an empty reply. Only allow an answer.
        structural_tag = _assemble_tag(
            allow_analysis=True,
            allow_commentary=True,
            content=OrFormat(
                elements=[
                    TagFormat(
                        begin=_FINAL_BEGIN.format(constrain=constrain),
                        content=_ANY_CONTENT,
                        end=_END_TAG,
                    )
                    for constrain in _JSON_CONSTRAINS + [""]
                ]
            ),
        )
        request.structured_outputs = StructuredOutputsParams(
            structural_tag=json.dumps(structural_tag.model_dump())
        )
        return request

    final_content = structured_outputs_to_format(params)
    if final_content is None:
        return request

    if isinstance(final_content, JSONSchemaFormat):
        content = OrFormat(
            elements=[
                TagFormat(
                    begin=_FINAL_BEGIN.format(constrain=constrain),
                    content=final_content,
                    end=_END_TAG,
                )
                for constrain in _JSON_CONSTRAINS
            ]
        )
    else:
        content = TagFormat(
            begin=_FINAL_BEGIN.format(constrain=""), content=final_content, end=_END_TAG
        )
    tool_calls = (
        get_model_structural_tag(
            GptOssToolParser.structural_tag_model,
            request.tools,
            "required",
            reasoning=True,
            strict_level=ToolStrictLevel.FUNCTION,
        )
        if add_tool_calls and request.tool_choice == "auto"
        else None
    )
    if tool_calls is not None:
        # The response format applies to the answer; tools stay callable.
        # Otherwise `_apply_structural_tag` adds the tool calls itself.
        # The last element of the Harmony tag holds the tool call tags.
        content = OrFormat(elements=[content, tool_calls.format.elements[-1]])
    structural_tag = _assemble_tag(
        allow_analysis=True, allow_commentary=tool_calls is not None, content=content
    )

    request.structured_outputs = replace(
        params,
        json=None,
        regex=None,
        choice=None,
        grammar=None,
        json_object=None,
        structural_tag=json.dumps(structural_tag.model_dump()),
    )
    if isinstance(request, ResponsesRequest):
        request.text = None
    else:
        request.response_format = None
    return request
