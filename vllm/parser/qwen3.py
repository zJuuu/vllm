# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3 parser for tool calls and reasoning.

Qwen3 XML tool call format::

    <tool_call>
    <function=func_name>
    <parameter=key>value</parameter>
    </function>
    </tool_call>

The argument body consists of `<parameter=NAME>VALUE</parameter>` tags.
The `_qwen3_arg_converter` parses these into a JSON object.
"""

from __future__ import annotations

import dataclasses
import functools
import json
from collections.abc import Callable
from typing import TYPE_CHECKING

import regex as re

from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ArgValueExtender,
    ParserEngineConfig,
    ParserState,
    Transition,
)

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

THINK_START = "<think>"
THINK_END = "</think>"
TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
CHATML_TURN_BOUNDARIES = frozenset(("<|im_start|>", "<|im_end|>"))
FUNC_PREFIX = "<function="
FUNC_END = "</function>"
PARAM_START = "<parameter="
PARAM_END = "</parameter>"

_PARAM_RE = re.compile(
    r"<\s*parameter\s*=\s*([^>]*)>"
    r"(.*?)"
    r"(?:<\s*/\s*parameter\s*>|(?=<\s*parameter\s*=))",
    re.DOTALL,
)
_PARTIAL_PARAM_RE = re.compile(r"<\s*parameter\s*=\s*([^>]+)>(.*)$", re.DOTALL)
_PARAM_OPENER_RE = re.compile(r"<\s*parameter\s*=")


def _trim_wrapping_newlines(value: str) -> str:
    """Strip one leading and one trailing newline (the Qwen3 template markup)."""
    if value.startswith("\n"):
        value = value[1:]
    if value.endswith("\n"):
        value = value[:-1]
    return value


class _ClosedParams:
    """Closed parameters of a raw argument text, reused by longer texts.

    Raw arguments only grow while a call streams, so a parse of a text that
    extends `text` resumes after its last closed parameter.
    """

    __slots__ = ("text", "end", "params", "outside")

    def __init__(self) -> None:
        self.text = ""  # raw text the closed parameters depend on
        self.end = 0  # end of the last closed parameter in ``text``
        self.params: dict[str, str] = {}
        self.outside = ""  # text before ``end`` outside the closed parameters


def _qwen3_parse_params(
    raw_args: str, partial: bool, closed: _ClosedParams | None = None
) -> tuple[dict[str, str], re.Match | None]:
    """Return the closed parameters, trimmed, and the open trailing one."""
    params: dict[str, str] = {}
    outside: list[str] = []
    pos = 0
    if closed is not None and raw_args.startswith(closed.text):
        params.update(closed.params)
        outside.append(closed.outside)
        pos = closed.end
    start = pos

    for match in _PARAM_RE.finditer(raw_args, pos):
        outside.append(raw_args[pos : match.start()])
        params[match.group(1)] = _trim_wrapping_newlines(match.group(2))
        pos = match.end()

    if closed is not None and pos > start:
        # A value closed by the next opener holds only while that opener follows.
        opener = _PARAM_OPENER_RE.match(raw_args, pos)
        closed.text = raw_args[: opener.end() if opener else pos]
        closed.end = pos
        closed.params = dict(params)
        closed.outside = "".join(outside)

    open_param = None
    if partial:
        open_param = _PARTIAL_PARAM_RE.search("".join(outside) + raw_args[pos:])
    return params, open_param


def _qwen3_arg_converter(
    raw_args: str, partial: bool, closed: _ClosedParams | None = None
) -> str:
    params, open_param = _qwen3_parse_params(raw_args, partial, closed)
    if open_param is not None and open_param.group(1):
        params[open_param.group(1)] = _trim_wrapping_newlines(open_param.group(2))

    return json.dumps(params, ensure_ascii=False)


# What ends an open value in ``_PARAM_RE``: a closing tag or the next opener.
_PARAM_VALUE_END_RE = re.compile(r"<\s*/\s*parameter\s*>|<\s*parameter\s*=")


def _param_value_end_window(text: str) -> str:
    """Return the suffix of *text* that may still grow into a value end."""
    start = text.rfind("<")
    if start == -1:
        return ""
    tail = text[start:]
    return tail if _PARAM_VALUE_END_RE.fullmatch(tail, partial=True) else ""


def _param_value_extender(
    raw_args: str, trim_newlines: bool, closed: _ClosedParams | None = None
) -> ArgValueExtender | None:
    """Extend the open trailing `<parameter=...>` value of *raw_args*.

    The extender streams exactly what the converter would for the same text.
    With *trim_newlines* it follows `_trim_wrapping_newlines`: it drops a
    leading newline of the value and holds back a trailing one.
    """
    closed_params, match = _qwen3_parse_params(raw_args, True, closed)
    if match is None:
        return None
    name, value = match.group(1), match.group(2)
    # A repeated key keeps its first position, so its value is not last.
    if name in closed_params:
        return None

    window = _param_value_end_window(value)
    started = bool(value)
    held = "\n" if trim_newlines and value.endswith("\n") and value != "\n" else ""

    def extend(delta: str) -> str | None:
        nonlocal window, started, held
        text = window + delta
        if _PARAM_VALUE_END_RE.search(text):
            return None
        window = _param_value_end_window(text)
        if trim_newlines:
            text = held + delta
            if not started and text:
                started = True
                text = text.removeprefix("\n")
            held = "\n" if text.endswith("\n") else ""
            text = text[: len(text) - len(held)]
        else:
            text = delta
        return json.dumps(text, ensure_ascii=False)[1:-1]

    return extend


def _qwen3_arg_value_extender(
    raw_args: str, closed: _ClosedParams | None = None
) -> ArgValueExtender | None:
    return _param_value_extender(raw_args, trim_newlines=True, closed=closed)


# Tags that structure a call body.  Inside a value they are text unless the
# rules in `_ParamScan` make them structural.  Spaces and the parameter name
# are bounded, so a partial tag held back from a value stays short.
_ARG_TAG_RE = re.compile(
    r"<[ \t]{0,8}parameter[ \t]{0,8}=[ \t]{0,8}([^<>\n]{0,128})>"
    r"|(<[ \t]{0,8}/[ \t]{0,8}parameter[ \t]{0,8}>)"
    r"|(<[ \t]{0,8}function[ \t]{0,8}=)"
    r"|(<tool_call>)"
    r"|(</tool_call>)"
    r"|<[ \t]{0,8}/[ \t]{0,8}function[ \t]{0,8}>"
)
_QUOTES = ("```", '"""', "'''")


class _ParamScan:
    """Incremental parse of a call body, with markup quoted in values.

    The model writes values verbatim, so a value may hold the call markup
    itself (code, templates, docs).  Inside a value:

    * `</parameter>` closes it only when whitespace and then the next tag or
      the end of the call follow; other text reopens the value.
    * `<parameter=` at the start of a line starts the next parameter (the
      value lacked `</parameter>`).
    * `<function=` opens a quoted call, whose parameters and `</function>`
      are text, unless it names one of the request's tools at the start of
      a line, after a `<tool_call>` line or after `</function>`, outside a
      code fence or triple-quoted string: then the value was left open and
      a new call starts.  A `</parameter>` with no quoted parameter open
      still closes the value.
    * Other `</function>` ends the call if `</tool_call>` follows, and a
      `</tool_call>` line ends it at the end of the output.

    A value that quotes a call to a request tool at the start of a line,
    outside fences, is therefore cut there, as is one ending in a bare
    `</function>` and `</tool_call>`.

    `feed` takes the text added since the previous call, so a call body
    costs time linear in its length.  The engine uses one scan per call to
    decide the call end (`allows`); the converter and the value extender
    share another.
    """

    __slots__ = (
        "is_tool_name",
        "params",
        "name",
        "value_id",
        "parts",
        "length",
        "end",
        "func_end",
        "call_start",
        "call_end",
        "end_char",
        "nested",
        "pending",
        "last_char",
    )

    def __init__(self, is_tool_name: Callable[[str], bool] | None = None) -> None:
        self.is_tool_name = is_tool_name
        self.params: dict[str, str] = {}
        self.name: str | None = None
        self.value_id = 0
        self.parts: list[str] = []
        self.length = 0
        # Where the value may end, while only whitespace follows: its
        # `</parameter>`, a `</function>`, or a `<tool_call>` or
        # `</tool_call>` line.
        self.end = -1
        self.func_end = -1
        self.call_start = -1
        self.call_end = -1
        self.end_char = ""
        self.nested: list[int] = []
        self.pending = ""
        self.last_char = ""

    def feed(self, text: str) -> None:
        if not self.pending and "<" not in text:
            self._text(text)
            return
        text = self.pending + text
        self.pending = ""
        pos = 0
        for match in _ARG_TAG_RE.finditer(text):
            self._text(text[pos : match.start()])
            self._tag(match)
            pos = match.end()
        rest = text[pos:]
        cut = rest.rfind("<")
        if cut >= 0 and _ARG_TAG_RE.fullmatch(rest[cut:], partial=True):
            self.pending = rest[cut:]
            rest = rest[:cut]
        else:
            self.pending = ""
        self._text(rest)

    def _append(self, text: str) -> None:
        self.parts.append(text)
        self.length += len(text)

    def _text(self, text: str) -> None:
        if not text:
            return
        if self.name is not None:
            self._append(text)
            if not text.isspace():
                # Text after `</parameter>` reopens the value.
                self.end = self.func_end = self.call_start = self.call_end = -1
        self.last_char = text[-1]

    def _mark(self, attr: str) -> None:
        if getattr(self, attr) >= 0:
            return
        if self.value_end() == self.length:
            self.end_char = self.last_char
        setattr(self, attr, self.length)

    @property
    def _line_start(self) -> bool:
        return not self.nested and self.last_char == "\n"

    def _tag(self, match: re.Match) -> None:
        name, param_end, func_start, call_start, call_end = match.group(1, 2, 3, 4, 5)
        in_value = self.name is not None and self.end < 0
        if name is not None and (not in_value or self._line_start):
            if self.name is not None:
                self.params.setdefault(self.name, self.value())
            self.name = name
            self.value_id += 1
            self.parts = []
            self.length = 0
            self.end = self.func_end = self.call_start = self.call_end = -1
            self.nested.clear()
        elif in_value:
            if name is not None:
                if self.nested:
                    self.nested[-1] += 1
            elif param_end is not None:
                if self.nested and self.nested[-1]:
                    self.nested[-1] -= 1
                else:
                    self.nested.clear()
                    self._mark("end")
            elif func_start is not None:
                self.nested.append(0)
            elif call_start is not None or call_end is not None:
                if self._line_start:
                    self._mark("call_start" if call_start else "call_end")
            elif self.nested:
                self.nested.pop()
            else:
                self._mark("func_end")
            self._append(match.group(0))
        elif self.name is not None:
            self._append(match.group(0))
        self.last_char = ">" if match.group(0).endswith(">") else "="

    def value_end(self) -> int:
        """End of the last value: its closing tag, or the text so far."""
        marks = (self.end, self.func_end, self.call_start, self.call_end)
        return min((m for m in marks if m >= 0), default=self.length)

    def value_text(self) -> str:
        if len(self.parts) > 1:
            self.parts = ["".join(self.parts)]
        return self.parts[0] if self.parts else ""

    def value(self) -> str:
        """The last value, without a trailing tag that is still open."""
        return _trim_wrapping_newlines(self.value_text()[: self.value_end()])

    def allows(self, terminal: str, following: str, finished: bool) -> bool | None:
        """Whether a call-markup terminal after the text fed ends the call.

        None asks for the text after it first: the name after `<function=`,
        or whether anything follows a `</tool_call>` line.
        """
        if terminal not in ("FUNC_END", "FUNC_PREFIX", "TOOL_END"):
            return True
        if self.name is None or self.end >= 0:
            return True
        if terminal == "FUNC_END":
            return False
        if terminal == "TOOL_END" and self.func_end >= 0:
            return True
        if self.nested or any(self.value_text().count(q) % 2 for q in _QUOTES):
            return False
        if terminal == "TOOL_END":
            if not self._line_start or following.strip():
                return False
            return True if finished else None
        if not (self._line_start or self.func_end >= 0 or self.call_start >= 0):
            return False
        name, closed, _ = following.partition(">")
        if not closed:
            return (
                None if len(name) < 128 and "\n" not in name and not finished else False
            )
        assert self.is_tool_name is not None
        return self.is_tool_name(name.strip())


class _ArgsScanCache:
    """The scan of the raw arguments last converted or extended."""

    __slots__ = ("scan", "fed")

    def __init__(self) -> None:
        self.scan = _ParamScan()
        self.fed: list[str] = []

    def scan_args(self, raw_args: str) -> _ParamScan:
        fed = "".join(self.fed)
        if raw_args.startswith(fed):
            self.scan.feed(raw_args[len(fed) :])
        else:
            self.scan = _ParamScan()
            self.scan.feed(raw_args)
        self.fed = [raw_args]
        return self.scan


def _qwen3_literal_arg_converter(
    raw_args: str, partial: bool, cache: _ArgsScanCache | None = None
) -> str:
    """Convert like `_qwen3_arg_converter`, keeping markup quoted in values.

    An unclosed last value is kept in both modes; a cut call reports
    `finish_reason: length`.
    """
    scan = (cache or _ArgsScanCache()).scan_args(raw_args)
    params = dict(scan.params)
    if scan.name is not None:
        params.setdefault(scan.name, scan.value())
    return json.dumps(params, ensure_ascii=False)


def _qwen3_literal_arg_value_extender(
    raw_args: str, cache: _ArgsScanCache | None = None
) -> ArgValueExtender | None:
    """Extend the last value of *raw_args*, through quoted markup.

    Each delta is fed to the scan; the text streams up to where the value
    may end, minus a trailing newline.  Text outside a value or in a
    repeated parameter streams nothing.  It stops once a parameter starts.
    """
    if cache is None:
        cache = _ArgsScanCache()
    scan = cache.scan_args(raw_args)
    value_id = scan.value_id
    streams = scan.name is not None and scan.name not in scan.params
    lead_checked = scan.length > 0
    streamed = _streamed_end(scan)
    if scan.value_text().startswith("\n"):
        streamed = max(streamed, 1)

    def extend(delta: str) -> str | None:
        nonlocal streamed, lead_checked
        cache.fed.append(delta)
        scan.feed(delta)
        if scan.value_id != value_id:
            return None
        if not streams:
            return ""
        if not lead_checked and scan.length:
            lead_checked = True
            if _value_slice(scan, 0, 1) == "\n":
                streamed = 1
        end = _streamed_end(scan)
        if end <= streamed:
            return ""
        part = _value_slice(scan, streamed, end)
        streamed = end
        return json.dumps(part, ensure_ascii=False)[1:-1]

    return extend


def _streamed_end(scan: _ParamScan) -> int:
    """End of the last value's text the converter has emitted."""
    end = scan.value_end()
    char = scan.last_char if end == scan.length else scan.end_char
    if end > 1 and char == "\n":
        end -= 1
    return end


def _value_slice(scan: _ParamScan, start: int, end: int) -> str:
    """`scan.value_text()[start:end]`, read from the last parts only."""
    pieces: list[str] = []
    offset = scan.length
    for part in reversed(scan.parts):
        if offset <= start:
            break
        offset -= len(part)
        if offset < end:
            pieces.append(part[max(start - offset, 0) : end - offset])
    return "".join(reversed(pieces))


@functools.cache
def qwen3_config(
    thinking: bool = True,
    *,
    name: str = "qwen3",
    think_start: str = THINK_START,
    think_end: str = THINK_END,
    tool_start: str = TOOL_CALL_START,
    tool_end: str = TOOL_CALL_END,
    turn_boundary_tokens: frozenset[str] = frozenset(),
    literal_value_markup: bool = False,
    prose_tool_opener_is_content: bool = False,
) -> ParserEngineConfig:
    config = ParserEngineConfig(
        name=name,
        initial_state=ParserState.REASONING if thinking else ParserState.CONTENT,
        wait_for_reasoning=thinking,
        turn_boundary_tokens=turn_boundary_tokens,
        terminals={
            # Reasoning terminals
            "THINK_START": think_start,
            "THINK_END": think_end,
            # Tool call terminals
            "TOOL_START": tool_start,
            "TOOL_END": tool_end,
            "FUNC_PREFIX": FUNC_PREFIX,
            "FUNC_END": FUNC_END,
            "PARAM_START": PARAM_START,
            "PARAM_END": PARAM_END,
            "CLOSE_ANGLE": ">",
        },
        token_id_terminals={
            "THINK_START": think_start,
            "THINK_END": think_end,
            "TOOL_START": tool_start,
            "TOOL_END": tool_end,
        },
        transitions={
            # -- Reasoning transitions --
            (ParserState.REASONING, "THINK_START"): Transition(
                ParserState.REASONING,
                (),
            ),
            (ParserState.REASONING, "THINK_END"): Transition(
                ParserState.CONTENT,
                (EventType.REASONING_END,),
            ),
            # Absorb duplicate </think> — model may emit it after
            # already transitioning to CONTENT; drop it silently.
            (ParserState.CONTENT, "THINK_END"): Transition(
                ParserState.CONTENT,
                (),
            ),
            # Tool call directly from reasoning (implicit end)
            (ParserState.REASONING, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                (EventType.REASONING_END, EventType.TOOL_CALL_START),
            ),
            # -- Tool call transitions --
            (ParserState.CONTENT, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                (EventType.REASONING_END, EventType.TOOL_CALL_START),
            ),
            # Fallback: <function= without a preceding <tool_call>
            (ParserState.CONTENT, "FUNC_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
            ),
            (ParserState.TOOL_PREAMBLE, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (EventType.TOOL_CALL_END,),
            ),
            (ParserState.TOOL_PREAMBLE, "FUNC_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (),
            ),
            (ParserState.TOOL_NAME, "CLOSE_ANGLE"): Transition(
                ParserState.TOOL_ARGS,
                (),
            ),
            # Malformed: </function> while still in TOOL_NAME (no closing >)
            (ParserState.TOOL_NAME, "FUNC_END"): Transition(
                ParserState.TOOL_BETWEEN,
                (EventType.TOOL_CALL_END,),
            ),
            # regression issue 58227
            # <function=...</tool_call>
            # function name is not closed, go to content
            (ParserState.TOOL_NAME, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (EventType.TOOL_CALL_END,),
            ),
            # regression issue 58227
            # <function=...<tool_call>
            # unclosed tool call followed by another tool call
            (ParserState.TOOL_NAME, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                (EventType.TOOL_CALL_END, EventType.TOOL_CALL_START),
            ),
            # regression issue 58227
            # <function=...<function=
            # unclosed function name followed by another function name
            (ParserState.TOOL_NAME, "FUNC_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_END, EventType.TOOL_CALL_START),
            ),
            (ParserState.TOOL_ARGS, "FUNC_END"): Transition(
                ParserState.TOOL_BETWEEN,
                (EventType.TOOL_CALL_END,),
            ),
            # regression issue 58227
            # <function=abc>...<function=
            (ParserState.TOOL_ARGS, "FUNC_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_END, EventType.TOOL_CALL_START),
            ),
            (ParserState.TOOL_ARGS, "PARAM_START"): Transition(
                ParserState.TOOL_ARGS,
                (EventType.ARG_VALUE_CHUNK,),
            ),
            (ParserState.TOOL_ARGS, "PARAM_END"): Transition(
                ParserState.TOOL_ARGS,
                (EventType.ARG_VALUE_CHUNK,),
            ),
            (ParserState.TOOL_BETWEEN, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (),
            ),
            # Consecutive tool call without closing </tool_call>
            (ParserState.TOOL_BETWEEN, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                (EventType.TOOL_CALL_START,),
            ),
            (ParserState.TOOL_BETWEEN, "FUNC_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
            ),
        },
        arg_converter=_qwen3_arg_converter,
        arg_value_extender=_qwen3_arg_value_extender,
        stream_arg_deltas=True,
        strip_trailing_reasoning_whitespace=False,
        tool_args_json=False,
        prose_tool_opener_is_content=prose_tool_opener_is_content,
    )
    if not literal_value_markup:
        return config
    return dataclasses.replace(
        config,
        transitions={
            **config.transitions,
            # The checker decides when </tool_call> ends a call whose last
            # value lacks </parameter>.
            (ParserState.TOOL_ARGS, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (EventType.TOOL_CALL_END,),
            ),
        },
        arg_converter=_qwen3_literal_arg_converter,
        arg_value_extender=_qwen3_literal_arg_value_extender,
        tool_args_checker=_ParamScan,
    )


class Qwen3Parser(ParserEngine):
    """Qwen3 parser: `<think>`/`</think>` reasoning +
    `<tool_call>` XML tool calls in a single engine.

    - `<tool_call>` as implicit reasoning end (a grammar transition, so it
      also feeds `is_reasoning_end` and the structured-output gate)

    Subclasses that share the grammar but differ only in the four wrapper
    token strings (reasoning + tool-call) override the class attributes
    below; everything else is inherited unchanged.
    """

    CONFIG_NAME = "qwen3"
    THINK_START = THINK_START
    THINK_END = THINK_END
    TOOL_START = TOOL_CALL_START
    TOOL_END = TOOL_CALL_END
    TURN_BOUNDARIES: frozenset[str] = CHATML_TURN_BOUNDARIES

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        self.thinking_enabled = chat_kwargs.get("enable_thinking", True)
        # The rules for values, prose markup, names and the prompt below are
        # checked on Qwen3 models only; models sharing the format through a
        # subclass or their own config keep the earlier behaviour.
        self._qwen3_rules = (
            self.CONFIG_NAME == "qwen3" and "parser_engine_config" not in kwargs
        )
        kwargs.setdefault(
            "parser_engine_config",
            qwen3_config(
                thinking=self.thinking_enabled,
                name=self.CONFIG_NAME,
                think_start=self.THINK_START,
                think_end=self.THINK_END,
                tool_start=self.TOOL_START,
                tool_end=self.TOOL_END,
                turn_boundary_tokens=self.TURN_BOUNDARIES,
                literal_value_markup=self._qwen3_rules,
                prose_tool_opener_is_content=self._qwen3_rules,
            ),
        )
        super().__init__(
            tokenizer,
            tools,
            **kwargs,
        )
        # Each streaming conversion resumes after the last closed parameter.
        if self._arg_converter is _qwen3_arg_converter:
            closed = _ClosedParams()
            self._arg_converter = functools.partial(_qwen3_arg_converter, closed=closed)
            self._arg_value_extender = functools.partial(
                _qwen3_arg_value_extender, closed=closed
            )
        elif self._arg_converter is _qwen3_literal_arg_converter:
            cache = _ArgsScanCache()
            self._arg_converter = functools.partial(
                _qwen3_literal_arg_converter, cache=cache
            )
            self._arg_value_extender = functools.partial(
                _qwen3_literal_arg_value_extender, cache=cache
            )

    def _accept_tool_name(self, name: str) -> bool:
        if self._qwen3_rules and len(name.split()) > 1:
            # Words after `<function=` are prose that quotes the syntax.
            return False
        return super()._accept_tool_name(name)

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        if not self.thinking_enabled:
            return None, model_output
        return super().extract_reasoning(model_output, request)
