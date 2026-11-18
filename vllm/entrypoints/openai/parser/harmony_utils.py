# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import datetime
import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from openai.types.responses.tool import Tool
from openai_harmony import (
    Author,
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    ReasoningEffort,
    RenderConversationConfig,
    Role,
    StreamableParser,
    SystemContent,
    TextContent,
    ToolDescription,
    load_harmony_encoding,
)

from vllm import envs
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

logger = init_logger(__name__)


def is_function_recipient(
    recipient: str,
    allowed_function_tool_names: frozenset[str] | None = None,
) -> bool:
    """Check whether *recipient* refers to a function tool call.

    The optional *allowed_function_tool_names* parameter is used by the
    Responses API to distinguish bare function-call recipients (missing the
    `functions.` prefix) from MCP tool calls.  When provided, a bare
    recipient is only treated as a function call if it appears in the set.
    The Chat Completions path omits this parameter so that all bare
    recipients are accepted as function calls (the heuristic fallback).
    """
    if not recipient:
        return False
    if recipient.startswith("<|"):
        return False
    if recipient.startswith("functions."):
        return len(recipient) > len("functions.")
    if recipient == "assistant":
        return False
    if recipient in BUILTIN_TOOL_TO_MCP_SERVER_LABEL:
        return False
    first_segment = recipient.split(".", 1)[0]
    if first_segment in BUILTIN_TOOL_TO_MCP_SERVER_LABEL:
        return False
    if allowed_function_tool_names is not None:
        return recipient in allowed_function_tool_names
    return True


def extract_function_from_recipient(recipient: str) -> str:
    return recipient.removeprefix("functions.")


REASONING_EFFORT = {
    "high": ReasoningEffort.HIGH,
    "medium": ReasoningEffort.MEDIUM,
    "low": ReasoningEffort.LOW,
}

_harmony_encoding = None

# Builtin tools that should be included in the system message when
# they are available and requested by the user.
# Tool args are provided by MCP tool descriptions. Output
# of the tools are stringified.
BUILTIN_TOOL_TO_MCP_SERVER_LABEL: dict[str, str] = {
    "python": "code_interpreter",
    "browser": "web_search_preview",
    "container": "container",
}

# Derive MCP_BUILTIN_TOOLS from the canonical mapping
MCP_BUILTIN_TOOLS: set[str] = set(BUILTIN_TOOL_TO_MCP_SERVER_LABEL.values())


def has_custom_tools(tool_types: set[str]) -> bool:
    """Checks if the given tool types are custom tools
    (i.e. any tool other than MCP builtin tools)
    """
    return not tool_types.issubset(MCP_BUILTIN_TOOLS)


def get_encoding():
    global _harmony_encoding
    if _harmony_encoding is None:
        _harmony_encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return _harmony_encoding


# Without it GPT-OSS answers a request it can't complete with invented tool
# results or a call written out as text.
NO_TOOL_CALL_INSTRUCTIONS = (
    "Tools cannot be called for this reply. Answer only from the information "
    "already in the conversation, and say plainly what you could not look up."
)


def forbids_tool_calls(request: "ChatCompletionRequest | ResponsesRequest") -> bool:
    """Whether no tool call may be returned for `request`.

    That is `tool_choice="none"` with tools, or a chat request without tools
    whose history holds tool calls or results.
    """
    if request.tools:
        return request.tool_choice == "none"
    if not isinstance(request, ChatCompletionRequest):
        return False
    return any(
        message.get("role") == "tool" or message.get("tool_calls")
        for message in request.messages
    )


def get_system_message(
    model_identity: str | None = None,
    reasoning_effort: str | None = None,
    start_date: str | None = None,
    browser_description: str | None = None,
    python_description: str | None = None,
    container_description: str | None = None,
    instructions: str | None = None,
    with_custom_tools: bool = False,
) -> Message:
    sys_msg_content = SystemContent.new()
    if model_identity is not None:
        sys_msg_content = sys_msg_content.with_model_identity(model_identity)
    if instructions is not None and envs.VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS:
        current_identity = sys_msg_content.model_identity
        new_identity = (
            f"{current_identity}\n{instructions}" if current_identity else instructions
        )
        sys_msg_content = sys_msg_content.with_model_identity(new_identity)
    if reasoning_effort is not None:
        if reasoning_effort not in REASONING_EFFORT:
            supported_values = ", ".join(REASONING_EFFORT)
            raise VLLMValidationError(
                f"reasoning_effort={reasoning_effort!r} is not supported by "
                f"Harmony. Supported values are: {supported_values}.",
                parameter="reasoning_effort",
            )
        sys_msg_content = sys_msg_content.with_reasoning_effort(
            REASONING_EFFORT[reasoning_effort]
        )
    if start_date is None:
        # NOTE(woosuk): This brings non-determinism in vLLM.
        # Set VLLM_SYSTEM_START_DATE to pin it.
        start_date = envs.VLLM_SYSTEM_START_DATE or datetime.datetime.now().strftime(
            "%Y-%m-%d"
        )
    sys_msg_content = sys_msg_content.with_conversation_start_date(start_date)
    if browser_description is not None:
        sys_msg_content = sys_msg_content.with_tools(browser_description)
    if python_description is not None:
        sys_msg_content = sys_msg_content.with_tools(python_description)
    if container_description is not None:
        sys_msg_content = sys_msg_content.with_tools(container_description)
    sys_msg = Message.from_role_and_content(Role.SYSTEM, sys_msg_content)
    return sys_msg


_RENDERED_AS_NOTE = (
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "pattern",
    "format",
    "minItems",
    "maxItems",
    "uniqueItems",
)


def _schema_for_harmony(
    schema: Any, defs: dict[str, Any], seen_refs: frozenset[str] = frozenset()
) -> Any:
    """Rewrite a JSON schema into what the Harmony TypeScript renderer shows.

    The renderer drops `$ref`, `const`, `anyOf` of objects, enums without a
    string type, `integer` and every constraint, so GPT-OSS never sees them.
    Inline local refs, turn `const` and untyped enums into string enums, and
    move what a TypeScript type can't express into the property comment.
    """
    if not isinstance(schema, dict):
        return schema
    schema = dict(schema)
    for key in ("anyOf", "oneOf"):
        variants = schema.get(key)
        if (
            isinstance(variants, list)
            and len(variants) == 2
            and {"type": "null"} in variants
        ):
            # Optional[X]: render X, keeping its enum and constraints.
            value = next(v for v in variants if v != {"type": "null"})
            if isinstance(value, dict):
                rest = {k: v for k, v in schema.items() if k != key}
                optional = _schema_for_harmony({**value, **rest}, defs, seen_refs)
                optional["description"] = "; ".join(
                    filter(None, [optional.get("description"), "or null"])
                )
                return optional
    notes: list[str] = []

    ref = schema.pop("$ref", None)
    if isinstance(ref, str):
        name = ref.rsplit("/", 1)[-1]
        if ref in seen_refs or name not in defs:
            notes.append(f"a {name} object")
        else:
            schema = {**defs[name], **schema}
            seen_refs = seen_refs | {ref}

    if "const" in schema:
        schema["enum"] = [schema.pop("const")]

    enum = schema.get("enum")
    if isinstance(enum, list):
        values = [v for v in enum if v is not None]
        if len(values) < len(enum):
            notes.append("or null")
        if values and all(isinstance(v, str) for v in values):
            schema["type"] = "string"
            schema["enum"] = values
        else:
            schema.pop("enum")
            notes.append("one of " + ", ".join(json.dumps(v) for v in enum))

    for key in ("anyOf", "oneOf"):
        variants = schema.pop(key, None)
        if not isinstance(variants, list):
            continue
        variants = [_schema_for_harmony(v, defs, seen_refs) for v in variants]
        types = [v.get("type") for v in variants if isinstance(v, dict)]
        if len(types) == len(variants) and all(
            isinstance(t, str) and t not in ("object", "array") for t in types
        ):
            schema["type"] = types
        else:
            notes.append(
                "one of: "
                + " | ".join(json.dumps(v, separators=(",", ":")) for v in variants)
            )

    schema_type = schema.get("type")
    if schema_type == "integer" or (
        isinstance(schema_type, list) and "integer" in schema_type
    ):
        notes.append("integer")
    for key in _RENDERED_AS_NOTE:
        if key in schema:
            notes.append(f"{key}: {json.dumps(schema[key])}")

    if isinstance(schema.get("properties"), dict):
        schema.setdefault("type", "object")
        schema["properties"] = {
            name: _schema_for_harmony(prop, defs, seen_refs)
            for name, prop in schema["properties"].items()
        }
    if "items" in schema:
        schema["items"] = _schema_for_harmony(schema["items"], defs, seen_refs)
    if isinstance(schema.get("additionalProperties"), dict):
        value = _schema_for_harmony(schema.pop("additionalProperties"), defs, seen_refs)
        notes.append(
            "map of string keys to " + json.dumps(value, separators=(",", ":"))
        )

    if notes:
        description = " ".join(str(schema.get("description") or "").split())
        schema["description"] = "; ".join(filter(None, [description, *notes]))
    return schema


def _parameters_for_harmony(parameters: dict[str, Any] | None) -> Any:
    if not parameters:
        return parameters
    parameters = dict(parameters)
    defs = {
        **parameters.pop("definitions", {}),
        **parameters.pop("$defs", {}),
    }
    return _schema_for_harmony(parameters, defs)


def create_tool_definition(tool: ChatCompletionToolsParam | Tool):
    if isinstance(tool, ChatCompletionToolsParam):
        return ToolDescription.new(
            name=tool.function.name,
            description=tool.function.description or "",
            parameters=_parameters_for_harmony(tool.function.parameters),
        )
    return ToolDescription.new(
        name=tool.name,
        description=tool.description or "",
        parameters=_parameters_for_harmony(tool.parameters),
    )


def get_developer_message(
    instructions: str | None = None,
    tools: list[Tool | ChatCompletionToolsParam] | None = None,
) -> Message:
    dev_msg_content = DeveloperContent.new()
    if instructions is not None and not envs.VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS:
        dev_msg_content = dev_msg_content.with_instructions(instructions)
    if tools is not None:
        function_tools: list[Tool | ChatCompletionToolsParam] = []
        for tool in tools:
            if tool.type in (
                "web_search_preview",
                "code_interpreter",
                "container",
            ):
                pass

            elif tool.type == "function":
                function_tools.append(tool)
            else:
                raise VLLMValidationError(
                    f"tool type {tool.type!r} is not supported.",
                    parameter="tools",
                )
        if function_tools:
            function_tool_descriptions = [
                create_tool_definition(tool) for tool in function_tools
            ]
            dev_msg_content = dev_msg_content.with_function_tools(
                function_tool_descriptions
            )
    dev_msg = Message.from_role_and_content(Role.DEVELOPER, dev_msg_content)
    return dev_msg


def get_user_message(content: str) -> Message:
    return Message.from_role_and_content(Role.USER, content)


def get_system_or_developer_message(role: str, instructions: str) -> Message:
    if role == "system" and envs.VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS:
        return get_system_message(instructions=instructions)
    return get_developer_message(instructions=instructions)


def parse_chat_inputs_to_harmony_messages(chat_msgs: list) -> list[Message]:
    """Parse a list of messages from request.messages in the Chat Completion API to
    Harmony messages.
    """
    msgs: list[Message] = []
    tool_id_names: dict[str, str] = {}

    # Collect tool id to name mappings for tool response recipient values
    for chat_msg in chat_msgs:
        for tool_call in chat_msg.get("tool_calls", []):
            tool_id_names[tool_call.get("id")] = tool_call.get("function", {}).get(
                "name"
            )

    # A tool result without a known `tool_call_id` answers the earliest
    # unanswered call of the preceding assistant turn.
    unanswered: list[tuple[str | None, str]] = []
    for chat_msg in chat_msgs:
        role = chat_msg.get("role")
        if role == "assistant" and chat_msg.get("tool_calls"):
            unanswered = [
                (tool_call.get("id"), tool_call.get("function", {}).get("name"))
                for tool_call in chat_msg["tool_calls"]
            ]
        elif role == "tool":
            tool_call_id = chat_msg.get("tool_call_id")
            name = chat_msg.get("name")
            if tool_call_id in tool_id_names:
                unanswered = [c for c in unanswered if c[0] != tool_call_id]
            elif name:
                answered = next((c for c in unanswered if c[1] == name), None)
                if answered is not None:
                    unanswered.remove(answered)
            elif unanswered:
                chat_msg = {**chat_msg, "name": unanswered.pop(0)[1]}
        msgs.extend(parse_chat_input_to_harmony_message(chat_msg, tool_id_names))

    return msgs


def auto_drop_analysis_messages(msgs: list[Message]) -> list[Message]:
    """Harmony models expect the analysis messages (representing raw chain of thought)
    to be dropped after an assistant message to the final channel is produced from the
    reasoning of those messages.

    The openai-harmony library does this if the very last assistant message is to the
    final channel, but it does not handle the case where we're in longer multi-turn
    conversations and the client gave us reasoning content from previous turns of
    the conversation with multiple assistant messages to the final channel in the
    conversation.

    So, we find the index of the last assistant message to the final channel and drop
    all analysis messages that precede it, leaving only the analysis messages that
    are relevant to the current part of the conversation.
    """
    last_assistant_final_index = -1
    for i in range(len(msgs) - 1, -1, -1):
        msg = msgs[i]
        if msg.author.role == "assistant" and msg.channel == "final":
            last_assistant_final_index = i
            break

    cleaned_msgs: list[Message] = []
    for i, msg in enumerate(msgs):
        if i < last_assistant_final_index and msg.channel == "analysis":
            continue
        cleaned_msgs.append(msg)

    return cleaned_msgs


def flatten_input_text_content(content: Any) -> str | None:
    """Extract text parts from a Chat Completion or Responses API content field and
    flatten them into a single string. Returns None if no text content is found.
    """
    if content is None or isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None

    texts: list[str] = []
    for item in content:
        if isinstance(item, str):
            texts.append(item)
            continue
        if isinstance(item, dict):
            text = item.get("text")
            if text is not None:
                texts.append(text)
    return "".join(texts) if texts else None


def extract_instructions_from_messages(
    messages: Sequence[Any],
) -> tuple[str | None, list[Any]]:
    """Peel a leading system/developer Chat Completion or Responses message and
    flatten its instruction text.
    """
    remaining_messages = list(messages)
    if not remaining_messages:
        return None, remaining_messages

    first_message = remaining_messages[0]
    if not isinstance(first_message, dict):
        if hasattr(first_message, "to_dict"):
            # Handle OpenAI Harmony Message
            first_message = first_message.to_dict()
        elif hasattr(first_message, "model_dump"):
            first_message = first_message.model_dump(exclude_none=True)
        else:
            raise VLLMValidationError(
                f"Unknown message type: {type(first_message)}",
                parameter="input",
            )

    if first_message.get("role") not in (
        "system",
        "developer",
    ):
        return None, remaining_messages

    instructions = flatten_input_text_content(first_message.get("content"))
    return instructions, remaining_messages[1:]


def build_harmony_preamble(
    *,
    instructions: str | None = None,
    tools: list[Tool | ChatCompletionToolsParam] | None = None,
    reasoning_effort: str | None = None,
    browser_description: str | None = None,
    python_description: str | None = None,
    container_description: str | None = None,
    with_custom_tools: bool = False,
) -> list[Message]:
    """Build the standard Harmony system/developer prefix for a request."""
    developer_instructions = system_instructions = None
    if envs.VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS:
        system_instructions = instructions
    else:
        developer_instructions = instructions

    messages = [
        get_system_message(
            reasoning_effort=reasoning_effort,
            browser_description=browser_description,
            python_description=python_description,
            container_description=container_description,
            instructions=system_instructions,
            with_custom_tools=with_custom_tools,
        )
    ]
    if developer_instructions or tools:
        messages.append(
            get_developer_message(
                instructions=developer_instructions,
                tools=tools,
            )
        )
    return messages


def parse_chat_input_to_harmony_message(
    chat_msg, tool_id_names: dict[str, str] | None = None
) -> list[Message]:
    """Parse a message from request.messages in the Chat Completion API to
    Harmony messages.
    """
    tool_id_names = tool_id_names or {}

    if not isinstance(chat_msg, dict):
        # Handle Pydantic models
        chat_msg = chat_msg.model_dump(exclude_none=True)

    role = chat_msg.get("role")
    msgs: list[Message] = []

    # Assistant message with tool calls
    tool_calls = chat_msg.get("tool_calls", [])

    if role == "assistant" and tool_calls:
        content = flatten_input_text_content(chat_msg.get("content"))
        if content:
            commentary_msg = Message.from_role_and_content(Role.ASSISTANT, content)
            commentary_msg = commentary_msg.with_channel("commentary")
            msgs.append(commentary_msg)

        reasoning = chat_msg.get("reasoning")
        if reasoning:
            analysis_msg = Message.from_role_and_content(Role.ASSISTANT, reasoning)
            analysis_msg = analysis_msg.with_channel("analysis")
            msgs.append(analysis_msg)

        for call in tool_calls:
            func = call.get("function", {})
            name = func.get("name", "")
            arguments = func.get("arguments", "") or ""
            msg = Message.from_role_and_content(Role.ASSISTANT, arguments)
            msg = msg.with_channel("commentary")
            msg = msg.with_recipient(f"functions.{name}")
            # Officially, this should be `<|constrain|>json` but there is not clear
            # evidence that improves accuracy over `json` and some anecdotes to the
            # contrary. Further testing of the different content_types is needed.
            msg = msg.with_content_type("json")
            msgs.append(msg)
        return msgs

    # Tool role message (tool output)
    if role == "tool":
        tool_call_id = chat_msg.get("tool_call_id", "")
        name = tool_id_names.get(tool_call_id) or chat_msg.get("name") or ""
        content = flatten_input_text_content(chat_msg.get("content")) or ""

        msg = (
            Message.from_author_and_content(
                Author.new(Role.TOOL, f"functions.{name}"), content
            )
            .with_channel("commentary")
            .with_recipient("assistant")
        )
        return [msg]

    # Non-tool reasoning content
    reasoning = chat_msg.get("reasoning")
    if role == "assistant" and reasoning:
        analysis_msg = Message.from_role_and_content(Role.ASSISTANT, reasoning)
        analysis_msg = analysis_msg.with_channel("analysis")
        msgs.append(analysis_msg)

    # Default: user/assistant/system messages with content
    content = chat_msg.get("content") or ""
    if content is None:
        content = ""
    if isinstance(content, str):
        contents = [TextContent(text=content)]
    else:
        # TODO: Support refusal.
        contents = [TextContent(text=c.get("text", "")) for c in content]

    # Only add assistant messages if they have content, as reasoning or tool calling
    # assistant messages were already added above.
    if role == "assistant" and contents and contents[0].text:
        msg = Message.from_role_and_contents(role, contents)
        # Send non-tool assistant messages to the final channel
        msg = msg.with_channel("final")
        msgs.append(msg)
    elif role in ("system", "developer"):
        instructions = flatten_input_text_content(chat_msg.get("content"))
        if instructions is not None:
            msg = get_system_or_developer_message(role, instructions)
            msgs.append(msg)
    # For user messages, add them directly even if no content.
    elif role != "assistant":
        msg = Message.from_role_and_contents(role, contents)
        msgs.append(msg)

    return msgs


def render_for_completion(messages: list[Message]) -> list[int]:
    messages = auto_drop_analysis_messages(messages)
    conversation = Conversation.from_messages(messages)
    encoding = get_encoding()
    token_ids = encoding.render_conversation_for_completion(
        conversation,
        Role.ASSISTANT,
        config=RenderConversationConfig(auto_drop_analysis=False),
    )
    return _use_gpt_oss_tool_headers(token_ids)


def _use_gpt_oss_tool_headers(token_ids: list[int]) -> list[int]:
    """Keep prior tool headers in the format expected by GPT-OSS.

    oss-harmony 0.0.10 changed a rendered built-in tool-call header from::

        assistant to=python<|channel|>commentary code

    to::

        assistant<|channel|>commentary to=python <|constrain|>code

    GPT-OSS can echo the latter as a malformed header with the recipient in
    both positions. Rewrite only that unambiguous token-level header shape,
    while retaining oss-harmony's embedded vocabulary and parser.

    Function calls keep oss-harmony's header, which is the one GPT-OSS emits
    itself. In the legacy shape, a few prior calls are enough for most
    samples to open with a malformed header that fails to parse.

    Tool results are rendered as in the Harmony format documentation::

        functions.get_weather to=assistant<|channel|>commentary

    instead of oss-harmony's::

        functions.get_weather<|channel|>commentary to=assistant
    """
    encoding = get_encoding()
    start = encoding.encode("<|start|>", allowed_special="all")[0]
    channel_token = encoding.encode("<|channel|>", allowed_special="all")[0]
    constrain = encoding.encode("<|constrain|>", allowed_special="all")[0]
    message = encoding.encode("<|message|>", allowed_special="all")[0]

    result = token_ids.copy()
    index = 0
    while index < len(result):
        if result[index] != start:
            index += 1
            continue

        try:
            header_end = result.index(message, index + 1)
            channel_index = result.index(channel_token, index + 1, header_end)
        except ValueError:
            index += 1
            continue

        author = encoding.decode(result[index + 1 : channel_index])
        if author != "assistant":
            content_type = result[channel_index + 1 : header_end]
            if constrain in content_type:
                content_type = content_type[content_type.index(constrain) :]
            else:
                content_type = []
            channel_and_recipient = encoding.decode(
                result[channel_index + 1 : header_end - len(content_type)]
            )
            if " to=" not in channel_and_recipient:
                index = header_end + 1
                continue
            channel, recipient = channel_and_recipient.split(" to=", 1)
            tool_result_header = (
                [start]
                + encoding.encode(f"{author} to={recipient.rstrip()}")
                + [channel_token]
                + encoding.encode(channel + (" " if content_type else ""))
                + content_type
                + [message]
            )
            result[index : header_end + 1] = tool_result_header
            index += len(tool_result_header)
            continue

        try:
            constrain_index = result.index(constrain, channel_index + 1, header_end)
        except ValueError:
            index = header_end + 1
            continue

        channel_and_recipient = encoding.decode(
            result[channel_index + 1 : constrain_index]
        )
        content_type = encoding.decode(result[constrain_index + 1 : header_end])
        if (
            " to=" not in channel_and_recipient
            or " to=functions." in channel_and_recipient
            or not channel_and_recipient.endswith(" ")
        ):
            index = header_end + 1
            continue

        channel, recipient = channel_and_recipient[:-1].split(" to=", 1)
        if not channel or not recipient or not content_type:
            index = header_end + 1
            continue

        legacy_header = (
            [start]
            + encoding.encode(f"assistant to={recipient}")
            + [channel_token]
            + encoding.encode(f"{channel} {content_type}")
            + [message]
        )
        result[index : header_end + 1] = legacy_header
        index += len(legacy_header)

    return result


def get_streamable_parser_for_assistant() -> StreamableParser:
    return StreamableParser(get_encoding(), role=Role.ASSISTANT)
