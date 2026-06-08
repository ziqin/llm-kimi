from __future__ import annotations

import json
from typing import Any, Iterator, TYPE_CHECKING, override

import httpx
import llm
from llm.default_plugins.openai_models import Chat, ImageDetailEnum, combine_chunks
from llm.parts import StreamEvent
from llm.utils import remove_dict_none_values

if TYPE_CHECKING:
    from openai import OpenAI
    from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall


API_BASE = 'https://api.moonshot.cn/v1'


@llm.hookimpl
def register_models(register):
    register(KimiModel(model_id='kimi-k2.6'), aliases=('k2.6',))
    register(KimiModel(model_id='kimi-k2.5'), aliases=('k2.5',))


class KimiModel(Chat):
    needs_key = 'kimi'
    key_env_var = 'MOONSHOT_API_KEY'

    class Options(llm.Options):
        json_object: bool = False
        thinking: bool = False
        max_tokens: int | None = None

    @override
    def __init__(self, model_id: str):
        super().__init__(
            model_id=model_id,
            model_name=model_id,
            api_base=API_BASE,
            supports_schema=True,
            supports_tools=True,
        )
        self._last_reasoning_content = None

    @override
    def __str__(self) -> str:
        return 'KimiModel: ' + self.model_id

    @override
    def execute(self, prompt: llm.Prompt, stream: bool, response: llm.Response, conversation: llm.Conversation | None = None, key: str | None = None) -> Iterator[StreamEvent]:
        messages = self.build_messages(prompt, conversation)
        kwargs = self.build_kwargs(prompt, stream)
        client: OpenAI = self.get_client(key=key)
        usage = None
        completion = client.chat.completions.create(
            model=self.model_name or self.model_id,
            messages=messages,
            stream=stream,
            extra_body={'thinking': {'type': 'enabled' if prompt.options.thinking else 'disabled'}},
            **kwargs,
        )
        if stream:
            chunks = []
            tool_calls = {}
            reasoning_content = ''
            for chunk in completion:
                chunks.append(chunk)
                if chunk.usage:
                    usage = chunk.usage.model_dump()
                try:
                    delta = chunk.choices[0].delta
                    KimiModel._merge_tool_calls(tool_calls, delta.tool_calls or [])
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                        yield StreamEvent(type='reasoning', chunk=delta.reasoning_content)
                        reasoning_content += delta.reasoning_content
                    if delta.content:
                        yield StreamEvent(type='text', chunk=delta.content)
                except IndexError:
                    pass
            response_dict = combine_chunks(chunks)
            if tool_calls:
                for value in tool_calls.values():
                    response.add_tool_call(llm.ToolCall(
                        name=value.function.name,
                        arguments=json.loads(value.function.arguments),
                        tool_call_id=value.id))
        else:
            if completion.usage:
                usage = completion.usage.model_dump()
            response_dict = completion.model_dump()
            try:
                message = completion.choices[0].message
                reasoning_content = getattr(message, 'reasoning_content', default='')
                if reasoning_content:
                    yield StreamEvent(type='reasoning', chunk=reasoning_content)
                yield StreamEvent(type='text', chunk=message.content)
            except IndexError:
                pass
        self._last_reasoning_content = reasoning_content
        response.response_json = remove_dict_none_values(response_dict)
        self.set_usage(response, usage)

    @staticmethod
    def _merge_tool_calls(output: dict[int, ChoiceDeltaToolCall], tool_calls: list[ChoiceDeltaToolCall]):
        for tool_call in tool_calls:
            if tool_call.function.arguments is None:
                tool_call.function.arguments = ''
            index = tool_call.index
            if index not in output:
                output[index] = tool_call
            else:
                output[index].function.arguments += tool_call.function.arguments

    @override
    def build_messages(self, prompt: llm.Prompt, conversation: llm.Conversation | None, image_detail: ImageDetailEnum | None = None) -> list[dict[str, Any]]:
        messages = super().build_messages(prompt, conversation, image_detail)
        # Monkey patching
        for message in messages:
            if message['role'] == 'assistant' and 'tool_calls' in message:
                for tool_call in message['tool_calls']:
                    if tool_call['function']['name'] == '$web_search':
                        tool_call['type'] = 'builtin_function'
        if prompt.options.thinking:
            for message in reversed(messages):
                if message['role'] == 'assistant' and 'tool_calls' in message and 'reasoning_content' not in message:
                    # In thinking mode, Kimi requires reasoning_content for tool_calls
                    message['reasoning_content'] = self._last_reasoning_content or ''
                    break
        return messages

    @override
    def build_kwargs(self, prompt: llm.Prompt, stream: bool) -> dict[str, Any]:
        kwargs = super().build_kwargs(prompt, stream)
        if 'thinking' in kwargs:
            del kwargs['thinking']
        if 'tools' in kwargs:
            for tool in kwargs['tools']:
                if tool['function']['name'] == '$web_search':
                    tool['type'] = 'builtin_function'
                    del tool['function']['description']
                    del tool['function']['parameters']
        return kwargs


# def web_search(arguments: dict[str, Any]):
def web_search(**kwargs):
    """Kimi's builtin web searching tool"""
    # https://platform.kimi.com/docs/guide/use-web-search
    return kwargs


@llm.hookimpl
def register_tools(register):
    register(web_search, name='$web_search')


class BearerAuth(httpx.Auth):
    def __init__(self, token: str):
        self.token = token

    def auth_flow(self, request: httpx.Request):
        request.headers['Authorization'] = 'Bearer ' + self.token
        yield request


@llm.hookimpl
def register_commands(cli):
    @cli.group()
    def kimi():
        """Commands for working directly with the Kimi API"""

    @kimi.command()
    def balance():
        """Display balance of the Moonshot account"""
        key = llm.get_key(None, KimiModel.needs_key, KimiModel.key_env_var)
        response = httpx.get(API_BASE + '/users/me/balance', auth=BearerAuth(key))
        response.raise_for_status()
        result = response.json()
        if result['code'] != 0:
            raise ValueError(f"code {result['code']}")
        data = result['data']
        print(f"Available balance: {data['available_balance']:>10.2f}")
        print(f"Voucher balance:   {data['voucher_balance']:>10.2f}")
        print(f"Cash balance:      {data['cash_balance']:>10.2f}")
