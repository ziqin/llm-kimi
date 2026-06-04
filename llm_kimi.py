import json
from typing import Iterator, TYPE_CHECKING, override

import httpx
import llm
from llm.default_plugins.openai_models import Chat, combine_chunks
from llm.utils import remove_dict_none_values
from rich.console import Console
from rich.style import Style

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
        self._console = Console()
        self._header_style = Style(color='cyan', bold=True)
        self._reasoning_style = Style(color='cyan', dim=True)
        self._last_reasoning_content = None

    @override
    def __str__(self) -> str:
        return 'KimiModel: ' + self.model_id

    @override
    def execute(self, prompt: llm.Prompt, stream: bool, response: llm.Response, conversation: llm.Conversation | None = None, key: str | None = None) -> Iterator[str]:
        messages = self.build_messages(prompt, conversation)
        kwargs = self.build_kwargs(prompt, stream)
        reasoning_enabled = kwargs.pop('thinking')
        self._patch_tool_call_reasoning_message(messages, reasoning_enabled)
        client: OpenAI = self.get_client(key=key)
        usage = None
        completion = client.chat.completions.create(
            model=self.model_name or self.model_id,
            messages=messages,
            stream=stream,
            extra_body={'thinking': {'type': 'enabled' if reasoning_enabled else 'disabled'}},
            **kwargs,
        )
        if stream:
            chunks = []
            tool_calls = {}
            reasoning_started = False
            reasoning_content = None
            for chunk in completion:
                chunks.append(chunk)
                if chunk.usage:
                    usage = chunk.usage.model_dump()
                try:
                    delta = chunk.choices[0].delta
                    KimiModel._merge_tool_calls(tool_calls, delta.tool_calls or [])
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                        if not reasoning_started:
                            reasoning_started = True
                            reasoning_content = ''
                            self._print_reasoning(delta.reasoning_content, print_header=True)
                        else:
                            self._print_reasoning(delta.reasoning_content, print_header=False)
                        reasoning_content += delta.reasoning_content
                    if delta.content:
                        if reasoning_started:
                            reasoning_started = False
                            self._console.print('\n\n 💬 Answer:', style=self._header_style, end='\n\n')
                        yield delta.content
                except IndexError:
                    pass
            response_dict = combine_chunks(chunks)
            response_dict['reasoning_content'] = reasoning_content
            self._last_reasoning_content = reasoning_content
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
                if hasattr(message, 'reasoning_content') and message.reasoning_content:
                    self._print_reasoning(message.reasoning_content, print_header=True)
                    self._console.print('\n\n 💬 Answer:', style=self._header_style, end='\n\n')
                    self._last_reasoning_content = message.reasoning_content
                else:
                    self._last_reasoning_content = None
                yield message.content
            except IndexError:
                pass
        response.response_json = remove_dict_none_values(response_dict)
        self.set_usage(response, usage)

    def _print_reasoning(self, reasoning_content: str, print_header: bool):
        if print_header:
            self._console.print('\n 💭 Thinking:', style=self._header_style, end='\n\n')
        self._console.print(reasoning_content, style=self._reasoning_style, end='')

    def _patch_tool_call_reasoning_message(self, messages: list[dict], reasoning: bool):
        if reasoning:
            for message in reversed(messages):
                if message['role'] == 'assistant' and 'tool_calls' in message and 'reasoning_content' not in message:
                    message['reasoning_content'] = self._last_reasoning_content
                    break

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


class BearerAuth(httpx.Auth):
    def __init__(self, token: str):
        self.token = token

    def auth_flow(self, request: httpx.Request):
        request.headers["Authorization"] = 'Bearer ' + self.token
        yield request


@llm.hookimpl
def register_commands(cli):
    @cli.group()
    def kimi():
        "Commands relating to the llm-kimi plugin"

    @kimi.command()
    def balance():
        """Display balance of the Moonshot account"""
        key = llm.get_key(None, 'kimi', 'MOONSHOT_API_KEY')
        response = httpx.get(API_BASE + '/users/me/balance', auth=BearerAuth(key))
        response.raise_for_status()
        result = response.json()
        if result['code'] != 0:
            raise ValueError(f"code {result['code']}")
        data = result['data']
        print(f"Available balance: {data['available_balance']:>10.2f}")
        print(f"Voucher balance:   {data['voucher_balance']:>10.2f}")
        print(f"Cash balance:      {data['cash_balance']:>10.2f}")
