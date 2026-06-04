# llm-kimi

An [LLM](https://llm.datasette.io/) plugin to access [Moonshots's Kimi models](https://platform.kimi.com/)

## Installation

```shell
git clone https://github.com/ziqin/llm-kimi.git
cd llm-kimi
llm install --editable .
```

## Usage

Configure the model by setting a key called "kimi" to your [API key](https://platform.kimi.com/console/api-keys):
```shell
llm keys set kimi
```

You can also set the API key by assigning it to the environment variable `MOONSHOT_API_KEY`.

Now run the model using `-m kimi-k2.6`, for example:
```shell
llm -m kimi-k2.6 "Hi"
```

To enable thinking mode, use `-o thinking true`:
```shell
llm -m kimi-k2.6 -o thinking true "I need to get my car washed. The car wash is 100m away. Should I go by car or by foot?"
```

To use Kimi's Internet search functionality, use `-T '$web_search'`:
```shell
llm -m kimi-k2.6 -T '$web_search' "Introduce the first astronaut from Hong Kong"
```

Display account balance:
```shell
llm kimi balance
```

## Available models

| Model ID  | Alias |
| --------- | ----- |
| kimi-k2.6 | k2.6  |
| kimi-k2.5 | k2.5  |
