from abc import ABC, abstractmethod
from openai import OpenAI
from together import Together
import anthropic
import subprocess

class LLMClient(ABC):
    @abstractmethod
    def generate(self, system, user_request):
        pass

class OpenAIClient(LLMClient):
    def __init__(self, api_key, model):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, system, user_request):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_request},
            ],
            temperature=0.8,
            max_tokens=2000,
            top_p=0.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        return response.choices[0].message.content

class TogetherClient(LLMClient):
    def __init__(self, api_key, model):
        self.client = Together(api_key=api_key)
        self.model = model

    def generate(self, system, user_request):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_request},
            ],
            temperature=0.8,
            max_tokens=2000,
            top_p=0.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        return response.choices[0].message.content

class AnthropicClient(LLMClient):
    def __init__(self, api_key, model):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(self, system, user_request):
        message = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            temperature=0.8,
            system=system,
            messages=[
                {"role": "user", "content": user_request}
            ]
        )
        return message.content[0].text

class CodexClient(LLMClient):
    """Simple wrapper around the Codex CLI agent."""

    def __init__(self, cmd: str):
        self.cmd = cmd

    def generate(self, system, user_request):
        prompt = f"{system}\n{user_request}"
        result = subprocess.run(
            [self.cmd],
            input=prompt,
            text=True,
            capture_output=True,
        )
        return result.stdout

def get_client(config):
    if config["model"].startswith("gpt"):
        return OpenAIClient(config["openai_api_key"], config["model"])
    elif config["model"].startswith("deepseek"):
        return TogetherClient(config["together_api_key"], config["model"])
    elif config["model"].startswith("claude"):
        return AnthropicClient(config["anthropic_api_key"], config["model"])
    elif config["model"].lower() == "codex":
        return CodexClient(config.get("codex_cmd", "codex"))
    else:
        raise ValueError(f"Unsupported model: {config['model']}")