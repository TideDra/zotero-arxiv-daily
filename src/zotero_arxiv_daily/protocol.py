from dataclasses import dataclass
from typing import Optional, TypeVar
from datetime import datetime
import re
import tiktoken
from openai import OpenAI
from loguru import logger
import json

try:
    import litellm
except ImportError:  # optional dependency, only needed when llm.use_litellm is true
    litellm = None

RawPaperItem = TypeVar('RawPaperItem')


def _litellm_credentials(llm_params: dict) -> tuple[Optional[str], Optional[str]]:
    """Read (api_key, api_base) from llm.api, treating unset/blank as None.

    Returning None lets LiteLLM fall back to the target provider's own
    environment variable (e.g. ANTHROPIC_API_KEY, GEMINI_API_KEY, AWS creds),
    which is what makes non-OpenAI-compatible providers work.
    """
    api = llm_params.get("api") or {}

    def _get(name: str) -> Optional[str]:
        try:
            value = api.get(name)
        except Exception:
            # OmegaConf raises on unset mandatory (???) values.
            return None
        return value if value not in (None, "", "???") else None

    return _get("key"), _get("base_url")


def _request_litellm(llm_params: dict, messages: list[dict]) -> str:
    if litellm is None:
        raise ImportError(
            "litellm is required when llm.use_litellm is true. "
            "Install it with `uv sync --extra litellm` or `pip install litellm`."
        )

    generation_kwargs = dict(llm_params.get('generation_kwargs', {}))
    # drop_params lets LiteLLM discard kwargs a given provider rejects (e.g.
    # Anthropic on seed/frequency_penalty), so one config works everywhere.
    call_kwargs = {"messages": messages, "drop_params": True, **generation_kwargs}

    api_key, api_base = _litellm_credentials(llm_params)
    if api_key:
        call_kwargs["api_key"] = api_key
    if api_base:
        call_kwargs["api_base"] = api_base

    response = litellm.completion(**call_kwargs)
    return response.choices[0].message.content


def _chat_completion(openai_client: OpenAI, llm_params: dict, messages: list[dict]) -> str:
    """Send a chat request and return the message content.

    Routes through LiteLLM when ``llm.use_litellm`` is set - one provider/model
    string then reaches 100+ providers, including Bedrock/Vertex/Azure AD whose
    auth is not OpenAI-compatible. Otherwise uses the OpenAI-compatible client.
    """
    if llm_params.get('use_litellm'):
        return _request_litellm(llm_params, messages)
    response = openai_client.chat.completions.create(
        messages=messages,
        **llm_params.get('generation_kwargs', {}),
    )
    return response.choices[0].message.content

@dataclass
class Paper:
    source: str
    title: str
    authors: list[str]
    abstract: str
    url: str
    pdf_url: Optional[str] = None
    full_text: Optional[str] = None
    tldr: Optional[str] = None
    affiliations: Optional[list[str]] = None
    score: Optional[float] = None

    def _generate_tldr_with_llm(self, openai_client:OpenAI,llm_params:dict) -> str:
        lang = llm_params.get('language', 'English')
        prompt = f"Given the following information of a paper, generate a one-sentence TLDR summary in {lang}:\n\n"
        if self.title:
            prompt += f"Title:\n {self.title}\n\n"

        if self.abstract:
            prompt += f"Abstract: {self.abstract}\n\n"

        if self.full_text:
            prompt += f"Preview of main content:\n {self.full_text}\n\n"

        if not self.full_text and not self.abstract:
            logger.warning(f"Neither full text nor abstract is provided for {self.url}")
            return "Failed to generate TLDR. Neither full text nor abstract is provided"
        
        # use gpt-4o tokenizer for estimation
        enc = tiktoken.encoding_for_model("gpt-4o")
        prompt_tokens = enc.encode(prompt)
        prompt_tokens = prompt_tokens[:4000]  # truncate to 4000 tokens
        prompt = enc.decode(prompt_tokens)
        
        tldr = _chat_completion(
            openai_client,
            llm_params,
            [
                {
                    "role": "system",
                    "content": f"You are an assistant who perfectly summarizes scientific paper, and gives the core idea of the paper to the user. Your answer should be in {lang}.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return tldr
    
    def generate_tldr(self, openai_client:OpenAI,llm_params:dict) -> str:
        try:
            tldr = self._generate_tldr_with_llm(openai_client,llm_params)
            self.tldr = tldr
            return tldr
        except Exception as e:
            logger.warning(f"Failed to generate tldr of {self.url}: {e}")
            tldr = self.abstract
            self.tldr = tldr
            return tldr

    def _generate_affiliations_with_llm(self, openai_client:OpenAI,llm_params:dict) -> Optional[list[str]]:
        if self.full_text is not None:
            prompt = f"Given the beginning of a paper, extract the affiliations of the authors in a python list format, which is sorted by the author order. If there is no affiliation found, return an empty list '[]':\n\n{self.full_text}"
            # use gpt-4o tokenizer for estimation
            enc = tiktoken.encoding_for_model("gpt-4o")
            prompt_tokens = enc.encode(prompt)
            prompt_tokens = prompt_tokens[:2000]  # truncate to 2000 tokens
            prompt = enc.decode(prompt_tokens)
            affiliations = _chat_completion(
                openai_client,
                llm_params,
                [
                    {
                        "role": "system",
                        "content": "You are an assistant who perfectly extracts affiliations of authors from a paper. You should return a python list of affiliations sorted by the author order, like [\"TsingHua University\",\"Peking University\"]. If an affiliation is consisted of multi-level affiliations, like 'Department of Computer Science, TsingHua University', you should return the top-level affiliation 'TsingHua University' only. Do not contain duplicated affiliations. If there is no affiliation found, you should return an empty list [ ]. You should only return the final list of affiliations, and do not return any intermediate results.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            affiliations = re.search(r'\[.*?\]', affiliations, flags=re.DOTALL).group(0)
            affiliations = json.loads(affiliations)
            affiliations = list(set(affiliations))
            affiliations = [str(a) for a in affiliations]

            return affiliations
    
    def generate_affiliations(self, openai_client:OpenAI,llm_params:dict) -> Optional[list[str]]:
        try:
            affiliations = self._generate_affiliations_with_llm(openai_client,llm_params)
            self.affiliations = affiliations
            return affiliations
        except Exception as e:
            logger.warning(f"Failed to generate affiliations of {self.url}: {e}")
            self.affiliations = None
            return None
@dataclass
class CorpusPaper:
    title: str
    abstract: str
    added_date: datetime
    paths: list[str]