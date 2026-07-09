from dataclasses import dataclass
from typing import Optional, TypeVar
from datetime import datetime
import re
import tiktoken
from openai import OpenAI
from loguru import logger
import json
RawPaperItem = TypeVar('RawPaperItem')

# Default section labels for the structured TLDR, used when config omits them.
DEFAULT_TLDR_LABELS = {"problem": "Problem", "idea": "Idea", "result": "Result"}

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

        if not self.full_text and not self.abstract:
            logger.warning(f"Neither full text nor abstract is provided for {self.url}")
            return "Failed to generate TLDR. Neither full text nor abstract is provided"

        # Assemble the paper context, then truncate the context (not the instructions) to 4000 tokens.
        context = ""
        if self.title:
            context += f"Title:\n {self.title}\n\n"
        if self.abstract:
            context += f"Abstract: {self.abstract}\n\n"
        if self.full_text:
            context += f"Preview of main content:\n {self.full_text}\n\n"
        enc = tiktoken.encoding_for_model("gpt-4o")  # gpt-4o tokenizer for estimation
        context = enc.decode(enc.encode(context)[:4000])

        tldr_cfg = llm_params.get('tldr', {}) or {}
        style = tldr_cfg.get('style', 'structured')
        generation_kwargs = llm_params.get('generation_kwargs', {})

        if style == 'plain':
            system_prompt = (
                "You are an assistant who perfectly summarizes scientific paper, and gives the "
                f"core idea of the paper to the user. Your answer should be in {lang}."
            )
            user_prompt = (
                f"Given the following information of a paper, generate a one-sentence TLDR "
                f"summary in {lang}:\n\n{context}"
            )
            response = openai_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                **generation_kwargs,
            )
            return response.choices[0].message.content

        # style == 'structured' (default): Problem / Idea / Result in plain language.
        max_words = tldr_cfg.get('max_words', 60)
        system_prompt = (
            "You explain scientific papers to a curious non-expert. Use plain, simple language, "
            "and add a short everyday analogy or example only when it genuinely aids understanding. "
            "Be concrete and avoid jargon. "
            f"Write your answer in {lang}. "
            "Respond with ONLY a JSON object and nothing else: no markdown, no code fences, no commentary."
        )
        user_prompt = (
            "Summarize the following paper as a JSON object with exactly these keys:\n"
            '- "problem": one plain sentence describing the problem or gap the paper tackles.\n'
            '- "idea": the core idea or approach in simple terms, with a brief example if it helps.\n'
            '- "result": the main outcome or finding if the paper states one, otherwise an empty string "".\n\n'
            f"Keep the entire summary under {max_words} words. Write the values in {lang}.\n\n"
            f"{context}"
        )
        response = openai_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **generation_kwargs,
        )
        return self._format_structured_tldr(response.choices[0].message.content, tldr_cfg)

    def _format_structured_tldr(self, raw:str, tldr_cfg:dict) -> str:
        # Robustly extract the JSON object, mirroring the affiliation parsing approach.
        match = re.search(r'\{.*\}', raw, flags=re.DOTALL)
        if match is None:
            raise ValueError(f"No JSON object found in TLDR response: {raw!r}")
        data = json.loads(match.group(0))

        problem = str(data.get("problem", "") or "").strip()
        idea = str(data.get("idea", "") or "").strip()
        result = str(data.get("result", "") or "").strip()
        if not problem and not idea:
            raise ValueError(f"TLDR JSON missing both problem and idea: {raw!r}")

        labels = {**DEFAULT_TLDR_LABELS, **dict(tldr_cfg.get('labels', {}) or {})}
        fields = [("problem", problem), ("idea", idea), ("result", result)]
        lines = [f"<strong>{labels.get(key, key.capitalize())}:</strong> {value}"
                 for key, value in fields if value]
        return "<br>".join(lines)

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
            affiliations = openai_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant who perfectly extracts affiliations of authors from a paper. You should return a python list of affiliations sorted by the author order, like [\"TsingHua University\",\"Peking University\"]. If an affiliation is consisted of multi-level affiliations, like 'Department of Computer Science, TsingHua University', you should return the top-level affiliation 'TsingHua University' only. Do not contain duplicated affiliations. If there is no affiliation found, you should return an empty list [ ]. You should only return the final list of affiliations, and do not return any intermediate results.",
                    },
                    {"role": "user", "content": prompt},
                ],
                **llm_params.get('generation_kwargs', {})
            )
            affiliations = affiliations.choices[0].message.content

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