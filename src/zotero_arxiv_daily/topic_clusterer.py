import json
import re
from dataclasses import dataclass
from typing import Any

from loguru import logger
from openai import OpenAI, OpenAIError

from .protocol import Paper


@dataclass
class PaperGroup:
    label: str
    summary: str | None
    papers: list[Paper]


class TopicClusterer:
    GENERIC_LABEL_PATTERN = re.compile(r"^group\s+\d+$", re.IGNORECASE)

    def __init__(self, openai_client: OpenAI, llm_params: dict):
        self.openai_client = openai_client
        self.llm_params = llm_params

    @classmethod
    def _is_generic_label(cls, label: str) -> bool:
        return bool(cls.GENERIC_LABEL_PATTERN.fullmatch(label.strip()))

    def cluster_papers(self, papers: list[Paper]) -> list[PaperGroup]:
        if len(papers) < 6:
            logger.warning("Skipping topic clustering because there are too few papers")
            return [self._fallback_group(papers)]

        groups = self._request_groups(papers)
        if groups is None:
            logger.warning("Falling back to a single topic group after clustering failure")
            return [self._fallback_group(papers)]
        return groups

    def _request_groups(self, papers: list[Paper]) -> list[PaperGroup] | None:
        payload = self._build_prompt_payload(papers)
        system_prompts = [
            self._system_prompt(strict=False),
            self._system_prompt(strict=True),
        ]

        for attempt, system_prompt in enumerate(system_prompts):
            try:
                response = self.openai_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ],
                    response_format={"type": "json_object"},
                    **self.llm_params.get("generation_kwargs", {}),
                )
                content = response.choices[0].message.content
                parsed_groups = self._parse_groups(content)
                self._validate_groups(parsed_groups, len(papers))
                return self._materialize_groups(parsed_groups, papers)
            except OpenAIError as exc:
                logger.warning(f"Topic clustering request failed without retry: {exc}")
                return None
            except (ValueError, TypeError, KeyError, IndexError) as exc:
                if attempt == 0:
                    logger.warning(f"Invalid topic clustering output; retrying once with stricter instructions: {exc}")
                    continue
                logger.warning(f"Topic clustering failed after retry: {exc}")
        return None

    def _build_prompt_payload(self, papers: list[Paper]) -> dict[str, Any]:
        return {
            "instruction": "Cluster these papers into 3 to 5 topic groups when feasible.",
            "papers": [
                {
                    "index": index,
                    "title": paper.title,
                    "summary_text": paper.tldr if paper.tldr else paper.abstract,
                }
                for index, paper in enumerate(papers)
            ],
        }

    def _system_prompt(self, strict: bool) -> str:
        prompt = (
            "You group scientific papers by topic. Return a JSON object with a top-level 'groups' array. "
            "Each group must include 'label', 'summary', and 'paper_indices'."
        )
        if strict:
            prompt += (
                " Return JSON only. Use concise, non-generic topic labels. "
                "Provide a non-empty summary string for every group. "
                "Every paper index must appear exactly once, groups must be non-empty, and produce 3 to 5 groups."
            )
        return prompt

    def _parse_groups(self, content: str | None) -> list[dict[str, Any]]:
        if not content:
            raise ValueError("empty response")
        payload = json.loads(content)
        groups = payload.get("groups")
        if not isinstance(groups, list):
            raise ValueError("groups must be a list")
        return groups

    def _validate_groups(self, groups: list[dict[str, Any]], paper_count: int) -> None:
        if paper_count >= 6 and not 3 <= len(groups) <= 5:
            raise ValueError("group count must be between 3 and 5")

        seen_indices: list[int] = []
        for group in groups:
            if not isinstance(group, dict):
                raise ValueError("group must be an object")

            label = group.get("label")
            summary = group.get("summary")
            paper_indices = group.get("paper_indices")

            if not isinstance(label, str) or not label.strip():
                raise ValueError("group label must be non-empty")
            if self._is_generic_label(label):
                raise ValueError("group label must be non-generic")
            if not isinstance(summary, str) or not summary.strip():
                raise ValueError("group summary must be non-empty")
            if not isinstance(paper_indices, list) or len(paper_indices) == 0:
                raise ValueError("group paper_indices must be a non-empty list")

            for index in paper_indices:
                if not isinstance(index, int):
                    raise ValueError("paper index must be an int")
                if index < 0 or index >= paper_count:
                    raise ValueError("paper index out of range")
                if index in seen_indices:
                    raise ValueError("paper index duplicated across groups")
                seen_indices.append(index)

        expected_indices = list(range(paper_count))
        if sorted(seen_indices) != expected_indices:
            raise ValueError("paper indices must cover every paper exactly once")

    def _materialize_groups(self, groups: list[dict[str, Any]], papers: list[Paper]) -> list[PaperGroup]:
        materialized: list[tuple[int, PaperGroup]] = []
        for group in groups:
            sorted_indices = sorted(group["paper_indices"])
            materialized.append(
                (
                    min(sorted_indices),
                    PaperGroup(
                        label=group["label"].strip(),
                        summary=group["summary"].strip(),
                        papers=[papers[index] for index in sorted_indices],
                    ),
                )
            )
        materialized.sort(key=lambda item: item[0])
        return [group for _, group in materialized]

    def _fallback_group(self, papers: list[Paper]) -> PaperGroup:
        return PaperGroup(label="Relevant papers today", summary=None, papers=papers)
