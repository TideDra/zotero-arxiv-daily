"""Feishu webhook card notification for daily arXiv papers."""

import json
from datetime import datetime

import requests
from loguru import logger
from omegaconf import DictConfig

from .protocol import Paper

FEISHU_WEBHOOK_TIMEOUT = (10, 30)


def _build_card(papers: list[Paper]) -> dict:
    """Build a Feishu interactive card JSON payload for the given papers."""

    today = datetime.now().strftime("%Y/%m/%d")

    if not papers:
        return {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {"tag": "plain_text", "content": f"📚 Daily arXiv - {today}"},
                    "template": "blue",
                },
                "elements": [
                    {
                        "tag": "div",
                        "text": {
                            "tag": "lark_md",
                            "content": "今日无新论文，休息一天～ 🎉",
                        },
                    }
                ],
            },
        }

    elements: list[dict] = []
    for i, p in enumerate(papers):
        lines: list[str] = []

        # Title
        title = p.title.replace("\n", " ").strip()
        lines.append(f"**📄 {title}**")

        # Authors
        author_list = [a for a in p.authors]
        if len(author_list) <= 5:
            authors = ", ".join(author_list)
        else:
            authors = ", ".join(author_list[:3] + ["..."] + author_list[-2:])
        lines.append(f"*{authors}*")

        # Affiliations
        if p.affiliations:
            affs = p.affiliations[:3]
            aff_text = ", ".join(affs)
            if len(p.affiliations) > 3:
                aff_text += ", ..."
            lines.append(f"🏛 {aff_text}")

        # Relevance score
        score = f"{p.score:.1f}" if p.score is not None else "N/A"
        lines.append(f"⭐ 相关性: **{score}**")

        # TL;DR
        tldr = (p.tldr or p.abstract or "暂无摘要").replace("\n", " ").strip()
        if len(tldr) > 300:
            tldr = tldr[:300] + "..."
        lines.append(f"💡 {tldr}")

        # Links
        links: list[str] = []
        if p.pdf_url:
            links.append(f"[PDF]({p.pdf_url})")
        links.append(f"[arXiv]({p.url})")
        lines.append(" | ".join(links))

        elements.append(
            {
                "tag": "div",
                "text": {"tag": "lark_md", "content": "\n".join(lines)},
            }
        )

        # Horizontal rule between papers
        if i < len(papers) - 1:
            elements.append({"tag": "hr"})

    # Footer note
    elements.append({"tag": "hr"})
    elements.append(
        {
            "tag": "note",
            "elements": [
                {
                    "tag": "plain_text",
                    "content": f"共 {len(papers)} 篇论文 | 由 zotero-arxiv-daily 自动生成",
                }
            ],
        }
    )

    return {
        "msg_type": "interactive",
        "card": {
            "header": {
                "title": {"tag": "plain_text", "content": f"📚 Daily arXiv - {today}"},
                "template": "blue",
            },
            "elements": elements,
        },
    }


def send_feishu_card(papers: list[Paper], config: DictConfig) -> bool:
    """Send a Feishu card notification via webhook.

    Returns True on success, False on failure (never raises).
    """
    webhook_url = config.feishu.webhook_url

    if not webhook_url:
        logger.warning("Feishu webhook URL is empty, skipping notification")
        return False

    card = _build_card(papers)

    try:
        resp = requests.post(
            webhook_url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(card, ensure_ascii=False).encode("utf-8"),
            timeout=FEISHU_WEBHOOK_TIMEOUT,
        )
        resp.raise_for_status()
        result = resp.json()
        if result.get("code") != 0:
            logger.warning(f"Feishu webhook returned error: {result}")
            return False
        return True
    except Exception as e:
        logger.warning(f"Failed to send Feishu notification: {e}")
        return False
