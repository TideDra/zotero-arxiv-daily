from .protocol import Paper
from html import escape
import math


framework = """
<!DOCTYPE HTML>
<html>
<head>
  <style>
    .star-wrapper {
      font-size: 1.3em; /* 调整星星大小 */
      line-height: 1; /* 确保垂直对齐 */
      display: inline-flex;
      align-items: center; /* 保持对齐 */
    }
    .half-star {
      display: inline-block;
      width: 0.5em; /* 半颗星的宽度 */
      overflow: hidden;
      white-space: nowrap;
      vertical-align: middle;
    }
    .full-star {
      vertical-align: middle;
    }
  </style>
</head>
<body>

<div>
    __CONTENT__
</div>

<br><br>
<div>
To unsubscribe, remove your email in your Github Action setting.
</div>

</body>
</html>
"""

def get_empty_html():
  block_template = """
  <table border="0" cellpadding="0" cellspacing="0" width="100%" style="font-family: Arial, sans-serif; border: 1px solid #ddd; border-radius: 8px; padding: 16px; background-color: #f9f9f9;">
  <tr>
    <td style="font-size: 20px; font-weight: bold; color: #333;">
        No Papers Today. Take a Rest!
    </td>
  </tr>
  </table>
  """
  return block_template

def get_block_html(
    title: str,
    authors: str,
    rate: str,
    tldr: str | None,
    pdf_url: str | None,
    affiliations: str | None = None,
    *,
    paper_url: str | None = None,
    journal: str | None = None,
    source: str | None = None,
    translated_title: str | None = None,
):
    title_html = escape(title)
    if paper_url:
        title_html = f'<a href="{escape(paper_url, quote=True)}">{title_html}</a>'
    if translated_title:
        title_html += f'<br><span style="font-size: 16px; color: #666;">{escape(translated_title)}</span>'
    metadata = " · ".join(part for part in (source.upper() if source else None, journal) if part)
    summary = escape(tldr) if tldr else "Metadata-only record; open the paper link for details."
    pdf_button = ""
    if pdf_url:
        pdf_button = (
            f'<a href="{escape(pdf_url, quote=True)}" style="display: inline-block; text-decoration: none; '
            'font-size: 14px; font-weight: bold; color: #fff; background-color: #d9534f; '
            'padding: 8px 16px; border-radius: 4px;">PDF</a>'
        )
    block_template = """
    <table border="0" cellpadding="0" cellspacing="0" width="100%" style="font-family: Arial, sans-serif; border: 1px solid #ddd; border-radius: 8px; padding: 16px; background-color: #f9f9f9;">
    <tr>
        <td style="font-size: 20px; font-weight: bold; color: #333;">
            {title_html}
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #666; padding: 8px 0;">
            {authors}
            <br>
            <i>{affiliations}</i><br>
            <span>{metadata}</span>
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>Relevance:</strong> {rate}
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>Summary:</strong> {summary}
        </td>
    </tr>

    <tr>
        <td style="padding: 8px 0;">
            {pdf_button}
        </td>
    </tr>
</table>
"""
    return block_template.format(
        title_html=title_html,
        authors=escape(authors),
        rate=escape(str(rate)),
        summary=summary,
        pdf_button=pdf_button,
        affiliations=escape(affiliations or "Unknown Affiliation"),
        metadata=escape(metadata),
    )

def get_stars(score:float):
    full_star = '<span class="full-star">⭐</span>'
    half_star = '<span class="half-star">⭐</span>'
    low = 6
    high = 8
    if score <= low:
        return ''
    elif score >= high:
        return full_star * 5
    else:
        interval = (high-low) / 10
        star_num = math.ceil((score-low) / interval)
        full_star_num = int(star_num/2)
        half_star_num = star_num - full_star_num * 2
        return '<div class="star-wrapper">'+full_star * full_star_num + half_star * half_star_num + '</div>'


def render_email(papers:list[Paper]) -> str:
    parts = []
    if len(papers) == 0 :
        return framework.replace('__CONTENT__', get_empty_html())
    
    for p in papers:
        #rate = get_stars(p.score)
        rate = round(p.score, 1) if p.score is not None else 'Unknown'
        author_list = [a for a in p.authors]
        num_authors = len(author_list)
        if num_authors <= 5:
            authors = ', '.join(author_list)
        else:
            authors = ', '.join(author_list[:3] + ['...'] + author_list[-2:])
        if p.affiliations is not None:
            affiliations = p.affiliations[:5]
            affiliations = ', '.join(affiliations)
            if len(p.affiliations) > 5:
                affiliations += ', ...'
        else:
            affiliations = 'Unknown Affiliation'
        parts.append(
            get_block_html(
                p.title,
                authors,
                rate,
                p.tldr,
                p.pdf_url,
                affiliations,
                paper_url=p.url,
                journal=p.journal,
                source=p.source,
                translated_title=p.translated_title,
            )
        )

    content = '<br>' + '</br><br>'.join(parts) + '</br>'
    return framework.replace('__CONTENT__', content)
