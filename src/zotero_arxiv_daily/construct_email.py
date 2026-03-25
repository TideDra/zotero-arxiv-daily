import math
from html import escape

from .protocol import Paper
from .topic_clusterer import PaperGroup


GROUP_WRAPPER_STYLE = "margin: 0 0 32px 0;"
GROUP_HEADING_STYLE = "font-family: Arial, sans-serif; font-size: 24px; font-weight: bold; color: #222; margin: 0 0 8px 0;"
GROUP_SUMMARY_STYLE = "font-family: Arial, sans-serif; font-size: 14px; color: #555; margin: 0 0 16px 0; line-height: 1.5;"
PAPER_SPACING = '<br></br><br>'




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

def get_block_html(title:str, authors:str, rate:str, tldr:str, pdf_url:str, affiliations:str=None):
    block_template = """
    <table border="0" cellpadding="0" cellspacing="0" width="100%" style="font-family: Arial, sans-serif; border: 1px solid #ddd; border-radius: 8px; padding: 16px; background-color: #f9f9f9;">
    <tr>
        <td style="font-size: 20px; font-weight: bold; color: #333;">
            {title}
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #666; padding: 8px 0;">
            {authors}
            <br>
            <i>{affiliations}</i>
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>Relevance:</strong> {rate}
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>TLDR:</strong> {tldr}
        </td>
    </tr>

    <tr>
        <td style="padding: 8px 0;">
            <a href="{pdf_url}" style="display: inline-block; text-decoration: none; font-size: 14px; font-weight: bold; color: #fff; background-color: #d9534f; padding: 8px 16px; border-radius: 4px;">PDF</a>
        </td>
    </tr>
</table>
"""
    return block_template.format(title=title, authors=authors,rate=rate, tldr=tldr, pdf_url=pdf_url, affiliations=affiliations)

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


def _render_paper_html(paper: Paper) -> str:
    rate = round(paper.score, 1) if paper.score is not None else 'Unknown'
    author_list = [author for author in paper.authors]
    num_authors = len(author_list)
    if num_authors <= 5:
        authors = ', '.join(author_list)
    else:
        authors = ', '.join(author_list[:3] + ['...'] + author_list[-2:])
    if paper.affiliations is not None:
        affiliations = paper.affiliations[:5]
        affiliations = ', '.join(affiliations)
        if len(paper.affiliations) > 5:
            affiliations += ', ...'
    else:
        affiliations = 'Unknown Affiliation'
    return get_block_html(
        escape(paper.title),
        escape(authors),
        rate,
        escape(paper.tldr or ''),
        escape(paper.pdf_url),
        escape(affiliations),
    )


def get_group_html(label: str, summary: str | None, paper_html: str) -> str:
    escaped_label = escape(label)
    summary_html = f'<div style="{GROUP_SUMMARY_STYLE}">{escape(summary)}</div>' if summary else ''
    return (
        f'<div style="{GROUP_WRAPPER_STYLE}">'
        f'<h2 style="{GROUP_HEADING_STYLE}">{escaped_label}</h2>'
        f'{summary_html}'
        f'{paper_html}'
        '</div>'
    )


def render_email(groups:list[PaperGroup]) -> str:
    if len(groups) == 0 :
        return framework.replace('__CONTENT__', get_empty_html())

    rendered_groups = []
    for group in groups:
        paper_parts = [_render_paper_html(paper) for paper in group.papers]
        paper_html = '<br>' + PAPER_SPACING.join(paper_parts) + '</br>'
        rendered_groups.append(get_group_html(group.label, group.summary, paper_html))

    content = '<br>' + PAPER_SPACING.join(rendered_groups) + '</br>'
    return framework.replace('__CONTENT__', content)
