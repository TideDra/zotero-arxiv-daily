from paper import BasePaper, ArxivPaper, MedrxivPaper
import math
from tqdm import tqdm
from email.header import Header
from email.mime.text import MIMEText
from email.utils import parseaddr, formataddr
import smtplib
import datetime
from loguru import logger

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

def get_block_html(title:str, authors:str, rate:str,paper_id:str, tldr:str, pdf_url:str, code_url:str=None, affiliations:str=None):
    code = f'<a href="{code_url}" style="display: inline-block; text-decoration: none; font-size: 14px; font-weight: bold; color: #fff; background-color: #5bc0de; padding: 8px 16px; border-radius: 4px; margin-left: 8px;">Code</a>' if code_url else ''
    affiliations_html = f"<br><i>{affiliations}</i>" if affiliations else ""
    block_template = """
    <table border="0" cellpadding="0" cellspacing="0" width="100%" style="font-family: Arial, sans-serif; border: 1px solid #ddd; border-radius: 8px; padding: 16px; background-color: #f9f9f9; margin-bottom: 1em;">
    <tr>
        <td style="font-size: 20px; font-weight: bold; color: #333;">
            {title}
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #666; padding: 8px 0;">
            {authors}
            {affiliations_html}
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>Relevance:</strong> {rate}
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>ID:</strong> {paper_id}
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
            {code}
        </td>
    </tr>
</table>
"""
    return block_template.format(title=title, authors=authors, rate=rate, paper_id=paper_id, tldr=tldr, pdf_url=pdf_url, code=code, affiliations_html=affiliations_html)

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


def _generate_paper_html(paper: BasePaper) -> str:
    """Helper function to generate HTML block for a single paper."""
    rate = get_stars(paper.score)
    
    author_list = paper.authors
    authors_str = ', '.join(author_list[:5])
    if len(author_list) > 5:
        authors_str += ', et al.'

    affiliations_str = ""
    if hasattr(paper, 'affiliations') and paper.affiliations:
        affiliations_list = paper.affiliations[:5]
        affiliations_str = ', '.join(affiliations_list)
        if len(paper.affiliations) > 5:
            affiliations_str += ', ...'
    else:
        affiliations_str = 'Unknown Affiliation'
        
    tldr_str = ""
    if hasattr(paper, 'tldr'):
        tldr_str = paper.tldr
    else:
        tldr_str = paper.summary
        
    paper_id_str = ""
    if isinstance(paper, ArxivPaper):
        paper_id_str = paper.arxiv_id
    elif isinstance(paper, MedrxivPaper):
        paper_id_str = paper._data.get('doi', 'N/A')

    code_url = paper.code_url if hasattr(paper, 'code_url') else None

    return get_block_html(paper.title, authors_str, rate, paper_id_str, tldr_str, paper.pdf_url, code_url, affiliations_str)

def render_email(papers:list[BasePaper]):
    if not papers :
        return framework.replace('__CONTENT__', get_empty_html())
    
    arxiv_papers = [p for p in papers if isinstance(p, ArxivPaper)]
    medrxiv_papers = [p for p in papers if isinstance(p, MedrxivPaper)]
    
    html_parts = []
    
    if arxiv_papers:
        html_parts.append("<h2 style='font-family: Arial, sans-serif; color: #333;'>arXiv Papers</h2>")
        for p in tqdm(arxiv_papers, desc='Rendering arXiv Email'):
            html_parts.append(_generate_paper_html(p))

    if arxiv_papers and medrxiv_papers:
        html_parts.append("<hr style='border: none; border-top: 1px solid #ddd; margin: 2em 0;'>")

    if medrxiv_papers:
        html_parts.append("<h2 style='font-family: Arial, sans-serif; color: #333;'>medRxiv Papers</h2>")
        for p in tqdm(medrxiv_papers, desc='Rendering medRxiv Email'):
            html_parts.append(_generate_paper_html(p))

    content = ''.join(html_parts)
    return framework.replace('__CONTENT__', content)


def send_email(sender:str, receiver:str, password:str,smtp_server:str,smtp_port:int, html:str,):
    def _format_addr(s):
        name, addr = parseaddr(s)
        return formataddr((Header(name, 'utf-8').encode(), addr))

    msg = MIMEText(html, 'html', 'utf-8')
    msg['From'] = _format_addr('Zotero Daily Papers <%s>' % sender)
    msg['To'] = _format_addr('You <%s>' % receiver)
    today = datetime.datetime.now().strftime('%Y/%m/%d')
    msg['Subject'] = Header(f'Daily Paper Digest {today}', 'utf-8').encode()

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
    except Exception as e:
        logger.warning(f"Failed to use TLS. {e}")
        logger.warning(f"Try to use SSL.")
        server = smtplib.SMTP_SSL(smtp_server, smtp_port)

    server.login(sender, password)
    server.sendmail(sender, [receiver], msg.as_string())
    server.quit()
