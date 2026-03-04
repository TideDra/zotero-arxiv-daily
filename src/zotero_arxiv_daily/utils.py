import tarfile
import re
import glob
import smtplib
from email.header import Header
from email.mime.text import MIMEText
from email.utils import parseaddr, formataddr
from loguru import logger
import datetime
from omegaconf import DictConfig
import pymupdf.layout
pymupdf.layout.activate()
import pymupdf4llm
def extract_tex_code_from_tar(file_path:str, paper_id:str) -> dict[str,str]:
    try:
        tar = tarfile.open(file_path)
    except tarfile.ReadError:
        logger.debug(f"Failed to find main tex file of {paper_id}: Not a tar file.")
        return None
 
    tex_files = [f for f in tar.getnames() if f.endswith('.tex')]
    if len(tex_files) == 0:
        logger.debug(f"Failed to find main tex file of {paper_id}: No tex file.")
        tar.close()
        return None
    
    bbl_file = [f for f in tar.getnames() if f.endswith('.bbl')]
    match len(bbl_file) :
        case 0:
            if len(tex_files) > 1:
                logger.debug(f"Cannot find main tex file of {paper_id} from bbl: There are multiple tex files while no bbl file.")
                main_tex = None
            else:
                main_tex = tex_files[0]
        case 1:
            main_name = bbl_file[0].replace('.bbl','')
            main_tex = f"{main_name}.tex"
            if main_tex not in tex_files:
                logger.debug(f"Cannot find main tex file of {paper_id} from bbl: The bbl file does not match any tex file.")
                main_tex = None
        case _:
            logger.debug(f"Cannot find main tex file of {paper_id} from bbl: There are multiple bbl files.")
            main_tex = None

    if main_tex is None:
        logger.debug(f"Trying to choose tex file containing the document block as main tex file of {paper_id}")
    #read all tex files
    file_contents = {}
    for t in tex_files:
        f = tar.extractfile(t)
        content = f.read().decode('utf-8',errors='ignore')
        #remove comments
        content = re.sub(r'%.*\n', '\n', content)
        content = re.sub(r'\\begin{comment}.*?\\end{comment}', '', content, flags=re.DOTALL)
        content = re.sub(r'\\iffalse.*?\\fi', '', content, flags=re.DOTALL)
        #remove redundant \n
        content = re.sub(r'\n+', '\n', content)
        content = re.sub(r'\\\\', '', content)
        #remove consecutive spaces
        content = re.sub(r'[ \t\r\f]{3,}', ' ', content)
        if main_tex is None and re.search(r'\\begin\{document\}', content):
            main_tex = t
            logger.debug(f"Choose {t} as main tex file of {paper_id}")
        file_contents[t] = content
    
    if main_tex is not None:
        main_source:str = file_contents[main_tex]
        #find and replace all included sub-files
        include_files = re.findall(r'\\input\{(.+?)\}', main_source) + re.findall(r'\\include\{(.+?)\}', main_source)
        for f in include_files:
            if not f.endswith('.tex'):
                file_name = f + '.tex'
            else:
                file_name = f
            main_source = main_source.replace(f'\\input{{{f}}}', file_contents.get(file_name, ''))
        file_contents["all"] = main_source
    else:
        logger.debug(f"Failed to find main tex file of {paper_id}: No tex file containing the document block.")
        file_contents["all"] = None
        
    tar.close()
    return file_contents

def extract_markdown_from_pdf(file_path:str) -> str:
    return pymupdf4llm.to_markdown(file_path,use_ocr=False,header=False,footer=False,ignore_code=True)

def glob_match(path:str, pattern:str) -> bool:
    re_pattern = glob.translate(pattern,recursive=True)
    return re.match(re_pattern, path) is not None

def send_email(config:DictConfig, html:str):
    sender = config.email.sender
    receiver = config.email.receiver
    password = config.email.sender_password
    smtp_server = config.email.smtp_server
    smtp_port = config.email.smtp_port
    def _format_addr(s):
        name, addr = parseaddr(s)
        return formataddr((Header(name, 'utf-8').encode(), addr))

    msg = MIMEText(html, 'html', 'utf-8')
    msg['From'] = _format_addr('Github Action <%s>' % sender)
    msg['To'] = _format_addr('You <%s>' % receiver)
    today = datetime.datetime.now().strftime('%Y/%m/%d')
    msg['Subject'] = Header(f'Daily arXiv {today}', 'utf-8').encode()

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
    except Exception as e:
        logger.debug(f"Failed to use TLS. {e}\nTry to use SSL.")
        try:
            server = smtplib.SMTP_SSL(smtp_server, smtp_port)
        except Exception as e:
            logger.debug(f"Failed to use SSL. {e}\nTry to use plain text.")
            server = smtplib.SMTP(smtp_server, smtp_port)

    server.login(sender, password)
    server.sendmail(sender, [receiver], msg.as_string())
    server.quit()