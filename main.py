import arxiv
import argparse
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pyzotero import zotero
from recommender import rerank_paper
from construct_email import render_email, send_email
from tqdm import trange,tqdm
from loguru import logger
from gitignore_parser import parse_gitignore
from tempfile import mkstemp
from paper import ArxivPaper
from llm import set_global_llm
import feedparser
from config import Config

def get_zotero_corpus(zotero_id:str, zotero_key:str) -> list[dict]:
    zot = zotero.Zotero(zotero_id, 'user', zotero_key)
    collections = zot.everything(zot.collections())
    collections = {c['key']:c for c in collections}
    corpus = zot.everything(zot.items(itemType='conferencePaper || journalArticle || preprint'))
    corpus = [c for c in corpus if c['data']['abstractNote'] != '']
    def get_collection_path(col_key:str) -> str:
        if p := collections[col_key]['data']['parentCollection']:
            return get_collection_path(p) + '/' + collections[col_key]['data']['name']
        else:
            return collections[col_key]['data']['name']
    for c in corpus:
        paths = [get_collection_path(col) for col in c['data']['collections']]
        c['paths'] = paths
    return corpus

def filter_corpus(corpus:list[dict], pattern:str) -> list[dict]:
    _,filename = mkstemp()
    with open(filename,'w') as file:
        file.write(pattern)
    matcher = parse_gitignore(filename,base_dir='./')
    new_corpus = []
    for c in corpus:
        match_results = [matcher(p) for p in c['paths']]
        if not any(match_results):
            new_corpus.append(c)
    os.remove(filename)
    return new_corpus


def get_arxiv_paper(query:str, debug:bool=False) -> list[ArxivPaper]:
    client = arxiv.Client(num_retries=10,delay_seconds=10)
    feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
    if 'Feed error for query' in feed.feed.title:
        raise ValueError(f"Invalid ARXIV_QUERY: {query}.")
    if not debug:
        papers = []
        all_paper_ids = [i.id.removeprefix("oai:arXiv.org:") for i in feed.entries if i.arxiv_announce_type == 'new']
        bar = tqdm(total=len(all_paper_ids),desc="Retrieving Arxiv papers")
        for i in range(0,len(all_paper_ids),50):
            search = arxiv.Search(id_list=all_paper_ids[i:i+50])
            batch = [ArxivPaper(p) for p in client.results(search)]
            bar.update(len(batch))
            papers.extend(batch)
        bar.close()

    else:
        logger.debug("Retrieve 5 arxiv papers regardless of the date.")
        search = arxiv.Search(query='cat:cs.AI', sort_by=arxiv.SortCriterion.SubmittedDate)
        papers = []
        for i in client.results(search):
            papers.append(ArxivPaper(i))
            if len(papers) == 5:
                break

    return papers



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Recommender system for academic papers')
    parser.add_argument('--zotero_id', type=str, help='Zotero user ID')
    parser.add_argument('--zotero_key', type=str, help='Zotero API key')
    parser.add_argument('--zotero_ignore',type=str,help='Zotero collection to ignore, using gitignore-style pattern.')
    parser.add_argument('--send_empty', action='store_true', help='If get no arxiv paper, send empty email')
    parser.add_argument('--max_paper_num', type=int, help='Maximum number of papers to recommend')
    parser.add_argument('--arxiv_query', type=str, help='Arxiv search query')
    parser.add_argument('--smtp_server', type=str, help='SMTP server')
    parser.add_argument('--smtp_port', type=int, help='SMTP port')
    parser.add_argument('--sender', type=str, help='Sender email address')
    parser.add_argument('--receiver', type=str, help='Receiver email address')
    parser.add_argument('--sender_password', type=str, help='Sender email password')
    parser.add_argument(
        "--use_llm_api",
        action='store_true',
        help="Use OpenAI API to generate TLDR",
    )
    parser.add_argument(
        "--openai_api_key",
        type=str,
        help="OpenAI API key",
    )
    parser.add_argument(
        "--openai_api_base",
        type=str,
        help="OpenAI API base URL",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="LLM Model Name",
    )
    parser.add_argument(
        "--language",
        type=str,
        help="Language of TLDR",
    )
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()

    config = Config(
        zotero_id=args.zotero_id,
        zotero_key=args.zotero_key,
        zotero_ignore=args.zotero_ignore,
        send_empty=args.send_empty,
        max_paper_num=args.max_paper_num,
        arxiv_query=args.arxiv_query,
        smtp_server=args.smtp_server,
        smtp_port=args.smtp_port,
        sender=args.sender,
        receiver=args.receiver,
        sender_password=args.sender_password,
        use_llm_api=args.use_llm_api,
        openai_api_key=args.openai_api_key,
        openai_api_base=args.openai_api_base,
        model_name=args.model_name,
        language=args.language,
        debug=args.debug
    )

    logger.remove()
    logger.add(sys.stdout, level="DEBUG" if config.debug else "INFO")

    logger.info("Retrieving Zotero corpus...")
    corpus = get_zotero_corpus(config.zotero_id, config.zotero_key)
    logger.info(f"Retrieved {len(corpus)} papers from Zotero.")
    if config.zotero_ignore:
        logger.info(f"Ignoring papers in:\n {config.zotero_ignore}...")
        corpus = filter_corpus(corpus, config.zotero_ignore)
        logger.info(f"Remaining {len(corpus)} papers after filtering.")
    logger.info("Retrieving Arxiv papers...")
    papers = get_arxiv_paper(config.arxiv_query, config.debug)
    if len(papers) == 0:
        logger.info("No new papers found. Yesterday maybe a holiday and no one submit their work :). If this is not the case, please check the ARXIV_QUERY.")
        if not config.send_empty:
          exit(0)
    else:
        logger.info("Reranking papers...")
        papers = rerank_paper(papers, corpus)
        if config.use_llm_api:
            logger.info("Using OpenAI API as global LLM.")
            llm_instance = LLM(use_llm_api=True, api_key=config.openai_api_key, base_url=config.openai_api_base, model=config.model_name, lang=config.language)
        else:
            logger.info("Using Local LLM as global LLM.")
            llm_instance = LLM(use_llm_api=False, lang=config.language)

        papers = rerank_paper(papers, corpus, llm_instance)
        if config.max_paper_num != -1:
            papers = papers[:config.max_paper_num]

    html = render_email(papers, llm_instance)
    logger.info("Sending email...")
    send_email(config.sender, config.receiver, config.sender_password, config.smtp_server, config.smtp_port, html)
    logger.success("Email sent successfully! If you don't receive the email, please check the configuration and the junk box.")