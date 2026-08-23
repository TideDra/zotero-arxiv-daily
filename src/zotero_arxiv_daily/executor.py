from loguru import logger
from pyzotero import zotero
from pyzotero.zotero_errors import CouldNotReachURLError, HTTPError, TooManyRequestsError, TooManyRetriesError
from omegaconf import DictConfig, ListConfig
from .utils import glob_match
from .retriever import get_retriever_cls
from .protocol import CorpusPaper, Paper
import random
from datetime import datetime
from time import sleep
from .reranker import get_reranker_cls
from .construct_email import render_email
from .construct_feed import write_atom_feed
from .utils import send_email
from .state import AdsState
from .retriever.arxiv_retriever import enrich_papers_from_arxiv
from openai import OpenAI
from tqdm import tqdm


SCIX_SOURCE_ALIASES = frozenset({"scix", "ads"})


def normalize_path_patterns(patterns: list[str] | ListConfig | None, config_key: str) -> list[str] | None:
    if patterns is None:
        return None

    if not isinstance(patterns, (list, ListConfig)):
        raise TypeError(
            f"config.zotero.{config_key} must be a list of glob patterns or null, "
            'for example ["2026/survey/**"]. Single strings are not supported.'
        )

    if any(not isinstance(pattern, str) for pattern in patterns):
        raise TypeError(f"config.zotero.{config_key} must contain only glob pattern strings.")

    return list(patterns)


class Executor:
    def __init__(self, config:DictConfig):
        self.config = config
        self.include_path_patterns = normalize_path_patterns(config.zotero.include_path, "include_path")
        self.ignore_path_patterns = normalize_path_patterns(config.zotero.ignore_path, "ignore_path")
        configured_sources = list(config.executor.source)
        scix_sources = [source for source in configured_sources if source in SCIX_SOURCE_ALIASES]
        if len(scix_sources) > 1:
            raise ValueError("Configure only one of executor.source=scix or ads; ads is a legacy alias for SciX.")
        if scix_sources and config.executor.reranker != "local":
            raise ValueError(
                "SciX/ADS records require executor.reranker=local so licensed abstracts never leave the process."
            )
        self.retrievers = {
            source: get_retriever_cls(source)(config) for source in config.executor.source
        }
        self.reranker = get_reranker_cls(config.executor.reranker)(config)
        self.openai_client = OpenAI(api_key=config.llm.api.key, base_url=config.llm.api.base_url)
        self.ads_state: AdsState | None = None
        self.ads_state_path: str | None = None
        self.scix_source = scix_sources[0] if scix_sources else None
        self.scix_state_retention_days = 90
        if self.scix_source:
            source_config = getattr(config.source, self.scix_source)
            self.ads_state_path = str(source_config.state_path)
            self.scix_state_retention_days = int(source_config.get("state_retention_days", 90))
            self.ads_state = AdsState.load(self.ads_state_path)

    @staticmethod
    def _zotero_request(operation, description: str):
        retryable = (CouldNotReachURLError, HTTPError, TooManyRequestsError, TooManyRetriesError)
        for attempt in range(3):
            try:
                return operation()
            except retryable as exc:
                if attempt == 2:
                    raise RuntimeError(f"Zotero {description} failed after 3 attempts") from exc
                wait = 2 ** attempt
                logger.warning(f"Zotero {description} failed; retrying in {wait}s: {exc}")
                sleep(wait)

    def fetch_zotero_corpus(self) -> list[CorpusPaper]:
        logger.info("Fetching zotero corpus")
        zot = zotero.Zotero(self.config.zotero.user_id, 'user', self.config.zotero.api_key)
        collections = self._zotero_request(
            lambda: zot.everything(zot.collections()),
            "collection retrieval",
        )
        collections = {c['key']:c for c in collections}
        corpus = self._zotero_request(
            lambda: zot.everything(zot.items(itemType='conferencePaper || journalArticle || preprint')),
            "item retrieval",
        )
        corpus = [c for c in corpus if c['data']['abstractNote'] != '']
        def get_collection_path(col_key:str) -> str:
            if p := collections[col_key]['data']['parentCollection']:
                return get_collection_path(p) + '/' + collections[col_key]['data']['name']
            else:
                return collections[col_key]['data']['name']
        for c in corpus:
            paths = [get_collection_path(col) for col in c['data']['collections']]
            c['paths'] = paths
        logger.info(f"Fetched {len(corpus)} zotero papers")
        return [CorpusPaper(
            title=c['data']['title'],
            abstract=c['data']['abstractNote'],
            added_date=datetime.strptime(c['data']['dateAdded'], '%Y-%m-%dT%H:%M:%SZ'),
            paths=c['paths']
        ) for c in corpus]
    
    def filter_corpus(self, corpus:list[CorpusPaper]) -> list[CorpusPaper]:
        if self.include_path_patterns:
            logger.info(f"Selecting zotero papers matching include_path: {self.include_path_patterns}")
            corpus = [
                c for c in corpus
                if any(
                    glob_match(path, pattern)
                    for path in c.paths
                    for pattern in self.include_path_patterns
                )
            ]
        if self.ignore_path_patterns:
            logger.info(f"Excluding zotero papers matching ignore_path: {self.ignore_path_patterns}")
            corpus = [
                c for c in corpus
                if not any(
                    glob_match(path, pattern)
                    for path in c.paths
                    for pattern in self.ignore_path_patterns
                )
            ]
        if self.include_path_patterns or self.ignore_path_patterns:
            samples = random.sample(corpus, min(5, len(corpus)))
            samples = '\n'.join([c.title + ' - ' + '\n'.join(c.paths) for c in samples])
            logger.info(f"Selected {len(corpus)} zotero papers:\n{samples}\n...")
        return corpus

    
    def run(self) -> list[Paper]:
        corpus = self.fetch_zotero_corpus()
        corpus = self.filter_corpus(corpus)
        if len(corpus) == 0:
            logger.error(f"No zotero papers found. Please check your zotero settings:\n{self.config.zotero}")
            return []
        all_papers: list[Paper] = []
        for source, retriever in self.retrievers.items():
            logger.info(f"Retrieving {source} papers...")
            papers = retriever.retrieve_papers()
            if len(papers) == 0:
                logger.info(f"No {source} papers found")
                continue
            if source in SCIX_SOURCE_ALIASES and self.ads_state is not None:
                unseen = self.ads_state.filter_new(papers)
                logger.info(
                    f"Selected {len(unseen)} unseen SciX records from {len(papers)} retrieved records"
                )
                papers = unseen
            logger.info(f"Retrieved {len(papers)} {source} papers")
            all_papers.extend(papers)
        logger.info(f"Total {len(all_papers)} papers retrieved from all sources")
        reranked_papers = []
        if len(all_papers) > 0:
            logger.info("Reranking papers...")
            reranked_papers = self.reranker.rerank(all_papers, corpus)
            reranked_papers = reranked_papers[:self.config.executor.max_paper_num]
            selected_scix_papers = [
                paper for paper in reranked_papers if "ads" in paper.external_ids
            ]
            logger.info("Enriching selected SciX records that have explicit arXiv identifiers...")
            enrich_papers_from_arxiv(selected_scix_papers)
            logger.info("Generating TLDR and affiliations...")
            for p in tqdm(reranked_papers):
                p.generate_tldr(self.openai_client, self.config.llm)
                p.generate_affiliations(self.openai_client, self.config.llm)
                p.translate_title(self.openai_client, self.config.llm)
        atom_config = self.config.get("output", {}).get("atom")
        if atom_config and atom_config.get("enabled", False):
            output_path = write_atom_feed(reranked_papers, atom_config)
            logger.info(f"Atom feed written to {output_path}")
        if not reranked_papers and not self.config.executor.send_empty:
            logger.info("No new papers found. No email will be sent.")
            return []
        logger.info("Sending email...")
        email_content = render_email(reranked_papers)
        send_email(self.config, email_content)
        logger.info("Email sent successfully")
        if self.ads_state is not None and self.ads_state_path is not None:
            delivered_scix_papers = [
                paper for paper in reranked_papers if "ads" in paper.external_ids
            ]
            self.ads_state.mark_seen(delivered_scix_papers)
            self.ads_state.prune(self.scix_state_retention_days)
            self.ads_state.save(self.ads_state_path)
            logger.info(f"SciX/ADS-compatible state written to {self.ads_state_path}")
        return reranked_papers
