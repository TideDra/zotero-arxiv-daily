from dataclasses import dataclass
from typing import Optional, TypeVar
from datetime import datetime
import re
import tiktoken
from openai import OpenAI
from loguru import logger
import json
RawPaperItem = TypeVar('RawPaperItem')

# QS World University Rankings Top 100 reference list (2024)
QS_TOP100_UNIVERSITIES = """
MIT, Stanford University, Harvard University, University of Cambridge, University of Oxford,
ETH Zurich, Imperial College London, UCL (University College London), University of Chicago,
National University of Singapore (NUS), University of Pennsylvania, EPFL,
Yale University, Caltech (California Institute of Technology), Princeton University,
Cornell University, University of Michigan, Johns Hopkins University, Columbia University,
University of Edinburgh, University of Toronto, Nanyang Technological University (NTU),
Duke University, Northwestern University, Tsinghua University, Peking University,
University of Tokyo, HKUST (Hong Kong University of Science and Technology),
University of Hong Kong (HKU), KAIST, Seoul National University, Carnegie Mellon University,
McGill University, University of Melbourne, University of Sydney, Australian National University (ANU),
Monash University, University of Queensland, University of Bristol, University of Amsterdam,
TU Munich (Technical University of Munich), KU Leuven, Delft University of Technology,
University of Zurich, King's College London, University of Glasgow, University of Birmingham,
University of Warwick, University of Sheffield, University of Leeds, University of Nottingham,
University of Southampton, Queen Mary University of London, Kyoto University,
Chinese University of Hong Kong (CUHK), City University of Hong Kong, Seoul National University,
POSTECH, Fudan University, Shanghai Jiao Tong University, Zhejiang University,
University of Science and Technology of China (USTC), Nanjing University,
University of California Berkeley, UCLA, UCSD (UC San Diego), University of Washington,
University of Texas at Austin, Georgia Tech, University of Illinois Urbana-Champaign (UIUC),
University of Wisconsin-Madison, University of Maryland, Purdue University,
Ohio State University, Penn State University, University of Minnesota,
Rice University, Vanderbilt University, University of North Carolina,
Boston University, University of California Davis, University of California Santa Barbara,
KTH Royal Institute of Technology, Chalmers University of Technology,
University of Copenhagen, University of Helsinki, Lund University,
Utrecht University, Leiden University, Ghent University, University of Geneva,
Paris Sciences et Lettres (PSL University), Sorbonne University, ENS Paris
"""

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
    has_top_university_author: Optional[bool] = None  # Whether first author is from QS top 100
    notable_corresponding_author: Optional[bool] = None  # Whether corresponding author is a notable researcher

    def _generate_tldr_with_llm(self, openai_client:OpenAI,llm_params:dict) -> str:
        lang = llm_params.get('language', 'English')
        prompt = f"Given the following information of a paper, generate a one-sentence TLDR summary in {lang}:\n\n"
        if self.title:
            prompt += f"Title:\n {self.title}\n\n"
        if self.full_text:
            prompt += f"Preview of main content:\n {self.full_text}\n\n"
        elif self.abstract:
            prompt += f"Abstract: {self.abstract}\n\n"
        else:
            logger.warning(f"Neither full text nor abstract is provided for {self.url}")
            return "Failed to generate TLDR. Neither full text nor abstract is provided"

        # use gpt-4o tokenizer for estimation
        enc = tiktoken.encoding_for_model("gpt-4o")
        prompt_tokens = enc.encode(prompt)
        prompt_tokens = prompt_tokens[:4000]  # truncate to 4000 tokens
        prompt = enc.decode(prompt_tokens)

        response = openai_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"You are an assistant who perfectly summarizes scientific paper, and gives the core idea of the paper to the user. Your answer should be in {lang}.",
                },
                {"role": "user", "content": prompt},
            ],
            **llm_params.get('generation_kwargs', {})
        )
        tldr = response.choices[0].message.content
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

    def _generate_affiliations_with_llm(self, openai_client:OpenAI,llm_params:dict) -> Optional[dict]:
        if self.full_text is not None:
            prompt = f"Given the beginning of a paper, extract the affiliations of the authors and assess whether the first author's institution is a top-ranked university:\n\n{self.full_text}"
            # use gpt-4o tokenizer for estimation
            enc = tiktoken.encoding_for_model("gpt-4o")
            prompt_tokens = enc.encode(prompt)
            prompt_tokens = prompt_tokens[:2000]  # truncate to 2000 tokens
            prompt = enc.decode(prompt_tokens)

            system_prompt = f"""You are an assistant who extracts author affiliations from papers and evaluates their institutional prestige.

Given the beginning of a paper:
1. Extract the unique top-level affiliations sorted by author order. If an affiliation has multiple levels (e.g., "Department of CS, MIT"), return only the top-level institution ("MIT"). Do not include duplicates.
2. Identify the first author's institution.
3. Determine if the first author's institution is among the QS World University Rankings Top 100.

The following are examples of QS Top 100 universities (not exhaustive):
{QS_TOP100_UNIVERSITIES}

Return ONLY a valid JSON object in this exact format:
{{
    "affiliations": ["Institution1", "Institution2"],
    "first_author_institution": "Institution name or null",
    "is_qs_top100": true
}}

Set "is_qs_top100" to true if the first author's institution is in QS top 100, false if it clearly is not, or null if you cannot determine with confidence.
If no affiliations are found, return {{"affiliations": [], "first_author_institution": null, "is_qs_top100": null}}."""

            response = openai_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                **llm_params.get('generation_kwargs', {})
            )
            result = response.choices[0].message.content
            json_match = re.search(r'\{.*?\}', result, flags=re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        return None

    def generate_affiliations(self, openai_client:OpenAI,llm_params:dict) -> Optional[list[str]]:
        try:
            result = self._generate_affiliations_with_llm(openai_client, llm_params)
            if result:
                affiliations = result.get('affiliations', [])
                affiliations = [str(a) for a in affiliations]
                self.affiliations = affiliations
                self.has_top_university_author = result.get('is_qs_top100')
            return self.affiliations
        except Exception as e:
            logger.warning(f"Failed to generate affiliations of {self.url}: {e}")
            self.affiliations = None
            return None

    def _check_notable_corresponding_author_with_llm(self, openai_client:OpenAI, llm_params:dict) -> Optional[bool]:
        # Build context from available information
        enc = tiktoken.encoding_for_model("gpt-4o")
        if self.full_text:
            text_tokens = enc.encode(self.full_text)
            text = enc.decode(text_tokens[:2000])
            content = f"Beginning of paper:\n{text}"
        elif self.authors:
            # Fallback: use title + authors + abstract
            fallback = f"Title: {self.title}\nAuthors: {', '.join(self.authors)}\nAbstract: {self.abstract or ''}"
            content = fallback
        else:
            return None

        system_prompt = """You are an assistant who evaluates the academic reputation of researchers.

Given information about a paper:
1. Identify the corresponding author. The corresponding author is usually:
   - Explicitly marked with *, †, or labeled "Corresponding author" / "Equal contribution"
   - If not explicitly marked, assume the last author is the corresponding author
2. Based on your knowledge, determine if this person is a well-known and influential researcher in their field.
   A "notable" researcher typically meets one or more of these criteria:
   - Has a high citation count (e.g., h-index > 30, total citations > 5000)
   - Holds or has held a faculty position at a prestigious research institution
   - Has received major awards (e.g., Turing Award, Fields Medal, NeurIPS best paper, fellowship of major academies)
   - Is widely recognized as a leading expert in their subfield

Return ONLY a valid JSON object in this exact format:
{
    "corresponding_author": "Author name or null",
    "is_notable": true
}

Set "is_notable" to true if the author is well-known, false if they are clearly not, or null if you cannot determine with confidence."""

        prompt = f"Given the following information about a paper, identify and evaluate the corresponding author:\n\n{content}"

        response = openai_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            **llm_params.get('generation_kwargs', {})
        )
        result = response.choices[0].message.content
        json_match = re.search(r'\{.*?\}', result, flags=re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            return data.get('is_notable')
        return None

    def generate_notable_corresponding_author(self, openai_client:OpenAI, llm_params:dict) -> Optional[bool]:
        try:
            is_notable = self._check_notable_corresponding_author_with_llm(openai_client, llm_params)
            self.notable_corresponding_author = is_notable
            return is_notable
        except Exception as e:
            logger.warning(f"Failed to check notable corresponding author of {self.url}: {e}")
            self.notable_corresponding_author = None
            return None

@dataclass
class CorpusPaper:
    title: str
    abstract: str
    added_date: datetime
    paths: list[str]
