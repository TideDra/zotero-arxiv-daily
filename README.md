<p align="center">
  <a href="" rel="noopener">
 <img width=200px height=200px src="assets/logo.svg" alt="logo"></a>
</p>

<h3 align="center">Zotero-arXiv-Daily</h3>

<div align="center">

  [![Status](https://img.shields.io/badge/status-active-success.svg)]()
  ![Stars](https://img.shields.io/github/stars/TideDra/zotero-arxiv-daily?style=flat)
  [![GitHub Issues](https://img.shields.io/github/issues/TideDra/zotero-arxiv-daily)](https://github.com/TideDra/zotero-arxiv-daily/issues)
  [![GitHub Pull Requests](https://img.shields.io/github/issues-pr/TideDra/zotero-arxiv-daily)](https://github.com/TideDra/zotero-arxiv-daily/pulls)
  [![License](https://img.shields.io/github/license/TideDra/zotero-arxiv-daily)](/LICENSE)
  [<img src="https://api.gitsponsors.com/api/badge/img?id=893025857" height="20">](https://api.gitsponsors.com/api/badge/link?p=PKMtRut1dWWuC1oFdJweyDSvJg454/GkdIx4IinvBblaX2AY4rQ7FYKAK1ZjApoiNhYEeduIEhfeZVIwoIVlvcwdJXVFD2nV2EE5j6lYXaT/RHrcsQbFl3aKe1F3hliP26OMayXOoZVDidl05wj+yg==)

</div>

---

<p align="center"> Recommend new astronomy papers from SciX according to your Zotero library.
    <br> 
</p>

> [!IMPORTANT]
> Please keep an eye on this repo, and merge your forked repo in time when there is any update of this upstream, in order to enjoy new features and fix found bugs.

## 🧐 About <a name = "about"></a>

> Track new scientific researches of your interest by just forking (and staring) this repo!😊

This fork uses [NASA SciX](https://scixplorer.org/) as the professional discovery layer for astronomy. SciX currently uses the same API, endpoints, authentication, and existing tokens as ADS, so the adapter keeps the official ADS-compatible production API while moving user-facing links and configuration to SciX. It ranks new SciX records against your Zotero library locally, enriches selected records with open arXiv content when an explicit arXiv identifier is available, sends a private email digest, and publishes an arXiv-backed Atom feed through GitHub Pages.

The upstream arXiv, bioRxiv, medRxiv, and ChemRxiv retrievers remain available. The astronomy configuration in `config/custom.yaml` is deliberately kept as a thin policy layer so that upstream fixes can continue to be merged.

## ✨ Features
- Totally free! All the calculation can be done in the Github Action runner locally within its quota (for public repo).
- AI-generated TL;DR for you to quickly pick up target papers.
- Affiliations of the paper are resolved and presented.
- Links of PDF and code implementation (if any) presented in the e-mail.
- List of papers sorted by relevance with your recent research interest.
- Fast deployment via fork this repo and set environment variables in the Github Action Page.
- Support LLM API for generating TL;DR of papers.
- Ignore unwanted Zotero papers using a list of glob patterns.
- Support multiple sources of papers to retrieve:
  - NASA SciX (with a legacy `ads` configuration alias)
  - arxiv
  - biorxiv
  - medrxiv
  - chemrxiv
- Publish a valid Atom feed that can be followed by any RSS reader.
- Keep SciX-only abstracts inside the workflow: ranking remains local and remote LLM calls are disabled unless the selected record is successfully enriched from arXiv.
- Exclude SciX-only records from the public Atom feed by default; they remain available in the private email digest as metadata-only entries.
- Keep an incremental SciX delivery cursor without committing generated state to the repository. Its historical `ads:<bibcode>` IDs and `.runtime/state/ads.json` path are intentionally preserved.

## 📷 Screenshot
![screenshot](./assets/screenshot.png)

## 🚀 Usage
### Quick Start
1. Fork (and star😘) this repo.
![fork](./assets/fork.png)

2. Set Github Action environment variables.
![secrets](./assets/secrets.png)

Below are all the secrets you need to set. They are invisible to anyone including you once they are set, for security.

| Key |Description | Example |
| :---  | :---  | :--- |
| ZOTERO_ID  | User ID of your Zotero account. **User ID is not your username, but a sequence of numbers**Get your ID from [here](https://www.zotero.org/settings/security). You can find it at the position shown in this [screenshot](https://github.com/TideDra/zotero-arxiv-daily/blob/main/assets/userid.png). | 12345678  |
| ZOTERO_KEY | An Zotero API key with read access. Get a key from [here](https://www.zotero.org/settings/security).  | AB5tZ877P2j7Sm2Mragq041H   |
| SENDER | The email account of the SMTP server that sends you email. | abc@qq.com |
| SENDER_PASSWORD | The password of the sender account. Note that it's not necessarily the password for logging in the e-mail client, but the authentication code for SMTP service. Ask your email provider for this.   | abcdefghijklmn |
| RECEIVER | One e-mail address, or a comma-separated list, that receives the paper list. | abc@outlook.com |
| SCIX_API_TOKEN | Personal SciX API token. An existing ADS token continues to work; copy it from your [SciX account](https://scixplorer.org/scixhelp/userpreferences-scix/scix-account). Never place it in `CUSTOM_CONFIG` or commit it. | `...` |
| ADS_API_TOKEN | Optional legacy secret. The workflows fall back to it, so existing forks do not need to rotate or rename their token immediately. | `...` |
| OPENAI_API_KEY | API Key when using the API to access LLMs. You can get FREE API for using advanced open source LLMs in [SiliconFlow](https://cloud.siliconflow.cn/i/b3XhBRAm). | sk-xxx |
| OPENAI_API_BASE | API URL when using the API to access LLMs. | https://api.siliconflow.cn/v1 |

The checked-in `config/custom.yaml` contains the astronomy defaults. Either edit it in your fork or set the public variable `CUSTOM_CONFIG` to replace it at runtime. If `CUSTOM_CONFIG` is empty, the workflow keeps the checked-in file.
![vars](./assets/repo_var.png)
![custom_config](./assets/config_var.png)
Paste the following content into the value of `CUSTOM_CONFIG` variable:
```yaml
zotero:
  user_id: ${oc.env:ZOTERO_ID}
  api_key: ${oc.env:ZOTERO_KEY}
  include_path: null

email:
  sender: ${oc.env:SENDER}
  receiver: ${oc.env:RECEIVER}
  smtp_server: smtp.qq.com
  smtp_port: 465
  sender_password: ${oc.env:SENDER_PASSWORD}

llm:
  api:
    key: ${oc.env:OPENAI_API_KEY}
    base_url: ${oc.env:OPENAI_API_BASE}
  generation_kwargs:
    model: gpt-4o-mini
  language: ${oc.env:LANGUAGE,English}
  translate_title: ${oc.decode:${oc.env:TRANSLATE_TITLE,false}}

source:
  scix:
    api_token: ${oc.env:SCIX_API_TOKEN,null}
    api_base: https://api.adsabs.harvard.edu/v1
    query: 'database:astronomy abs:("Milky Way" OR "galactic dynamics" OR "stellar halo" OR "gravitational potential" OR "Large Magellanic Cloud")'
    lookback_days: 7
    max_results: 200
    state_path: .runtime/state/ads.json
    state_retention_days: 90

executor:
  debug: ${oc.env:DEBUG,null}
  source: ['scix']
  reranker: local

output:
  atom:
    enabled: true
    scix_only_policy: exclude
    path: public/index.xml
    title: ${oc.env:RSS_TITLE,Astronomy paper recommendations}
    description: ${oc.env:RSS_DESCRIPTION,SciX papers ranked from a Zotero library}
    site_url: ${oc.env:RSS_SITE_URL,https://example.github.io}
    feed_url: ${oc.env:RSS_FEED_URL,https://example.github.io/index.xml}
```
>[!NOTE]
> `${oc.env:XXX,yyy}` means the value of the environment variable `XXX`. If the variable is not set, the default value `yyy` will be used.

The SciX query uses the familiar [SciX/ADS search syntax](https://scixplorer.org/scixhelp/search-scix/search-syntax/). Do not add a rolling date clause: the retriever appends `entdate:[NOW-NDAYS TO *]` from `lookback_days`. Keep `max_results` bounded and narrow the scientific query if SciX returns more candidates.

The privacy boundary is content-based, not source-name based. SciX records with an explicit arXiv identifier are supplemented from arXiv after local ranking and may then use the configured LLM. SciX-only abstracts remain local and never enter email, Atom, or a remote model. Their bibliographic metadata appears in the private email, but the complete entry is excluded from the public Atom feed by default. Set `scix_only_policy: include` only after confirming your redistribution use with SciX.

`source.ads` and `executor.source: ['ads']` remain supported as a legacy alias. Do not enable `scix` and `ads` together: they address the same service and the executor rejects that configuration to prevent duplicate queries.

Here is the full configuration, `???` means the value must be filled in:
```yaml
zotero:
  user_id: ??? # User ID of your Zotero account.
  api_key: ??? # An Zotero API key with read access.
  include_path: null # A list of glob patterns marking the Zotero collections that should be included. Example: ["2026/survey/**", "2026/reading-group/**"]

source:
  scix:
    api_token: ??? # SciX token; prefer ${oc.env:SCIX_API_TOKEN}. ADS_API_TOKEN remains a runtime fallback.
    api_base: https://api.adsabs.harvard.edu/v1
    query: ??? # SciX/ADS-compatible search expression without the rolling entdate clause.
    lookback_days: 7 # Candidate discovery window. State prevents repeated delivery.
    max_results: 200 # Hard safety bound; narrow the query if it is exceeded.
    state_path: .runtime/state/ads.json
    state_retention_days: 90
  arxiv:
    category: null # The categories of target arxiv papers. Find the abbr of your research area from [here](https://arxiv.org/category_taxonomy). Example: ["cs.AI","cs.CV","cs.LG","cs.CL"]
    include_cross_list: false # Whether to include arXiv cross-list papers in subscribed categories. Example: true
  biorxiv:
    category: null # The categories of target biorxiv papers. Find categories from [here](https://www.biorxiv.org/). Example: ["biochemistry","animal behavior and cognition"]
  medrxiv:
    category: null # The categories of target medrxiv papers. Find categories from [here](https://www.medrxiv.org/) Example: ["psychiatry and clinical psychology", "neurology"]
  chemrxiv:
    include_new_versions: false # Whether to include revised versions (v2, v3, ...) of previously posted chemrxiv preprints in addition to new first postings. chemrxiv has no category filter: all new preprints (a few dozen per day) are retrieved via Crossref and left to the reranker. Example: true

email:
  sender: ??? # The email account of the SMTP server that sends you email. Example: abc@qq.com
  receivers: null # Preferred list form. Example: [abc@outlook.com, lab@example.edu]
  receiver: ??? # Backward-compatible single or comma-separated receiver string.
  smtp_server: ??? # The SMTP server that sends the email. Ask your email provider (Gmail, QQ, Outlook, ...) for its SMTP server. Example: smtp.qq.com
  smtp_port: ??? # The port of SMTP server. Example: 465
  sender_password: ??? # The password of the sender account. Note that it's not necessarily the password for logging in the e-mail client, but the authentication code for SMTP service. Ask your email provider for this. Example: abcdefghijklmn

llm:
  api:
    key: ??? # API Key of your LLM API. Example: sk-xxx
    base_url: ??? # API URL of your LLM API. Example: https://api.openai.com/v1
  generation_kwargs:
  # Arguments for the LLM API. See [here](https://platform.openai.com/docs/api-reference/chat/create) for more details.
    max_tokens: 16384
    model: ???
  language: English # Preferred language for the TL;DR. Example: English
  translate_title: false # Generate a translated title only when explicitly enabled.

reranker:
  local:
    model: jinaai/jina-embeddings-v5-text-nano # The Hugging Face model name of the local embedding model. Example: jinaai/jina-embeddings-v5-text-nano
    encode_kwargs:
    # The kwargs for the encode method of the local embedding model. Details see [here](https://www.sbert.net/docs/package_reference/SentenceTransformer.html#sentence_transformers.SentenceTransformer.encode)
      task: retrieval
      prompt_name: document
  api:
    key: null # API Key of your embedding model API. Example: sk-xxx
    base_url: null # API URL of your embedding model API. Example: https://api.openai.com/v1
    model: null # The model name of the embedding model. Example: text-embedding-3-large
    batch_size: null # The batch size for embedding API requests. Adjust to match your provider's limit. Example: 64

executor:
  debug: false # Whether to use debug mode. Example: true
  send_empty: false # Whether to send an empty email even if no new papers today. Example: true
  max_paper_num: 100 # The maximum number of the papers presented in the email. Example: 100
  source: ??? # The sources of papers to retrieve. Example: ['scix']
  reranker: local # SciX requires the local reranker to preserve its local-only boundary.

output:
  atom:
    enabled: false
    scix_only_policy: exclude # Safe default. Use include only with explicit redistribution permission.
    path: public/index.xml
    title: Astronomy paper recommendations
    description: SciX papers ranked from a Zotero library
    site_url: https://example.github.io
    feed_url: https://example.github.io/index.xml
```

That's all! Now you can test the workflow by manually triggering it:
![test](./assets/test.png)

> [!NOTE]
> The Test-Workflow Action runs the same configured source in debug mode and limits the digest to five papers. A successful scheduled SciX run updates a private GitHub Actions cache; a missing cursor is treated as a first run.

Then check the log and the receiver email after it finishes.

By default, the main workflow runs at 14:00 UTC (22:00 Asia/Shanghai) every day. You can change this time by editing `.github/workflows/main.yml`.

Enable **Settings → Pages → Build and deployment → GitHub Actions** once. The workflow publishes only `public/index.xml`. The SciX delivery cursor stays at the compatibility path `.runtime/state/ads.json`, is restored through the private Actions cache, and is neither committed nor deployed to Pages.

### Local Running
Supported by [uv](https://github.com/astral-sh/uv), this workflow can easily run on your local device if uv is installed:
```bash
# set all the environment variables
# export ZOTERO_ID=xxxx
# ...
cd zotero-arxiv-daily
uv sync --locked
uv run src/zotero_arxiv_daily/main.py
```

## 🚀 Sync with the latest version
This project is in active development. You can subscribe this repo via `Watch` so that you can be notified once we publish new release.

![Watch](./assets/subscribe_release.png)


## 📖 How it works
The workflow retrieves your Zotero library and a bounded window of candidate records from SciX through its ADS-compatible API. A local embedding model ranks each candidate against recent Zotero papers. Only selected records are enriched: a SciX record explicitly linked to arXiv receives arXiv abstract/full text and may be summarized by the configured LLM; a SciX-only record remains a metadata-only local result. The email is sent first, then only the records actually delivered are written to the compatibility cursor, so a failed delivery can be retried. The public Atom feed includes native open sources and successfully arXiv-enriched SciX records, while excluding SciX-only entries by default.

## 📌 Limitations
- The recommendation algorithm is very simple, it may not accurately reflect your interest. Welcome better ideas for improving the algorithm!
- High `MAX_PAPER_NUM` can lead the execution time exceed the limitation of Github Action runner (6h per execution for public repo, and 2000 mins per month for private repo). Commonly, the quota given to public repo is definitely enough for individual use. If you have special requirements, you can deploy the workflow in your own server, or use a self-hosted Github Action runner, or pay for the exceeded execution time.
- SciX API use is subject to the [SciX API documentation](https://scixplorer.org/scixhelp/api-scix/), [terms of use](https://scixplorer.org/scixhelp/policies-scix/terms), and [AI policy](https://scixplorer.org/help/policies/ai-policy). The client logs rate-limit headers, does not bulk-harvest, and fails closed when the configured result bound is exceeded.
- SciX-only metadata is intended for a private, personal digest in the default configuration. Confirm broader redistribution or multi-recipient use with `help@scixplorer.org`.


## 📃 License
Distributed under the AGPLv3 License. See `LICENSE` for detail.

## ❤️ Acknowledgement
- [pyzotero](https://github.com/urschrei/pyzotero)
- [arxiv](https://github.com/lukasschwab/arxiv.py)
- [NASA SciX](https://scixplorer.org/)
- [sentence_transformers](https://github.com/UKPLab/sentence-transformers)

## ☕ Buy Me A Coffee
If you find this project helpful, welcome to sponsor me via WeChat or via [ko-fi](https://ko-fi.com/tidedra).
![wechat_qr](assets/wechat_sponsor.JPG)


## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=TideDra/zotero-arxiv-daily&type=Date)](https://star-history.com/#TideDra/zotero-arxiv-daily&Date)
