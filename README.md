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

<p align="center"> Recommend new arxiv papers of your interest daily according to your Zotero library.
    <br> 
</p>

> [!IMPORTANT]
> Please keep an eye on this repo, and merge your forked repo in time when there is any update of this upstream, in order to enjoy new features and fix found bugs.

## 🧐 About <a name = "about"></a>

> Track new scientific researches of your interest by just forking (and staring) this repo!😊

*Zotero-arXiv-Daily* finds arxiv papers that may attract you based on the context of your Zotero library, and then sends the result to your mailbox📮. It can be deployed as Github Action Workflow with **zero cost**, **no installation**, and **few configuration** of Github Action environment variables for daily **automatic** delivery.

## ✨ Features
- Totally free! All the calculation can be done in the Github Action runner locally within its quota (for public repo).
- AI-generated TL;DR for you to quickly pick up target papers.
- Affiliations of the paper are resolved and presented.
- Links of PDF and code implementation (if any) presented in the e-mail.
- List of papers sorted by relevance with your recent research interest.
- Fast deployment via fork this repo and set environment variables in the Github Action Page.
- Support LLM API for generating TL;DR of papers.
- Ignore unwanted Zotero papers using gitignore-style pattern.

## 📷 Screenshot
![screenshot](./assets/screenshot.png)

## 🚀 Usage
### Quick Start
1. Fork (and star😘) this repo.
![fork](./assets/fork.png)

2. Set Github Action environment variables.
![secrets](./assets/secrets.png)

Below are all the secrets you need to set. They are invisible to anyone including you once they are set, for security.

| Key | Required | Type |Description | Example |
| :--- | :---: | :---  | :---  | :--- |
| ZOTERO_ID | ✅ | str  | User ID of your Zotero account. **User ID is not your username, but a sequence of numbers**Get your ID from [here](https://www.zotero.org/settings/security). You can find it at the position shown in this [screenshot](https://github.com/TideDra/zotero-arxiv-daily/blob/main/assets/userid.png). | 12345678  |
| ZOTERO_KEY | ✅ | str  | An Zotero API key with read access. Get a key from [here](https://www.zotero.org/settings/security).  | AB5tZ877P2j7Sm2Mragq041H   |
| ARXIV_QUERY | ✅ | str  | The categories of target arxiv papers. Use `+` to concatenate multiple categories. The example retrieves papers about AI, CV, NLP, ML. Find the abbr of your research area from [here](https://arxiv.org/category_taxonomy).  | cs.AI+cs.CV+cs.LG+cs.CL |
| BIORXIV_QUERY | ✅ | str  | The categories of target arxiv papers. Use `+` to concatenate multiple categories. The example retrieves papers about cell biology, neuroscience, biophysics. Find the abbr of your research area from [here](https://www.biorxiv.org/). Note to replace " " with "-"  | cell_biology+neuroscience |
| SMTP_SERVER | ✅ | str | The SMTP server that sends the email. I recommend to utilize a seldom-used email for this. Ask your email provider (Gmail, QQ, Outlook, ...) for its SMTP server| smtp.qq.com |
| SMTP_PORT | ✅ | int | The port of SMTP server. | 465 |
| SENDER | ✅ | str | The email account of the SMTP server that sends you email. | abc@qq.com |
| SENDER_PASSWORD | ✅ | str | The password of the sender account. Note that it's not necessarily the password for logging in the e-mail client, but the authentication code for SMTP service. Ask your email provider for this.   | abcdefghijklmn |
| RECEIVER | ✅ | str | The e-mail address that receives the paper list. | abc@outlook.com |
| MAX_PAPER_NUM | | int | The maximum number of the papers presented in the email. This value directly affects the execution time of this workflow, because it takes about 70s to generate TL;DR for one paper. `-1` means to present all the papers retrieved. | 50 |
| MAX_BIORXIV_NUM| | int | Max number of biorxiv papers. | 50 |
| SEND_EMPTY | | bool | Whether to send an empty email even if no new papers today. | False |
| USE_LLM_API | | bool | Whether to use the LLM API in the cloud or to use local LLM. If set to `1`, the API is used. Else if set to `0`, the workflow will download and deploy an open-source LLM. Default to `0`. | 0 |
| OPENAI_API_KEY | | str | API Key when using the API to access LLMs. You can get FREE API for using advanced open source LLMs in [SiliconFlow](https://cloud.siliconflow.cn/i/b3XhBRAm). | sk-xxx |
| OPENAI_API_BASE | | str | API URL when using the API to access LLMs. If not filled in, the default is the OpenAI URL. | https://api.siliconflow.cn/v1 |
| MODEL_NAME | | str | Model name when using the API to access LLMs. If not filled in, the default is gpt-4o. Qwen/Qwen2.5-7B-Instruct is recommended when using [SiliconFlow](https://cloud.siliconflow.cn/i/b3XhBRAm). | Qwen/Qwen2.5-7B-Instruct |

There are also some public variables (Repository Variables) you can set, which are easy to edit.
![vars](./assets/repo_var.png)

| Key | Required | Type | Description | Example |
| :--- | :---  | :---  | :--- | :--- |
| ZOTERO_IGNORE | | str | Gitignore-style patterns marking the Zotero collections that should be ignored. One rule one line. Learn more about [gitignore](https://git-scm.com/docs/gitignore). | AI Agent/<br>**/survey<br>!LLM/survey |
| REPOSITORY | | str | The repository that provides the workflow. If set, the value can only be `TideDra/zotero-arxiv-daily`, in which case, the workflow always pulls the latest code from this upstream repo, so that you don't need to sync your forked repo upon each update, unless the workflow file is changed. | `TideDra/zotero-arxiv-daily` |
| REF | | str | The specified ref of the workflow to run. Only valid when REPOSITORY is set to `TideDra/zotero-arxiv-daily`. Currently supported values include `main` for stable version, `dev` for development version which has new features and potential bugs. | `main` |
| LANGUAGE | | str | The language of TLDR; Its value is directly embeded in the prompt passed to LLM | Chinese |

That's all! Now you can test the workflow by manually triggering it:
![test](./assets/test.png)

> [!NOTE]
> The Test-Workflow Action is the debug version of the main workflow (Send-emails-daily), which always retrieve 5 arxiv papers regardless of the date. While the main workflow will be automatically triggered everyday and retrieve new papers released yesterday. There is no new arxiv paper at weekends and holiday, in which case you may see "No new papers found" in the log of main workflow.

Then check the log and the receiver email after it finishes.

By default, the main workflow runs on 22:00 UTC everyday. You can change this time by editting the workflow config `.github/workflows/main.yml`.

### Local Running
Supported by [uv](https://github.com/astral-sh/uv), this workflow can easily run on your local device if uv is installed:
```bash
# set all the environment variables
# export ZOTERO_ID=xxxx
# ...
cd zotero-arxiv-daily
uv run main.py
```
To run the demo that support biorxiv recommendation, you can run
```bash
uv run biorxiv_demo.py
```
> [!IMPORTANT]
> The workflow will download and run an LLM (Qwen2.5-3B, the file size of which is about 3G). Make sure your network and hardware can handle it.

> [!WARNING]
> Other package managers like pip or conda are not tested. You can still use them to install this workflow because there is a `pyproject.toml`, while potential problems exist.

## 🚀 Sync with the latest version
This project is in active development. You can subscribe this repo via `Watch` so that you can be notified once we publish new release.

![Watch](./assets/subscribe_release.png)


## 📖 How it works
*Zotero-arXiv-Daily* firstly retrieves all the papers in your Zotero library and all the papers released in the previous day, via corresponding API. Then it calculates the embedding of each paper's abstract via an embedding model. The score of a paper is its weighted average similarity over all your Zotero papers (newer paper added to the library has higher weight).

The TLDR of each paper is generated by a lightweight LLM (Qwen2.5-3b-instruct-q4_k_m), given its title, abstract, introduction, and conclusion (if any). The introduction and conclusion are extracted from the source latex file of the paper.

## 📌 Limitations
- The recommendation algorithm is very simple, it may not accurately reflect your interest. Welcome better ideas for improving the algorithm!
- This workflow deploys an LLM on the cpu of Github Action runner, and it takes about 70s to generate a TLDR for one paper. High `MAX_PAPER_NUM` can lead the execution time exceed the limitation of Github Action runner (6h per execution for public repo, and 2000 mins per month for private repo). Commonly, the quota given to public repo is definitely enough for individual use. If you have special requirements, you can deploy the workflow in your own server, or use a self-hosted Github Action runner, or pay for the exceeded execution time.

## 👯‍♂️ Contribution
Any issue and PR are welcomed! But remember that **each PR should merge to the `dev` branch**.

## 📃 License
Distributed under the AGPLv3 License. See `LICENSE` for detail.

## ❤️ Acknowledgement
- [pyzotero](https://github.com/urschrei/pyzotero)
- [arxiv](https://github.com/lukasschwab/arxiv.py)
- [sentence_transformers](https://github.com/UKPLab/sentence-transformers)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)

## ☕ Buy Me A Coffee
If you find this project helpful, welcome to sponsor me via WeChat or via [ko-fi](https://ko-fi.com/tidedra).
![wechat_qr](assets/wechat_sponsor.JPG)


## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=TideDra/zotero-arxiv-daily&type=Date)](https://star-history.com/#TideDra/zotero-arxiv-daily&Date)
