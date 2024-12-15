from tempfile import TemporaryDirectory
import arxiv
import tarfile
import re
from llama_cpp import Llama
from openai import OpenAI
def get_paper_summary(paper:arxiv.Result) -> str:
    with TemporaryDirectory() as tmpdirname:
        file = paper.download_source(dirpath=tmpdirname)
        with tarfile.open(file) as tar:
            tex_files = [f for f in tar.getnames() if f.endswith('.tex')]
            if len(tex_files) == 0:
                return None
            #read all tex files
            introduction = ""
            conclusion = ""
            for t in tex_files:
                f = tar.extractfile(t)
                content = f.read().decode('utf-8')
                #remove comments
                content = re.sub(r'%.*\n', '\n', content)
                content = re.sub(r'\\begin{comment}.*?\\end{comment}', '', content, flags=re.DOTALL)
                content = re.sub(r'\\iffalse.*?\\fi', '', content, flags=re.DOTALL)
                #remove redundant \n
                content = re.sub(r'\n+', '\n', content)
                #remove cite
                content = re.sub(r'~?\\cite.?\{.*?\}', '', content)
                #remove figure
                content = re.sub(r'\\begin\{figure\}.*?\\end\{figure\}', '', content, flags=re.DOTALL)
                #remove table
                content = re.sub(r'\\begin\{table\}.*?\\end\{table\}', '', content, flags=re.DOTALL)
                #find introduction and conclusion
                # end word can be \section or \end{document} or \bibliography or \appendix
                match = re.search(r'\\section\{Introduction\}.*?(\\section|\\end\{document\}|\\bibliography|\\appendix|$)', content, flags=re.DOTALL)
                if match:
                    introduction = match.group(0)
                match = re.search(r'\\section\{Conclusion\}.*?(\\section|\\end\{document\}|\\bibliography|\\appendix|$)', content, flags=re.DOTALL)
                if match:
                    conclusion = match.group(0)
                
    return introduction, conclusion

def get_paper_tldr(paper:arxiv.Result, model:Llama | OpenAI, **kwargs) -> str:
    try:
        introduction, conclusion = get_paper_summary(paper)
    except:
        introduction, conclusion = "", ""
    prompt = """Given the title, abstract, introduction and the conclusion (if any) of a paper in latex format, generate a one-sentence TLDR summary:
    
    \\title{__TITLE__}
    \\begin{abstract}__ABSTRACT__\\end{abstract}
    __INTRODUCTION__
    __CONCLUSION__
    """
    prompt = prompt.replace('__TITLE__', paper.title)
    prompt = prompt.replace('__ABSTRACT__', paper.summary)
    prompt = prompt.replace('__INTRODUCTION__', introduction)
    prompt = prompt.replace('__CONCLUSION__', conclusion)
    if isinstance(model, Llama):
        prompt_tokens = model.tokenize(prompt.encode("utf-8"))
        prompt_tokens = prompt_tokens[:3800]  # truncate to 3800 tokens
        prompt = model.detokenize(prompt_tokens).decode("utf-8")
        response = model.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant who perfectly summarizes scientific paper, and gives the core idea of the paper to the user.",
                },
                {"role": "user", "content": prompt},
            ],
            top_k=0,
            temperature=0,
        )
        return response["choices"][0]["message"]["content"]
    elif isinstance(model, OpenAI):
        response = model.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant who perfectly summarizes scientific paper, and gives the core idea of the paper to the user.",
                },
                {"role": "user", "content": prompt},
            ],
            # Only set the temperature to 0 (According to OpenAI's Suggestions)
            temperature=0,
            model=kwargs.get("model_name"),
        )
        return response.choices[0].message.content
