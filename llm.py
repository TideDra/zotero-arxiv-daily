from llama_cpp import Llama
from openai import OpenAI
from loguru import logger

GLOBAL_LLM = None

class LLM:
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        if api_key:
            self.llm = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.llm = Llama.from_pretrained(
                repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
                filename="qwen2.5-3b-instruct-q4_k_m.gguf",
                n_ctx=5_000,
                n_threads=4,
                verbose=False,
            )
        self.model = model

    def generate(self, messages: list[dict]) -> str:
        if isinstance(self.llm, OpenAI):
            response = self.llm.chat.completions.create(messages=messages,temperature=0,model=self.model)
            return response.choices[0].message.content
        else:
            response = self.llm.create_chat_completion(messages=messages,temperature=0)
            return response["choices"][0]["message"]["content"]

def set_global_llm(api_key: str = None, base_url: str = None, model: str = None):
    global GLOBAL_LLM
    GLOBAL_LLM = LLM(api_key=api_key, base_url=base_url, model=model)

def get_llm() -> LLM:
    if GLOBAL_LLM is None:
        logger.info("No global LLM found, creating a default one. Use `set_global_llm` to set a custom one.")
        set_global_llm()
    return GLOBAL_LLM