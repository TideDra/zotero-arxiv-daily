import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

@dataclass
class Config:
    # Zotero settings
    zotero_id: str = field(default_factory=lambda: os.getenv('ZOTERO_ID'))
    zotero_key: str = field(default_factory=lambda: os.getenv('ZOTERO_KEY'))
    zotero_ignore: str = field(default_factory=lambda: os.getenv('ZOTERO_IGNORE'))

    # Arxiv settings
    arxiv_query: str = field(default_factory=lambda: os.getenv('ARXIV_QUERY'))
    max_paper_num: int = 100

    # Email settings
    smtp_server: str = field(default_factory=lambda: os.getenv('SMTP_SERVER'))
    smtp_port: int = field(default_factory=lambda: int(os.getenv('SMTP_PORT', 587)))
    sender: str = field(default_factory=lambda: os.getenv('SENDER'))
    receiver: str = field(default_factory=lambda: os.getenv('RECEIVER'))
    sender_password: str = field(default_factory=lambda: os.getenv('SENDER_PASSWORD'))
    send_empty: bool = False

    # LLM settings
    use_llm_api: bool = False
    openai_api_key: str = field(default_factory=lambda: os.getenv('OPENAI_API_KEY'))
    openai_api_base: str = "https://api.openai.com/v1"
    model_name: str = "gpt-4o"
    language: str = "English"

    # Debug settings
    debug: bool = False

    def __post_init__(self):
        # Load environment variables from .env file
        load_dotenv(override=True)

        # Override with environment variables if set and not already set by argparse
        for f in self.__dataclass_fields__:
            env_var = os.getenv(f.upper())
            if env_var is not None and getattr(self, f) is None:
                # Handle boolean and integer types
                if self.__dataclass_fields__[f].type == bool:
                    setattr(self, f, env_var.lower() in ['true', '1'])
                elif self.__dataclass_fields__[f].type == int:
                    setattr(self, f, int(env_var))
                else:
                    setattr(self, f, env_var)

        # Ensure required fields are set
        if self.zotero_id is None:
            raise ValueError("Zotero ID is not set. Please provide it via --zotero_id or ZOTERO_ID environment variable.")
        if self.zotero_key is None:
            raise ValueError("Zotero Key is not set. Please provide it via --zotero_key or ZOTERO_KEY environment variable.")
        if self.arxiv_query is None:
            raise ValueError("Arxiv Query is not set. Please provide it via --arxiv_query or ARXIV_QUERY environment variable.")
        if self.smtp_server is None:
            raise ValueError("SMTP Server is not set. Please provide it via --smtp_server or SMTP_SERVER environment variable.")
        if self.sender is None:
            raise ValueError("Sender email is not set. Please provide it via --sender or SENDER environment variable.")
        if self.receiver is None:
            raise ValueError("Receiver email is not set. Please provide it via --receiver or RECEIVER environment variable.")
        if self.sender_password is None:
            raise ValueError("Sender password is not set. Please provide it via --sender_password or SENDER_PASSWORD environment variable.")
        if self.use_llm_api and self.openai_api_key is None:
            raise ValueError("OpenAI API Key is required when use_llm_api is True.")
