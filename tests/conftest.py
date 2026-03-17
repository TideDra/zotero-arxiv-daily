import pytest
import hydra

@pytest.fixture(scope="package")
def config():
    with hydra.initialize(config_path='../config',version_base=None):
        config = hydra.compose(config_name="default")
    config.email.smtp_server = "localhost"
    config.email.smtp_port = 1025
    config.email.sender = "test@example.com"
    config.email.receiver = "test@example.com"
    config.email.sender_password = "test"
    config.llm.api.base_url = "http://localhost:30000/v1"
    config.llm.api.key = "sk-xxx"
    config.reranker.api.base_url = "http://localhost:30000/v1"
    config.reranker.api.key = "sk-xxx"
    config.reranker.api.model = "text-embedding-3-large"
    return config

