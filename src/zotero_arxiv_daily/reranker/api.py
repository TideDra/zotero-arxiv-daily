from .base import BaseReranker, register_reranker
from ..protocol import _litellm_credentials
from openai import OpenAI
import numpy as np

try:
    import litellm
except ImportError:  # optional dependency, only needed when reranker.api.use_litellm is true
    litellm = None


def _litellm_embed(api_params, batch: list[str]) -> list[list[float]]:
    """Embed a batch through LiteLLM. Routes one model string to 100+ embedding
    providers (Voyage, Cohere, Bedrock, Vertex, ...); credentials are omitted
    when blank so LiteLLM falls back to each provider's own env var."""
    if litellm is None:
        raise ImportError(
            "litellm is required when reranker.api.use_litellm is true. "
            "Install it with `uv sync --extra litellm` or `pip install litellm`."
        )
    call_kwargs = {"model": api_params.model, "input": batch, "drop_params": True}
    api_key, api_base = _litellm_credentials({"api": api_params})
    if api_key:
        call_kwargs["api_key"] = api_key
    if api_base:
        call_kwargs["api_base"] = api_base
    response = litellm.embedding(**call_kwargs)
    return [item["embedding"] for item in response.model_dump()["data"]]


@register_reranker("api")
class ApiReranker(BaseReranker):
    def get_similarity_score(self, s1: list[str], s2: list[str]) -> np.ndarray:
        api = self.config.reranker.api
        use_litellm = api.get("use_litellm")
        client = None if use_litellm else OpenAI(api_key=api.key, base_url=api.base_url)
        batch_size = api.get("batch_size") or 64
        all_texts = s1 + s2
        all_embeddings = []
        for i in range(0, len(all_texts), batch_size):
            batch = all_texts[i:i + batch_size]
            if use_litellm:
                all_embeddings.extend(_litellm_embed(api, batch))
            else:
                response = client.embeddings.create(
                    input=batch,
                    model=api.model
                )
                all_embeddings.extend([r.embedding for r in response.data])
        s1_embeddings = np.array(all_embeddings[:len(s1)])           # [n_s1, d]
        s2_embeddings = np.array(all_embeddings[len(s1):])           # [n_s2, d]
        s1_embeddings_normalized = s1_embeddings / np.linalg.norm(s1_embeddings, axis=1, keepdims=True)
        s2_embeddings_normalized = s2_embeddings / np.linalg.norm(s2_embeddings, axis=1, keepdims=True)
        sim = np.dot(s1_embeddings_normalized, s2_embeddings_normalized.T) # [n_s1, n_s2]
        return sim
