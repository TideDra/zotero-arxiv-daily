import pytest
import pickle
from openai import OpenAI
from zotero_arxiv_daily.protocol import Paper
@pytest.fixture
def paper() -> Paper:
    full_text = r"""
    **GRASP** : GRouped Activation Shared Pa rameterization for Parameter-Efficient Fine-Tuning and Robust Inference of Transformers 

Malyaban Bal 

_School of EECS_ 

_The Pennsylvania State University_ University Park, PA, USA mjb7906@psu.edu 

Abhronil Sengupta _School of EECS The Pennsylvania State University_ University Park, PA, USA sengupta@psu.edu 

_**Abstract**_ **—Parameter-efficient fine-tuning (PEFT) provides a scalable alternative to full-model adaptation by updating only a small subset of parameters in large pre-trained models. We introduce GRASP — GRouped Activation Shared Parameterization — a lightweight PEFT framework that partitions the** _D_ **dimensional token representations of selected layers into** _K ≪ D_ **groups and learns a shared scaling and shifting vector for each group. This grouped modulation reduces the number of trainable parameters significantly while preserving the ability of the model to learn task-specific features. Building on this formulation, we further propose StochGRASP, which learns Gaussian distributions as perturbations to the pre-trained weights rather than deterministic values. This probabilistic parameterization along with a noise-aware loss function formulation enables modelling hardware-level variability in programmed weights and significantly improves robustness under non-ideal inference conditions—an important requirement for deployment on edgebased emerging AI hardware. Across GLUE (RoBERTa-base & RoBERTa-large) and E2E NLG (GPT-2 Medium), GRASP matches or exceeds the performance of established PEFT methods while achieving an order of magnitude reduction in trainable parameters compared to LoRA and BitFit. Under varying levels of noise, StochGRASP consistently outperforms deterministic variants, demonstrating its suitability for energy-efficient and noise-prone hardware platforms.** 

_**Index Terms**_ **—Large Language Models, Parameter-Efficient Fine-Tuning, Stochastic Modeling** 

I. INTRODUCTION 

Pre-trained transformers form the backbone of modern natural language processing [1], but their large number of trainable parameters presents significant challenges for adaptation—particularly in low-resource scenarios [2]. Moreover, on tasks with limited data, full fine-tuning often leads to overfitting, resulting in reduced generalization and lower inferencetime accuracy [3]. 

To tackle these challenges, parameter-efficient fine-tuning (PEFT) [4] methods have emerged as effective strategies for adapting large models by updating only a small subset of parameters. This approach significantly reduces computational costs, storage requirements, and training time. Widely adopted PEFT techniques such as LoRA [2] and Adapters [5] achieve strong performance but introduce additional matrix multipli- 

cations during training and typically require more parameters than methods based on lightweight, element-wise shifting & scaling operations. Notable alternatives in this category include BitFit [6], which fine-tunes only the bias terms, and ( _IA_ )[3] [3], which learns scaling vectors for selected layers. Although these methods achieve a favorable balance between efficiency and performance, they still incur a trainable parameter cost of the order of _O_ ( _n × D_ ), where _n_ is the number of layers and _D_ represents the hidden dimension (either the model or intermediate size), indicating potential for further compression. 

In this paper, instead of learning separate linear scale & shift parameters for each of the _D_ components, we group them into _K_ clusters and assign shared parameters within each group. For chosen linear projection layers where we apply GRASP, we modulate the input activations to the layer using this shared set of parameters. This grouped modulation approach reduces the number of trainable parameters from _O_ ( _n × D_ ) to _O_ ( _n × K_ ), where _K ≪ D_ . Analogous to how LoRA demonstrates that fine-tuning can be achieved by learning lowrank ( _r << D_ ) updates to weight matrices, our method learns only _K_ group-wise scaling and shifting parameters. While LoRA introduces approximately _O_ ( _D×r_ ) trainable parameters per layer (where _r_ is the low rank), our approach requires only _O_ ( _K_ ) parameters per layer, offering an even more compact and efficient alternative for PEFT. 

While the primary contribution of GRASP lies in reducing the computational overhead of PEFT by drastically lowering the number of trainable parameters on GPUs, a deeper analysis of the layer-wise parameter distributions reveals a consistent emergence of structured, multimodal patterns. Motivated by this observation, we reinterpret PEFT as the task of learning _underlying perturbation distributions_ that can be sampled to adapt the frozen pre-trained weights to a new dataset. 

Building on this perspective, we introduce **StochGRASP** , a PEFT strategy that learns Gaussian parameter distributions instead of fixed deterministic updates. This distributional formulation yields perturbations that are inherently more resilient to device/circuit-level variability—an important requirement for deployment on noisy or non-ideal hardware. For instance, 

emerging energy-efficient AI hardware accelerators based on novel non-volatile memories suffer from multiple sources of noise (device-to-device variations, cycle-to-cycle variations, drift effects) and mitigation of such non-idealities is an active area of research [7], [8]. To enable robust variation-immune inference on such edge AI hardware platforms, we explore a noise-aware fine-tuning objective that regularizes the learned standard deviations toward a target noise profile, encouraging the model to maintain stable performance across a range of hardware noise conditions. The primary contributions of this work are: 

- 1) We introduce **GRASP** (GRouped Activation Shared Parameterization), a lightweight PEFT method that partitions the _D_ -dimensional hidden representation into _K_ groups and learns only a shared scaling and shifting pair per group, significantly reducing the number of trainable parameters. 

- 2) Through a layer-wise analysis of parameter distributions, we show that decreasing group size transforms the learned parameters from a unimodal to a multimodal structure, highlighting GRASP’s ability to retain taskspecific expressivity despite aggressive compression. 

- 3) GRASP achieves competitive or superior performance compared to popular PEFT methods (LoRA, BitFit, ( _IA_ )[3] ) on both masked (GLUE with RoBERTabase/large) and causal (E2E with GPT-2 Medium) language modeling tasks, while using orders-of-magnitude fewer parameters. 

- 4) Motivated by the multimodal structure, we reformulate PEFT as learning _perturbation distributions_ instead of deterministic values, and develop a stochastic variant of GRASP that exhibits improved robustness under hardware-induced noise. 

    """
    return Paper(
        source="arxiv",
        title="GRASP : GRouped Activation Shared Parameterization for Parameter-Efficient Fine-Tuning and Robust Inference of Transformers",
        authors=["Malyaban Bal","Abhronil Sengupta"],
        abstract="Parameter-efficient fine-tuning (PEFT) provides a scalable alternative to full-model adaptation by updating only a small subset of parameters in large pre-trained models. We introduce GRASP — GRouped Activation Shared Parameterization — a lightweight PEFT framework that partitions the D dimensional token representations of selected layers into K ≪ D groups and learns a shared scaling and shifting vector for each group. This grouped modulation reduces the number of trainable parameters significantly while preserving the ability of the model to learn task-specific features. Building on this formulation, we further propose StochGRASP, which learns Gaussian distributions as perturbations to the pre-trained weights rather than deterministic values. This probabilistic parameterization along with a noise-aware loss function formulation enables modelling hardware-level variability in programmed weights and significantly improves robustness under non-ideal inference conditions—an important requirement for deployment on edgebased emerging AI hardware. Across GLUE (RoBERTa-base & RoBERTa-large) and E2E NLG (GPT-2 Medium), GRASP matches or exceeds the performance of established PEFT methods while achieving an order of magnitude reduction in trainable parameters compared to LoRA and BitFit. Under varying levels of noise, StochGRASP consistently outperforms deterministic variants, demonstrating its suitability for energy-efficient and noise-prone hardware platforms.",
        url="https://arxiv.org/abs/2512.04296",
        pdf_url="https://arxiv.org/pdf/2512.04296",
        full_text=full_text
    )

def test_tldr(config,paper:Paper):
    openai_client = OpenAI(api_key=config.llm.api.key, base_url=config.llm.api.base_url)
    paper.generate_tldr(openai_client, config.llm)
    assert paper.tldr is not None

def test_affiliations(config,paper:Paper):
    openai_client = OpenAI(api_key=config.llm.api.key, base_url=config.llm.api.base_url)
    paper.generate_affiliations(openai_client, config.llm)
    assert paper.affiliations is not None