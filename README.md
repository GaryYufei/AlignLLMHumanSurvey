# Awesome-Align-LLM-Human

A collection of papers and resources about aligning large language models (LLMs) with human.

We hope this repository can help researchers and practitioners to get a better understanding of this emerging field. If this repository is helpful for you, plase help us by citing this paper:
```bash


```

## Table of Contents
- [Awesome-Aligning-LLM-Human](#awesome-align-llm-human)
    - [Alignment Data](#alignment-data)
        - [Instructions From Human](#instructions-from-human)
        - [Instructions From Strong LLMs](#instructions-from-strong-llms)
        - [Instructions Data Management](#instructions-data-management)
    - [Alignment Training](#alignment-training)
        - [Online Human Alignment](#online-human-alignment)
        - [Offline Human Alignment](#offline-human-alignment)
        - [Parameter-Efficient Training](#parameter-efficient-training)
    - [Alignment Evaluation](#alignment-evaluation)
        - [Evaluation Benchmarks](#evaluation-benchmarks)
        - [Evaluation Paradigms](#evaluation-paradigms)


## Alignment Data
### Instructions From Human
#### NLP Benchmarks
- PromptSource: An Integrated Development Environment and Repository for Natural Language Prompts [[paper]](https://aclanthology.org/2022.acl-demo.9/)
- Super-NaturalInstructions: Generalization via Declarative Instructions on 1600+ NLP Tasks [[paper]](https://aclanthology.org/2022.emnlp-main.340/)
- The FLAN collection: Designing data and methods for effective instruction tuning [[paper]](https://arxiv.org/abs/2301.13688)
- Unnatural Instructions: Tuning Language Models with (Almost) No Human Labor [[paper]](https://aclanthology.org/2023.acl-long.806/)
- The OIG Dataset [[Blog]](https://laion.ai/blog/oig-dataset/)
#### Hand-crafted Instructions 
- Free Dolly: Introducing the World's First Truly Open Instruction-Tuned LLM [[Blog]](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm)
- OpenAssistant Conversations -- Democratizing Large Language Model Alignment [[Paper]](https://arxiv.org/abs/2304.07327)
- Chinese open instruction generalist: A preliminary release [[Paper]](https://arxiv.org/abs/2304.07987)
- ShareGPT [[Blog]](https://lmsys.org/blog/2023-03-30-vicuna/)

### Instructions From Strong LLMs
#### Multi-Turn Instructions 
- Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90%* ChatGPT Quality [[Blog]](https://lmsys.org/blog/2023-03-30-vicuna/)
- Enhancing Chat Language Models by Scaling High-quality Instructional Conversations [[Paper]](https://arxiv.org/abs/2305.14233)
- CAMEL: Communicative Agents for "Mind" Exploration of Large Scale Language Model Society [[Paper]](https://arxiv.org/abs/2303.17760)
- Selfee: Iterative self-revising llm empowered by self-feedback generation [[Blog]](https://kaistai.github.io/SelFee/)
#### Multilingual Instructions
- Phoenix: Democratizing ChatGPT across Languages [[Paper]](https://arxiv.org/abs/2304.10453)
- BayLing: Bridging Cross-lingual Alignment and Instruction Following through Interactive Translation for Large Language Models [[Paper]](https://arxiv.org/abs/2306.10968)
- Bactrian-X : A Multilingual Replicable Instruction-Following Model with Low-Rank Adaptation [[Paper]](https://arxiv.org/abs/2305.15011)

### Instructions Data Management
#### Instruction Implications
- How far can camels go? exploring the state of instruction tuning on open resources [[Paper]](https://arxiv.org/abs/2306.04751)
- Flacuna: Unleashing the problem solving power of vicuna using flan fine-tuning [[Paper]](https://arxiv.org/abs/2307.02053)
- Scaling data-constrained language models [[Paper]](https://arxiv.org/abs/2305.16264)
- Towards Better Instruction Following Language Models for Chinese: Investigating the Impact of Training Data and Evaluation [[Paper]](https://arxiv.org/abs/2304.07854)
#### Instruction Quantity
- Becoming self-instruct: introducing early stopping criteria for minimal instruct tuning [[Paper]](https://arxiv.org/abs/2307.03692)
- LIMA: Less Is More for Alignment [[Paper]](https://arxiv.org/abs/2305.11206)
- Instruction Mining: High-Quality Instruction Data Selection for Large Language Models [[Paper]](https://arxiv.org/abs/2307.06290)
- AlpaGasus: Training A Better Alpaca with Fewer Data [[Paper]](https://arxiv.org/abs/2307.08701)

## Alignment Training
### Online Human Alignment
- Training language models to follow instructions with human feedback [[Paper]](https://openreview.net/forum?id=TG8KACxEON)
- RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment [[Paper]](https://arxiv.org/abs/2304.06767)
### Offline Human Alignment
#### Rank-based Training
- Direct Preference Optimization: Your Language Model is Secretly a Reward Model [[Paper]](https://arxiv.org/abs/2305.18290)
- Preference Ranking Optimization for Human Alignment [[Paper]](https://arxiv.org/abs/2306.17492)
- RRHF: Rank Responses to Align Language Models with Human Feedback without tears [[Paper]](https://arxiv.org/abs/2304.05302)
- Calibrating Sequence likelihood Improves Conditional Language Generation [[Paper]](https://openreview.net/forum?id=0qSOodKmJaN)
#### Language-based Training
- OpenChat: Less is More for Open-source Models [[Github]](https://github.com/imoneoi/openchat)
- Languages are rewards: Hindsight finetuning using human feedback [[Paper]](https://arxiv.org/abs/2302.02676)
- Second Thoughts are Best: Learning to Re-Align With Human Values from Text Edits [[Paper]](https://openreview.net/forum?id=u6OfmaGIya1)
- Training Socially Aligned Language Models in Simulated Human Society [[Paper]](https://arxiv.org/abs/2305.16960)
- Selfee: Iterative self-revising llm empowered by self-feedback generation [[Blog]](https://kaistai.github.io/SelFee/)
### Parameter-Efficient Training
- LoRA: Low-Rank Adaptation of Large Language Models [[Paper]](https://openreview.net/forum?id=nZeVKeeFYf9)
- QLoRA: Efficient Finetuning of Quantized LLMs [[Paper]](https://arxiv.org/abs/2305.14314)
- Prefix-Tuning: Optimizing Continuous Prompts for Generation [[Paper]](https://aclanthology.org/2021.acl-long.353/)
- The Power of Scale for Parameter-Efficient Prompt Tuning [[Paper]](https://aclanthology.org/2021.emnlp-main.243/)
- Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning [[Paper]](https://openreview.net/forum?id=lq62uWRJjiY)
- Parameter-Efficient Fine-Tuning Design Spaces [[Paper]](https://openreview.net/forum?id=XSRSWxyJIC)

## Alignment Evaluation
### Evaluation Benchmarks
#### Closed-set Benchmarks
##### General Knowledge
##### Reasoning
##### Coding
#### Open-set Benchmarks
- Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90%* ChatGPT Quality [[Blog]](https://lmsys.org/blog/2023-03-30-vicuna/)
- Self-Instruct: Aligning Language Models with Self-Generated Instructions [[Paper]](https://aclanthology.org/2023.acl-long.754/)
- OpenAssistant Conversations -- Democratizing Large Language Model Alignment [[Paper]](https://arxiv.org/abs/2304.07327)
- FLASK: Fine-grained Language Model Evaluation based on Alignment Skill Sets [[Paper]](https://arxiv.org/abs/2307.10928)
- Judging LLM-as-a-judge with MT-Bench and Chatbot Arena [[Paper]](https://arxiv.org/abs/2306.05685)
- AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback [[Paper]](https://arxiv.org/abs/2305.14387)
### Evaluation Paradigms
#### Human-based Evaluation
- Self-Instruct: Aligning Language Models with Self-Generated Instructions [[Paper]](https://aclanthology.org/2023.acl-long.754/)
- LaMini-LM: A Diverse Herd of Distilled Models from Large-Scale Instructions [[Paper]](https://arxiv.org/abs/2304.14402)
- Training language models to follow instructions with human feedback [[Paper]](https://openreview.net/forum?id=TG8KACxEON)
- Judging LLM-as-a-judge with MT-Bench and Chatbot Arena [[Paper]](https://arxiv.org/abs/2306.05685)
#### LLMs-based Evaluation
##### Reference-free Evaluation
##### LLMs bias in Evaluation
- Large Language Models are not Fair Evaluators [[Paper]](https://arxiv.org/abs/2305.17926)
- Style Over Substance: Evaluation Biases for Large Language Models [[Paper]](https://arxiv.org/abs/2307.03025)
- Judging LLM-as-a-judge with MT-Bench and Chatbot Arena [[Paper]](https://arxiv.org/abs/2306.05685)
##### LLMs for Evaluation
- PandaLM: An Automatic Evaluation Benchmark for LLM Instruction Tuning Optimization [[Paper]](https://arxiv.org/abs/2306.05087)

