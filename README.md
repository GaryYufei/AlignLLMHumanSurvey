# Awesome-Align-LLM-Human

A collection of papers and resources about aligning large language models (LLMs) with human.

Large Language Models (LLMs) trained on extensive textual corpora have emerged as leading solutions for a broad array of Natural Language Processing (NLP) tasks. Despite their notable performance, these models are prone to certain limitations such as misunderstanding human instructions, generating potentially biased content, or factually incorrect (hallucinated) information. Hence, aligning LLMs with human expectations has become an active area of interest within the research community. This survey presents a comprehensive overview of these alignment technologies, including the following aspects. (1) Data collection (2) Training methodologies (3) Model Evaluation. In conclusion, we collate and distill our findings, shedding light on several promising future research avenues in the field. This survey, therefore, serves as a valuable resource for anyone invested in understanding and advancing the alignment of LLMs to better suit human-oriented tasks and expectations.

We hope this repository can help researchers and practitioners to get a better understanding of this emerging field. If this repository is helpful for you, plase help us by citing this paper:
```bash
@article{aligning_llm_human,
    title={Aligning Large Language Models with Human: A Survey},
    author={Yufei Wang and Wanjun Zhong and Liangyou Li and Fei Mi and Xingshan Zeng and Wenyong Huang and Lifeng Shang and Xin Jiang and Qun Liu},
    journal={arXiv preprint arXiv:2307.12966},
    year={2023}
}
```
## News
🔭 This project is under development. You can hit the **STAR** and **WATCH** to follow the updates.
- 2023/07/25: Our initial survey paper [Aligning Large Language Models with Human: A Survey](arxiv.org/abs/2307.12966) becomes available.

## Table of Contents
- [News](#news)
- [Awesome-Aligning-LLM-Human](#awesome-align-llm-human)
    - [Related Surveys](#related-surveys)
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

## Related Surveys
- A Survey of Large Language Models [[Paper]](https://arxiv.org/abs/2303.18223)
- A Survey on Multimodal Large Language Models [[Paper]](https://arxiv.org/abs/2306.13549)
- A Survey on Evaluation of Large Language Models [[Paper]](https://arxiv.org/abs/2307.03109)
- Challenges and Applications of Large Language Models [[Paper]](https://arxiv.org/abs/2307.10169)
- Harnessing the Power of LLMs in Practice: A Survey on ChatGPT and Beyond [[Paper]](https://arxiv.org/abs/2304.13712)
- Domain Specialization as the Key to Make Large Language Models Disruptive: A Comprehensive Survey [[Paper]](https://arxiv.org/abs/2305.18703)

## Alignment Data
### Instructions From Human
#### NLP Benchmarks
- PromptSource: An Integrated Development Environment and Repository for Natural Language Prompts [[paper]](https://aclanthology.org/2022.acl-demo.9/)
- Super-NaturalInstructions: Generalization via Declarative Instructions on 1600+ NLP Tasks [[paper]](https://aclanthology.org/2022.emnlp-main.340/)
- The FLAN collection: Designing data and methods for effective instruction tuning [[paper]](https://arxiv.org/abs/2301.13688)
- Unnatural Instructions: Tuning Language Models with (Almost) No Human Labor [[paper]](https://aclanthology.org/2023.acl-long.806/)
- The OIG Dataset [[Blog]](https://laion.ai/blog/oig-dataset/)
- ChatPLUG: Open-Domain Generative Dialogue System with Internet-Augmented Instruction Tuning for Digital Human [[Paper]](https://arxiv.org/abs/2304.07849)
  
#### Hand-crafted Instructions 
- Free Dolly: Introducing the World's First Truly Open Instruction-Tuned LLM [[Blog]](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm)
- OpenAssistant Conversations -- Democratizing Large Language Model Alignment [[Paper]](https://arxiv.org/abs/2304.07327)
- Chinese open instruction generalist: A preliminary release [[Paper]](https://arxiv.org/abs/2304.07987)
- ShareGPT [[Blog]](https://lmsys.org/blog/2023-03-30-vicuna/)
- Let's Verify Step by Step [[Paper]](https://arxiv.org/abs/2305.20050)

### Instructions From Strong LLMs
#### General Instructions
##### Improving Input Quality
- Self-Instruct: Aligning Language Models with Self-Generated Instructions [[Paper]](https://aclanthology.org/2023.acl-long.754/)
- LaMini-LM: A Diverse Herd of Distilled Models from Large-Scale Instructions [[Paper]](https://arxiv.org/abs/2304.14402)
- Baize: An Open-Source Chat Model with Parameter-Efficient Tuning on Self-Chat Data [[Paper]](https://arxiv.org/abs/2304.01196)
- Large Language Model as Attributed Training Data Generator: A Tale of Diversity and Bias [[Paper]](https://arxiv.org/abs/2306.15895)
- WizardLM: Empowering Large Language Models to Follow Complex Instructions [[Paper]](https://arxiv.org/abs/2304.12244)
- Unnatural Instructions: Tuning Language Models with (Almost) No Human Labor [[paper]](https://aclanthology.org/2023.acl-long.806/)
- WizardCoder: Empowering Code Large Language Models with Evol-Instruct [[Paper]](https://arxiv.org/abs/2306.08568)

##### Improving Output Quality
- Chain-of-Thought Prompting Elicits Reasoning in Large Language Models [[Paper]](https://openreview.net/forum?id=_VjQlMeSB_J)
- Orca: Progressive Learning from Complex Explanation Traces of GPT-4 [[Paper]](https://arxiv.org/abs/2306.02707)
- Lion: Adversarial Distillation of Closed-Source Large Language Model [[Paper]](https://arxiv.org/abs/2305.12870)
- Principle-Driven Self-Alignment of Language Models from Scratch with Minimal Human Supervision [[Paper]](https://arxiv.org/abs/2305.03047)
- ExpertPrompting: Instructing Large Language Models to be Distinguished Experts [[Paper]](https://arxiv.org/abs/2305.14688)
- Phoenix: Democratizing ChatGPT across Languages [[Paper]](https://arxiv.org/abs/2304.10453)

#### Reasoning Instructions
##### General Reasoning
- Specializing Smaller Language Models towards Multi-Step Reasoning [[Paper]](https://openreview.net/forum?id=MXuLl38AEm)
- Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes [[Paper]](https://aclanthology.org/2023.findings-acl.507/)
- Knowledge-Augmented Reasoning Distillation for Small Language Models in Knowledge-Intensive Tasks [[Paper]](https://arxiv.org/abs/2305.18395)
- PaD: Program-aided Distillation Specializes Large Models in Reasoning [[Paper]](https://arxiv.org/abs/2305.13888)
##### Code
- Textbooks Are All You Need [[Paper]](https://arxiv.org/abs/2306.11644)
- WizardCoder: Empowering Code Large Language Models with Evol-Instruct [[Paper]](https://arxiv.org/abs/2306.08568)
- Code Alpaca: An Instruction-following LLaMA model for code generation [[Github]](https://github.com/sahil280114/codealpaca)
##### Maths
- MinT: Boosting Generalization in Mathematical Reasoning via Multi-View Fine-Tuning [[Paper]](https://arxiv.org/abs/2307.07951)
- Goat: Fine-tuned LLaMA Outperforms GPT-4 on Arithmetic Tasks [[Paper]](https://arxiv.org/abs/2305.14201)
  
#### Conversation Instructions 
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
- The False Promise of Imitating Proprietary LLMs [[Paper]](https://arxiv.org/abs/2305.15717)
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
- Measuring Massive Multitask Language Understanding [[Paper]](https://openreview.net/forum?id=d7KBjmI3GmQ)
- Holistic Evaluation of Language Models [[Paper]](https://arxiv.org/abs/2211.09110)
- CMMLU: Measuring massive multitask language understanding in Chinese [[Paper]](https://arxiv.org/abs/2306.09212)
- C-Eval: A Multi-Level Multi-Discipline Chinese Evaluation Suite for Foundation Models [[Paper]](https://arxiv.org/abs/2305.08322)
- KoLA: Carefully Benchmarking World Knowledge of Large Language Models [[Paper]](https://arxiv.org/abs/2306.09296)
- M3KE: A Massive Multi-Level Multi-Subject Knowledge Evaluation Benchmark for Chinese Large Language Models [[Paper]](https://arxiv.org/abs/2305.10263)
- AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models [[Paper]](https://arxiv.org/abs/2304.06364)
- Measuring Massive Multitask Chinese Understanding [[Paper]](https://arxiv.org/abs/2304.12986)
##### Reasoning
- Training Verifiers to Solve Math Word Problems [[Paper]](https://arxiv.org/abs/2110.14168)
- Measuring Massive Multitask Language Understanding [[Paper]](https://openreview.net/forum?id=d7KBjmI3GmQ)
- CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge [[Paper]](https://aclanthology.org/N19-1421/)
- Did Aristotle Use a Laptop? A Question Answering Benchmark with Implicit Reasoning Strategies [[Paper]](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00370/100680/Did-Aristotle-Use-a-Laptop-A-Question-Answering)
- Chain-of-Thought Prompting Elicits Reasoning in Large Language Models [[Paper]](https://openreview.net/forum?id=_VjQlMeSB_J)
- Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them [[Paper]](https://arxiv.org/abs/2210.09261)
- Program Synthesis with Large Language Models [[Paper]](https://arxiv.org/abs/2108.07732)
- DS-1000: A Natural and Reliable Benchmark for Data Science Code Generation [[Paper]](https://arxiv.org/abs/2211.11501)
- Evaluating Large Language Models Trained on Code [[Paper]](https://arxiv.org/abs/2107.03374)
- Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation [[Paper]](https://arxiv.org/abs/2305.01210)
  
##### Safety
- Safety Assessment of Chinese Large Language Models [[Paper]](https://arxiv.org/abs/2304.10436)
- CValues: Measuring the Values of Chinese Large Language Models from Safety to Responsibility [[Paper]](https://arxiv.org/abs/2307.09705)
  
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
##### LLMs for Evaluation
- G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment [[Paper]](https://arxiv.org/abs/2303.16634)
- GPTScore: Evaluate as You Desire [[Paper]](https://arxiv.org/abs/2302.04166)
- Exploring the Use of Large Language Models for Reference-Free Text Quality Evaluation: A Preliminary Empirical Study [[Paper]](https://arxiv.org/abs/2304.00723)
- Can Large Language Models Be an Alternative to Human Evaluations? [[Paper]](https://arxiv.org/abs/2305.01937)
- FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation [[Paper]](https://arxiv.org/abs/2305.14251)
- AlignScore: Evaluating Factual Consistency with A Unified Alignment Function [[Paper]](https://aclanthology.org/2023.acl-long.634/)
- Error Analysis Prompting Enables Human-Like Translation Evaluation in Large Language Models: A Case Study on ChatGPT [[Paper]](https://arxiv.org/abs/2303.13809)
- Human-like Summarization Evaluation with ChatGPT [[Paper]](https://arxiv.org/abs/2304.02554)
- Large Language Models Are State-of-the-Art Evaluators of Code Generation [[Paper]](https://arxiv.org/abs/2304.14317)
- Benchmarking Foundation Models with Language-Model-as-an-Examiner [[Paper]](https://arxiv.org/abs/2306.04181)
- LLM-Eval: Unified Multi-Dimensional Automatic Evaluation for Open-Domain Conversations with Large Language Models [[Paper]](https://aclanthology.org/2023.nlp4convai-1.5/)
##### LLMs bias in Evaluation
- Large Language Models are not Fair Evaluators [[Paper]](https://arxiv.org/abs/2305.17926)
- Style Over Substance: Evaluation Biases for Large Language Models [[Paper]](https://arxiv.org/abs/2307.03025)
- Judging LLM-as-a-judge with MT-Bench and Chatbot Arena [[Paper]](https://arxiv.org/abs/2306.05685)
##### Evaluation-specific LLMs
- PandaLM: An Automatic Evaluation Benchmark for LLM Instruction Tuning Optimization [[Paper]](https://arxiv.org/abs/2306.05087)


