# Evaluating LLMs on Hugging Face
The AVID (AI Vulnerability Database) team is examining a few large language models (LLMs) on Hugging Face. We will develop a way to evaluate and catalog their vulnerabilities in the hopes of encouraging the community to contribute. As a first step, we’re going to pick a single model and try to evaluate it for vulnerabilities on a specific task. Once we have done one model, we’ll see if we can generalize our data sets and tools to function broadly on the Hugging Face platform.

## Vision
Build a foundation for evaluating LLMs using the Hugging Face platform and start populating our database with real incidents.

## Goals
* Build, test, and refine our own data sets for evaluating models 
* Identify existing data sets we want to use for evaluating models (Ex. Stereoset, wino_bias, etc.)
* Test different tools and methods for evaluating LLMs so we can start to create and support some for cataloging vulnerabilities in our database
* Start populating the database with known, verified, and discovered vulnerabilities for models hosted on Hugging Face

## Resources
The links below should help anyone who wants to support the project find a place to start. They are not exhaustive, and people should feel free to add anything relevant.
* [Huggingface.co](https://huggingface.co/) - platform for hosting data sets, models, etc.
* [Papers With Code](https://paperswithcode.com/) - a platform for the ML community to share research, it may have additional data sets or papers
* Potential Models
  * [xlm-roberta-base](https://huggingface.co/xlm-roberta-base)
  * [Bert-base-uncased](https://huggingface.co/bert-base-uncased)
  * [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased)
  * [gpt2](https://huggingface.co/gpt2)
* Data Sets
  * [StereoSet](https://stereoset.mit.edu/) -  StereoSet is a dataset that measures stereotype bias in language models. StereoSet consists of 17,000 sentences that measure model preferences across gender, race, religion, and profession.
  * [Wino_bias](https://huggingface.co/datasets/wino_bias) - WinoBias, a Winograd-schema dataset for coreference resolution focused on gender bias.
  * [Jigsaw_unintended_bias](https://huggingface.co/distilbert-base-uncased) - The main target for this dataset is toxicity prediction. Several toxicity subtypes are also available, so the dataset can be used for multi-attribute prediction. 
  * [BigScienceBiasEval/bias-shades](https://huggingface.co/datasets/BigScienceBiasEval/bias-shades) - This dataset was curated by hand-crafting stereotype sentences by native speakers from the culture which is being targeted. (Seems incomplete)
  * [md_gender_bias](https://huggingface.co/datasets/md_gender_bias) - The dataset can be used to train a model for classification of various kinds of gender bias.  
  * [social_bias_frames](https://huggingface.co/datasets/social_bias_frames) - This dataset supports both classification and generation. Sap et al. developed several models using the SBIC.
  * [BIG-bench/keywords_to_tasks.md at main](https://github.com/google/BIG-bench/blob/main/bigbench/benchmark_tasks/keywords_to_tasks.md#pro-social-behavior) - includes many options for testing bias of different types (gender, religion, etc.)
  * [FB Fairscore](https://github.com/facebookresearch/ResponsibleNLP/tree/main/fairscore) - Has a wide selection of sources, focuses on gender (including non-binary).
* Papers
  * [Evaluate & Evaluation on the Hub: Better Best Practices for Data and Model Measurement](https://arxiv.org/abs/2210.01970)
  * [On the Dangers of Stochastic Parrots](https://dl.acm.org/doi/10.1145/3442188.3445922)
  * [Language (Technology) is Power: A Critical Survey of “Bias” in NLP](https://aclanthology.org/2020.acl-main.485/)
  * [Measuring Fairness with Biased Rulers: A Comparative Study on Bias Metrics for Pre-trained Language Models](https://aclanthology.org/2022.naacl-main.122/)
  * [Harms of Gender Exclusivity and Challenges in Non-Binary Representation in Language Technologies](https://aclanthology.org/2021.emnlp-main.150/)
