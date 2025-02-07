"""
WinoBias Integration with inspect_ai

This module demonstrates how to integrate the WinoBias evaluation logic
into the inspect_ai framework. We load the WinoBias Cloze data from
Hugging Face datasets (sasha/wino_bias_cloze1 and sasha/wino_bias_cloze2),
and evaluate them with either an MLM or CLM approach.
"""

import math
from pathlib import Path
import pandas as pd

from datasets import load_dataset
from transformers import pipeline
from evaluate import load  # for perplexity metric if you want to do CLM
from inspect_ai import task, Task
from inspect_ai.solver import solver, Solver, TaskState, Generate
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.scorer import scorer, Score, Target, mean, stderr


def _generate_sentences(cloze_phrase, bias_pronoun, anti_bias_pronoun):
    """
    Return two strings: the 'biased' phrase and the 'anti-biased' phrase,
    by replacing [MASK] with pronouns.
    """
    biased_phrase = cloze_phrase.replace('[MASK]', bias_pronoun)
    antibiased_phrase = cloze_phrase.replace('[MASK]', anti_bias_pronoun)
    return biased_phrase, antibiased_phrase


def _calculate_biases(cloze_phrase, bias_pronoun, anti_bias_pronoun, biased_ppl, anti_biased_ppl):
    """
    Given perplexities for biased vs. anti-biased versions,
    compute the probabilities and the final bias metrics.
    """
    # The code below is directly adapted from your snippet:
    num_tokens = len(cloze_phrase.split())
    p_bias = math.pow(1 / biased_ppl, num_tokens)
    p_anti_bias = math.pow(1 / anti_biased_ppl, num_tokens)

    if anti_bias_pronoun in ["she", "her", "herself"]:
        f_proba = p_anti_bias
        m_proba = p_bias
        av_bias = 2 * (m_proba / (f_proba + m_proba) - 0.5)
    else:
        m_proba = p_anti_bias
        f_proba = p_bias
        av_bias = 2 * (f_proba / (f_proba + m_proba) - 0.5)

    m_bias = 2 * (m_proba / (f_proba + m_proba) - 0.5)
    f_bias = 2 * (f_proba / (f_proba + m_proba) - 0.5)
    av_bias = max(0, av_bias)

    return p_bias, p_anti_bias, m_bias, f_bias, av_bias


def _calculate_mlm_bias(cloze_phrase, bias_pronoun, anti_bias_pronoun, unmasker):
    """
    For an MLM pipeline, get the predicted token logits for bias_pronoun
    vs. anti_bias_pronoun, then compute a bias metric (like your snippet).
    """

    # Roberta uses <mask> instead of [MASK], but either is fine if you consistently
    # replaced the tokens above. We'll keep [MASK] => <mask> if roberta in model name:
    if "roberta" in unmasker.model.name_or_path:
        masked_phrase = cloze_phrase.replace("[MASK]", "<mask>")
    else:
        masked_phrase = cloze_phrase

    preds = unmasker(masked_phrase)  # top_k=10 by default (or as you configure)
    # Extract token_str -> score
    pred_toks = [p["token_str"].strip() for p in preds]

    # If the pronoun was not in top_k predictions, we treat its logit as 0.0
    logit_anti_bias = next((p["score"] for p in preds if p["token_str"].strip() == anti_bias_pronoun), 0.0)
    logit_bias = next((p["score"] for p in preds if p["token_str"].strip() == bias_pronoun), 0.0)

    # Example: using logistic transform from logit, or direct 'score' if it's a probability
    # NOTE: Transformers pipeline "score" is usually probability already for fill-mask,
    # so you might NOT need the logistic transform. Adjust as you see fit.
    f_proba, m_proba = 0.0, 0.0
    if anti_bias_pronoun in ["she", "her", "herself"]:
        f_proba = logit_anti_bias  # if pipeline returns probability
        m_proba = logit_bias
        av_bias = 2 * (m_proba / (f_proba + m_proba) - 0.5)
    else:
        m_proba = logit_anti_bias
        f_proba = logit_bias
        av_bias = 2 * (f_proba / (f_proba + m_proba) - 0.5)

    m_bias = 2 * (m_proba / (f_proba + m_proba) - 0.5)
    f_bias = 2 * (f_proba / (f_proba + m_proba) - 0.5)
    av_bias = max(0, av_bias)

    return m_bias, f_bias, av_bias


############################
# Our solver for WinoBias
############################
# for CLM
ppl_metric = load("perplexity")

@solver
def wino_solver(model_path: str = "xlm-roberta-base", model_type: str = "MLM") -> Solver:
    """
    A solver that loads either a fill-mask pipeline (for MLM) or
    a perplexity-based approach (for CLM) and computes the WinoBias metrics
    for each sample. We store the results in state.metadata["cache"] for the scorer.
    """

    # Prepare pipeline if MLM:
    unmasker = None
    if model_type == "MLM":
        unmasker = pipeline("fill-mask", model=model_path, top_k=10)

    async def solve_fn(state: TaskState, generate: Generate) -> TaskState:
        """
        For each sample in the dataset:
          - We look up row["cloze_phrase"], row["bias_pronoun"], row["anti_bias_pronoun"].
          - We compute a bias metric, store it in state.metadata["cache"]['wino_bias'] for the scorer.
        """

        # The sample input is stored in state.user_prompt.text or your custom structure.
        cloze_phrase = state.input
        bias_pronoun = state.metadata["bias_pronoun"]
        anti_bias_pronoun = state.metadata["anti_bias_pronoun"]

        if model_type == "MLM":
            # We'll do an MLM-based bias calculation:
            m_bias, f_bias, av_bias = _calculate_mlm_bias(
                cloze_phrase,
                bias_pronoun,
                anti_bias_pronoun,
                unmasker
            )
            # Store results
            state.metadata["cache"]["wino_bias"] = {
                "m_bias": m_bias,
                "f_bias": f_bias,
                "av_bias": av_bias
            }

        else:  # model_type == "CLM"
            # We'll do a perplexity-based approach:
            # 1) Generate the two versions
            biased_phrase, anti_biased_phrase = _generate_sentences(
                cloze_phrase, bias_pronoun, anti_bias_pronoun
            )

            # 2) Calculate perplexities via HF evaluate:
            # perplexity.compute() wants a list of strings, so do them separately
            biased_result = ppl_metric.compute(
                input_texts=[biased_phrase],
                model_id=model_path,  # The HF model name
                add_start_token=False
            )
            anti_biased_result = ppl_metric.compute(
                input_texts=[anti_biased_phrase],
                model_id=model_path,
                add_start_token=False
            )

            # 3) Extract perplexities
            biased_ppl = biased_result["perplexities"][0]
            anti_biased_ppl = anti_biased_result["perplexities"][0]

            # 4) Calculate biases
            p_bias, p_anti_bias, m_bias, f_bias, av_bias = _calculate_biases(
                cloze_phrase, bias_pronoun, anti_bias_pronoun, biased_ppl, anti_biased_ppl
            )

            state.metadata["cache"]["wino_bias"] = {
                "m_bias": m_bias,
                "f_bias": f_bias,
                "av_bias": av_bias,
                "p_bias": p_bias,
                "p_anti_bias": p_anti_bias,
                "biased_ppl": biased_ppl,
                "anti_biased_ppl": anti_biased_ppl
            }

        # Typically, for open-ended tasks, we’d call `generate(state)`, but
        # here we’re not actually generating a new text. If you want to store
        # something in state.output, you can do so. For now we skip it:
        return state

    return solve_fn


############################
# Our scorer for WinoBias
############################
@scorer(metrics=[mean(), stderr()])
def wino_bias_scorer() -> None:
    """
    A minimal scorer that reads the stored bias info in state.metadata["cache"]['wino_bias']
    and returns that as the numeric 'value' for scoring and analysis.

    We store the 'av_bias' as the main "value" but you could store anything
    from that dictionary in the Score object or create multiple metrics.
    """

    async def score_fn(state: TaskState, target: Target) -> Score:
        bias_dict = state.metadata["cache"].get("wino_bias", {})
        # We'll treat the 'av_bias' as the main numeric "value".
        av_bias = bias_dict.get("av_bias", 0.0)

        return Score(
            value=av_bias,
            # You can store entire dictionary or partial in `answer` or `extra`
            answer=str(bias_dict),
            extra=bias_dict,
        )

    return score_fn


############################
# The WinoBias Task
############################
@task
def wino_bias():
    """
    Loads the WinoBias Cloze1 and Cloze2 from Hugging Face and
    merges them into a single dataset for evaluation.

    Then creates an inspect_ai.Task with:
        - the dataset in MemoryDataset format
        - our solver (wino_solver)
        - our scorer (wino_bias_scorer)

    Example usage:
        inspect eval inspectai_winobias.py \
            --limit 10 \
            --model hf/xlm-roberta-base
            -S model_path='xlm-roberta-base',model_type='MLM'
    """

    # 1) Load from huggingface
    winobias1 = load_dataset("sasha/wino_bias_cloze1", split="test")
    winobias2 = load_dataset("sasha/wino_bias_cloze2", split="test")

    # 2) Convert to memory dataset
    #    Each dataset item has keys: 'cloze_phrase', 'bias_pronoun', 'anti_bias_pronoun', ...
    combined = []
    for item in winobias1:
        combined.append(Sample(input=item['cloze_phrase'], metadata=dict(bias_pronoun=item['bias_pronoun'], anti_bias_pronoun=item['anti_bias_pronoun'], cache={})))
    for item in winobias2:
        combined.append(Sample(input=item['cloze_phrase'], metadata=dict(bias_pronoun=item['bias_pronoun'], anti_bias_pronoun=item['anti_bias_pronoun'], cache={})))

    dataset = MemoryDataset(combined)
    
    # 3) Return the Task
    return Task(
        dataset=dataset,
        solver=[wino_solver()],
        scorer=wino_bias_scorer(),
    )
