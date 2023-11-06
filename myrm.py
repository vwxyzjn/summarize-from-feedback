import json
import os
from dataclasses import dataclass, field

import blobfile as bf
import numpy as np
import torch
from rich.pretty import pprint
import fire
import datasets
from tqdm import tqdm

from summarize_from_feedback.datasets import jsonl_encoding
from summarize_from_feedback.query_response_model import ModelSpec
from summarize_from_feedback.reward_model import RewardModel
from summarize_from_feedback.task_data import make_jsonl_samples_iter
from summarize_from_feedback.tasks import TaskHParams
from summarize_from_feedback.utils import Timer, hyperparams
from summarize_from_feedback.utils.assertions import assert_shape_eq, assert_eq
from summarize_from_feedback.utils.logging_utils import setup_logging_with_pacific_tz
from summarize_from_feedback.utils.torch_utils import to_numpy
from summarize_from_feedback import eval_rm
from summarize_from_feedback.utils import experiment_helpers as utils
from summarize_from_feedback.utils.combos import combos, bind, bind_nested
from summarize_from_feedback.utils.experiments import experiment_def_launcher
from transformers import AutoTokenizer


"""
Evaluates a reward model on a set of query-responses examples. The output will contain the same
json data as the input along with an extra key containing the predicted reward.
"""


@dataclass
class HParams(hyperparams.HParams):
    reward_model_spec: ModelSpec = field(default_factory=ModelSpec)
    task: TaskHParams = field(default_factory=TaskHParams)
    input_path: str = None  # Should contain files samples.0.jsonl, samples.1.jsonl, ...
    fp16_activations: bool = True
    output_key: str = "predicted_reward"


def main(H: HParams):
    layout = H.reward_model_spec.run_params.all_gpu_layout()
    H.reward_model_spec.device = "cpu"
    H.fp16_activations = False
    pprint(H)

    reward_model = RewardModel(task_hparams=H.task, spec=H.reward_model_spec, layout=layout)
    act_dtype = torch.float16 if H.fp16_activations else torch.float32
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # breakpoint()
    # ds = datasets.load_dataset("vwxyzjn/summarize_from_feedback_tldr_3_filtered_oai_preprocessing", split="train")
    ds = datasets.load_dataset("vwxyzjn/summarize_from_feedback_oai_preprocessing", split="validation")

    all_rewards = []
    count = 0
    correct = 0
    i = 0
    for item in tqdm(ds):
        count += 1
        query_tokens = torch.tensor(item["query_token"])
        query_tokens[query_tokens == 50257] = 0
        response0_tokens = torch.tensor([item["response0_token"]])
        response0_tokens[response0_tokens == 50257] = 0
        response1_tokens = torch.tensor([item["response1_token"]])
        response1_tokens[response1_tokens == 50257] = 0
        response_tokens = torch.cat([response0_tokens, response1_tokens], dim=0)
        with Timer() as timer:

            
            # query_tokens = torch.tensor(input["context_tokens"])
            assert_shape_eq(query_tokens, (H.task.query.length,), "Context tokens shape mismatch")
            # response_tokens = torch.tensor(input["sample_tokens"])
            assert_eq(response_tokens.dim(), 2)

            n_responses = response_tokens.size(0)

            results = reward_model.reward(
                query_tokens=query_tokens.unsqueeze(0),
                response_tokens=response_tokens.unsqueeze(0),
                act_dtype=act_dtype,
            )

            rewards = to_numpy(results["reward"].reshape((n_responses,)))
            if rewards.argmax() == item["choice"]:
                correct += 1
            if i % 20 == 0:
                print("accuracy", correct / count, "count", count)
            all_rewards.append(rewards)
            i += 1


        print(f"Took {timer.interval} seconds", np.mean(all_rewards), np.std(all_rewards))
    breakpoint()



def experiment_definitions():
    rm4 = combos(
        bind_nested("task", utils.tldr_task),
        bind("mpi", 1),
        bind_nested("reward_model_spec", utils.rm4()),
        bind("input_path", "https://openaipublic.blob.core.windows.net/summarize-from-feedback/samples/sup4_ppo_rm4"),
    )

    test = combos(
        bind_nested("task", utils.test_task),
        bind_nested("reward_model_spec", utils.random_teeny_model_spec()),
        bind("mpi", 1),
        bind("input_path", "https://openaipublic.blob.core.windows.net/summarize-from-feedback/samples/test"),
    )
    test_cpu = combos(
        test,
        bind_nested("reward_model_spec", utils.stub_model_spec()),
    )
    tldrtest = combos(
        bind_nested("task", utils.test_tldr_task),
        bind_nested("reward_model_spec", utils.random_teeny_model_spec(n_shards=2)),
        bind("mpi", 2),
    )
    return locals()


if __name__ == "__main__":
    fire.Fire(
        experiment_def_launcher(
            experiment_dict=experiment_definitions(), main_fn=main, mode="local"
        )
    )
