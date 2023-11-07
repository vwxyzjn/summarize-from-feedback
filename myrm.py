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
# from transformers import AutoTokenizer
from summarize_from_feedback.utils.torch_utils import first_true_indices, gather_one
from torch.utils.data import DataLoader


"""
Evaluates a reward model on a set of query-responses examples. The output will contain the same
json data as the input along with an extra key containing the predicted reward.
"""


test_queries = torch.tensor([  220,   220,   220,   220,   220,   220,   220,   220,   220,   220,
           220,   220,   220,   220,   220,   220,   220,    50, 10526, 22083,
         49828,    25,   374,    14, 22856,   501,   198,   198, 49560,  2538,
            25, 21235,  1104,   329,   616,  3956,   357, 39363, 19991,   291,
             8,   198,   198, 32782,    25, 10750,    13,   314,  1101,   287,
           281,  7016,   537,   385,  1706, 31699,    13,  5949,  5823,   329,
          5794,  1806,    11,   314,  1101,   319,  5175,    13,   314,   761,
          5608,   287,  5742,  3613,   616,  3956,   422, 38968,    11,   290,
          2192, 28822,    13,   317,  1310,  4469,    26,   220,   198,  1544,
           338,  1542,   812,  1468,    11, 14641, 19991,   291,   416,   347,
         17184,    11,   290,   484,   423,  3421,   465, 17638,   517,  1661,
           621,   314,   460,  3505,    13,   314,  1101,  2048,   257,  5707,
          7099,   621,   339,   318,    11,   290,   423,  7334,   510,  4379,
           683, 31736,    11,   466,  1588,  6867,   286,  5010,    11,   307,
          5169,    11,   423,  3294, 35078,  6266,    11,  3503,    13,  3423,
           338,   262,  1517,    11,   339,  3011, 30285,    11,  5300,  6639,
           422,   465,  1117,  1229,   283,   295,    11,   290,  9911,  2263,
           340,    13,  1320,   338,   644,  5640,  8640,    13,  2011,  6621,
          1595,   470,   892,   340,   338, 22794,    11,  1719,   587,   691,
           257,   614,  5475,   422,   683,    13,   679,  3160,   351,   616,
          3397,    11,   290,   314,   423,  7620,   587,   287,   257,  3518,
         38880,   351,   683,    13,  1649,   314,   373,    11,   339, 25711,
           502,   866,   284,  4405,   616,  1986,   287,   290,  5025,   262,
          2589,   314,   531, 30629,  3672,   828,   345,   836,   470,   765,
           284,  5938,   502,  1911,   314,   760,    11,   422,  3957,   510,
           351,   683,    11,   326,   339,   318, 21757,   290,   339, 21046,
            13,  2399,  1266,  1545,  2823,  2241,   618,   484,   547,   287,
          1029,  1524,   290,   314,   760,   339,  8020,   470,   587,   262,
           976,  1201,    13,   679,   373, 18605,  7287,   355,   257,  1200,
           290,   468,   587,   832,   257,  1256,    13, 34668,   616,  2802,
          1444,   502,   284,  1309,   502,   760,   326,   484,   423,   257,
         35078,  1502,   319,   683,   780,   484,   466,   407,  1254,  3338,
           351,   683,   612,   357,  1544,   468,  2192,   407,  2077, 14103,
           287,   257,  3155,  1933,   290, 18240,   477,   286,   262, 23110,
            11,   587,  7650,    11,   550, 11418,   737,   775,   836,   470,
           760,   810,   339,   318,    11,   475,  2184,   318,   319,   262,
          2534,   358,   290,   314,   836,   470,   760,   644,   481,  1645,
           284,   683,    13,   347, 17184,   468,   587,   257, 17123,   290,
           339,   338,   587,   319,   262,  4953,  1351,   351,   606,   284,
           651,   281,  7962,   329,  9337,    13,   679,   468,   587,  7195,
           257,  6808, 29365,   329,  1933,   290,   318,   635,   319,   257,
          4953,  1351,   329,  9934,    13,   679,  2067,  9216,   284, 19222,
           262,  2356,    13,  1318,   338,   523,   881,   517,   284,   340,
            11,   290,   616,  6621,   290,   314,  3377,   938,  1755, 13774,
           290,  7722,   780,   356,   389,  5000, 21144,    13,   775,   836,
           470,   765,   284,   766,   683,   319,   262,  4675,    13,  3423,
           338,   810,   314,  1101,  4737,   329,  5608,    13,  4231,   612,
           597,  1900,  4056,   393,  2628,   326,   460,  1037,   514,   651,
           683,   319,   465,  3625,    30,   775,   821,  2048,  1728,   339,
           338, 10463,   783,   290,   339,  1595,   470,   423,   257,  2685,
          3072,   351,  1366,   393,  2431,    13,   198,   198, 14990,    26,
          7707,    25])

test_responses = torch.tensor([[ 2011,  3956,   318, 19991,   291,    11,   319,  1117,    82,   290,
            845, 11557,    13,   679,  1244,   307, 10463,   783,    13, 15616,
            329,   597,  1900,  4056,   393,  2628,   326,   460,  1037,   683,
            651,   683,   319,   465,  3625,   290,   503,   319,   465,  3625,
             13, 50256,     -1,     -1,     -1,     -1,     -1,     -1]])


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
    # tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # breakpoint()
    # ds = datasets.load_dataset("vwxyzjn/summarize_from_feedback_tldr_3_filtered_oai_preprocessing", split="train")
    ds = datasets.load_dataset("vwxyzjn/summarize_from_feedback_oai_preprocessing", split="validation")

    def ds_process(example):
        query_tokens = torch.tensor(example["query_token"])
        non_padded_idx = first_true_indices(query_tokens != 50257)
        query_tokens[:non_padded_idx] = 220
        response0_tokens = torch.tensor([example["response0_token"]])
        non_padded_idx = first_true_indices(response0_tokens == 50257)
        if non_padded_idx != response0_tokens.size(1):
            response0_tokens[:,non_padded_idx:] = -1
            response0_tokens[:,non_padded_idx] = 50256
        response1_tokens = torch.tensor([example["response1_token"]])
        non_padded_idx = first_true_indices(response1_tokens == 50257)
        if non_padded_idx != response0_tokens.size(1):
            response1_tokens[:,non_padded_idx:] = -1
            response1_tokens[:,non_padded_idx] = 50256
        response_tokens = torch.cat([response0_tokens, response1_tokens], dim=0)
        example["query_token"] = query_tokens
        example["response_tokens"] = response_tokens
        return example

    ds = ds.map(ds_process)
    ds.set_format("torch", columns=["query_token", "response_tokens", "choice"])

    loader = DataLoader(ds, batch_size=16)

    query_tokens = test_queries
    response_tokens = test_responses
    assert_shape_eq(query_tokens, (H.task.query.length,), "Context tokens shape mismatch")
    assert_eq(response_tokens.dim(), 2)

    n_responses = response_tokens.size(0)

    results = reward_model.reward(
        query_tokens=query_tokens.unsqueeze(0),
        response_tokens=response_tokens.unsqueeze(0),
        act_dtype=act_dtype,
    )

    rewards = to_numpy(results["reward"].reshape((n_responses,)))
    print(rewards)

    all_rewards = []
    count = 0
    correct = 0
    i = 0
    for item in tqdm(loader):
        count += 1
        query_tokens = torch.tensor(item["query_token"])
        response_tokens = torch.tensor(item["response_tokens"])
        
        # query_tokens = torch.tensor(input["context_tokens"])
        # assert_shape_eq(query_tokens, (H.task.query.length,), "Context tokens shape mismatch")
        # response_tokens = torch.tensor(input["sample_tokens"])
        # breakpoint()
        # assert_eq(response_tokens.dim(), 2)

        n_responses = response_tokens.size(0)
        results = reward_model.reward(
            query_tokens=query_tokens,
            response_tokens=response_tokens,
            act_dtype=act_dtype,
        )

        rewards = results["reward"]
        correct += (rewards.argmax(1) == item["choice"]).sum().item()
        count += query_tokens.size(0)
        if i % 20 == 0:
            print("accuracy", correct / count, "count", count)
        all_rewards.append(rewards)
        print(rewards, item["choice"], rewards.argmax() == item["choice"])
        i += 1


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
