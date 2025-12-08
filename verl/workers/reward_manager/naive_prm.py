# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict

import torch
import re
from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.utils.prm import RemoteRewardModelConfig, RMRemoteCaller
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed, is_equiv
import functools
import json
import os

def add_tags(response_text, split_step_token_lst):

    # add <think> if not exist, and add <step> after <think>
    if "<think>" not in response_text:
        response_text = "<think>\n<step>" + response_text
    else:
        response_text = re.sub(r'(<think>)', r'\1\n<step>', response_text, count=1)
    
    # add </think> if not exist, and add </step> behind </think>
    if "</think>" in response_text:
        response_text = re.sub(r'(</think>)', r'</step>\n\1', response_text, count=1)
    elif "**Final Answer**" in response_text:
        # if "**Final Answer**" exists, add </step></think> behind it
        response_text = re.sub(r'(\*\*Final Answer\*\*)', r'</step>\n</think>\1', response_text, count=1)
    else:
        # if "**Final Answer**" not exists, add </step></think> after the last '\n\n'
        match_two_endlines = list(re.finditer(r'(\n\n)', response_text))
        if match_two_endlines:
            response_text = response_text[:match_two_endlines[-1].end()] + "</step>\n</think>" + response_text[match_two_endlines[-1].end():]

    # add <answer> after </think> if not exist, and add <step> after <answer>
    if "<answer>" not in response_text:
        response_text = re.sub(r'(</think>)', r'\1\n<answer>\n<step>', response_text, count=1)
    else:
        response_text = re.sub(r'(<answer>)', r'\1\n<step>', response_text, count=1)
    
    # if </answer> not exists, add </answer> at the end of response_text
    if "</answer>" not in response_text:
        response_text = response_text + "</step>\n</answer>"
    else:
        response_text = re.sub(r'(</answer>)', r'</step>\n\1', response_text, count=1)
    
    if split_step_token_lst == ['\n\n']:
        match_think = list(re.finditer(r'(<think>\n<step>)', response_text))
        match_slash_think = list(re.finditer(r'(</step>\n</think>)', response_text))
        match_answer = list(re.finditer(r'(<answer>\n<step>)', response_text))
        match_slash_answer = list(re.finditer(r'(</step>\n</answer>)', response_text))
        assert match_think[0].end() < match_slash_think[0].start() < match_answer[0].end() < match_slash_answer[0].start()

        #response_text = response_text[:match_think[0].end()] + response_text[match_think[0].end():match_slash_think[0].start()].strip() + response_text[match_slash_think[0].start():match_answer[0].end()] + response_text[match_answer[0].end():match_slash_answer[0].start()].strip() + response_text[match_slash_answer[0].start():]

        response_text = "<answer>\n<step>" + response_text[match_answer[0].end():match_slash_answer[0].start()].strip() + response_text[match_slash_answer[0].start():]

        response_text = re.sub(r'(\n\n)', r'</step>\1<step>', response_text)
    else:
        pattern = '|'.join([re.escape(token) for token in split_step_token_lst])
        response_text = re.sub(f'({pattern})', r'</step>\n<step> \1', response_text)
        match_think = list(re.finditer(r'(<think>\n<step>)', response_text))
        if response_text[match_think[0].end():].strip().startswith("</step>\n<step>"):
            response_text = response_text[:match_think[0].end()] + re.sub(r'(</step>\n<step>)', r'', response_text[match_think[0].end():], count=1)
        match_answer = list(re.finditer(r'(<answer>\n<step>)', response_text))
        if response_text[match_answer[0].end():].strip().startswith("</step>\n<step>"):
            response_text = response_text[:match_answer[0].end()] + re.sub(r'(</step>\n<step>)', r'', response_text[match_answer[0].end():], count=1)
        
        response_text = response_text[match_answer[0].start():]

    return response_text

@register("naive")
class NaiveRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, train_batch_size, num_generation, beta, compute_score=None, reward_fn_key="data_source") -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source
        self.train_batch_size = train_batch_size
        self.num_generation = num_generation
        self.beta = beta
    
    def check_correctness_and_length(self, data):
        """in GRPO, the input data here should have len=train_batch_size * num_generations(i.e. config.actor_rollout_ref.rollout.n)"""
        train_batch_size = self.train_batch_size
        num_generation = self.num_generation
        
        assert len(data) == train_batch_size * num_generation, f"Implementation Error: the input data should have len={train_batch_size * num_generation}, where train_batch_size={train_batch_size}, num_generation={num_generation}, while len(data)={len(data)}"
        
        # Extract unique problem IDs to group responses
        problem_ids = [item.non_tensor_batch['index'] for item in data] # ids of problems of this {batch*rollout_n}
        unique_problem_ids = []
        for pid in problem_ids:
            if pid not in unique_problem_ids:
                unique_problem_ids.append(pid) # ids of problem of this batch (unique)
        
        # Should have train_batch_size unique problems
        assert len(unique_problem_ids) == train_batch_size, f"Expected {train_batch_size} unique problems, got {len(unique_problem_ids)}"
        
        # Create a mapping from data index to results
        correctness_map = {}
        length_map = {}
        optimal_length_map = {}
        
        # Process each unique problem
        for problem_id in unique_problem_ids:
            # Find all instances of this problem in the batch
            indices = [i for i in range(len(data)) if data[i].non_tensor_batch['index'] == problem_id]
            completions_given_prompt = [data[i] for i in indices]
            
            # Should have num_generation repetitions of each problem
            assert len(completions_given_prompt) == num_generation, f"Expected {num_generation} repetitions for problem {problem_id}, got {len(completions_given_prompt)}"
            
            prompts = []  # they should be the same
            response_lengths_given_prompt = [] # the response length of each y_j  ( l(y_j) )
            correctnesses_given_prompt = [] # the correctnesses of each y_j
            
            for idx, completion in zip(indices, completions_given_prompt):
                prompt_ids = completion.batch['prompts'] # the token ids of each same prompt for this problem
                
                prompt_length = prompt_ids.shape[-1]
                
                valid_prompt_length = completion.batch['attention_mask'][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:] # valid token ids of each same prompt for this problem
                
                response_ids = completion.batch['responses']
                valid_response_length = completion.batch['attention_mask'][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length] # valid token ids of each same response for this problem
                
                # decode
                prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                
                ground_truth = completion.non_tensor_batch['reward_model']['ground_truth'] # get the ground truth of this problem
                
                # check correctness
                correct_or_not = verify_correctness(
                    solution_str=response_str,
                    ground_truth=ground_truth,
                )
                correctnesses_given_prompt.append(correct_or_not)
                response_lengths_given_prompt.append(valid_response_length)
                
                # Store in the map
                correctness_map[idx] = correct_or_not
                length_map[idx] = valid_response_length
                
                prompts.append(prompt_str) # the prompts str of each y_j, this should be same
            
            # the shortest correct response length is selected as optimal length
            correct_lengths = [l for c, l in zip(correctnesses_given_prompt, response_lengths_given_prompt) if c]


            # --------------------------------------------------
            # TEST: WHAT IF JUST CHOOSE THE MIN LENGTH, REGRARDLESS OF CORRECTNESS?
            optimal_length = min(correct_lengths) if correct_lengths else sum(response_lengths_given_prompt)/len(response_lengths_given_prompt)
            # optimal_length = min(response_lengths_given_prompt) else avg
            # --------------------------------------------------



            # Just print a compact summary line for each problem
            correct_count = sum(correctnesses_given_prompt)
            print(f"Lengths={response_lengths_given_prompt}, Correct={correct_lengths}", flush=True)
            
            # Store optimal length for all instances of this problem
            for idx in indices:
                optimal_length_map[idx] = optimal_length # same problem_id, same opt_length
            
            assert len(set(prompts)) == 1, f"Implementation Error: in this given dataproto prompts should be the same, while prompts={prompts}"
        
        
        # Convert maps back to lists in the original order
        correctness_set = [correctness_map[i] for i in range(len(data))]
        length_set = [length_map[i] for i in range(len(data))]
        optimal_length_set = [optimal_length_map[i] for i in range(len(data))]
        
        assert len(correctness_set) == len(length_set) == len(optimal_length_set) == len(data), f"Implementation Error: the length of correctness_set={len(correctness_set)}, length_set={len(length_set)}, optimal_length_set={len(optimal_length_set)}, data={len(data)}, they should all be the same length"
        
        return correctness_set, length_set, optimal_length_set
    
    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        # Check correctness and length for each sample in batch
        correctness_set, length_set, optimal_length_set = self.check_correctness_and_length(data)

        prompt_response_str_lst = []

        for i in range(len(data)):
            data_item = data[i] # DataProtoItem
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            response_str = add_tags(response_str, ["\n\n"])
            prompt_response_str_lst.append((prompt_str, response_str))

        prm_step_tag="\n\n"
        prm_format_str = "{question}\n\n\n\n{answer}"
        rm_config = RemoteRewardModelConfig(
                    step_tag=prm_step_tag,
                    format_str=prm_format_str,
                    model_name="/mnt/luoyingfeng/model_card/Qwen2.5-Math-PRM-7B",
                    # controller_addr=config.controller_addr,
                )
        rm_call = RMRemoteCaller(rm_config)
        rm_call_fn = functools.partial(rm_call, lm_step_tag=prm_step_tag)
        step_rewards_lst = rm_call_fn(prompt_response_str_lst)

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            correct_or_not = correctness_set[i] # correctness: I(y_j=y_i*)
            completion_length = length_set[i] # l(y_j)
            optimal_length = optimal_length_set[i] # l^SOL(G(x_i))

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()

            prompt_str = prompt_response_str_lst[i][0]
            response_str = prompt_response_str_lst[i][1]

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["num_turns"] = num_turns

            #score = self.compute_score(
            #    data_source=data_source,
            #    solution_str=response_str,
            #    ground_truth=ground_truth,
            #    extra_info=extra_info,
            #)
            
            step_correctness_lst = list(map(lambda x: 1 if x > 0.8 else 0, step_rewards_lst[i]))
            step_accuracy = sum(step_correctness_lst) / len(step_correctness_lst)
            score = float(correct_or_not) - self.beta * completion_length * max(step_accuracy, 1 / self.num_generation)

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)
            
            # save jsonl log
            log_dict = {
                "prompt": prompt_str,
                "response": response_str,
                "ground_truth": ground_truth,
                "score": score.item(),
                "outcome_score": float(correct_or_not),
                "step_accuracy": step_accuracy,
                "process_rewards": step_correctness_lst
            }

            file_path = "Process_Verification/output_log.jsonl"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_dict, ensure_ascii=False) + '\n')


        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor


# Below are the helper functions to verify the mathematical equivalence of two strings
def verify_correctness(solution_str, ground_truth) -> float:
    retval = False
    try:
        string_in_last_boxed = last_boxed_only_string(solution_str)
        if string_in_last_boxed is not None:
            answer = remove_boxed(string_in_last_boxed)
            if is_equiv(answer, ground_truth):
                retval = True
    except Exception as e:
        print(e)

    return retval