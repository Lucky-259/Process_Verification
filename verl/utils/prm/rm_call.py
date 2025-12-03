import re
import torch
import functools
import numpy as np
import requests
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass



def _value_inference(model,
    tokenizer,
    input_str: Union[List[str], str],
):
    import pprint
    def make_step_rewards(logits, token_masks):
        probabilities = F.softmax(logits, dim=-1)
        probabilities = probabilities * token_masks.unsqueeze(-1) # bs, seq_len, num_labels
        
        all_scores_res = []
        for i in range(probabilities.size(0)):
            sample = probabilities[i] # seq_len, num_labels
            positive_probs = sample[sample != 0].view(-1, 2)[:, 1] # valid_tokens, num_labels
            non_zero_elements_list = positive_probs.cpu().tolist()
            all_scores_res.append(non_zero_elements_list)
        return all_scores_res
    
    def post_http_request(prompt: dict, api_url: str) -> requests.Response:
        headers = {"User-Agent": "Test Client"}
        response = requests.post(api_url, headers=headers, json=prompt)
        return response

    api_url = f"http://localhost:8034/pooling" #"http://localhost:8012/v1"
    step_reward = []
    for input in input_str:
        question_answer_pair = input.split("\n\n\n\n")
        data = {
        "system": "Please reason step by step, and put your final answer within \\boxed{}.",
        "query": question_answer_pair[0],
        "response": [
            answer.strip() for answer in question_answer_pair[1].split("\n\n") if answer != ""
            ]
        }
        
        messages = [
            {"role": "system", "content": data['system']},
            {"role": "user", "content": data['query']},
            {"role": "assistant", "content": "<extra_0>".join(data['response']) + "<extra_0>"},
        ]
        conversation_str = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )

        input_ids = tokenizer.encode(
            conversation_str, 
            return_tensors="pt", 
        )
        #print(input_ids)
        prompt = {
            "model": model,
            "messages": messages
        }
        pooling_response = post_http_request(prompt=prompt, api_url=api_url)

        outputjson = pooling_response.json()
        #print(outputjson)
        try:
            outputdata = outputjson['data'][0]['data']
        except Exception as e:
            print(f"出现报错：{outputjson}")
            raise e
        step=[]
        for i in outputdata:
            step.append(i[1])
        step_reward.append(step)
    return step_reward

@dataclass
class RewardModelBaseConfig:
    step_tag: str
    # a format string that takes in question and answer
    #  need to have {question} and {answer} in the string
    format_str: str


class RewardModelCallingFunction:
    def __init__(self, config: RewardModelBaseConfig):
        self.config = config
        self.step_tag = config.step_tag
        self.format_str = config.format_str

    def __call__(
        self,
        question_answer_pairs: Union[Tuple[str, str], List[Tuple[str, str]]],
        lm_step_tag: str,
    ) -> Union[List[int], List[List[int]]]:
        raise NotImplementedError

    def replace_step_tag(self, answer: str, lm_step_tag: str):
        splits = answer.split(lm_step_tag)
        splits = [s.strip() for s in splits]
        # add a whitespace to avoid tokenization issue
        response = f" {self.step_tag}".join([s for s in splits if s != ""])
        response += f" {self.step_tag}"
        return response


@dataclass
class RemoteRewardModelConfig(RewardModelBaseConfig):
    model_name: str
    # controller_addr: str


class RMRemoteCaller(RewardModelCallingFunction):
    def __init__(self, config: RemoteRewardModelConfig):
        self.model_name = config.model_name
        # self.controller_addr = config.controller_addr
        # print("RM loading")
        #self.model = AutoModel.from_pretrained(
        #    self.model_name, 
        #    device_map="auto", 
        #    torch_dtype=torch.bfloat16,
        #    trust_remote_code=True,
        #).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        super().__init__(config)

    def __call__(
        self,
        question_answer_pairs: Union[Tuple[str, str], List[Tuple[str, str]]],
        lm_step_tag: str,
    ) -> Union[List[int], List[List[int]]]:
                
        if isinstance(question_answer_pairs[0], str):
            response = self.replace_step_tag(question_answer_pairs[1], lm_step_tag)
            input_str = self.format_str.format(
                question=question_answer_pairs[0], answer=response
            )
        else:
            input_str = [
                self.format_str.format(
                    question=s[0],
                    answer=self.replace_step_tag(s[1], lm_step_tag),
                )
                for s in question_answer_pairs
            ]

        return _value_inference(
            self.model_name,
            self.tokenizer,
            input_str=input_str,
        )



## Using Example
#prm_step_tag="\n\n"
#prm_format_str = "{question}\n\n\n\n{answer}"
#rm_config = RemoteRewardModelConfig(
#            step_tag=prm_step_tag,
#            format_str=prm_format_str,
#            model_name="/mnt/luoyingfeng/model_card/Qwen2.5-Math-PRM-7B",
#            # controller_addr=config.controller_addr,
#        )
#rm_call = RMRemoteCaller(rm_config)
#rm_call_fn = functools.partial(rm_call, lm_step_tag=prm_step_tag)
#
#question =  """Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakesmuffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 perfresh duck egg. How much in dollars does she make every day at the farmers' market?"""

#answer = """Step 1: Janet's ducks lay 16 eggs per day. \n\nStep 2: She eats three for breakfast every morning, soshe has 16 - 3 = 13 eggs left. \n\nStep 3: She bakes muffins for her friends every day with four eggs, so she has13 - 4 = 9 eggs left. \n\nStep 4: She sells the remainder at the farmers' market daily for $2 per fresh duck egg,so she makes 9 * $2 = $17 every day at the farmers' market. The answer is: 17 \n\n""" 

#print(rm_call_fn([(question,answer), (question,answer)]))
