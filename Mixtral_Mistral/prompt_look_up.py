"""
本项目为Mixtral的预测脚本
"""

import time
import copy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteriaList, MaxLengthCriteria
from transformers.utils import ModelOutput
from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    MaxLengthCriteria,
    StoppingCriteriaList,
)

from transformers.generation.utils import _crop_past_key_values
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, TaskType
from transformers import (
    BitsAndBytesConfig,
)


device = torch.device(device='cuda:6')

model_name = "/mnt/data3/models/Mixtral-8x7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # tokenizer.unk_token
tokenizer.pad_token_id = tokenizer.eos_token_id  # tokenizer.unk_token_id
tokenizer.padding_side = 'left' #Necessary for FlashAttention compatibility

compute_dtype = getattr(torch, "float16")  # getattr(torch, "bfloat16")
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name, quantization_config=bnb_config, device_map={"": 6}, trust_remote_code=True)
model = prepare_model_for_kbit_training(model=model)
model.config.pad_token_id = tokenizer.pad_token_id  # Configure the pad token in the model
model.config.use_cache = False

print(model)  # 查看模型结构

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False, 
    r=8, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=32, 
    lora_dropout=0.1
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
model.to(device)


@dataclass
class GreedySearchDecoderOnlyOutput(ModelOutput):
    """
    Base class for outputs of decoder-only generation models using greedy search.


    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, generated_length, hidden_size)`.
    """

    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


@torch.no_grad()
# we modify speculative decoding (aka assisted generation in hf transformers) by swapping out the “draft model” with this function.
def find_candidate_pred_tokens(input_ids, max_ngram_size=3, num_pred_tokens=10):
    """"
    Input to this function is the same as to the draft model - all the tokens till the current generation step (input_ids). 
    It then tries to match last few tokens to somewhere earlier in the prompt. 
    If found, it returns the next-k token continuation as candidate_input_ids or candidate sequence. 
        - The 2 parameters are max_ngram_size, which is the maximum ngram to use when looking for matches in the prompt. 
        - num_pred_tokens is the candidate sequence length to return after match is found.

    We modify speculative decoding where we replace the draft model with simple string matching in the prompt to generate candidate token sequences. This results in significant speedups (2x-4x) in input-grounded tasks, with no effect on output quality. This method can be used with any decoder model without model changes or external datastore, and with both greedy and sampling techniques.
    """
    input_length = input_ids.size(1)
    for ngram_size in range(max_ngram_size, 0, -1):
        ngram = input_ids[0, -ngram_size:].tolist()  # Extract the last n tokens as our search ngram
        windows = input_ids.unfold(dimension=1, size=ngram_size, step=1)  # Create sliding windows of size ngram_size        
        ngram_tensor = torch.tensor(ngram, device=input_ids.device).unsqueeze(0)  # Convert ngram to a tensor for comparison
        matches = (windows == ngram_tensor).all(dim=2)  # Find where the windows match the ngram
        match_indices = matches.nonzero(as_tuple=True)[1]   # Get the indices of matches

        # Iterate through match indices to find a valid continuation
        for idx in match_indices:
            start_idx = idx + ngram_size
            end_idx = start_idx + num_pred_tokens
            if end_idx <= input_length and start_idx < input_length - ngram_size:  # Ensure we don't go beyond the length of input_ids and avoid self-match
                return input_ids[0, start_idx:end_idx]
    return torch.tensor([], dtype=torch.long, device=input_ids.device)  # If no match is found, return an empty tensor

COLORS = ["\x1b[31m", "\x1b[32m", "\x1b[34m", "\x1b[35m"]  # Red, Green, Blue, Magenta
UNDERLINE = "\x1b[4m"
RESET = "\x1b[0m"

@torch.no_grad()
def greedy_search_pld(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        draft_matching_window_size = 3,
        draft_num_candidate_tokens = 10,
        print_output=True,
        **model_kwargs,
    ):
        global tokenizer

        # init values
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None

        # # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        max_len = stopping_criteria[0].max_length
        i = 0
        current_color_index = 0

        while True:
            i += 1
            cur_len = input_ids.shape[-1]
            candidate_pred_tokens = find_candidate_pred_tokens(input_ids, draft_matching_window_size, draft_num_candidate_tokens)
            if len(candidate_pred_tokens) == 0:
                candidate_pred_tokens = torch.tensor([100], device=input_ids.device).unsqueeze(0)
            else:
                candidate_pred_tokens = candidate_pred_tokens.unsqueeze(0)
            
            candidate_input_ids = torch.cat((input_ids, candidate_pred_tokens), dim=1)
            candidate_length = candidate_input_ids.shape[1] - input_ids.shape[1]

            candidate_kwargs = copy.copy(model_kwargs)

            # extension = torch.ones(1, candidate_input_ids.shape[1] - 264).to(device)
            # candidate_kwargs['attention_mask'] = torch.cat((candidate_kwargs['attention_mask'], extension), dim=1)

            candidate_kwargs['attention_mask'] = torch.tensor([[1 for _ in range(candidate_input_ids.shape[1])]]).to(device)
            # candidate_kwargs = self._extend_attention_mask(candidate_kwargs, candidate_input_ids.shape[1])  # ori
            # candidate_kwargs = self._extend_token_type_ids(candidate_kwargs, candidate_input_ids.shape[1])  # ori
            model_inputs = self.prepare_inputs_for_generation(candidate_input_ids, **candidate_kwargs)
            
            # prepare model inputs
            # model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            new_logits = outputs.logits[:, -candidate_length - 1 :]  # excludes the input prompt if present
            selected_tokens = new_logits.argmax(dim=-1)
            candidate_new_tokens = candidate_input_ids[:, -candidate_length:]
            n_matches = ((~(candidate_new_tokens == selected_tokens[:, :-1])).cumsum(dim=-1) < 1).sum()

            
            # if last_assistant_token_is_eos and n_matches == candidate_length: # todo: do this earlier somehow
            #     n_matches -= 1
            
            n_matches = min(n_matches, max_len - cur_len - 1)

            # print(n_matches)
            # i+= n_matches.item()

            if print_output:
                current_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            
            valid_tokens = selected_tokens[:, : n_matches + 1]
            input_ids = torch.cat((input_ids, valid_tokens), dim=-1)
            new_cur_len = input_ids.shape[-1]

            if print_output:
                updated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                # Find and print the newly added text
                if updated_text != current_text:
                    new_text = updated_text[len(current_text):]
                    if len(valid_tokens[0]) > 1:
                        color = COLORS[current_color_index]
                        print(f"{color}{new_text}{RESET}", end='')
                        # Update color for next generation
                        current_color_index = (current_color_index + 1) % len(COLORS)
                    else:
                        print(f"{new_text}", end='')

            new_cache_size = new_cur_len - 1
            outputs.past_key_values = _crop_past_key_values(self, outputs.past_key_values, new_cache_size)

        
            model_kwargs["past_key_values"] = outputs.past_key_values

            # stop if we exceed the maximum length

            if (valid_tokens == eos_token_id_tensor.item()).any():
                break
            
            if stopping_criteria(input_ids, scores):
                break


        if return_dict_in_generate:
            return GreedySearchDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                # attentions=decoder_attentions,
                # hidden_states=decoder_hidden_states,
            )
        else:
            return input_ids

# breakpoint()
model.greedy_search_pld = greedy_search_pld.__get__(model, type(model))
code_text = """import numpy as np
import matplotlib.pyplot as plt

# Calculate the average
average_throughput = np.mean(tokens_per_sec_arr)
print(f"Average Throughput: {average_throughput} tokens/sec")

# Plotting the histogram
plt.hist(tokens_per_sec_arr, bins=20, color='blue', edgecolor='black', alpha=0.7)
plt.title('Histogram of Throughput Values')
plt.xlabel('Tokens per Second')
plt.ylabel('Frequency')
plt.axvline(average_throughput, color='red', linestyle='dashed', linewidth=1)
plt.text(average_throughput*0.9, max(plt.ylim())*0.9, f'Average: {average_throughput:.2f}', color = 'red')
plt.show()
"""
question = "Can you please change x axis to start from 0"
prompt = "[INST] Code:```python\n{code_text}``` \n\n Question: {question} \n\n Modified code:[/INST]".format(code_text=code_text, question=question)

inputs = tokenizer(prompt, return_tensors="pt")

# Move all tensor values in the inputs to GPU
for key in inputs:
    inputs[key] = inputs[key].to(device)



# Define the variable for max_new_tokens
max_new_tokens = 500
use_new_generate = True

# Start timing
start_time = time.time()

# Generate the output

if not use_new_generate:
    out = model.generate(inputs=inputs.input_ids, max_new_tokens=max_new_tokens, use_cache=True, pad_token_id=0,
                         do_sample=False,
                         return_dict_in_generate=True)
else:
    out = model.greedy_search_pld(inputs.input_ids, 
                              attention_mask = inputs.attention_mask,
                              stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=len(inputs.input_ids[0]) + max_new_tokens)]),
                              draft_matching_window_size = 3,
                              draft_num_candidate_tokens = 10,
                              use_cache=True, 
                              pad_token_id=0,
                              eos_token_id=2,
                              return_dict_in_generate=True)

end_time = time.time()

out_text = tokenizer.batch_decode(out.sequences, skip_special_tokens=True)[0]
num_tokens_generated = len(out.sequences[0]) - len(inputs['input_ids'][0])

total_time = end_time - start_time
tokens_per_sec = num_tokens_generated / total_time

print(f"\n\nTotal time: {total_time} seconds")
print(f"Tokens per second: {tokens_per_sec} tokens/sec")
print(f"Total tokens generated: {num_tokens_generated}")


