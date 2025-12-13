import copy
import logging
from typing import Optional, Literal, Tuple, Union, List, Dict
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
import jinja2
from transformers import AutoTokenizer, AutoModel
import accelerate

from lm_eval.api.model import LM
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from lm_eval.models.utils import (
    Collator,
    handle_stop_sequences,
    postprocess_generated_text
)


eval_logger = logging.getLogger(__name__)


# copied from https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Base
def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


# copied from https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Base
def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


# based on example from https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Base
@torch.no_grad()
def generate(
        model: AutoModel,
        input_ids: torch.Tensor,
        mask_token_id: int,
        attention_mask: Optional[torch.Tensor] = None,
        steps: int = 128,
        gen_length: int = 128,
        block_length: int = 128,
        temperature: float = 0.,
        cfg_scale: float = 0.,
        remasking: Literal['random', 'low_confidence'] = 'low_confidence'
) -> torch.Tensor:
    x = torch.full((input_ids.shape[0], input_ids.shape[1] + gen_length), mask_token_id, dtype=torch.long).to(model.device)
    x[:, :input_ids.shape[1]] = input_ids.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((input_ids.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)

    prompt_index = (x != mask_token_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, input_ids.shape[1] + num_block * block_length: input_ids.shape[1] + (num_block + 1) * block_length:] == mask_token_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_token_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_token_id
                x_ = torch.cat([x, un_x], dim=0)
                if attention_mask is not None:
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                logits = model(x_, attention_mask=attention_mask_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, input_ids.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


# based on https://github.com/ML-GSAI/LLaDA/blob/main/eval_llada.py
@register_model("llada")
class LLaDA(LM):
    def __init__(
            self,
            pretrained_model_name_or_path: str,
            batch_size: int = 32,
            steps: int = 1024,
            gen_length: int = 1024,
            block_length: int = 1024,
            cfg_scale: float = 0.,
            remasking: str = 'low_confidence',
            device: str | torch.device = 'cuda',
            **kwargs,
    ):
        super().__init__()

        accelerator = accelerate.Accelerator()
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None

        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({'device_map': {'': f'{self.accelerator.device}'}})

        self.model = AutoModel.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True,
                                               dtype=torch.bfloat16, **model_kwargs)
        self.model.eval()

        self.device = torch.device(device)
        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            self.device = torch.device(f'{self.accelerator.device}')
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else: 
            self.model = self.model.to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
        if self.tokenizer.padding_side != "left":
            self.tokenizer.padding_side = "left"
        assert self.tokenizer.pad_token_id != self.tokenizer.mask_token_id

        self.batch_size = int(batch_size)
        self.steps = steps
        self.gen_length = gen_length
        self.block_length = block_length
        self.cfg_scale = cfg_scale
        self.remasking = remasking

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def tok_batch_encode(self, contexts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        encoding = self.tokenizer(
            contexts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt"
        )
        return encoding["input_ids"], encoding["attention_mask"]

    def loglikelihood(self, requests: List[Instance]) -> List[float]:
        raise NotImplementedError

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        raise NotImplementedError

    # based on lm_eval.models.huggingface.HFLM
    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False) -> List[str]:
        generated_texts = []

        def _collate(req: tuple[str, dict]):
            """Defines the key for the sorted method"""
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tokenizer.encode(req[0], add_special_tokens=False)
            return -len(toks), req[0]

        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_until requests",
        )

        collator = Collator(
            [req.args for req in requests],
            sort_fn=_collate,
            group_by="gen_kwargs",
            group_fn=lambda x: x[1],
        )

        for batch in collator.get_batched(n=self.batch_size):
            prompts, all_gen_kwargs = zip(*batch)

            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            # unpack our keyword arguments.
            if isinstance(gen_kwargs, dict):
                kwargs = copy.deepcopy(gen_kwargs)  # edge case for repeats > 1
                # add EOS token to stop sequences
                until = handle_stop_sequences(kwargs.pop("until", None), eos=self.tokenizer.eos_token)
            else:
                raise TypeError(
                    f"Expected `kwargs` to be of type `dict` but got {type(gen_kwargs)}"
                )
   
            input_ids, attention_mask = self.tok_batch_encode(prompts)
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

            gen_ids = generate(
                self.model,
                input_ids=input_ids,
                mask_token_id=self.tokenizer.mask_token_id,
                attention_mask=attention_mask,
                steps=self.steps,
                gen_length=self.gen_length,
                block_length=self.block_length,
                temperature=0,
                cfg_scale=self.cfg_scale,
                remasking=self.remasking
            )
            
            gen_ids = gen_ids[:, input_ids.shape[1]:]

            gen_texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=False)

            for gen_text in gen_texts: 
                gen_text = postprocess_generated_text(gen_text, stop=until, think_end_token=None)
                gen_text = self.tokenizer.decode(self.tokenizer(gen_text)["input_ids"], skip_special_tokens=True)
                generated_texts.append(gen_text)

                pbar.update(1)

        # reorder this group of results back to original unsorted form
        generated_texts = collator.get_original(generated_texts)
        
        pbar.close()

        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

        return generated_texts

    def chat_template(self, chat_template: Union[bool, str] = False) -> str:
        if chat_template is False or chat_template is None:
            eval_logger.warning(
                "model.chat_template was called with the chat_template set to False or None. "
                "Therefore no chat template will be applied. Make sure this is an intended behavior."
            )
            return None

        return self.tokenizer.chat_template

    def apply_chat_template(self, chat_history: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
        try:
            chat_templated = self.tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt,
            )
        except jinja2.exceptions.TemplateError:
            eval_logger.warning(
                "Failed to apply chat template. removing the system role in chat history."
            )
            chat_history = [msg for msg in chat_history if msg["role"] != "system"]
            chat_templated = self.tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt,
            )

        return chat_templated
