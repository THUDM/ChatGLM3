from __future__ import annotations

from collections.abc import Iterable
import os
from typing import Any, Protocol

from huggingface_hub.inference._text_generation import TextGenerationStreamResponse, Token
import streamlit as st
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig

from conversation import Conversation

TOOL_PROMPT = 'Answer the following questions as best as you can. You have access to the following tools:'

MODEL_PATH = os.environ.get('MODEL_PATH', 'THUDM/chatglm3-6b')
PT_PATH = os.environ.get('PT_PATH', None)
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# for Mac Computer like M1
# You Need Use Pytorch compiled with Metal
# DEVICE = 'mps'

# for AMD gpu likes MI100 (Not Official Steady Support yet)
# You Need Use Pytorch compiled with ROCm
# DEVICE = 'cuda'

# for Intel gpu likes A770 (Not Official Steady Support yet)
# You Need Use Pytorch compiled with oneDNN and install intel-extension-for-pytorch
# import intel_extension_for_pytorch as ipex
# DEVICE = 'xpu'

# for Moore Threads gpu like MTT S80 (Not Official Steady Support yet)
# You Need Use Pytorch compiled with Musa
# DEVICE = 'musa'


@st.cache_resource
def get_client() -> Client:
    client = HFClient(MODEL_PATH, TOKENIZER_PATH, PT_PATH, DEVICE)
    return client


class Client(Protocol):
    def generate_stream(self,
                        system: str | None,
                        tools: list[dict] | None,
                        history: list[Conversation],
                        **parameters: Any
                        ) -> Iterable[TextGenerationStreamResponse]:
        ...


def stream_chat(self, tokenizer, query: str, history: list[tuple[str, str]] = None, role: str = "user",
                past_key_values=None, max_length: int = 8192, do_sample=True, top_p=0.8, temperature=0.8,
                repetition_penalty=1.0, length_penalty=1.0, num_beams=1,
                logits_processor=None, return_past_key_values=False, **kwargs):
    from transformers.generation.logits_process import LogitsProcessor
    from transformers.generation.utils import LogitsProcessorList

    class InvalidScoreLogitsProcessor(LogitsProcessor):
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
            if torch.isnan(scores).any() or torch.isinf(scores).any():
                scores.zero_()
                scores[..., 5] = 5e4
            return scores

    if history is None:
        history = []
    if logits_processor is None:
        logits_processor = LogitsProcessorList()
    logits_processor.append(InvalidScoreLogitsProcessor())
    eos_token_id = [tokenizer.eos_token_id, tokenizer.get_command("<|user|>"),
                    tokenizer.get_command("<|observation|>")]
    gen_kwargs = {"max_length": max_length,
                  "do_sample": do_sample,
                  "top_p": top_p,
                  "temperature": temperature,
                  "logits_processor": logits_processor,
                  "repetition_penalty": repetition_penalty,
                  "length_penalty": length_penalty,
                  "num_beams": num_beams,
                  **kwargs
                  }

    print(gen_kwargs)
    if past_key_values is None:
        inputs = tokenizer.build_chat_input(query, history=history, role=role)
    else:
        inputs = tokenizer.build_chat_input(query, role=role)
    inputs = inputs.to(self.device)
    if past_key_values is not None:
        past_length = past_key_values[0][0].shape[0]
        if self.transformer.pre_seq_len is not None:
            past_length -= self.transformer.pre_seq_len
        inputs.position_ids += past_length
        attention_mask = inputs.attention_mask
        attention_mask = torch.cat((attention_mask.new_ones(1, past_length), attention_mask), dim=1)
        inputs['attention_mask'] = attention_mask
    history.append({"role": role, "content": query})
    print("input_shape>", inputs['input_ids'].shape)

    input_sequence_length = inputs['input_ids'].shape[1]

    if max_length < input_sequence_length <= self.config.seq_length:
        yield "Current input sequence length {} exceeds sequence length set in generation parameters {}. The maximum model sequence length is {}. You may adjust the generation parameter to enable longer chat history.".format(
            input_sequence_length, max_length, self.config.seq_length
        ), history
        return

    if input_sequence_length > self.config.seq_length:
        yield "Current input sequence length {} exceeds maximum model sequence length {}. Unable to generate tokens.".format(
            input_sequence_length, self.config.seq_length
        ), history
        return

    for outputs in self.stream_generate(**inputs, past_key_values=past_key_values,
                                        eos_token_id=eos_token_id, return_past_key_values=return_past_key_values,
                                        **gen_kwargs):
        if return_past_key_values:
            outputs, past_key_values = outputs
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
        response = tokenizer.decode(outputs)
        if response and response[-1] != "�":
            new_history = history
            if return_past_key_values:
                yield response, new_history, past_key_values
            else:
                yield response, new_history


class HFClient(Client):
    def __init__(self, model_path: str, tokenizer_path: str, pt_checkpoint: str | None = None, DEVICE = 'cpu'):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

        if pt_checkpoint is not None:
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, pre_seq_len=128)
            self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, config=config)
            prefix_state_dict = torch.load(os.path.join(pt_checkpoint, "pytorch_model.bin"))
            new_prefix_state_dict = {}
            for k, v in prefix_state_dict.items():
                if k.startswith("transformer.prefix_encoder."):
                    new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
            print("Loaded from pt checkpoints", new_prefix_state_dict.keys())
            self.model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
        else:
            self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

        self.model = self.model.to(DEVICE).eval() if 'cuda' in DEVICE else self.model.float().to(DEVICE).eval()


    def generate_stream(self,
                        system: str | None,
                        tools: list[dict] | None,
                        history: list[Conversation],
                        **parameters: Any
                        ) -> Iterable[TextGenerationStreamResponse]:
        chat_history = [{
            'role': 'system',
            'content': system if not tools else TOOL_PROMPT,
        }]

        if tools:
            chat_history[0]['tools'] = tools

        for conversation in history[:-1]:
            chat_history.append({
                'role': str(conversation.role).removeprefix('<|').removesuffix('|>'),
                'content': conversation.content,
            })

        query = history[-1].content
        role = str(history[-1].role).removeprefix('<|').removesuffix('|>')

        text = ''

        for new_text, _ in stream_chat(self.model,
                                       self.tokenizer,
                                       query,
                                       chat_history,
                                       role,
                                       **parameters,
                                       ):
            word = new_text.removeprefix(text)
            word_stripped = word.strip()
            text = new_text
            yield TextGenerationStreamResponse(
                generated_text=text,
                token=Token(
                    id=0,
                    logprob=0,
                    text=word,
                    special=word_stripped.startswith('<|') and word_stripped.endswith('|>'),
                )
            )
