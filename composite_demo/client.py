from __future__ import annotations

from collections.abc import Iterable
import os
from typing import Any, Protocol

from huggingface_hub.inference._text_generation import TextGenerationStreamResponse, Token
import streamlit as st
import torch
from transformers import AutoModel, AutoTokenizer

from conversation import Conversation

TOOL_PROMPT = 'Answer the following questions as best as you can. You have access to the following tools:'

MODEL_PATH = os.environ.get('MODEL_PATH', 'THUDM/chatglm3-6b')

@st.cache_resource
def get_client() -> Client:
    client = HFClient(MODEL_PATH)
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
                    past_key_values=None,max_length: int = 8192, do_sample=True, top_p=0.8, temperature=0.8,
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
    gen_kwargs = {"max_length": max_length, "do_sample": do_sample, "top_p": top_p,
                    "temperature": temperature, "logits_processor": logits_processor, **kwargs}
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
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(
            'cuda' if torch.cuda.is_available() else
            'mps' if torch.backends.mps.is_available() else
            'cpu'
        )
        self.model = self.model.eval()

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
