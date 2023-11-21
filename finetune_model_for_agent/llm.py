from typing import List, Optional, Mapping, Any
from functools import partial
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from transformers import AutoModel, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import os
import torch

class ChatGLM3(LLM):

    model_path: str
    max_length: int = 8192
    temperature: float = 0.1
    top_p: float = 0.7
    history: List = []
    streaming: bool = True
    model: object = None
    tokenizer: object = None

    @property
    def _llm_type(self) -> str:
        return "chatglm3-6B"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        add_history: bool = False
    ) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Must call `load_model()` to load model and tokenizer!")

        if self.streaming:
            text_callback = partial(StreamingStdOutCallbackHandler().on_llm_new_token, verbose=True)
            resp = self.generate_resp(prompt, text_callback, add_history=add_history)
        else:
            resp = self.generate_resp(self, prompt, add_history=add_history)
        return resp

    def generate_resp(self, prompt, text_callback=None, add_history=True):
        resp = ""
        index = 0
        if text_callback:
            for i, (resp, _) in enumerate(self.model.stream_chat(
                    self.tokenizer,
                    prompt,
                    self.history,
                    max_length=self.max_length,
                    top_p=self.top_p,
                    temperature=self.temperature
            )):
              if add_history:
                    if i == 0:
                        self.history += [[prompt, resp]]
                    else:
                        self.history[-1] = [prompt, resp]
                text_callback(resp[index:])
                index = len(resp)
        else:
            resp, _ = self.model.chat(
                self.tokenizer,
                prompt,
                self.history,
                max_length=self.max_length,
                top_p=self.top_p,
                temperature=self.temperature
            )
            if add_history:
                self.history += [[prompt, resp]]
        return resp

    def load_model(self):
        if self.model is not None or self.tokenizer is not None:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True).half().cuda().eval()-

    def load_model_from_checkpoint(self, checkpoint=None):
        if self.model is not None or self.tokenizer is not None:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True).half()
        peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=8,
                target_modules=['query_key_value'],
                lora_alpha=32,
                lora_dropout=0.1,
        )
        self.model = get_peft_model(self.model, peft_config).to("cuda")
        if checkpoint=="text_classification":
            model_dir = "./output/checkpoint-3000/"
            peft_path = "{}/chatglm-lora.pt".format(model_dir)
            if os.path.exists(peft_path):
                 self.model.load_state_dict(torch.load(peft_path), strict=False)
