import os
from typing import Optional, Union
from transformers import AutoModel, AutoTokenizer, LogitsProcessorList

MODEL_PATH = os.environ.get('MODEL_PATH', 'THUDM/chatglm3-6b')
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, device_map="auto").eval()

def batch(
        model,
        tokenizer,
        prompts: Union[str, list[str]],
        max_length: int = 8192,
        num_beams: int = 1,
        do_sample: bool = True,
        top_p: float = 0.8,
        temperature: float = 0.8,
        logits_processor: Optional[LogitsProcessorList] = LogitsProcessorList(),
):
    tokenizer.encode_special_tokens = True
    if isinstance(prompts, str):
        prompts = [prompts]
    batched_inputs = tokenizer(prompts, return_tensors="pt", padding="longest")
    batched_inputs = batched_inputs.to(model.device)

    eos_token_id = [
        tokenizer.eos_token_id,
        tokenizer.get_command("<|user|>"),
        tokenizer.get_command("<|assistant|>"),
    ]
    gen_kwargs = {
        "max_length": max_length,
        "num_beams": num_beams,
        "do_sample": do_sample,
        "top_p": top_p,
        "temperature": temperature,
        "logits_processor": logits_processor,
        "eos_token_id": eos_token_id,
    }
    batched_outputs = model.generate(**batched_inputs, **gen_kwargs)
    batched_response = []
    for input_ids, output_ids in zip(batched_inputs.input_ids, batched_outputs):
        decoded_text = tokenizer.decode(output_ids[len(input_ids):])
        batched_response.append(decoded_text.strip())
    return batched_response


def main(batch_queries):
    gen_kwargs = {
        "max_length": 2048,
        "do_sample": True,
        "top_p": 0.8,
        "temperature": 0.8,
        "num_beams": 1,
    }
    batch_responses = batch(model, tokenizer, batch_queries, **gen_kwargs)
    return batch_responses


if __name__ == "__main__":
    batch_queries = [
        "<|user|>\n讲个故事\n<|assistant|>",
        "<|user|>\n讲个爱情故事\n<|assistant|>",
        "<|user|>\n讲个开心故事\n<|assistant|>",
        "<|user|>\n讲个睡前故事\n<|assistant|>",
        "<|user|>\n讲个励志的故事\n<|assistant|>",
        "<|user|>\n讲个少壮不努力的故事\n<|assistant|>",
        "<|user|>\n讲个青春校园恋爱故事\n<|assistant|>",
        "<|user|>\n讲个工作故事\n<|assistant|>",
        "<|user|>\n讲个旅游的故事\n<|assistant|>",
    ]
    batch_responses = main(batch_queries)
    for response in batch_responses:
        print("=" * 10)
        print(response)
