from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset
from typing import Dict, List


def sanity_check(tokens: List[int], target: List[int], tokenizer: PreTrainedTokenizer):
    print("Sanity Check >>>>>>>>>>>>>")
    for t, m in zip(tokens, target):
        decoded =  tokenizer.tokenizer.index_special_tokens[t] \
            if t in tokenizer.tokenizer.index_special_tokens \
            else tokenizer.decode([t])
        print("%20s: %6d -> %6d" % (repr(decoded), t, m))
    print("<<<<<<<<<<<<< Sanity Check")

    assert len(tokens) == len(target), f"length mismatch: {len(tokens)} vs {len(target)}"


class InputOutputDataset(Dataset):
    def __init__(self, data: List[dict], tokenizer: PreTrainedTokenizer, max_source_length: int, max_target_length: int):
        super(InputOutputDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.max_seq_length = max_source_length + max_target_length + 1
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i) -> dict:
        data_item = self.data[i]

        a_ids = self.tokenizer.encode(text=data_item['context'], add_special_tokens=True, truncation=True,
                                         max_length=self.max_source_length)
        b_ids = self.tokenizer.encode(text=data_item['target'], add_special_tokens=False, truncation=True,
                                    max_length=self.max_target_length)

        context_length = len(a_ids)
        input_ids = a_ids + b_ids + [self.tokenizer.eos_token_id]
        labels = [self.tokenizer.pad_token_id] * context_length + b_ids + [self.tokenizer.eos_token_id]
        
        pad_len = self.max_seq_length - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
        labels = labels + [self.tokenizer.pad_token_id] * pad_len
        labels = [(l if l != self.tokenizer.pad_token_id else -100) for l in labels]

        assert len(input_ids) == len(labels), f"length mismatch: {len(input_ids)} vs {len(labels)}"

        return {
            "input_ids": input_ids,
            "labels": labels
        }
