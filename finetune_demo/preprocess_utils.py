import json
import ast
import astunparse
from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset
from copy import deepcopy
from typing import Dict, List

# text constants
FUNCTION_CALL_NAME     = 'tool_call'
FUNCTION_CALL_PREFIX   = '```python\n'
FUNCTION_CALL_POSTFIX  = '\n```'
TOOL_DEFINITION_PREFIX = 'Answer the following questions as best as you can. You have access to the following tools:\n'
CONVERSATOIN_KEY       = 'conversations'
TOOL_DESC_KEY          = 'tools'

def format_function_call(function_name: str, parameters: Dict[str, str]):
    function_name = ast.Name(id=function_name)
    keywords = [
        ast.keyword(arg=arg_name, value=ast.Constant(arg_value)) 
        for arg_name, arg_value in parameters.items()
    ]
    func_call = ast.Call(func=function_name, args=[], keywords=keywords)
    return astunparse.unparse(func_call).strip()

def format_conversation(item, tokenizer, conversation_key: str, tool_key: str):
    conversations = deepcopy(item[conversation_key])

    # Note: `loss_mask` here means whether *the prediction* of the token should take loss
    tokens, loss_masks = [tokenizer.get_command("[gMASK]"), tokenizer.get_command("sop")], [0, 0]

    def _update(_tokens: List[int], value: int = 1):
        value = int(value)
        tokens.extend(_tokens)
        loss_masks.extend([value] * len(_tokens))

    # insert system prompt for tools
    if tool_key in item:
        conversations.insert(0, 
            {
                "role": "system", 
                "content": TOOL_DEFINITION_PREFIX + json.dumps(item[tool_key], indent=4, ensure_ascii=False)
            }
        )
    
    for idx, conv in enumerate(conversations):
        loss = conv.get("loss", True)
        if conv['role'] in {'system', 'user'}:
            loss = False
        if conv['role'] == 'tool':
            # function call python code
            value = FUNCTION_CALL_PREFIX + format_function_call(FUNCTION_CALL_NAME, conv["parameters"]) + FUNCTION_CALL_POSTFIX
            text = tokenizer.build_single_message("assistant", conv["name"], value)
            _update(text, loss)

            # function call result
            value = conv.get('observation', None)
            if not isinstance(value, str):
                value = json.dumps(value, ensure_ascii=False)
            text = tokenizer.build_single_message("observation", "", value)
            _update(text, False)
        else:
            text = tokenizer.build_single_message(conv['role'], "", conv["content"])
            _update(text, loss)

    _update([tokenizer.eos_token_id], False)

    assert len(tokens) == len(loss_masks), f"length mismatch: {len(tokens)} vs {len(loss_masks)}"
    return tokens, loss_masks

def sanity_check(tokens: List[int], target: List[int], tokenizer: PreTrainedTokenizer):
    print("Sanity Check >>>>>>>>>>>>>")
    for t, m in zip(tokens, target):
        decoded =  tokenizer.tokenizer.index_special_tokens[t] \
            if t in tokenizer.tokenizer.index_special_tokens \
            else tokenizer.decode([t])
        print("%20s: %6d -> %6d" % (repr(decoded), t, m))
    print("<<<<<<<<<<<<< Sanity Check")

    assert len(tokens) == len(target), f"length mismatch: {len(tokens)} vs {len(target)}"

class MultiTurnDataset(Dataset):
    def __init__(self, data: List[dict], tokenizer: PreTrainedTokenizer, max_seq_length: int):
        super(MultiTurnDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i) -> dict:
        data_item = self.data[i]
        tokens, loss_masks = format_conversation(data_item, self.tokenizer, CONVERSATOIN_KEY, TOOL_DESC_KEY)

        # labels are used inside the model
        target_based_loss_mask = [False] + loss_masks[:-1]
        labels = [(t if m else -100) for t, m in zip(tokens, target_based_loss_mask)]

        tokens = tokens[:self.max_seq_length]
        labels = labels[:self.max_seq_length]
        tokens += [self.tokenizer.pad_token_id] * (self.max_seq_length - len(tokens))
        labels += [-100] * (self.max_seq_length - len(labels))

        assert len(tokens) == len(labels), f"length mismatch: {len(tokens)} vs {len(labels)}"

        return {
            "input_ids": tokens,
            "labels": labels
        }
    
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

        a_ids = self.tokenizer.encode(text=data_item['prompt'], add_special_tokens=True, truncation=True,
                                         max_length=self.max_source_length)
        b_ids = self.tokenizer.encode(text=data_item['response'], add_special_tokens=False, truncation=True,
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
