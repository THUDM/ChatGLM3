import ast
import json
from langchain.llms.base import LLM
from transformers import AutoTokenizer, AutoModel, AutoConfig
from typing import List, Optional


class ChatGLM3(LLM):
    max_token: int = 8192
    do_sample: bool = True
    temperature: float = 0.8
    top_p = 0.8
    tokenizer: object = None
    model: object = None
    history: List = []
    has_search: bool = False

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "ChatGLM3"

    def load_model(self, model_name_or_path=None):
        model_config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_name_or_path, config=model_config, trust_remote_code=True, device_map="auto").eval()

    def _tool_history(self, prompt: str):
        ans = []

        tool_prompts = prompt.split(
            "You have access to the following tools:\n\n")[1].split("\n\nUse a json blob")[0].split("\n")
        tools_json = []

        for tool_desc in tool_prompts:
            name = tool_desc.split(":")[0]
            description = tool_desc.split(", args:")[0].split(":")[0].strip()
            parameters_str = tool_desc.split("args:")[1].strip()
            parameters_dict = ast.literal_eval(parameters_str)
            params_cleaned = {}
            for param, details in parameters_dict.items():
                params_cleaned[param] = {'description': details['description'], 'type': details['type']}

            tools_json.append({
                "name": name,
                "description": description,
                "parameters": params_cleaned
            })

        ans.append({
            "role": "system",
            "content": "Answer the following questions as best as you can. You have access to the following tools:",
            "tools": tools_json
        })

        dialog_parts = prompt.split("Human: ")
        for part in dialog_parts[1:]:
            if "\nAI: " in part:
                user_input, ai_response = part.split("\nAI: ")
                ai_response = ai_response.split("\n")[0]
            else:
                user_input = part
                ai_response = None

            ans.append({"role": "user", "content": user_input.strip()})
            if ai_response:
                ans.append({"role": "assistant", "content": ai_response.strip()})

        query = dialog_parts[-1].split("\n")[0]
        return ans, query

    def _extract_observation(self, prompt: str):
        return_json = prompt.split("Observation: ")[-1].split("\nThought:")[0]
        self.history.append({
            "role": "observation",
            "content": return_json
        })
        return

    def _extract_tool(self):
        if len(self.history[-1]["metadata"]) > 0:
            metadata = self.history[-1]["metadata"]
            content = self.history[-1]["content"]

            lines = content.split('\n')
            for line in lines:
                if 'tool_call(' in line and ')' in line and self.has_search is False:
                    # 获取括号内的字符串
                    params_str = line.split('tool_call(')[-1].split(')')[0]

                    # 解析参数对
                    params_pairs = [param.split("=") for param in params_str.split(",") if "=" in param]
                    params = {pair[0].strip(): pair[1].strip().strip("'\"") for pair in params_pairs}
                    action_json = {
                        "action": metadata,
                        "action_input": params
                    }
                    self.has_search = True
                    print("*****Action*****")
                    print(action_json)
                    print("*****Answer*****")
                    return f"""
Action: 
```
{json.dumps(action_json, ensure_ascii=False)}
```"""
        final_answer_json = {
            "action": "Final Answer",
            "action_input": self.history[-1]["content"]
        }
        self.has_search = False
        return f"""
Action: 
```
{json.dumps(final_answer_json, ensure_ascii=False)}
```"""

    def _call(self, prompt: str, history: List = [], stop: Optional[List[str]] = ["<|user|>"]):
        if not self.has_search:
            self.history, query = self._tool_history(prompt)
        else:
            self._extract_observation(prompt)
            query = ""
        _, self.history = self.model.chat(
            self.tokenizer,
            query,
            history=self.history,
            do_sample=self.do_sample,
            max_length=self.max_token,
            temperature=self.temperature,
        )
        response = self._extract_tool()
        history.append((prompt, response))
        return response
