## ChatGLM3 Chat Format
To avoid injection attacks from user input, and to unify the input of Code Interpreter, Tool & Agent and other tasks, ChatGLM3 adopts a brand-new dialogue format.

### Regulations
#### Overall Structure
The format of the ChatGLM3 dialogue consists of several conversations, each of which contains a dialogue header and content. A typical multi-turn dialogue structure is as follows:
```text
<|system|>
You are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's instructions carefully. Respond using markdown.
<|user|>
Hello
<|assistant|>
Hello, I'm ChatGLM3. What can I assist you today?
```

#### Chat Header
The chat header occupies a complete line, formatted as:
```text
<|role|>{metadata}
```
Where `<|role|>` part is represented in a special token,  which can’t be encoded by the tokenizer from the text form to prevent injection attacks. The `metadata` part is represented in plain texts and is optional content.
* `<|system|>`: System information, which can be interspersed in the dialogue in design, **but currently only appears at the beginning**
* `<|user|>`: User
  - Multiple messages from `<|user|>` will not appear continuously
* `<|assistant|>`: AI assistant
  - There must be a message from `<|user|>` before it appears
* `<|observation|>`: External return result
  - Must be after the message from `<|assistant|>`

### Example Scenarios

For better readablity, an extra `\n` is added before each role special token. This extra `\n` should not be added in actual use and tokenizer implementation.

#### Multi-turn Dialogue
* There are only three roles: `<|user|>`, `<|assistant|>`, and `<|system|>`.
```text
<|system|>
You are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's instructions carefully. Respond using markdown.
<|user|>
Hello
<|assistant|>
Hello, I'm ChatGLM3. What can I assist you today?
```

#### Tool Calling
````
<|system|>
Answer the following questions as best as you can. You have access to the following tools:
[
    {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {"type": "string"},
            },
            "required": ["location"],
        },
    }
]
<|user|>
What's the weather in Beijing today?
<|assistant|>
Okay, let's look up the weather in Bejing today.
<|assistant|>get_current_weather
```python
tool_call(location="beijing", unit="celsius")
```
<|observation|>
{"temperature": 22}
<|assistant|>
According to the query results, the temperature in Beijing today is 22 degrees Celsius.
````

#### Code Execution
* Four available roles: `<|user|>`, `<|assistant|>`, `<|system|>`, and `<|observation|>`, where `metadata` of `<|assistant|>` can only be interpreter.

`````text
<|system|>
You are an intelligent AI assistant, named ChatGLM3. You are connected to a computer, but please note that you cannot connect to the internet. When using Python to solve tasks, you can run the code and get the results. If there are errors in the running results, you need to improve the code as much as possible. You can process files uploaded by users to your computer, and the default storage path for files is /mnt/data/.
<|user|>
#File: /mnt/data/metadata.jsonl
#Size: 35380
#File uploaded
Are there missing or abnormal values in the file?
<|assistant|>
Firstly, I will read the file you provided and review its contents to determine if there are missing or abnormal values.

We start by loading the file:
<|assistant|>interpreter
```python
import json

# Load the uploaded file
data = []
with open("/mnt/data/metadata.jsonl", 'r') as file:
    for line in file:
        data.append(json.loads(line))

# Display the first few records to get a sense of the data
data[:5]
```
<|observation|>
```result
[{'file_name': 'aceinthehole.png',
  'name': 'Ace in the Hole',
  'type': 'survivor',
  'description': 'Lady Luck always seems to be throwing something good your way.'},
 {'file_name': 'adrenaline.png',
  'name': 'Adrenaline',
  'type': 'survivor',
  'description': 'You are fuelled by unexpected energy when on the verge of escape.'},
 {'file_name': 'aftercare.png',
  'name': 'Aftercare',
  'type': 'survivor',
  'description': "Unlocks potential in one's aura reading ability."},
 {'file_name': 'agitation.png',
  'name': 'Agitation',
  'type': 'killer',
  'description': 'You get excited in anticipation of hooking your prey.'},
 {'file_name': 'alert.png',
  'name': 'Alert',
  'type': 'survivor',
  'description': 'Your acute senses are on high alert.'}]
```
<|assistant|>
This file appears to contain metadata about certain items, each with the following fields:

- `file_name`
- `name`
- `type` (e.g. "survivor" or "killer")
- `description`

Next, I will check for missing values in these fields.
<|assistant|>interpreter
```python
# Check for missing values in each column
missing_values = {key: sum(1 for item in data if key not in item) for key in data[0].keys()}

missing_values
```
<|observation|>
```result
{'file_name': 0, 'name': 0, 'type': 0, 'description': 0}
```
<|assistant|>
Each field has no missing values. This means that each entry is complete.

Next, we can check for any potential outliers. This usually requires deeper analysis and understanding of the data, but I can start by examining some basic statistics and distributions. To do this, we can check the value distribution of the "type" field to see if there are any unexpected categories.
<|assistant|>interpreter
```python
# Check the distribution of values in the 'type' column
type_distribution = {key: sum(1 for item in data if item['type'] == key) for key in set(item['type'] for item in data)}

type_distribution
```
<|observation|>
```result
{'survivor': 116, 'killer': 103}
```
<|assistant|>
The 'type' field has two values: 'survivor' and 'killer'. Among them, 'survivor' has 116 entries, while 'killer' has 103 entries.

From this preliminary examination, we did not find any obvious abnormal values.

If you have any specific standards or other fields that need to be checked, I can provide you with a more detailed analysis. What else do you want me to check?
<|user|>
Help me draw a heart
<|assistant|>interpreter
```python
import numpy as np
import matplotlib.pyplot as plt

# The parametric equation for the shape of a heart
def heart(t):
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    return x, y

t = np.linspace(0, 2 * np.pi, 1000)
x, y = heart(t)

plt.figure(figsize=(6, 6))
plt.plot(x, y, color='red')
plt.axis('equal')
plt.axis('off')
plt.show()
```
<|observation|>
```result
[Image]
```
<|assistant|>
This is a shape of heart. I used parametric equations to describe this shape and drew it using Matplotlib. If you have any other needs or questions, please feel free to let me know.
<|user|>  # End
````
