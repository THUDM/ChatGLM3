# coding=utf-8
# WebSocket API for ChatGLM3-6B
import asyncio
from contextlib import asynccontextmanager
from typing import Optional, Literal
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.websockets import WebSocket, WebSocketDisconnect, WebSocketState
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel


@asynccontextmanager
async def lifespan(app: FastAPI):  # collects GPU memory
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True


class History(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: str
    history: Optional[list[History]] = []
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.8
    max_tokens: Optional[int] = 1024
    max_length: Optional[int] = 8192
    repetition_penalty: Optional[float] = 1.1


@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket):
    global model, tokenizer
    await websocket.accept()
    # 建立WebSocket连接
    if websocket is not None:
        try:
            request = await websocket.receive_json()
            chat_request = ChatRequest(**request)
            stop_stream = False
            current_length = 0
            history_dict_list: list[dict[str, str]] = request["history"]
            for response, history, past_key_values in model.stream_chat(
                    tokenizer,
                    chat_request.messages,
                    history_dict_list,
                    past_key_values=None,
                    max_length=chat_request.max_length,
                    top_p=chat_request.top_p,
                    temperature=chat_request.temperature,
                    return_past_key_values=True
            ):
                # 如果模型输出完毕或者客户端断开了连接，就停止对话
                if stop_stream or websocket.client_state == WebSocketState.DISCONNECTED:
                    break
                else:
                    await websocket.send_text(response[current_length:])
                    # 让事件循环有机会处理其他挂起的任务
                    await asyncio.sleep(0)
                    current_length = len(response)
        except WebSocketDisconnect:
            pass
        finally:
            await websocket.close()
    else:
        raise HTTPException(status_code=400, detail="Invalid request")


if __name__ == "__main__":
    # 启动服务器
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True).cuda()
    # 多显卡支持，使用下面两行代替上面一行，将num_gpus改为你实际的显卡数量
    # from utils import load_model_on_gpus
    # model = load_model_on_gpus("THUDM/chatglm3-6b", num_gpus=2)
    # model = model.eval()
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
    # 通过websocket访问：ws://0.0.0.0:8000/ws/chat
