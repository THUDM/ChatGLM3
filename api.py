from contextlib import asynccontextmanager
from typing import List

import streamlit as st
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer


@st.cache_resource
def get_model():
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True).cuda()
    # 多显卡支持，使用下面两行代替上面一行，将num_gpus改为你实际的显卡数量
    # from utils import load_model_on_gpus
    # model = load_model_on_gpus("THUDM/chatglm3-6b", num_gpus=2)
    model = model.eval()
    return tokenizer, model


@asynccontextmanager
async def lifespan(app: FastAPI):  # collects GPU memory
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


tokenizer, model = get_model()

app = FastAPI(lifespan=lifespan)


class StreamChatReq(BaseModel):
    input: str
    history: list


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active_connections.append(ws)

    def disconnect(self, ws: WebSocket):
        self.active_connections.remove(ws)

    @staticmethod
    async def send_message(message: str, ws: WebSocket):
        await ws.send_text(message)

    async def broadcast(self, message: str):
        # broadcast
        for connection in self.active_connections:
            await connection.send_text(message)


manager = ConnectionManager()

max_length = 4096
top_p = 0.75
temperature = 0.6


@app.websocket('/stream_chat')
async def stream_chat(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            prompt = data.get('prompt')
            history = data.get('history', [])
            if prompt is None:
                await manager.send_message('', websocket)
            else:
                for response, *_ in model.stream_chat(tokenizer, prompt, history,
                                                      return_past_key_values=True,
                                                      max_length=max_length, top_p=top_p,
                                                      temperature=temperature):
                    await manager.send_message(response, websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.post('/chat')
async def chat(s: StreamChatReq):
    response, history = model.chat(tokenizer, s.input, s.history,
                                   max_length=max_length, top_p=top_p,
                                   temperature=temperature)
    return response, history


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
