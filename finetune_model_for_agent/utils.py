from pypdf import PdfReader
import docx
import os
import faiss
import numpy as np
from transformers import AutoModel,AutoTokenizer

def read_data_from_txt(root_path):
    all_content = []
    files = os.listdir(root_path)
    with open(path,encoding="utf-8") as f:
        lines = f.readlines()
        for content in lines:
            all_content.append(content)
    return all_content

class MyFaiss:
    def __init__(self):
        self.index = faiss.IndexFlatL2(4096)
        self.text_str_list = []

    def search(self, emb):
        D,I = self.index.search(emb.astype(np.float32), 100000)
        distance = 1e9
        if D[0][0] > distance:
            content = ""
        else:
            content = ""
            for i in range(len(self.text_str_list)):
                if D[0][i] < distance:
                    content += self.text_str_list[I[0][i]]
        return content

class Model_emb:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("./model", trust_remote_code=True)
        self.model = AutoModel.from_pretrained("./model", trust_remote_code=True).half().cuda()
        self.myfaiss = MyFaiss()

    def retrive(self, text):
        emb = self.get_sentence_emb(text,is_numpy=True)
        retrive_ans = self.myfaiss.search(emb)
        return retrive_ans

    def load_data(self,path):
        all_content = read_data_from_txt(path)
        for content in all_content:
            self.myfaiss.text_str_list.append(content)
            emb = self.get_sentence_emb(content,is_numpy=True)
            self.myfaiss.index.add(emb.astype(np.float32))

    def get_sentence_emb(self,text,is_numpy=False):
        idx = self.tokenizer([text],return_tensors="pt")
        idx = idx["input_ids"].to("cuda")
        emb = self.model.transformer(idx,return_dict=False)[0]
        emb = emb.transpose(0,1)
        emb = emb[:,-1]

        if is_numpy:
            emb = emb.detach().cpu().numpy()

        return emb
