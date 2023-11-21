from modelscope import snapshot_download
model_dir = snapshot_download('ZhipuAI/chatglm3-6b', cache_dir='./model', revision='v1.0.0')
