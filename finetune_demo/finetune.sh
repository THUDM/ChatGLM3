# CUDA_VISIBLE_DEVICES=0:
# 这个环境变量指定了一个GPU设备的列表，模型将只在这些设备上运行。
# 例如，CUDA_VISIBLE_DEVICES=0 表示模型将只在编号为0的GPU上运行。
# 如果您有多个GPU，并且希望模型在特定的GPU上运行，可以使用逗号分隔的列表，例如 CUDA_VISIBLE_DEVICES=0,1,2,3。
# 这个参数通常用于限制模型在特定数量的GPU上运行，以节省资源或避免在所有可用GPU上启动不必要的计算。
# NCCL_P2P_DISABLE=“1”:
# 这个环境变量控制NVIDIA的NCCL（NVIDIA Collective Communications Library）库是否允许GPU之间的直接内存访问（DMA）。
# 当设置为 "1" 时，NCCL将禁用GPU之间的DMA，这意味着GPU之间的数据传输将需要通过CPU。
# 禁用DMA可以提高安全性，因为它可以防止潜在的GPU之间的通信问题，但它可能会增加CPU的负载，从而影响整体性能。
# 默认情况下，NCCL通常启用DMA，因为它可以提高数据传输的效率。
# NCCL_IB_DISABLE=“1”:
# 这个环境变量控制NCCL是否使用InfiniBand网络接口。
# 当设置为 "1" 时，NCCL将禁用InfiniBand网络接口，这意味着它将只使用其他网络接口，如以太网。
# 禁用InfiniBand可能会导致性能下降，因为它通常提供了比以太网更快的数据传输速度。
# 默认情况下，NCCL通常使用InfiniBand网络接口，因为它可以提供更好的性能。

config_lora="configs/lora2.yaml"
config_modelpath="/Users/ragin/.cache/modelscope/hub/ZhipuAI/chatglm3-6b"

path_data=${1:-"data/AdvertiseGen"}
type_config_modelpath=${2:-"local"}
type_config_lora=${3:-"low"}
path_data_fix="${path_data}_fix"

if [[ $type_config_modelpath == "modelscope" ]]; then

    config_modelpath="/mnt/workspace/.cache/modelscope/ZhipuAI/chatglm3-6b"

fi

if [[ $type_config_lora == "high" ]]; then

    config_lora="configs/lora1.yaml"

fi

if [[ $path_data == "data/cognition" ]]; then
    path_data_fix= $path_data
else
    python convert.py $path_data
fi

CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" python finetune_hf.py $path_data_fix $config_modelpath $config_lora
