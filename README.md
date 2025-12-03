### 配置服务器A
8 GPUs，python==3.11

首先进入Process_Verification文件夹
```bash
cd Process_Verification
```
然后，执行命令安装所需的包
```bash
pip install -e .
```
最后，安装flash-attn
```bash
pip install flash-attn --no-build-isolation
```

### 配置服务器B
2 GPUs

下载vllm
```bash
pip install vllm==0.8.4
```

### 在服务器A上下载DeepSeek-R1-Distill-Qwen-1.5B
将${your_DeepSeek-R1-Distill-Qwen-1.5B_path}替换为您的DeepSeek-R1-Distill-Qwen-1.5B要下载到的本地路径
```bash
huggingface-cli download DeepSeek-R1-Distill-Qwen-1.5B \
  --local-dir ${your_DeepSeek-R1-Distill-Qwen-1.5B_path} \
  --local-dir-use-symlinks False
```

### 在服务器B上下载Qwen2.5-Math-PRM-7B
将${your_Qwen2.5-Math-PRM-7B_path}分别替换为您的Qwen2.5-Math-PRM-7B要下载到的本地路径
```bash
huggingface-cli download Qwen/Qwen2.5-Math-PRM-7B \
  --local-dir ${your_Qwen2.5-Math-PRM-7B_path} \
  --local-dir-use-symlinks False
```

### 在服务器B上启动vllm服务
在服务器B的命令行输入启动vllm服务命令，将${your_Qwen2.5-Math-PRM-7B_path}替换为您下载的Qwen2.5-Math-PRM-7B模型路径，部署服务的IP地址需要获取（利用hostname -I命令即可输出本机IPV4地址），端口号自行设置（我们以8034为例部署服务），设置完请到服务器A中的Process_Verification/verl/utils/prm/rm_call.py中把 api_url = f"http://localhost:8034/pooling" 这条语句中的localhost改为服务器B的IPV4地址；8034改为修改后的端口号

单卡：
```bash
CUDA_VISIBLE_DEVICES=0 vllm serve ${your_Qwen2.5-Math-PRM-7B_path} --task reward --max-model-len 16384 --host 0.0.0.0 --port 8034
```
多卡，修改显卡数量与-dp后面的数字对应：
```bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve ${your_Qwen2.5-Math-PRM-7B_path} --task reward --max-model-len 16384 --host 0.0.0.0 --port 8034 -dp 2
```

### 在服务器A上进行RL训练

在服务器B的vllm服务启动成功后，在服务器A中先执行下面命令回到上一行目录
```bash
cd ..
```

之后运行下面脚本，将${your_DeepSeek-R1-Distill-Qwen-1.5B_path}换成您下载的DeepSeek-R1-Distill-Qwen-1.5B模型路径
```bash
bash Process_Verification/scripts/rl_1.5B.sh --model ${your_DeepSeek-R1-Distill-Qwen-1.5B_path}
```