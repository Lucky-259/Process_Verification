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
在服务器B的命令行输入启动vllm服务命令，将${your_Qwen2.5-Math-PRM-7B_path}替换为您下载的Qwen2.5-Math-PRM-7B模型路径，部署服务的IP地址需要获取（利用hostname -I命令即可输出本机IPV4地址），端口号自行设置（我们以8034为例部署服务），设置完请到服务器A中的Process_Verification/verl/prm/discriminative_prm/reward_function.py中把 PRM_API_URL = "http://localhost:8034/pooling" 这条语句中的localhost改为服务器B的IPV4地址；8034改为修改后的端口号

单卡：
```bash
CUDA_VISIBLE_DEVICES=0 vllm serve ${your_Qwen2.5-Math-PRM-7B_path} --task reward --max-model-len 16384 --host 0.0.0.0 --port 8034
```
多卡，修改显卡数量与-dp后面的数字对应：
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve ${your_Qwen2.5-Math-PRM-7B_path} --task reward --max-model-len 16384 --host 0.0.0.0 --port 8034 -dp 4
```

### 在服务器A上进行RL训练

在Process_Verification目录下运行脚本，将${your_DeepSeek-R1-Distill-Qwen-1.5B_path}换成您下载的DeepSeek-R1-Distill-Qwen-1.5B模型路径
```bash
bash verl/prm/discriminative_prm/rl_disprm_1.5B.sh --model ${your_DeepSeek-R1-Distill-Qwen-1.5B_path}
```


### 评估
启动vllm服务进行评估，在eval目录下运行
- vllm_all.sh脚本用于测试整个项目的所有checkpoint文件
  - 汇总后的结果保存在Process_Verification/eval/eval_results的all_checkpoints_summary.csv中。
  - 需要修改vllm_all.sh前面的若干变量，指向需要评估的checkpoint文件夹、启动vllm服务的环境名（用于kill服务）
  - 运行命令中第1个参数为Process_Verification/deepscaler/data/test/路径下的数据集名称，第2个参数为pass@N中的N，第3个参数为总并发数。
    ```bash
    bash vllm_all.sh aime 64 150
    ```
- vllm_one.sh脚本用于测试单个checkpoint文件，使用方式同上