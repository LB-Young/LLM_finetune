### LLM_finetune
LLM_finetune
当前仓库是本人学习微调过程中的一些测试脚本，参考了很多其他博主的代码，也在其基础上做了一些调整和优化，并且处理了一些bug；
主要参考仓库：
transformers、trl、peft仓库和
https://github.com/yuanzhoulvpi2017/zero_nlp
https://github.com/mlabonne/llm-course

### 主要内容
### 主要模型

1. baichuan2：
```
        sft\dpo\kto
```
2. Qwen：
```
        sft\ppo\dpo\kto\orpo
```
3. ChatGLM：
```
        sft
```
4. Mixtral-Mistral
```
        sft
```

## 代码结构
```
1. 导入三方包；
导入数据集处理包
from dataclasses import dataclass, field
from datasets import Dataset, load_dataset
导入模型加载器、token加载器、参数解析器
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
导入生成参数类
from transformers.generation.utils import GenerationConfig
导入训练器、训练参数、peft模型封装函数
from trl import KTOConfig, KTOTrainer, ModelConfig, get_peft_config
导入peft配置和模型封装器
from peft import LoraConfig, get_peft_model

2. 定义ScriptArguments、ModelArguments、DatasetArguments、TrainArguments等，也可以从trainsformers和trl库导入默认的Arguments、基于默认的进行修改；
需要使用dataclass装饰器封装Arguments类，里面定义参数；

3. 定义数据集预处理器；
首先使用load_dataset加载数据集，再通过map函数将加载的数据集处理成当前训练器需要的数据格式，最后拆分成训练集和测试集。

4. 定义主函数，组织各部分代码进行训练；
        使用HfArgumentParser解析参数
        通过自定义数据集预处理器加载训练集和测试集
        加载模型和参考模型
        加载分词器、并且需要处理特殊token
        定义peft_config并封装peft_model
        定义训练器
```