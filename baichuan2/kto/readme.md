## 本代码为trl的官方kto示例代码
这个脚本是基于`trl`的示例脚本修改的,原版的链接为:[kto.py](https://github.com/huggingface/trl/blob/main/examples/scripts/kto.py)


### 主要是做了下面几点改动：
1. 使用baichuan2在微调时，peft的target_module参数需要依据模型的具体网络层进行修改；
2. 使用的模型是`Baichuan2-13B-Chat-v2`
3. 训练的框架使用的是`trl`包，这个是huggingface开发的，和`transformers`是一脉相承。
   - 现在训练大模型，支持最好的框架就是`transformers`。那么，基于这个框架做的二次开发的包，上手就简单很多。
   - 这个包在强化学习里面，确实也是最流行的。
4. peft的target_module参数需要依据Qwen模型的具体网络层进行修改；

## 使用教程

#### 使用自定义数据
```
训练的数据集来自在llama-factory的dpo、ppo训练数据集，每条数据包含`instruction`和`answers`，其中answer中存在两条答案，一条为接受、一条为拒绝的对比数据；在数据加载部分需要把每条数据的内容修改为kto默认的键值对形式；
```

### 训练模型

```shell
sh train_ds.sh

```
