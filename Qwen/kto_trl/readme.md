## 本代码为trl的官方kto示例代码
这个脚本是基于`trl`的示例脚本修改的,原版的链接为:[kto.py](https://github.com/huggingface/trl/blob/main/examples/scripts/kto.py)


### 主要是做了下面几点改动：
1. 使用Qwen在微调时，peft的target_module参数报错，需要依据Qwen模型的具体网络层进行修改；

## 使用教程

#### 使用自定义数据
```
待更新
```

### 训练模型

```shell
sh train_ds.sh

```