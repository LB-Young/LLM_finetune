# 当前仓库代码参考自https://github.com/yuanzhoulvpi2017/zero_nlp

1. 本仓库代码参考自https://github.com/yuanzhoulvpi2017/zero_nlp对chatglm-v2-6b模型训练的代码，实现对智谱AI开源`chatglm-v3-6b`大模型；
2. 基于`v2`的[官网代码](https://github.com/THUDM/ChatGLM2-6B/tree/main/ptuning)，做的修改；

# 修改内容
1. 使用chatglm-v2-6b的训练代码直接讲模型改为chatglm-v3-6b存在部分小问题：
   1、load_dataset中的参数use_auth_token被删除；
   2、config中的use_cache设置为False；
   3、preprocess_function_train函数中的tokenizer.build_prompt会报错：chatglm3的tokenizer没有这个方法，我直接删除了这一行，query赋值给prompt，因为我的输入数据就只有一列，不想需要拼接；
   4、trainer.train(resume_from_checkpoint=checkpoint)不能在传递checkpoint的同时，使用两张卡训练；（后续优化）

3. 其他修改内容
   1、train.sh脚本中的prompt_column和responses_column修改为数据集中的输入列和输出列；
