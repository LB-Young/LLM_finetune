本代码参考自：https://github.com/yuanzhoulvpi2017/zero_nlp/blob/main/Chatglm6b_ModelParallel/train_model_all.py
原始代码针对chatglm-v2实现的训练代码，在此基础上，实现对chatglm3的训练；
主要修改：
  device_map_dict中的网络层名称存在差异；需要修改成chatglm3对应的网络层keys；
