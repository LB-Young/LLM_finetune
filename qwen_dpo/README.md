由于当前脚本是针对baichuan2模型的微调基本，
直接使用当前脚本训练qwen，trl/trainer/utils.py会报错"NoneType" object cannot be interpreted as an integer:
需要对trl/trainer/utils.py报错位置修改：
将to_pad = [torch.LongTensor(ex[k][：:-1]) for ex in features]和to_pad = [torch.LongTensor(ex[k]) for ex in features]改成 to_pad = [torch.LongTensor(ex[k][1:-1]) for ex in features]
