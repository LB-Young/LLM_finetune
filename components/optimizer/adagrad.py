"""
SGD对应取值: m_t = g_t   v_t = (g_1**2 + g_2**2 + g_t**2);
故 η_t  = lr * g_t / (g_1**2 + g_2**2 + g_t**2);
w_new = w_t - lr * g_t / (g_1**2 + g_2**2 + g_t**2)
"""




import torch
import torch.nn as nn
import torch.optim as optim

# 定义一个简单的全连接层
class MyModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.fc(x)

# 定义一个自定义优化器
class CustomAdagrad(optim.Optimizer):
    def __init__(self, params, lr=0.01, β=0.9):
        defaults = dict(lr=lr)
        super(CustomAdagrad, self).__init__(params, defaults)
        self.m_post = []
        self.v_post = []
        # self.g_post = []
        self.β = β
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                # SGD
                g_t = param.grad.data
                m_t = g_t
                v_t = m_t ** 2
                if len(self.m_post) < len(group['params']):
                    pass
                else:
                    index = len(self.m_post) - len(group['params'])
                    while index >= 0:
                        v_t += self.m_post[index] ** 2
                        index -= len(group['params'])
                η_t  = -group['lr'] * m_t / torch.sqrt(v_t)
                param.data.add_(η_t)
                self.m_post.append(g_t)

                """
                param.data：这是一个张量，表示模型参数的当前值。通过.data属性，我们可以直接访问和修改这些值，而不需要计算图的跟踪（即不记录这些操作，以避免影响梯度计算）。
                add_：这是PyTorch中的一个原地（in-place）加法操作。它会直接修改param.data的值，而不是返回一个新的张量。add_函数的格式是param.data.add_(value)，其中value是要加到param.data上的值。
                """
        return loss

# 示例：使用全连接层和自定义优化器进行训练
input_size = 10
output_size = 2
model = MyModel(input_size, output_size)

# 随机输入数据和标签
inputs = torch.randn(5, input_size)
labels = torch.randint(0, output_size, (5,))

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 使用自定义优化器
optimizer = CustomAdagrad(model.parameters(), lr=0.01, β=0.9)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    optimizer.zero_grad()  # 清零梯度
    outputs = model(inputs)  # 前向传播
    loss = criterion(outputs, labels)  # 计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
