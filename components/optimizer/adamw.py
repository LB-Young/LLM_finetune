"""
SGD对应取值: m_t = β_1 * m_(t-1) + (1 - β_1) * g_t   v_t = β_2 * v_(t-1) + (1-β_2) * (g_t**2);
修正: m_t_hat = m_t / (1 - β_1**t)  v_t_hat = v_t / (1 - β_2**t)
故 η_t  = lr * m_t_hat / aqrt(v_t_hat);
w_new = w_t - lr * m_t_hat / aqrt(v_t_hat);
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
class CustomAdam(optim.Optimizer):
    def __init__(self, params, lr=0.01, β_1=0.9, β_2=0.999):
        defaults = dict(lr=lr)
        super(CustomAdam, self).__init__(params, defaults)
        self.m_post = []
        self.v_post = []
        # self.g_post = []
        self.t = 1
        self.β_1 = β_1
        self.β_2 = β_2
        
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
                if len(self.m_post) < len(group['params']):
                    m_t = (1-self.β_1) * g_t
                else:
                    m_t = self.β_1 * self.m_post[-len(group['params'])] + (1-self.β_1) * g_t

                if len(self.v_post) < len(group['params']):
                    v_t = (1 - self.β_2) * g_t ** 2
                else:
                    v_t = self.β_2 * self.v_post[-len(group['params'])] + (1 - self.β_2) * g_t ** 2

                m_t_hat = m_t / (1 - self.β_1**self.t)  
                v_t_hat = v_t / (1 - self.β_2**self.t)
                η_t  = -group['lr'] * m_t_hat / torch.sqrt(v_t_hat)
                param.data.add_(η_t)
                self.m_post.append(m_t)
                self.v_post.append(v_t)
                self.t += 1
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
optimizer = CustomAdam(model.parameters(), lr=0.01, β_1=0.9, β_2=0.999)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    optimizer.zero_grad()  # 清零梯度
    outputs = model(inputs)  # 前向传播
    loss = criterion(outputs, labels)  # 计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
