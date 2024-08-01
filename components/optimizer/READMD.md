# 已实现优化器列表
    - SGD、SGDM、Adagrad、RMSProp、Adam、AdamW

# 参数定义
    - 待优化参数：w
    - 损失函数：loss
    - 学习率：lr

# 主要计算流程
    1、计算t时刻损失函数关于参数的梯度g_t；
    2、计算t时刻一阶动量m_t和二阶动量v_t；
    3、计算t时刻下降梯度：η_t = lr * m_t / sqrt(v_t);
    4、计算t+时刻的参数：w_new = w_t - η_t;

    其中：
    一阶动量：与梯度有关；
    二阶动量：与梯度的平方有关；
