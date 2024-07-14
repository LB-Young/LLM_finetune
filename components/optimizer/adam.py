"""
SGD对应取值: m_t = β_1 * m_(t-1) + (1 - β_1) * g_t   v_t = β_2 * v_(t-1) + (1-β_2) * (g_t**2);
修正: m_t_hat = m_t / (1 - β_1**t)  v_t_hat = v_t / (1 - β_2**t)
故 η_t  = lr * m_t_hat / aqrt(v_t_hat);
w_new = w_t - lr * m_t_hat / aqrt(v_t_hat);
"""