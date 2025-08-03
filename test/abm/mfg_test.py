if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    from torch.autograd import grad


    class GeneratorNetwork(nn.Module):
        """
        生成器网络
        输入：初始观点采样z和时间t
        输出：时间t的观点分布
        """
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim + 1, hidden_dim),  # +1 for time dimension
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )

        def forward(self, z, t):
            # 拼接初始观点和时间信息
            inputs = torch.cat([z, t], dim=1)
            return (1 - t) * z + t * self.net(inputs)


    class DiscriminatorNetwork(nn.Module):
        """
        判别器网络
        输入：观点x和时间t
        输出：价值函数估计
        """
        def __init__(self, input_dim, hidden_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim + 1, hidden_dim),  # +1 for time dimension
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )

        def forward(self, x, t):
            inputs = torch.cat([x, t], dim=1)
            return self.net(inputs)


    class MFG_GAN:
        def __init__(self, dim, device='cuda'):
            self.dim = dim  # 观点维度
            self.device = device
            self.beta = 1.0  # 控制权重
            self.lambd = 0.2  # 固执强度
            self.v = 0.4  # 扩散常数
            self.c1 = 0.5  # 运行成本权重
            self.c2 = 1.0  # 终端成本权重

            # 神经网络
            self.G = GeneratorNetwork(dim, 256, dim).to(device)
            self.phi = DiscriminatorNetwork(dim, 256).to(device)

            # 优化器
            self.opt_G = optim.Adam(self.G.parameters(), lr=1e-4)
            self.opt_phi = optim.Adam(self.phi.parameters(), lr=4e-4)

            # 目标观点 (论文中设置为1.0)
            self.target_opinion = torch.ones(dim).to(device) * 1.0

        def hamiltonian(self, x, grad_phi, r_x_t):
            """
            计算Hamiltonian函数
            """
            mean_x = torch.mean(x, dim=0)
            control_term = -0.25 * self.beta ** 2 * torch.sum(grad_phi ** 2, dim=1)
            opinion_term = torch.sum(grad_phi * ((1 - self.lambd) * r_x_t * mean_x + (self.lambd - 1) * x))
            return control_term + opinion_term

        def compute_losses(self, batch_size):
            """
            计算生成器和判别器的损失
            """
            # 采样初始观点和时间
            z = torch.randn(batch_size, self.dim).to(self.device)  # 从初始分布中采样
            t = torch.rand(batch_size, 1).to(self.device)  # 均匀时间采样

            # 判别器训练
            # 计算生成输出
            x = self.G(z, t)

            # 计算价值函数及其导数
            x.requires_grad_(True)
            t.requires_grad_(True)

            phi_val = self.phi(x, t)

            # 计算梯度 (空间导数)
            grad_x = grad(phi_val, x, torch.ones_like(phi_val), create_graph=True, retain_graph=True)[0]

            # 计算时间导数
            dt_phi = grad(phi_val, t, torch.ones_like(phi_val), create_graph=True, retain_graph=True)[0]

            # 计算Laplacian (二阶空间导数)
            lap_phi = torch.zeros(batch_size, 1).to(self.device)
            for i in range(self.dim):
                grad_i = grad(grad_x[:, i], x, torch.ones_like(grad_x[:, i]), create_graph=True, retain_graph=True)[0][:, i]
                lap_phi += grad_i.unsqueeze(1)

            # 计算Hamiltonian
            r_x_t = torch.ones(batch_size).to(self.device)
            H = self.hamiltonian(x, grad_x, r_x_t)

            # 计算HJB残差
            hjb_residual = dt_phi + self.v * lap_phi + H.unsqueeze(1)

            # 判别器损失
            loss_phi = torch.mean(phi_val[:, 0]) + torch.mean(hjb_residual)

            # 正则项
            gamma = 0.1
            reg_term = gamma * torch.mean(torch.abs(hjb_residual))

            # 生成器训练
