if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    from torch.autograd import grad


    class GeneratorNetwork(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            """
            生成器网络 (G_θ)
            输入: 初始观点采样z和时间t
            输出: 时间t时的观点分布
            """
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
        判别器网络 (φ_ω)
        输入: 观点x和时间t
        输出: 价值函数估计
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
            # 模型参数
            self.dim = dim  # 观点维度
            self.device = device
            self.beta = 1.0  # 控制权重
            self.lambd = 0.2  # 固执强度
            self.v = 0.4  # 扩散常数
            self.c1 = 0.5  # 运行成本权重
            self.c2 = 1.0  # 终端成本权重


            self.G = GeneratorNetwork(dim, 256, dim).to(device)
            self.phi = DiscriminatorNetwork(dim, 256).to(device)

            # 优化器
            self.opt_G = optim.Adam(self.G.parameters(), lr=1e-4)
            self.opt_phi = optim.Adam(self.phi.parameters(), lr=4e-4)

            # 目标观点 (论文中设置为(1,0))
            self.target_opinion = torch.ones(dim).to(device) * 1.0

        def hamiltonian(self, x, grad_phi, r_x_t):
            """
            修正后的Hamiltonian计算函数
            符合公式：H = -β²/4 * ||∇φ||² + ∇φ·[(1-λ)r(x,t)E[y] + (λ-1)x]
            """
            # 控制项计算 (无变化)
            control_term = -0.25 * self.beta ** 2 * torch.sum(grad_phi ** 2, dim=1)

            # 观点项修正 (关键修改)
            mean_x = torch.mean(x, dim=0)

            # 显式扩展mean_x到与grad_phi相同形状
            expanded_mean_x = mean_x.unsqueeze(0).expand_as(grad_phi)  # [1, 50] → [256, 50]

            # 计算观点项 (维度兼容处理)
            social_effect = (1 - self.lambd) * r_x_t.unsqueeze(1) * expanded_mean_x
            personal_effect = (self.lambd - 1) * x
            opinion_term = torch.sum(
                grad_phi * (social_effect + personal_effect),
                dim=1
            )

            return control_term + opinion_term

        def compute_losses(self, batch_size):
            """
            计算生成器和判别器的损失
            """
            # 采样初始观点和时间
            z = torch.randn(batch_size, self.dim).to(self.device)  # 从初始分布采样
            t = torch.rand(batch_size, 1).to(self.device)  # 均匀时间采样

            # ===== 判别器训练 =====
            # 计算生成器输出
            x = self.G(z, t)

            # 计算价值函数及其导数
            x.requires_grad_(True)
            t.requires_grad_(True)

            phi_val = self.phi(x, t)

            # 计算梯度 (空间导数)
            grad_x = grad(phi_val, x, torch.ones_like(phi_val),
                          create_graph=True, retain_graph=True)[0]

            # 计算时间导数
            dt_phi = grad(phi_val, t, torch.ones_like(phi_val),
                          create_graph=True, retain_graph=True)[0]

            # 计算Laplacian (二阶空间导数)
            lap_phi = torch.zeros(batch_size, 1).to(self.device)
            for i in range(self.dim):
                grad_i = grad(grad_x[:, i], x, torch.ones_like(grad_x[:, i]),
                              create_graph=True, retain_graph=True)[0][:, i]
                lap_phi += grad_i.unsqueeze(1)

            # 计算Hamiltonian (需要实现r(x,t))
            # 此处简化: 假设r(x,t)=1 (对称社交网络)
            r_x_t = torch.ones(batch_size).to(self.device)
            H = self.hamiltonian(x, grad_x, r_x_t)

            # 计算HJB残差 (公式28)
            hjb_residual = dt_phi + self.v * lap_phi + H.unsqueeze(1)

            # 判别器损失
            loss_phi = torch.mean(phi_val[:, 0]) + torch.mean(hjb_residual)

            # 正则项 (可选)
            gamma = 0.1
            reg_term = gamma * torch.mean(torch.abs(hjb_residual))

            # ===== 生成器训练 =====
            # 固定判别器，计算生成器损失
            x_gen = self.G(z, t)
            phi_gen = self.phi(x_gen, t)

            # 计算生成器相关的导数
            x_gen.requires_grad_(True)
            grad_x_gen = grad(phi_gen, x_gen, torch.ones_like(phi_gen),
                              create_graph=True, retain_graph=True)[0]

            dt_phi_gen = grad(phi_gen, t, torch.ones_like(phi_gen),
                              create_graph=True)[0]

            lap_phi_gen = torch.zeros(batch_size, 1).to(self.device)
            for i in range(self.dim):
                grad_i = grad(grad_x_gen[:, i], x_gen, torch.ones_like(grad_x_gen[:, i]),
                              create_graph=True, retain_graph=True)[0][:, i]
                lap_phi_gen += grad_i.unsqueeze(1)

            H_gen = self.hamiltonian(x_gen, grad_x_gen, r_x_t)
            hjb_residual_gen = dt_phi_gen + self.v * lap_phi_gen + H_gen.unsqueeze(1)

            loss_G = torch.mean(hjb_residual_gen)

            return loss_phi + reg_term, loss_G

        def train(self, epochs, batch_size=128):
            """
            训练循环 (算法1)
            """
            for epoch in range(epochs):
                # 训练判别器
                self.opt_phi.zero_grad()
                loss_phi, _ = self.compute_losses(batch_size)
                loss_phi.backward()
                self.opt_phi.step()

                # 训练生成器
                self.opt_G.zero_grad()
                _, loss_G = self.compute_losses(batch_size)
                loss_G.backward()
                self.opt_G.step()

                if epoch % 100 == 0:
                    print(f'Epoch {epoch}/{epochs} | Phi Loss: {loss_phi.item():.4f} | G Loss: {loss_G.item():.4f}')

        def predict(self, z, t):
            """
            预测给定初始观点和时间的观点分布
            """
            with torch.no_grad():
                return self.G(z, t)


    # 使用示例
    if __name__ == "__main__":
        # 初始化模型 (50维观点空间)
        model = MFG_GAN(dim=50)

        # 训练模型
        model.train(epochs=10000, batch_size=256)

        # 预测示例
        z_test = torch.randn(10, 50).cuda()  # 10个样本
        t_test = torch.ones(10, 1).cuda()  # 终端时间
        predictions = model.predict(z_test, t_test)
        print("Predictions shape:", predictions.shape)