"""
Flow-GRPO (Flow-based Group Relative Policy Optimization) 训练器

基于 Flow Matching 的 GRPO 实现，使用 SDE 采样器进行探索。

关键特性：
1. SDE 采样器：使用特定的 SDE 更新公式注入随机性
2. 降噪缩减策略：训练时使用少步数，推理时使用多步数
3. 基于 SDE 的 KL 散度解析解：无需 Critic 网络

参考：Flow-GRPO 论文
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Callable
from tqdm import tqdm
import copy

from diffusion.nn import mean_flat
from diffusion.gaussian_diffusion import _extract_into_tensor


class FlowGRPOTrainer:
    """
    基于 Flow Matching 的组相对策略优化训练器。
    
    Flow-GRPO 使用 SDE 采样器进行探索，并通过组内相对优势替代绝对优势。
    """
    
    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        diffusion: 'GaussianDiffusion',
        optimizer: torch.optim.Optimizer,
        reward_fn: Callable[[torch.Tensor, List[str]], torch.Tensor],
        group_size: int = 4,
        clip_epsilon: float = 0.2,
        kl_penalty: float = 0.1,
        advantage_eps: float = 1e-8,
        device: str = 'cuda',
        use_checkpointing: bool = False,
        noise_scale: float = 0.7,  # SDE 噪声缩放系数 a
        train_timesteps: int = 10,  # 训练时的推理步数
        inference_timesteps: int = 40,  # 推理时的步数
    ):
        """
        初始化 Flow-GRPO 训练器。
        
        参数:
            model: 可训练的模型（带 LoRA）用于优化
            ref_model: 冻结的参考模型，用于 KL 惩罚
            diffusion: 扩散调度器
            optimizer: 优化器
            reward_fn: 计算奖励的函数: (motions, prompts) -> rewards
            group_size: 每个 prompt 的采样数量（论文中的 G）
            clip_epsilon: PPO 风格的裁剪参数
            kl_penalty: KL 散度惩罚权重（beta）
            advantage_eps: 优势归一化的数值稳定性小量
            device: 运行设备
            use_checkpointing: 是否使用梯度检查点以节省显存
            noise_scale: SDE 噪声缩放系数 a（默认 0.7）
            train_timesteps: 训练时的推理步数（默认 10）
            inference_timesteps: 推理时的步数（默认 40）
        """
        self.model = model
        self.ref_model = ref_model
        self.diffusion = diffusion
        self.optimizer = optimizer
        self.reward_fn = reward_fn
        self.group_size = group_size
        self.clip_epsilon = clip_epsilon
        self.kl_penalty = kl_penalty
        self.advantage_eps = advantage_eps
        self.device = device
        self.use_checkpointing = use_checkpointing
        self.noise_scale = noise_scale
        self.train_timesteps = train_timesteps
        self.inference_timesteps = inference_timesteps
        
        # 确保参考模型被冻结
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()
        
        # 内存优化：在计算 log prob 后立即清理轨迹
        self.clear_trajectory_after_logprob = True
    
    def _get_velocity_field(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        model_kwargs: Dict,
    ) -> torch.Tensor:
        """
        从模型输出获取速度场 v_theta(x_t, t)。
        
        如果模型预测 x_0，则从 x_0 推导速度场。
        在 Flow Matching 中，速度场与 x_0 的关系为：
        v_theta(x_t, t) = (x_0 - x_t) / (1 - t)
        
        参数:
            model: 模型
            x_t: 当前状态 [B, C, H, W]
            t: 时间步 [B]（整数，范围 [0, num_timesteps]）
            model_kwargs: 条件信息
            
        返回:
            v_theta: 速度场 [B, C, H, W]
        """
        # 获取模型输出（预测 x_0）
        model_output = model(x_t, self.diffusion._scale_timesteps(t), **model_kwargs)
        
        # 将时间步归一化到 [0, 1]（如果 diffusion 使用整数时间步）
        # 假设 diffusion.num_timesteps 是总步数
        t_normalized = t.float() / self.diffusion.num_timesteps
        
        # 确保 t 不会太接近 1（避免除零）
        t_normalized = torch.clamp(t_normalized, min=1e-6, max=1.0 - 1e-6)
        
        # 从 x_0 预测推导速度场
        # v_theta(x_t, t) = (x_0 - x_t) / (1 - t)
        velocity = (model_output - x_t) / (1.0 - t_normalized.view(-1, *([1] * (len(x_t.shape) - 1))))
        
        return velocity
    
    def _sde_noise_schedule(self, t: torch.Tensor) -> torch.Tensor:
        """
        计算 SDE 噪声调度 σ_t = a * sqrt(t / (1 - t))
        
        参数:
            t: 时间步 [B]（整数，范围 [0, num_timesteps]）
            
        返回:
            sigma_t: 噪声标准差 [B, 1, 1, 1]（广播形状）
        """
        # 归一化时间步到 [0, 1]
        t_normalized = t.float() / self.diffusion.num_timesteps
        t_normalized = torch.clamp(t_normalized, min=1e-6, max=1.0 - 1e-6)
        
        # σ_t = a * sqrt(t / (1 - t))
        sigma_t = self.noise_scale * torch.sqrt(t_normalized / (1.0 - t_normalized))
        
        # 扩展维度以匹配 x_t 的形状
        while len(sigma_t.shape) < 4:
            sigma_t = sigma_t.unsqueeze(-1)
        
        return sigma_t
    
    def _sde_step(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        model_kwargs: Dict,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        执行一步 SDE 更新。
        
        更新公式：
        x_{t+Δt} = x_t + [v_θ(x_t, t) + σ_t²/(2t) * (x_t + (1-t)v_θ(x_t, t))] * Δt + σ_t * sqrt(Δt) * ε
        
        其中 ε ~ N(0, I) 且 σ_t = a * sqrt(t / (1-t))
        
        参数:
            model: 模型
            x_t: 当前状态 [B, C, H, W]
            t: 时间步 [B]（整数）
            dt: 时间步长
            model_kwargs: 条件信息
            noise: 可选噪声（用于可重现性）
            
        返回:
            x_{t+Δt}: 下一步状态 [B, C, H, W]
        """
        batch_size = x_t.shape[0]
        device = x_t.device
        
        # 归一化时间步到 [0, 1]
        t_normalized = t.float() / self.diffusion.num_timesteps
        t_normalized = torch.clamp(t_normalized, min=1e-6, max=1.0 - 1e-6)
        
        # 获取速度场
        v_theta = self._get_velocity_field(model, x_t, t, model_kwargs)
        
        # 计算 σ_t
        sigma_t = self._sde_noise_schedule(t)
        
        # 计算漂移项：v_θ(x_t, t) + σ_t²/(2t) * (x_t + (1-t)v_θ(x_t, t))
        t_view = t_normalized.view(-1, *([1] * (len(x_t.shape) - 1)))
        drift_term = v_theta + (sigma_t ** 2) / (2.0 * t_view + 1e-8) * (x_t + (1.0 - t_view) * v_theta)
        
        # 计算扩散项：σ_t * sqrt(Δt) * ε
        if noise is None:
            noise = torch.randn_like(x_t)
        
        diffusion_term = sigma_t * torch.sqrt(torch.tensor(dt, device=device, dtype=torch.float32)) * noise
        
        # SDE 更新
        x_next = x_t + drift_term * dt + diffusion_term
        
        return x_next
    
    def sample_with_trajectory_sde(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        model_kwargs: Dict,
        noise: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        save_trajectory: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        使用 SDE 采样器从模型采样并保存完整轨迹。
        
        参数:
            model: 用于采样的模型
            shape: 采样形状 [B, C, H, W]
            model_kwargs: 条件信息
            noise: 可选的固定噪声（用于可重现性）
            num_steps: 采样步数（如果为 None，使用 train_timesteps）
            save_trajectory: 是否保存完整轨迹
            
        返回:
            字典包含:
                - 'samples': 生成的样本 [B, C, H, W]
                - 'latents_sequence': 完整轨迹序列（如果 save_trajectory=True）
                - 'timesteps_sequence': 时间步序列（如果 save_trajectory=True）
                - 'velocity_sequence': 速度场序列（如果 save_trajectory=True）
        """
        if num_steps is None:
            num_steps = self.train_timesteps
        
        batch_size = shape[0]
        device = next(model.parameters()).device
        
        if noise is None:
            noise = torch.randn(*shape, device=device)
        else:
            noise = noise.to(device)
        
        # 初始化：从 t=T 开始（完全噪声）
        x_t = noise.clone()
        
        # 时间步：从 T 到 0（整数时间步）
        # 使用线性时间步分布
        timesteps_float = np.linspace(self.diffusion.num_timesteps - 1, 0, num_steps + 1)
        timesteps = [int(t) for t in timesteps_float[:-1]]  # 不包括最后一个（t=0）
        dt = 1.0 / num_steps  # 归一化时间步长
        
        # 保存轨迹
        latents_sequence = [x_t.clone()] if save_trajectory else []
        timesteps_sequence = []
        velocity_sequence = []
        
        # SDE 采样循环
        for i, t_val in enumerate(timesteps):
            t = torch.full((batch_size,), t_val, device=device, dtype=torch.long)
            
            # 保存当前状态和时间步
            if save_trajectory:
                # 获取速度场用于 KL 散度计算
                with torch.set_grad_enabled(model.training):
                    v_theta = self._get_velocity_field(model, x_t, t, model_kwargs)
                velocity_sequence.append(v_theta.clone())
            
            # 生成下一步的噪声
            step_noise = torch.randn_like(x_t)
            
            # SDE 更新
            x_t = self._sde_step(
                model=model,
                x_t=x_t,
                t=t,
                dt=dt,
                model_kwargs=model_kwargs,
                noise=step_noise,
            )
            
            # 保存采样后的状态
            if save_trajectory:
                latents_sequence.append(x_t.clone())
                timesteps_sequence.append(t.clone())
        
        result = {
            'samples': x_t,
        }
        
        if save_trajectory:
            result['latents_sequence'] = latents_sequence
            result['timesteps_sequence'] = timesteps_sequence
            result['velocity_sequence'] = velocity_sequence
        
        return result
    
    def compute_sde_kl_divergence(
        self,
        v_theta: torch.Tensor,
        v_ref: torch.Tensor,
        t: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """
        计算基于 SDE 的高斯策略解析解 KL 散度。
        
        D_KL = (Δt/2) * ((σ_t(1-t))/(2t) + 1/σ_t)² * ||v_θ - v_ref||²
        
        参数:
            v_theta: 当前模型的速度场 [B, C, H, W]
            v_ref: 参考模型的速度场 [B, C, H, W]
            t: 时间步 [B]（整数）
            dt: 时间步长（归一化）
            
        返回:
            kl_divergence: KL 散度 [B]
        """
        # 归一化时间步到 [0, 1]
        t_normalized = t.float() / self.diffusion.num_timesteps
        t_normalized = torch.clamp(t_normalized, min=1e-6, max=1.0 - 1e-6)
        
        # 计算 σ_t
        sigma_t = self._sde_noise_schedule(t)
        
        # 计算 KL 散度系数
        # coef = (σ_t(1-t))/(2t) + 1/σ_t
        t_view = t_normalized.view(-1, *([1] * (len(v_theta.shape) - 1)))
        coef = (sigma_t * (1.0 - t_view)) / (2.0 * t_view + 1e-8) + 1.0 / (sigma_t + 1e-8)
        
        # 计算速度场差异的平方范数
        v_diff = v_theta - v_ref
        v_diff_squared = v_diff ** 2
        
        # 对空间维度求和，得到每个样本的 KL 散度
        kl_per_sample = (dt / 2.0) * (coef ** 2) * v_diff_squared.sum(dim=[1, 2, 3])
        
        return kl_per_sample
    
    def compute_group_advantages(
        self,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算组相对优势。
        
        对于每个组（来自同一 prompt 的 G 个样本），计算：
        A_i = (r_i - mean(r_group)) / (std(r_group) + eps)
        
        参数:
            rewards: 奖励张量 [B*G]，其中 B 是批次大小，G 是组大小
            
        返回:
            advantages: 组相对优势 [B*G]
        """
        batch_size = rewards.shape[0] // self.group_size
        rewards_reshaped = rewards.view(batch_size, self.group_size)
        
        # 计算组统计量
        group_mean = rewards_reshaped.mean(dim=1, keepdim=True)  # [B, 1]
        group_std = rewards_reshaped.std(dim=1, keepdim=True)  # [B, 1]
        
        # 检查组内奖励差异是否过小
        if (group_std < 1e-6).any():
            print("警告: 组内奖励标准差过小，advantage 可能接近 0")
        
        # 数值稳定性：确保 std 不会太小
        group_std = torch.clamp(group_std, min=self.advantage_eps)
        
        # 在每个组内归一化
        advantages = (rewards_reshaped - group_mean) / (group_std + self.advantage_eps)
        
        # 检查 NaN
        if torch.isnan(advantages).any():
            print("警告: advantages 计算中包含 NaN")
            advantages = torch.nan_to_num(advantages, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return advantages.view(-1)  # [B*G]
    
    def compute_flow_grpo_loss(
        self,
        log_prob_current: torch.Tensor,
        log_prob_ref: torch.Tensor,
        advantages: torch.Tensor,
        kl_divergence: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算 Flow-GRPO 损失。
        
        L = (1/G) * Σ_i [min(ratio_i * A_i, clip(ratio_i, 1-ε, 1+ε) * A_i) - β * KL_i]
        
        参数:
            log_prob_current: 当前模型下的 log probs [B*G]
            log_prob_ref: 参考模型下的 log probs [B*G]
            advantages: 组相对优势 [B*G]
            kl_divergence: KL 散度 [B*G]
            
        返回:
            loss: 标量损失值
            stats: 用于记录的统计信息字典
        """
        # 计算 ratio（数值稳定性：限制差值范围）
        log_ratio = log_prob_current - log_prob_ref  # [B*G]
        
        # 限制 log_ratio 范围，避免 exp 溢出
        log_ratio = torch.clamp(log_ratio, min=-10, max=10)
        ratio = torch.exp(log_ratio)  # [B*G]
        
        # 额外限制 ratio 的范围
        ratio = torch.clamp(ratio, min=1e-4, max=1e4)
        
        # 检查 NaN
        if torch.isnan(ratio).any():
            print("警告: ratio 包含 NaN")
            ratio = torch.nan_to_num(ratio, nan=1.0, posinf=1e6, neginf=1e-6)
        
        # PPO 风格的裁剪目标
        ratio_clipped = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        
        # 策略损失: min(ratio * A, clipped_ratio * A)
        policy_loss_1 = ratio * advantages
        policy_loss_2 = ratio_clipped * advantages
        policy_loss = torch.min(policy_loss_1, policy_loss_2)
        
        # 总损失: 策略损失 - KL 惩罚
        loss = -(policy_loss.mean() - self.kl_penalty * kl_divergence.mean())
        
        # 检查损失是否异常
        if torch.isnan(loss) or torch.isinf(loss):
            print("警告: loss 包含 NaN 或 Inf")
            loss = torch.tensor(0.0, device=loss.device, requires_grad=True)
        
        # 计算统计信息
        try:
            stats = {
                'loss': loss.item() if not torch.isnan(loss) else 0.0,
                'policy_loss': -policy_loss.mean().item() if not torch.isnan(policy_loss.mean()) else 0.0,
                'kl_penalty': kl_divergence.mean().item() if not torch.isnan(kl_divergence.mean()) else 0.0,
                'mean_ratio': ratio.mean().item() if not torch.isnan(ratio.mean()) else 1.0,
                'mean_advantage': advantages.mean().item() if not torch.isnan(advantages.mean()) else 0.0,
                'mean_log_prob_current': log_prob_current.mean().item() if not torch.isnan(log_prob_current.mean()) else 0.0,
                'mean_log_prob_ref': log_prob_ref.mean().item() if not torch.isnan(log_prob_ref.mean()) else 0.0,
            }
        except Exception as e:
            print(f"警告: 计算统计信息时出错: {e}")
            stats = {
                'loss': 0.0,
                'policy_loss': 0.0,
                'kl_penalty': 0.0,
                'mean_ratio': 1.0,
                'mean_advantage': 0.0,
                'mean_log_prob_current': 0.0,
                'mean_log_prob_ref': 0.0,
            }
        
        return loss, stats
    
    def get_batch_log_prob_sde(
        self,
        model: nn.Module,
        latents_sequence: List[torch.Tensor],
        timesteps_sequence: List[torch.Tensor],
        velocity_sequence: List[torch.Tensor],
        model_kwargs: Dict,
    ) -> torch.Tensor:
        """
        计算基于 SDE 的批量 log probability。
        
        对于 Flow Matching，我们使用速度场预测的准确性来计算 log prob。
        这是一个简化实现，基于速度场预测误差。
        
        参数:
            model: 用于计算的模型
            latents_sequence: 采样轨迹序列 [z_T, z_{T-1}, ..., z_0]
            timesteps_sequence: 时间步序列 [T, T-1, ..., 0]
            velocity_sequence: 速度场序列 [v_T, v_{T-1}, ..., v_1]
            model_kwargs: 条件信息
            
        返回:
            log_prob: 累积 log probability [B]
        """
        batch_size = latents_sequence[0].shape[0]
        device = latents_sequence[0].device
        
        # 对于 Flow Matching，我们使用速度场的一致性作为 log prob 的近似
        total_log_prob = torch.zeros(batch_size, device=device)
        
        num_steps = len(latents_sequence) - 1
        dt = 1.0 / self.train_timesteps
        
        for i in range(num_steps):
            x_t = latents_sequence[i]
            x_next = latents_sequence[i + 1]
            t = timesteps_sequence[i]
            v_pred = velocity_sequence[i]
            
            # 计算实际的速度（从轨迹推导）
            # 实际速度 = (x_next - x_t) / dt
            v_actual = (x_next - x_t) / dt
            
            # 计算速度预测的误差（作为 log prob 的负值）
            # 使用高斯分布的 log prob 近似
            v_error = (v_pred - v_actual) ** 2
            
            # 对空间维度求和，得到每个样本的 log prob（负误差）
            log_prob_step = -v_error.sum(dim=[1, 2, 3])  # [B]
            
            # 检查异常值
            if torch.isnan(log_prob_step).any():
                log_prob_step = torch.nan_to_num(log_prob_step, nan=0.0, posinf=0.0, neginf=-1e6)
            
            total_log_prob = total_log_prob + log_prob_step
        
        return total_log_prob
    
    def step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        执行一步 Flow-GRPO 训练。
        
        参数:
            batch: 批次数据，包含:
                - 'text': 文本提示列表 [B]
                - 其他条件信息
                
        返回:
            stats: 训练统计信息字典
        """
        self.model.train()
        
        # 提取 prompts
        prompts = batch.get('text', [])
        if isinstance(prompts, torch.Tensor):
            prompts = prompts.tolist()
        batch_size = len(prompts)
        
        # 扩展 prompts 用于组采样：每个 prompt 重复 G 次
        expanded_prompts = []
        for prompt in prompts:
            expanded_prompts.extend([prompt] * self.group_size)
        
        # 准备扩展批次的 model kwargs
        expanded_batch_size = batch_size * self.group_size
        model_kwargs = self._prepare_model_kwargs(batch, expanded_batch_size)
        
        # ========== 阶段 1: Rollout（使用 SDE 采样器）==========
        # 确定动作长度
        if 'lengths' in batch and isinstance(batch['lengths'], torch.Tensor):
            motion_length = int(batch['lengths'].max().item()) if len(batch['lengths']) > 0 else 196
        else:
            motion_length = 196  # 默认长度
        
        shape = (
            expanded_batch_size,
            self.model.njoints,
            self.model.nfeats,
            motion_length
        )
        
        # 使用固定噪声进行采样（用于可重现性）
        noise = torch.randn(*shape, device=self.device)
        
        # 使用当前模型进行 SDE 采样（保存轨迹和速度场）
        with torch.set_grad_enabled(True):
            current_result = self.sample_with_trajectory_sde(
                self.model,
                shape,
                model_kwargs,
                noise=noise,
                num_steps=self.train_timesteps,
                save_trajectory=True,
            )
        
        motions = current_result['samples']  # [B*G, C, H, W]
        latents_sequence_current = current_result['latents_sequence']
        timesteps_sequence = current_result['timesteps_sequence']
        velocity_sequence_current = current_result['velocity_sequence']
        
        # 计算当前模型的 log probability
        log_prob_current = self.get_batch_log_prob_sde(
            self.model,
            latents_sequence_current,
            timesteps_sequence,
            velocity_sequence_current,
            model_kwargs,
        )  # [B*G]
        
        # 检查 log_prob_current
        if torch.isnan(log_prob_current).any():
            print("警告: log_prob_current 包含 NaN")
            log_prob_current = torch.nan_to_num(log_prob_current, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if (log_prob_current.abs() > 1e7).any():
            log_prob_current = torch.clamp(log_prob_current, min=-1e7, max=1e7)
        
        # 清理中间变量
        del current_result
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 使用参考模型计算 log probability 和速度场（用于 KL 散度）
        with torch.no_grad():
            ref_result = self.sample_with_trajectory_sde(
                self.ref_model,
                shape,
                model_kwargs,
                noise=noise,  # 使用相同的噪声
                num_steps=self.train_timesteps,
                save_trajectory=True,
            )
            
            log_prob_ref = self.get_batch_log_prob_sde(
                self.ref_model,
                ref_result['latents_sequence'],
                ref_result['timesteps_sequence'],
                ref_result['velocity_sequence'],
                model_kwargs,
            )  # [B*G]
            
            velocity_sequence_ref = ref_result['velocity_sequence']
        
        # 检查 log_prob_ref
        if torch.isnan(log_prob_ref).any():
            print("警告: log_prob_ref 包含 NaN")
            log_prob_ref = torch.nan_to_num(log_prob_ref, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if (log_prob_ref.abs() > 1e7).any():
            log_prob_ref = torch.clamp(log_prob_ref, min=-1e7, max=1e7)
        
        # 计算 KL 散度（基于速度场差异）
        # 注意：需要在清理轨迹之前计算 KL 散度
        num_steps = len(velocity_sequence_current)
        dt = 1.0 / self.train_timesteps
        kl_divergence_total = torch.zeros(expanded_batch_size, device=self.device)
        
        for i in range(num_steps):
            v_theta = velocity_sequence_current[i]
            v_ref = velocity_sequence_ref[i]
            t = timesteps_sequence[i]
            
            kl_step = self.compute_sde_kl_divergence(v_theta, v_ref, t, dt)
            kl_divergence_total = kl_divergence_total + kl_step
        
        # 清理轨迹（在计算完 KL 散度之后）
        if self.clear_trajectory_after_logprob:
            del latents_sequence_current, timesteps_sequence, velocity_sequence_current
            del velocity_sequence_ref
            del ref_result
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # ========== 阶段 2: 奖励计算 ==========
        rewards = self.reward_fn(motions, expanded_prompts)  # [B*G]
        rewards = rewards.to(self.device)
        
        # 检查奖励
        if torch.isnan(rewards).any():
            print("警告: rewards 包含 NaN")
            rewards = torch.nan_to_num(rewards, nan=0.0, posinf=1.0, neginf=0.0)
        
        # ========== 阶段 3: 优势计算 ==========
        advantages = self.compute_group_advantages(rewards)  # [B*G]
        
        # 检查优势
        if torch.isnan(advantages).any():
            print("警告: advantages 包含 NaN")
            advantages = torch.nan_to_num(advantages, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # ========== 阶段 4: 损失计算和更新 ==========
        loss, stats = self.compute_flow_grpo_loss(
            log_prob_current,
            log_prob_ref,
            advantages,
            kl_divergence_total,
        )
        
        # 添加奖励统计信息
        stats['mean_reward'] = rewards.mean().item()
        stats['std_reward'] = rewards.std().item()
        stats['min_reward'] = rewards.min().item()
        stats['max_reward'] = rewards.max().item()
        
        # 计算每个 prompt 的平均奖励（用于监控）
        rewards_reshaped = rewards.view(batch_size, self.group_size)
        prompt_avg_rewards = rewards_reshaped.mean(dim=1)
        stats['prompt_avg_reward'] = prompt_avg_rewards.mean().item()
        
        # 反向传播
        loss.backward()
        
        # 计算梯度范数
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float('inf'))
        stats['grad_norm'] = grad_norm.item()
        stats['learning_rate'] = self.optimizer.param_groups[0]['lr']
        
        # 检查梯度异常
        if torch.isnan(grad_norm) or torch.isinf(grad_norm) or grad_norm > 1000:
            print(f"警告: 梯度范数异常 ({grad_norm.item():.2f})，跳过此步更新")
            self.optimizer.zero_grad()
            return stats
        
        # 梯度裁剪：使用更严格的值
        max_grad_norm = 1.0
        if grad_norm > max_grad_norm:
            print(f"警告: 梯度范数过大 ({grad_norm.item():.2f})，将被裁剪到 {max_grad_norm}")
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)
            grad_norm_after = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float('inf'))
            stats['grad_norm'] = grad_norm_after.item()
            
            # 如果裁剪后仍然很大，跳过更新
            if grad_norm_after > max_grad_norm * 10:
                print(f"错误: 梯度裁剪后仍然过大 ({grad_norm_after.item():.2f})，跳过此步")
                self.optimizer.zero_grad()
                return stats
        else:
            # 正常情况下的梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)
        
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return stats
    
    def _prepare_model_kwargs(
        self,
        batch: Dict[str, torch.Tensor],
        expanded_batch_size: int,
    ) -> Dict:
        """
        为扩展批次（组采样）准备 model kwargs。
        
        参数:
            batch: 原始批次
            expanded_batch_size: 新的批次大小（B * G）
            
        返回:
            model_kwargs: 模型参数字典
        """
        model_kwargs = {'y': {}}
        
        # 处理文本 prompts
        if 'text' in batch:
            prompts = batch['text']
            if isinstance(prompts, list):
                # 扩展 prompts
                expanded_prompts = []
                for prompt in prompts:
                    expanded_prompts.extend([prompt] * self.group_size)
                model_kwargs['y']['text'] = expanded_prompts
            else:
                # 张量情况：沿批次维度重复
                model_kwargs['y']['text'] = prompts.repeat_interleave(self.group_size, dim=0)
        
        # 处理其他条件
        for key in ['lengths', 'mask']:
            if key in batch:
                value = batch[key]
                if isinstance(value, torch.Tensor):
                    model_kwargs['y'][key] = value.repeat_interleave(self.group_size, dim=0)
                else:
                    model_kwargs['y'][key] = value
        
        return model_kwargs


def create_flow_grpo_trainer(
    model: nn.Module,
    ref_model: nn.Module,
    diffusion: 'GaussianDiffusion',
    reward_fn: Callable,
    learning_rate: float = 1e-5,
    group_size: int = 4,
    clip_epsilon: float = 0.2,
    kl_penalty: float = 0.1,
    noise_scale: float = 0.7,
    train_timesteps: int = 10,
    inference_timesteps: int = 40,
    **kwargs,
) -> FlowGRPOTrainer:
    """
    创建 Flow-GRPO 训练器的工厂函数。
    
    参数:
        model: 可训练模型
        ref_model: 参考模型
        diffusion: 扩散调度器
        reward_fn: 奖励函数
        learning_rate: 优化器学习率
        group_size: GRPO 的组大小
        clip_epsilon: 裁剪参数
        kl_penalty: KL 惩罚权重
        noise_scale: SDE 噪声缩放系数 a（默认 0.7）
        train_timesteps: 训练时的推理步数（默认 10）
        inference_timesteps: 推理时的步数（默认 40）
        **kwargs: FlowGRPOTrainer 的额外参数
        
    返回:
        FlowGRPOTrainer 实例
    """
    # 创建优化器（仅针对可训练参数，例如 LoRA）
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    # 验证只有 LoRA 参数是可训练的
    trainable_param_names = [name for name, param in model.named_parameters() if param.requires_grad]
    non_lora_trainable = [name for name in trainable_param_names if 'lora' not in name.lower()]
    if non_lora_trainable:
        print(f"警告: 发现非 LoRA 参数是可训练的: {non_lora_trainable[:5]}")
        if len(non_lora_trainable) > 5:
            print(f"  ... 还有 {len(non_lora_trainable) - 5} 个参数")
    else:
        print(f"✓ 确认: 只有 LoRA 参数是可训练的（共 {len(trainable_param_names)} 个参数）")
    
    if len(trainable_params) == 0:
        raise ValueError("错误: 没有可训练的参数！请检查模型是否正确配置了 LoRA")
    
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)
    
    return FlowGRPOTrainer(
        model=model,
        ref_model=ref_model,
        diffusion=diffusion,
        optimizer=optimizer,
        reward_fn=reward_fn,
        group_size=group_size,
        clip_epsilon=clip_epsilon,
        kl_penalty=kl_penalty,
        noise_scale=noise_scale,
        train_timesteps=train_timesteps,
        inference_timesteps=inference_timesteps,
        **kwargs,
    )