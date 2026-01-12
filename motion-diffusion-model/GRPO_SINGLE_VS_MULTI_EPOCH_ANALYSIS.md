# GRPO 单轮 vs 多轮更新分析

## 一、当前实现：**单轮更新**（Single-Epoch）

### 实现分析

**当前代码位置**：`model/GRPO/grpo_trainer.py` 的 `step()` 方法（第510-809行）

### 执行流程

1. **采样阶段**（第542-606行）：
   ```python
   # 使用当前模型采样
   current_result = self.sample_with_trajectory(...)
   motions = current_result['samples']
   latents_sequence_current = current_result['latents_sequence']
   
   # 计算当前模型的 log probability
   log_prob_current = self.get_batch_log_prob(
       self.model,  # 当前模型
       latents_sequence_current,
       ...
   )
   ```

2. **参考模型计算**（第608-633行）：
   ```python
   # 使用参考模型计算 log probability（同一轨迹）
   log_prob_ref = self.get_batch_log_prob(
       self.ref_model,  # 参考模型（冻结）
       latents_sequence_current,  # 使用相同的轨迹
       ...
   )
   ```

3. **Ratio 计算**（第401-423行，在 `compute_grpo_loss` 中）：
   ```python
   log_ratio = log_prob_current - log_prob_ref  # [B*G]
   ratio = torch.exp(log_ratio)
   ```

4. **损失计算和更新**（第441-807行）：
   ```python
   # 计算损失
   loss = -(policy_loss.mean() - self.kl_penalty * kl.mean())
   
   # 立即反向传播和更新
   self.optimizer.zero_grad()
   loss.backward()
   self.optimizer.step()  # 模型参数立即更新
   ```

### 关键特征

- ✅ **每个 batch 只更新一次**：调用 `trainer.step(batch)` 后，模型参数立即更新
- ✅ **Ratio 不是 1.0**：`ratio = exp(log_prob_current - log_prob_ref)`，其中：
  - `log_prob_current` 是当前模型在采样轨迹上的 log prob
  - `log_prob_ref` 是参考模型在**同一个轨迹**上的 log prob
  - 两者不同，所以 `ratio ≠ 1.0`
- ❌ **没有保存 old_log_prob**：没有在采样时保存初始的 `log_prob_current`
- ❌ **没有多轮循环**：每个 batch 只执行一次前向、反向、更新

### 与标准 PPO 的区别

标准 PPO 的问题（单轮更新）：
- 在第一次更新时，`old_log_prob = new_log_prob`（使用同一个模型）
- 导致 `ratio = 1.0`，PPO clipping 无效

当前 GRPO 实现的情况：
- **Ratio 不是 1.0**：因为我们用的是 `log_prob_current - log_prob_ref`，不是 `new_log_prob - old_log_prob`
- **但仍然是单轮更新**：每个 batch 只使用一次，没有多轮优化

---

## 二、多轮 GRPO 更新（Multi-Epoch PPO）

### 核心改进

根据提供的资料，多轮 GRPO 应该：

1. **在采样时保存 `old_log_prob`**：
   ```python
   # 第一次采样（使用当前模型参数）
   old_log_prob = self.get_batch_log_prob(self.model, ...)  # 保存为固定值
   motions = self.sample(...)  # 保存 motions 和轨迹
   ```

2. **循环 K 次更新**（K epochs）：
   ```python
   for epoch in range(K_epochs):  # 例如 K=3
       # 使用更新后的模型重新计算 log prob（不重新采样）
       new_log_prob = self.get_batch_log_prob(
           self.model,  # 模型参数已经更新
           latents_sequence,  # 使用相同的轨迹（不重新采样）
           ...
       )
       
       # 计算 ratio（相对于初始 old_log_prob）
       ratio = exp(new_log_prob - old_log_prob)  # 注意：用 old_log_prob
       
       # 计算损失和更新
       loss = compute_grpo_loss(ratio, advantages, ...)
       optimizer.step()
       
       # 检查 KL 散度是否超过阈值，提前终止
       kl = compute_kl_divergence(new_log_prob, log_prob_ref)
       if kl.mean() > kl_threshold:
           break  # 提前终止
   ```

### 优势

1. **提高样本效率**：每个 batch 的数据被使用 K 次（K epochs）
2. **PPO clipping 生效**：
   - 第 1 轮：`ratio = 1.0`（如果使用 `new_log_prob - old_log_prob`）
   - 第 2 轮及以后：`ratio ≠ 1.0`，clipping 开始生效
3. **KL 散度提前终止**：如果 KL 散度超过阈值，提前终止更新，防止模型偏离参考模型太远

---

## 三、当前实现的优缺点

### 优点

1. ✅ **Ratio 计算正确**：使用 `log_prob_current - log_prob_ref` 确保了 ratio 不是 1.0
2. ✅ **实现简单**：代码清晰，易于理解和调试
3. ✅ **内存效率高**：不需要保存多轮数据

### 缺点

1. ❌ **样本效率低**：每个 batch 的数据只使用一次
2. ❌ **PPO clipping 可能不够有效**：
   - 虽然 `ratio ≠ 1.0`，但只更新一次，clipping 的影响有限
   - 多轮更新可以让 clipping 在后续轮次中发挥作用
3. ❌ **没有 KL 散度提前终止机制**：无法在 KL 散度过大时提前停止更新
4. ❌ **可能不稳定**：单次更新可能导致模型参数变化过大

---

## 四、建议改进方案

### 方案 1：标准多轮更新（推荐）

**修改 `step()` 方法**：

```python
def step(self, batch, num_epochs=3, kl_threshold=0.01):
    """
    执行一步 GRPO 训练（多轮更新）
    
    参数:
        batch: 批次数据
        num_epochs: 每个 batch 的更新轮数（默认 3）
        kl_threshold: KL 散度阈值，超过则提前终止（默认 0.01）
    """
    # 1. 采样阶段（只执行一次）
    motions, latents_sequence, timesteps_sequence = self._sample_batch(batch)
    
    # 保存 old_log_prob（固定值，不随模型更新而改变）
    with torch.no_grad():
        old_log_prob = self.get_batch_log_prob(
            self.model,  # 使用当前模型参数
            latents_sequence,
            timesteps_sequence,
            ...
        )
    
    # 计算奖励和优势（只计算一次）
    rewards = self.reward_fn(motions, prompts)
    advantages = self.compute_group_advantages(rewards)
    
    # 计算参考模型的 log prob（只计算一次）
    with torch.no_grad():
        log_prob_ref = self.get_batch_log_prob(
            self.ref_model,
            latents_sequence,
            timesteps_sequence,
            ...
        )
    
    # 2. 多轮更新阶段
    total_loss = 0.0
    for epoch in range(num_epochs):
        # 重新计算 new_log_prob（使用更新后的模型）
        new_log_prob = self.get_batch_log_prob(
            self.model,  # 模型参数可能已经更新
            latents_sequence,  # 使用相同的轨迹（不重新采样）
            timesteps_sequence,
            ...
        )
        
        # 计算 ratio（相对于初始 old_log_prob）
        ratio = torch.exp(new_log_prob - old_log_prob)
        
        # 计算 KL 散度
        kl = self.compute_kl_penalty(new_log_prob, log_prob_ref)
        
        # 检查 KL 散度阈值，提前终止
        if kl.mean() > kl_threshold:
            print(f"警告: KL 散度超过阈值 ({kl.mean():.4f} > {kl_threshold})，提前终止更新")
            break
        
        # 计算损失
        loss, stats = self.compute_grpo_loss(ratio, advantages, kl)
        
        # 反向传播和更新
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        total_loss += loss.item()
        
        # 更新 old_log_prob 为当前 new_log_prob（可选：用于下一轮）
        # 或者保持 old_log_prob 不变（推荐：始终相对于初始采样）
    
    return stats
```

### 方案 2：渐进式 old_log_prob 更新

在每轮更新后，更新 `old_log_prob = new_log_prob`，这样每轮都是相对于上一轮的变化：

```python
for epoch in range(num_epochs):
    new_log_prob = self.get_batch_log_prob(...)
    ratio = torch.exp(new_log_prob - old_log_prob)  # 相对于上一轮
    # ... 更新模型 ...
    old_log_prob = new_log_prob  # 更新 old_log_prob
```

### 方案 3：混合方案（当前实现 + 多轮选项）

保留当前实现作为默认（单轮），添加可选的 `num_epochs` 参数：

```python
def step(self, batch, num_epochs=1, kl_threshold=0.01):
    """
    执行一步 GRPO 训练
    
    参数:
        num_epochs: 每个 batch 的更新轮数（默认 1，即单轮更新）
    """
    if num_epochs == 1:
        # 使用当前的单轮更新逻辑
        return self._step_single_epoch(batch)
    else:
        # 使用多轮更新逻辑
        return self._step_multi_epoch(batch, num_epochs, kl_threshold)
```

---

## 五、总结

### 当前实现状态

- **类型**：单轮更新（Single-Epoch）
- **Ratio 计算**：使用 `log_prob_current - log_prob_ref`，不是标准的 `new_log_prob - old_log_prob`
- **样本效率**：每个 batch 只使用一次
- **PPO clipping**：虽然 ratio ≠ 1.0，但只更新一次，clipping 的影响有限

### 建议

1. **如果追求更好的性能**：实现多轮更新（方案 1）
   - 提高样本效率
   - 让 PPO clipping 真正生效
   - 添加 KL 散度提前终止机制

2. **如果追求简单和稳定**：保持当前实现
   - 代码简单，易于调试
   - 当前实现已经能工作（ratio ≠ 1.0）

3. **折中方案**：添加 `num_epochs` 参数，默认 1，可选 3-5
   - 用户可以根据需要选择单轮或多轮
   - 向后兼容

