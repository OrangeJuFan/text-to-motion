# 多轮 GRPO 更新实现说明

## 一、实现概述

已成功实现多轮 GRPO 更新（Multi-Epoch PPO）功能，支持：
1. **保存 old_log_prob**：在采样时记录初始 log prob（固定不变）
2. **循环 K 次更新**：每个 batch 的数据被使用 K 次（K epochs）进行更新
3. **KL 散度提前终止**：如果 KL 散度超过阈值，提前终止更新
4. **向后兼容**：默认 `num_epochs=1`，保持单轮更新行为

## 二、核心改进

### 1. 单轮更新 vs 多轮更新

**单轮更新（num_epochs=1）**：
- 每个 batch 只使用一次
- `ratio = exp(log_prob_current - log_prob_ref)`
- 模型参数立即更新

**多轮更新（num_epochs > 1）**：
- 每个 batch 被使用 K 次（K epochs）
- 第 1 轮：`ratio = exp(new_log_prob - old_log_prob) ≈ 1.0`
- 第 2+ 轮：`ratio ≠ 1.0`，PPO clipping 开始生效
- 如果 KL 散度超过阈值，提前终止

### 2. 关键实现细节

#### 保存 old_log_prob
```python
# 在采样时保存 old_log_prob（固定不变）
old_log_prob = log_prob_current.detach().clone()  # 固定值，不随模型更新而改变
```

#### 多轮更新循环
```python
for epoch in range(self.num_epochs):
    # 重新计算 new_log_prob（使用更新后的模型，但使用相同的轨迹）
    new_log_prob = self.get_batch_log_prob(
        self.model,
        latents_sequence_current,  # 使用相同的轨迹（不重新采样）
        timesteps_sequence,
        model_kwargs,
    )
    
    # 计算 KL 散度，检查是否超过阈值
    kl = self.compute_kl_penalty(new_log_prob, log_prob_ref)
    if kl.mean() > self.kl_threshold:
        print(f"警告: KL 散度超过阈值，提前终止更新")
        break
    
    # 计算损失（使用 old_log_prob 计算 ratio）
    loss, epoch_stats = self.compute_grpo_loss(
        new_log_prob,
        log_prob_ref,
        advantages,
        old_log_prob=old_log_prob,  # 使用初始采样时的 log prob
    )
    
    # 更新模型参数
    optimizer.step()
```

#### compute_grpo_loss 修改
```python
def compute_grpo_loss(
    self,
    log_prob_current: torch.Tensor,
    log_prob_ref: torch.Tensor,
    advantages: torch.Tensor,
    old_log_prob: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    # 如果提供了 old_log_prob（多轮更新），则使用 old_log_prob 计算 ratio
    if old_log_prob is not None:
        log_ratio = log_prob_current - old_log_prob  # 多轮更新
    else:
        log_ratio = log_prob_current - log_prob_ref  # 单轮更新
    # ...
```

## 三、修改的文件

### 1. `model/GRPO/grpo_trainer.py`

**修改内容**：
- `__init__` 方法：添加 `num_epochs` 和 `kl_threshold` 参数
- `compute_grpo_loss` 方法：添加 `old_log_prob` 参数支持
- `step` 方法：实现多轮更新逻辑（if num_epochs == 1 使用单轮更新，否则使用多轮更新）

### 2. `utils/parser_util.py`

**修改内容**：
- 添加 `--num_epochs` 参数（默认 1，可选 3-5）
- 添加 `--kl_threshold` 参数（默认 0.01）

### 3. `train/train_grpo.py`

**修改内容**：
- 在 `create_grpo_trainer` 和 `create_flow_grpo_trainer` 调用中传递 `num_epochs` 和 `kl_threshold` 参数

## 四、使用方法

### 单轮更新（默认，向后兼容）
```bash
python -m train.train_grpo \
    --model_path ./save/pretrained_model/model000200000.pt \
    --save_dir ./save/grpo_finetuned \
    --group_size 4 \
    --batch_size 2 \
    --num_steps 10000 \
    --reward_type matching
    # num_epochs 默认为 1，无需指定
```

### 多轮更新（推荐）
```bash
python -m train.train_grpo \
    --model_path ./save/pretrained_model/model000200000.pt \
    --save_dir ./save/grpo_finetuned \
    --group_size 4 \
    --batch_size 2 \
    --num_steps 10000 \
    --reward_type matching \
    --num_epochs 3 \
    --kl_threshold 0.01
```

### 参数说明

- `--num_epochs`：每个 batch 的更新轮数
  - 默认：`1`（单轮更新）
  - 推荐：`3-5`（多轮更新，提高样本效率）
  
- `--kl_threshold`：KL 散度阈值（用于提前终止）
  - 默认：`0.01`
  - 如果 KL 散度超过此阈值，提前终止多轮更新，防止模型偏离参考模型太远

## 五、优势

1. **提高样本效率**：每个 batch 的数据被使用 K 次，而不是只使用一次
2. **PPO clipping 真正生效**：从第 2 轮开始，ratio ≠ 1.0，clipping 开始发挥作用
3. **KL 散度提前终止**：防止模型偏离参考模型太远，提高训练稳定性
4. **向后兼容**：默认 `num_epochs=1`，保持原有的单轮更新行为

## 六、注意事项

1. **内存使用**：多轮更新需要保留轨迹（`latents_sequence_current`, `timesteps_sequence`），因此内存使用会增加。如果内存不足，可以：
   - 减少 `batch_size`
   - 减少 `num_epochs`
   - 减少 `group_size`

2. **计算时间**：多轮更新会增加计算时间（每个 batch 需要计算 K 次 log prob 和更新 K 次）。但样本效率的提高通常可以抵消这个开销。

3. **KL 散度阈值**：如果 KL 散度经常超过阈值导致提前终止，可以：
   - 增加 `kl_threshold`（例如 0.02 或 0.05）
   - 增加 `kl_penalty`（增加 KL 惩罚权重）
   - 降低学习率

4. **Flow-GRPO**：Flow-GRPO 的参数传递已更新，但 Flow-GRPO 的多轮更新逻辑尚未实现（因为 Flow-GRPO 使用 SDE 和不同的 log prob 计算方式）。如果需要，可以后续实现。

## 七、测试建议

1. **单轮更新测试**：使用默认参数（`num_epochs=1`），确保向后兼容
2. **多轮更新测试**：使用 `--num_epochs 3 --kl_threshold 0.01`，观察：
   - 训练损失是否更稳定
   - KL 散度是否在合理范围内
   - 奖励是否提高
   - 内存使用是否可接受

## 八、参考

- 标准 PPO 多轮更新机制
- DanceGRPO 论文
- 当前实现基于用户提供的多轮 GRPO 更新资料

