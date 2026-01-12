# CUDA 内存溢出（OOM）问题解决方案

## 一、问题原因

在多轮更新（Multi-Epoch PPO）中，每个 batch 的数据被使用 K 次（K epochs），需要保留以下数据：

1. **轨迹数据**（占用最多内存）：
   - `latents_sequence_current`：完整的采样轨迹序列（包含多个时间步的状态）
   - `timesteps_sequence`：时间步序列
   - 这些数据在多轮更新中必须保留，因为需要在每一轮重新计算 log prob

2. **其他数据**：
   - `motions`：生成的动作样本 [B*G, C, H, W]
   - `old_log_prob`：初始 log prob
   - `log_prob_ref`：参考模型的 log prob
   - `advantages`：优势值
   - 模型参数和梯度

**内存使用估算**：
- 轨迹数据：每个样本的轨迹包含 ~50 个时间步，每个时间步的维度为 [B*G, 263, 1, 196]
- 如果 `batch_size=2`, `group_size=4`，则 `B*G=8`
- 单个时间步的内存：`8 * 263 * 1 * 196 * 4 bytes ≈ 1.65 MB`
- 50 个时间步：`1.65 MB * 50 ≈ 82.5 MB`
- 加上梯度、模型参数等，一个 batch 可能占用数百 MB 到数 GB

## 二、解决方案

### 方案 1：减少 batch_size（最简单有效）

```bash
# 原来
python -m train.train_grpo \
    --batch_size 2 \
    --num_epochs 3 \
    ...

# 改为
python -m train.train_grpo \
    --batch_size 1 \  # 减少到 1
    --num_epochs 3 \
    ...
```

**效果**：内存使用减少约 50%（如果 batch_size 从 2 减到 1）

### 方案 2：减少 num_epochs

```bash
# 原来
python -m train.train_grpo \
    --num_epochs 3 \
    ...

# 改为
python -m train.train_grpo \
    --num_epochs 2 \  # 减少到 2（仍然有多轮更新的好处）
    ...
```

**效果**：内存使用不变，但更新轮数减少（仍然比单轮更新好）

### 方案 3：减少 group_size

```bash
# 原来
python -m train.train_grpo \
    --group_size 4 \
    ...

# 改为
python -m train.train_grpo \
    --group_size 2 \  # 减少到 2
    ...
```

**效果**：内存使用减少约 50%（如果 group_size 从 4 减到 2）

### 方案 4：组合使用（推荐）

```bash
python -m train.train_grpo \
    --batch_size 1 \      # 从 2 减到 1
    --group_size 2 \      # 从 4 减到 2（可选）
    --num_epochs 2 \      # 从 3 减到 2（可选）
    --kl_threshold 0.01 \
    ...
```

**效果**：内存使用减少约 75%（如果 batch_size 和 group_size 都减半）

### 方案 5：使用单轮更新（如果内存仍然不足）

如果内存仍然不足，可以回到单轮更新（向后兼容）：

```bash
python -m train.train_grpo \
    --batch_size 1 \
    --num_epochs 1 \  # 单轮更新（不使用多轮更新）
    ...
```

**效果**：内存使用大幅减少（不需要保留轨迹），但失去多轮更新的好处

### 方案 6：清理 PyTorch 缓存（临时方案）

如果只是偶尔出现 OOM，可以尝试：

```python
# 在训练脚本中，每个 step 后清理缓存
import torch
torch.cuda.empty_cache()
```

或者在训练循环中添加：

```python
if step % 10 == 0:
    torch.cuda.empty_cache()
```

## 三、内存优化代码改进（可选）

如果上述方案仍然不够，可以考虑代码层面的优化：

### 优化 1：在每轮更新后清理中间变量

在多轮更新的循环中，每次计算完 log prob 后，可以清理一些中间变量：

```python
for epoch in range(self.num_epochs):
    new_log_prob = self.get_batch_log_prob(...)
    # ... 计算损失和更新 ...
    
    # 清理中间变量（但保留轨迹）
    if epoch < self.num_epochs - 1:  # 不是最后一轮
        del loss, epoch_stats
        torch.cuda.empty_cache()  # 清理缓存
```

### 优化 2：使用梯度累积（如果 batch_size=1 仍然不够）

如果 `batch_size=1` 仍然 OOM，可以使用梯度累积来模拟更大的 batch：

```python
# 累积多个小 batch 的梯度，然后一起更新
accumulation_steps = 4
for step in range(num_steps):
    total_loss = 0.0
    for acc_step in range(accumulation_steps):
        batch = next(data_loader)
        loss = compute_loss(batch)
        loss = loss / accumulation_steps  # 平均
        loss.backward()  # 累积梯度
        total_loss += loss.item()
    
    optimizer.step()  # 更新一次
    optimizer.zero_grad()
```

### 优化 3：使用混合精度训练（FP16）

使用 FP16 可以减少内存使用约 50%：

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for epoch in range(self.num_epochs):
    with autocast():  # 使用 FP16
        new_log_prob = self.get_batch_log_prob(...)
        loss, epoch_stats = self.compute_grpo_loss(...)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## 四、推荐配置

### 内存充足（24GB+ GPU）

```bash
python -m train.train_grpo \
    --batch_size 2 \
    --group_size 4 \
    --num_epochs 3 \
    --kl_threshold 0.01 \
    ...
```

### 内存中等（12-24GB GPU）

```bash
python -m train.train_grpo \
    --batch_size 1 \
    --group_size 4 \
    --num_epochs 3 \
    --kl_threshold 0.01 \
    ...
```

### 内存较少（8-12GB GPU）

```bash
python -m train.train_grpo \
    --batch_size 1 \
    --group_size 2 \
    --num_epochs 2 \
    --kl_threshold 0.01 \
    ...
```

### 内存很少（<8GB GPU）

```bash
python -m train.train_grpo \
    --batch_size 1 \
    --group_size 2 \
    --num_epochs 1 \  # 使用单轮更新
    ...
```

## 五、监控内存使用

可以添加代码来监控内存使用：

```python
import torch

def print_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        print(f"GPU 内存: 已分配 {allocated:.2f} GB, 已保留 {reserved:.2f} GB")

# 在训练循环中使用
for step in range(num_steps):
    print_memory_usage()
    # ... 训练代码 ...
```

## 六、总结

1. **最有效**：减少 `batch_size` 和 `group_size`
2. **次有效**：减少 `num_epochs`
3. **最后手段**：使用单轮更新（`num_epochs=1`）
4. **代码优化**：清理缓存、梯度累积、混合精度训练（需要修改代码）

建议从方案 4（组合使用）开始，如果仍然 OOM，再考虑方案 5（单轮更新）。

