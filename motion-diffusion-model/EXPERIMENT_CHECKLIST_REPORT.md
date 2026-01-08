# 实验清单检查报告

## 一、复合数据集构造模块 (Data Construction)

### ❌ 1. 确定性切片记录
**状态：未实现**

**问题：**
- 代码中**没有找到**保存以下字段的代码：
  - `composite_prompt`: 拼接后的长句子
  - `sub_prompts`: 原始的三个短文本描述
  - `durations`: 对应的帧数数组（上帝视角）
  - `source_ids`: 原始 HumanML3D 的 ID

**建议：**
- 需要创建一个数据构造脚本，生成包含上述字段的 `.npy` 索引文件
- 格式示例：
```python
{
    'composite_prompt': "First walk forward, then jump up, finally land",
    'sub_prompts': ["walk forward", "jump up", "land"],
    'durations': [60, 45, 50],  # 帧数
    'source_ids': ['000001', '000002', '000003']
}
```

### ❌ 2. 语义基准矩阵 $B$ 预计算
**状态：未实现**

**问题：**
- 代码中 `B_matrix` 是在**运行时**计算的（`reward_model.py:545`），而不是在数据构造时预计算
- 当前实现：每次调用 `compute_semantic_reward` 时都会重新计算文本间相似度

**当前代码位置：**
```python
# reward_model.py:545
B_matrix = torch.mm(text_embs_norm, text_embs_norm.t())  # [K, K]
```

**建议：**
- 在数据构造阶段预计算 `B_matrix`，保存到 `.npy` 文件中
- 这样可以：
  1. 避免重复计算
  2. 确保所有实验使用相同的基准矩阵
  3. 提高训练效率

### ⚠️ 3. 长度合法性检查
**状态：部分实现**

**问题：**
- 代码中有 `max_motion_length` 参数（默认 196），但没有检查 K 个动作的总长度是否接近 196
- 当前实现：如果超过 `max_motion_length`，会截断（`_truncate_motions`），但不会检查总长度是否合理

**建议：**
- 在数据构造时添加检查：
```python
L_sum = sum(durations)  # 总帧数
if abs(L_sum - 196) > 20:  # 允许 ±20 帧的误差
    # 警告或跳过该样本
```

---

## 二、奖励函数实现模块 (Reward Engineering)

### ✅ 1. $R_{pos}$ (正向匹配)
**状态：已实现**

**检查结果：**
- ✅ 代码按 `durations` 对生成的 $\hat{y}$ 进行了切片（`reward_model.py:463-487`）
- ✅ `TMR_Motion_Encoder` 和 `TMR_Text_Encoder` 处于 `eval()` 模式（通过 `with torch.no_grad()` 确保）
- ✅ 计算逻辑正确：对每个片段计算相似度，然后取平均（`reward_model.py:521`）

**代码位置：**
- 切片逻辑：`reward_model.py:463-487`
- 相似度计算：`reward_model.py:495-522`

### ⚠️ 2. $R_{neg}$ (自适应负向判别)
**状态：部分实现，需要改进**

**检查结果：**

**✅ 逻辑检查：**
- 代码实现了 `F.relu(s_kj - B_matrix[k])`（`reward_model.py:572`）
- 使用了 `torch.clamp` 的等价形式 `F.relu`，符合要求

**⚠️ 分母检查：**
- 代码中对所有 K 个分段取了平均（`reward_model.py:577`）
- 但需要确认公式：应该是 $\frac{1}{K} \sum_{k=1}^{K} \max_{j \neq k} [\max(0, s_{k,j} - B_{k,j})]$
- 当前实现：`R_neg_b = torch.stack(neg_penalties).mean()` ✅

**❌ 问题：**
- `B_matrix` 是在运行时计算的，不是预计算的
- 需要确保 `B_matrix` 的对角线元素被正确处理（不应该计算自己与自己的相似度）

**代码位置：**
- `reward_model.py:524-581`

### ⚠️ 3. $R_{phy}$ (全局物理正则)
**状态：部分实现，需要验证**

**检查结果：**

**⚠️ Jerk Penalty：**
- 代码计算了加速度变化（`reward_model.py:377-380`）
- **问题**：计算跨越了**整个序列**，但没有特别处理拼接点
- 建议：确保 Jerk 计算跨越拼接点，保证全序列平滑

**⚠️ Foot Skating：**
- 代码尝试提取脚部接触信息（`reward_model.py:308-316`）
- **问题**：假设最后 4 维是脚部接触信息，需要验证 HumanML 数据格式是否正确
- 当前实现：`foot_contact = motions_flat[:, :, -4:-2]`（提取最后 4 维中的前 2 维）

**代码位置：**
- `reward_model.py:321-390`

### ✅ 4. 奖励合成
**状态：已实现**

**检查结果：**
- ✅ 公式正确：`R_sem = R_pos - alpha * R_neg`（`reward_model.py:587`）
- ✅ 最终合成：`R_total = beta_s * R_sem + beta_p * R_phy`（`reward_model.py:631-637`）

---

## 三、GRPO 训练循环模块 (Training Loop)

### ❌ 1. 组采样 (Group Sampling)
**状态：未正确实现**

**问题：**
- **关键问题**：代码中使用**相同的噪声**进行采样（`grpo_trainer.py:558`）
```python
noise = torch.randn(*shape, device=self.device)  # 所有样本使用相同噪声
```
- 这导致同一组内的 G 个样本**完全相同**，无法产生多样性

**建议：**
- 应该为每个样本生成不同的噪声：
```python
# 为每个样本生成不同的噪声
noises = []
for i in range(expanded_batch_size):
    noise = torch.randn(shape[1:], device=self.device)  # 不同的噪声
    noises.append(noise)
noise = torch.stack(noises, dim=0)
```

**代码位置：**
- `grpo_trainer.py:556-568`

### ✅ 2. 相对优势计算 (Advantage Estimation)
**状态：已实现**

**检查结果：**
- ✅ Advantage $A_i$ 是在**当前 Prompt 的组内**进行标准化的（`grpo_trainer.py:649-650`）
- ✅ 实现：`compute_group_advantages` 方法将 rewards reshape 为 `[B, G]`，然后在组内计算均值和标准差

**代码位置：**
- `grpo_trainer.py:649-650`
- `grpo_trainer.py:700-730`（`compute_group_advantages` 方法）

### ⚠️ 3. 异构梯度估计
**状态：需要验证**

**检查结果：**
- ✅ **MDM 分支**：代码使用 DDPO 风格的 log prob 计算（`grpo_trainer.py:574-580`）
- ⚠️ **T2M-GPT 分支**：代码中没有找到 T2M-GPT 相关的实现，可能需要单独处理

**代码位置：**
- `grpo_trainer.py:574-580`（log prob 计算）

### ✅ 4. 参考模型约束 ($\pi_{ref}$)
**状态：已实现**

**检查结果：**
- ✅ 参考模型参数被冻结：`param.requires_grad = False`（`grpo_trainer.py:76-78`）
- ✅ 参考模型设置为 `eval()` 模式（`grpo_trainer.py:78`）
- ✅ 使用 `torch.no_grad()` 确保不计算梯度（`grpo_trainer.py:606`）

**代码位置：**
- `grpo_trainer.py:75-78`（初始化时冻结）
- `grpo_trainer.py:606`（使用时禁用梯度）

---

## 四、评估与评估对齐模块 (Evaluation Protocol)

### ❌ 1. 跨模型评估 (Cross-Evaluator)
**状态：未实现**

**问题：**
- **重要**：代码中**没有找到**使用 MotionCLIP 进行 R-Precision 评估的代码
- 当前评估使用 `EvaluatorMDMWrapper`（`eval_humanml.py`），这是 MDM 的评估器，不是 MotionCLIP

**建议：**
- 需要实现或集成 MotionCLIP 评估器
- 训练时使用 TMR 做 Reward，测试时使用 MotionCLIP 跑 R-Precision 指标

**代码位置：**
- `eval/eval_humanml.py:20-332`

### ❌ 2. Logic-Acc 指标计算
**状态：未实现**

**问题：**
- 代码中**没有找到** Logic-Acc 指标的计算
- 需要实现：对于第 $k$ 个片段，检查 $\text{Sim}(\hat{y}_{T_k}, x_k)$ 是否是该行相似度矩阵中的最大值

**建议：**
- 实现 Logic-Acc 计算函数：
```python
def compute_logic_acc(motion_segments, text_lists, evaluator):
    # 对每个样本的每个片段
    # 计算 motion_segment_k 与所有 text_j 的相似度
    # 检查是否 text_k 的相似度最高
    ...
```

---

## 特别提醒检查

### ⚠️ 1. 关于 $B_{k,j}$ 的零惩罚阈值
**状态：已实现**

**检查结果：**
- ✅ 代码使用了 `F.relu(s_kj - B_matrix[k])`（`reward_model.py:572`）
- ✅ 这等价于 `torch.clamp(s_kj - B_kj, min=0)`，符合要求

### ❌ 2. 关于数据加载
**状态：未实现**

**问题：**
- 数据加载器中**仍然有随机 Crop**（`dataset.py:176, 183, 197, 360`）
- 如果 `durations` 是固定的，一旦模型输入偏移了 1 帧，所有的分段奖励都会出错

**建议：**
- 在数据加载时设置 `disable_offset_aug=True`
- 或者为复合数据集创建专门的数据加载器，禁用所有随机操作

**代码位置：**
- `data_loaders/humanml/data/dataset.py:176, 183, 197, 360`

### ⚠️ 3. 关于训练步数
**状态：部分实现**

**检查结果：**
- ✅ 代码中有保存 checkpoint 的逻辑（`train_grpo.py`）
- ⚠️ 代码中**没有找到**实时绘制 $R_{pos}$ vs $R_{neg}$ 曲线的代码

**建议：**
- 添加实时监控和绘制功能
- 每 200 step 保存 checkpoint 并记录 $R_{pos}$ 和 $R_{neg}$ 的值

---

## 总结

### ✅ 已实现的功能
1. $R_{pos}$ 计算（使用 durations 切片）
2. $R_{neg}$ 基本逻辑（ReLU 和平均）
3. $R_{phy}$ 基本计算（Jerk 和 Foot Skating）
4. 奖励合成公式
5. 相对优势计算（组内标准化）
6. 参考模型冻结

### ⚠️ 需要改进的功能
1. $B_{matrix}$ 预计算（当前是运行时计算）
2. Jerk 计算需要确保跨越拼接点
3. Foot Skating 需要验证数据格式
4. 组采样需要使用不同的噪声种子
5. 添加 Logic-Acc 指标
6. 添加 MotionCLIP 评估器

### ❌ 缺失的功能
1. 复合数据集构造模块（composite_prompt, sub_prompts, durations, source_ids）
2. $B_{matrix}$ 预计算和保存
3. 长度合法性检查
4. 数据加载时禁用随机 Crop
5. Logic-Acc 指标计算
6. 实时绘制 $R_{pos}$ vs $R_{neg}$ 曲线

---

## 优先级建议

### 高优先级（必须实现）
1. **组采样使用不同噪声种子**（影响训练效果）
2. **数据加载禁用随机 Crop**（影响奖励计算准确性）
3. **复合数据集构造模块**（实验基础）

### 中优先级（建议实现）
1. **$B_{matrix}$ 预计算**（提高效率，确保一致性）
2. **Logic-Acc 指标**（证明逻辑提升）

### 低优先级（可选）
1. **实时绘制曲线**（监控工具）
2. **长度合法性检查**（数据质量保证）

