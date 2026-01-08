# 代码修复总结

根据实验清单检查报告（排除 MotionCLIP 评估器问题），已对以下问题进行了修复和改进：

**修复日期：** 2024年
**修复范围：** 组采样、数据加载、Logic-Acc 指标、实时绘制功能

## ✅ 已修复的问题

### 1. 组采样问题（已澄清）

**问题描述：**
- 原代码注释暗示所有样本使用相同噪声，但实际上 `torch.randn(*shape)` 已经为每个样本独立生成不同的噪声

**修复内容：**
- 更新了注释，明确说明 `torch.randn` 会为每个样本生成不同的噪声
- 位置：`model/GRPO/grpo_trainer.py:556-558`

**说明：**
- `torch.randn(*shape)` 其中 `shape = [B*G, C, H, W]`，会生成 `B*G` 个独立的随机噪声
- 每个样本的噪声都是不同的，这确保了组内多样性
- 参考模型使用相同的轨迹计算 log prob，这是正确的（用于 KL 散度计算）

---

### 2. 数据加载禁用随机 Crop

**问题描述：**
- 数据加载器中存在随机 Crop 操作，当使用固定的 `durations` 时，随机偏移会导致分段奖励计算错误

**修复内容：**

1. **添加命令行参数** (`utils/parser_util.py:443-445`):
   ```python
   grpo_group.add_argument('--disable_random_crop', action='store_true',
                           help='Disable random crop and offset augmentation in dataset loading. '
                                'Required when using fixed durations for composite prompts.')
   ```

2. **修改数据加载函数** (`data_loaders/get_data.py:36-48`):
   - `get_dataset()` 和 `get_dataset_loader()` 添加了 `disable_random_crop` 参数
   - 参数会传递给 `HumanML3D` 数据集类

3. **修改数据集类** (`data_loaders/humanml/data/dataset.py:778-779`):
   - `HumanML3D.__init__()` 接收 `disable_random_crop` 参数
   - 设置 `opt.disable_offset_aug = disable_random_crop or ...`

4. **修改数据加载逻辑** (`data_loaders/humanml/data/dataset.py:343-363`):
   - 当 `disable_offset_aug=True` 时：
     - 跳过随机选择 `coin2`（固定使用 'single'）
     - 固定起始位置 `idx = 0`（不使用随机偏移）

5. **更新训练脚本** (`train/train_grpo.py:217-223`):
   - 从命令行参数读取 `disable_random_crop`
   - 传递给 `get_dataset_loader()`

**使用方法：**
```bash
python train/train_grpo.py \
    --disable_random_crop \
    --use_dense_reward \
    ...
```

---

### 3. Logic-Acc 指标计算

**问题描述：**
- 缺少 Logic-Acc 指标：对于第 k 个片段，检查 `Sim(hat{y}_{T_k}, x_k)` 是否是该行相似度矩阵中的最大值

**修复内容：**

1. **添加 `compute_logic_accuracy` 方法** (`model/GRPO/reward_model.py:618-720`):
   - 计算每个片段的逻辑准确率
   - 检查每个动作片段与其对应文本的相似度是否最高
   - 返回整体逻辑准确率和每个片段的准确率

**方法签名：**
```python
def compute_logic_accuracy(
    self,
    motions: torch.Tensor,
    text_lists: List[List[str]],
    segments: Optional[List[List[Tuple[int, int]]]] = None,
    durations: Optional[List[List[float]]] = None,
) -> Dict[str, float]:
    """
    返回:
        {
            'logic_acc': 整体逻辑准确率,
            'avg_segment_acc': 平均片段准确率,
            'logic_acc_per_segment': 每个样本的片段准确率列表
        }
    """
```

**使用方法：**
```python
# 在评估时调用
logic_acc_dict = reward_fn.compute_logic_accuracy(
    motions=motions,
    text_lists=text_lists,
    durations=durations,
)
print(f"Logic Accuracy: {logic_acc_dict['logic_acc']:.4f}")
```

---

### 4. 实时绘制 R_pos vs R_neg 曲线

**问题描述：**
- 缺少实时监控 R_pos 和 R_neg 的功能，无法观察负向惩罚是否下降

**修复内容：**

1. **修改奖励函数返回值** (`model/GRPO/reward_model.py:789-797`, `reward_model_tmr.py:924-932`):
   - `MatchingScoreReward.__call__()` 现在返回 `(R_total, components)` 元组
   - `components` 包含 `R_pos`, `R_neg`, `R_sem`, `R_phy`

2. **修改训练器处理返回值** (`model/GRPO/grpo_trainer.py:641-645, 689-707`):
   - 检查奖励函数返回值是单个张量还是元组
   - 保存组件信息到 `self._last_reward_components`
   - 在 stats 中添加 `R_pos` 和 `R_neg`

3. **修改训练循环收集数据** (`train/train_grpo.py:372-375, 421-430`):
   - 添加 `R_pos_values` 和 `R_neg_values` 列表
   - 从 stats 中收集 `R_pos` 和 `R_neg` 值

4. **更新绘制函数** (`train/train_grpo.py:522-583`):
   - `plot_training_curves()` 添加 `R_pos_values` 和 `R_neg_values` 参数
   - 创建第三个子图显示 R_pos vs R_neg 曲线
   - 每 200 step 保存 checkpoint 时自动绘制

**效果：**
- 训练过程中会实时绘制三条曲线：
  1. Loss 曲线
  2. Motion Average Score 曲线
  3. **R_pos vs R_neg 曲线**（新增）

**示例输出：**
- 图片保存在：`{save_dir}/training_curves_step_{step:09d}.png`
- 最新版本：`{save_dir}/training_curves_latest.png`

---

## ⚠️ 需要用户实现的模块

### 1. 复合数据集构造模块

**状态：** 未实现（需要用户创建）

**建议实现：**
创建一个脚本 `scripts/construct_composite_dataset.py`，功能包括：

1. **生成复合数据**：
   - 从 HumanML3D 数据集中选择 K=3/4/5 个动作
   - 拼接文本描述
   - 计算每个片段的 duration（秒）和对应帧数

2. **保存格式**：
   ```python
   {
       'composite_prompt': "First walk forward, then jump up, finally land",
       'sub_prompts': [["walk forward"], ["jump up"], ["land"]],
       'durations': [3.0, 2.0, 2.5],  # 秒
       'durations_frames': [60, 40, 50],  # 帧数（duration * fps）
       'source_ids': ['000001', '000002', '000003'],
       'B_matrix': [[1.0, 0.3, 0.2], [0.3, 1.0, 0.4], [0.2, 0.4, 1.0]],  # 预计算的文本相似度矩阵
   }
   ```

3. **长度合法性检查**：
   ```python
   L_sum = sum(durations_frames)
   if abs(L_sum - 196) > 20:  # 允许 ±20 帧误差
       # 警告或跳过
   ```

---

### 2. B_matrix 预计算优化

**当前状态：**
- `B_matrix` 在运行时计算（`reward_model.py:545`）
- 每次调用 `compute_semantic_reward` 都会重新计算

**建议优化：**
- 在数据构造阶段预计算 `B_matrix`，保存到 `.npy` 文件
- 在训练时从文件加载，避免重复计算
- 确保所有实验使用相同的基准矩阵

**实现位置：**
- 数据构造脚本中计算并保存
- 训练时通过 `durations` 参数传递（或从数据文件加载）

---

## 📝 使用说明

### 训练命令示例（使用修复后的功能）

```bash
# 使用 Segment-Dense 模式，禁用随机 Crop
python train/train_grpo.py \
    --model_path ./path/to/pretrained/model.pt \
    --save_dir ./outputs/grpo_experiment \
    --reward_model_type tmr \
    --use_dense_reward \
    --disable_random_crop \
    --fps 20.0 \
    --alpha 0.5 \
    --beta_s 1.0 \
    --beta_p 0.1 \
    --lambda_skate 1.0 \
    --lambda_jerk 1.0 \
    --num_steps 1000 \
    --save_interval 200 \
    --device auto
```

### 关键参数说明

- `--disable_random_crop`: **必须**在使用固定 durations 时启用
- `--use_dense_reward`: 启用 Segment-Dense 模式
- `--fps 20.0`: HumanML 数据集帧率（KIT 使用 12.5）
- `--save_interval 200`: 每 200 step 保存 checkpoint 并绘制曲线

---

## 🔍 验证检查点

### 1. 验证组采样多样性
```python
# 在训练器中，检查生成的 motions 是否不同
motions = current_result['samples']  # [B*G, C, H, W]
# 同一组内的 G 个样本应该不同（由于不同的噪声）
```

### 2. 验证随机 Crop 已禁用
```python
# 在数据加载时，检查 idx 是否为 0
if opt.disable_offset_aug:
    assert idx == 0, "随机 Crop 未正确禁用"
```

### 3. 验证 R_pos 和 R_neg 收集
```python
# 在训练循环中，检查 stats 是否包含 R_pos 和 R_neg
assert 'R_pos' in stats, "R_pos 未收集"
assert 'R_neg' in stats, "R_neg 未收集"
```

### 4. 验证 Logic-Acc 计算
```python
# 在评估时调用
logic_acc = reward_fn.compute_logic_accuracy(motions, text_lists, durations=durations)
assert 'logic_acc' in logic_acc, "Logic-Acc 计算失败"
```

---

## 📊 预期效果

修复后，训练过程应该：

1. ✅ **组内多样性**：每个 prompt 的 G 个样本应该不同（由于不同的噪声）
2. ✅ **固定分段**：使用 `--disable_random_crop` 时，动作分段位置固定
3. ✅ **实时监控**：每 200 step 自动绘制 R_pos vs R_neg 曲线
4. ✅ **逻辑评估**：可以使用 `compute_logic_accuracy()` 评估逻辑准确性

---

## ⚠️ 注意事项

1. **向后兼容性**：
   - 奖励函数现在返回元组 `(rewards, components)`
   - 训练器已处理这种情况，但其他调用奖励函数的代码可能需要更新

2. **性能影响**：
   - Logic-Acc 计算需要额外的前向传播，建议仅在评估时使用
   - R_pos/R_neg 收集不会显著影响训练速度

3. **数据格式**：
   - 使用 `--disable_random_crop` 时，确保数据集的 motion 长度足够
   - 如果 motion 长度不足，可能需要 padding 或使用 `fixed_len` 参数

---

## 🎯 下一步建议

1. **创建复合数据集构造脚本**（高优先级）
2. **实现 B_matrix 预计算**（中优先级）
3. **添加更多评估指标**（可选）
4. **优化性能**（如果训练速度慢）

