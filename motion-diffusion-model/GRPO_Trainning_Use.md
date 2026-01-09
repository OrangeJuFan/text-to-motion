# GRPO 训练使用说明

本文档详细说明如何使用 GRPO (Group Relative Policy Optimization) 进行文本生成动作模型的强化学习微调。

## 目录

1. [快速开始](#快速开始)
2. [GRPO 核心参数](#grpo-核心参数)
3. [奖励模型参数](#奖励模型参数)
4. [奖励函数高级参数](#奖励函数高级参数)
5. [复合数据集参数](#复合数据集参数)
6. [Flow-GRPO 参数](#flow-grpo-参数)
7. [训练示例](#训练示例)
8. [参数说明表格](#参数说明表格)

## 快速开始

### 使用训练脚本

- 不挂lora的状态下

  ```bash
  python -m train.train_grpo --model_path ./save/official_humanml_enc_512_50steps/model000750000.pt --save_dir ./save/grpo_finetuned --dataset humanml --batch_size 1 --group_size 4 --num_steps 10000 --learning_rate 1e-6 --reward_model_type mdm  --reward_type matching --device 0 
  ```

- 挂lora的状态

  ```bash
  python -m train.train_grpo --model_path ./save/pretrained_model/model000200000.pt --save_dir ./save/grpo_finetuned --dataset humanml --batch_size 1 --group_size 4 --num_steps 10000 --learning_rate 5e-7 --use_lora --lora_r 8 lora_alpha 16 --reward_model_type mdm \ --reward_type matching --device 0 
  ```

- 使用mdm评估函数作为reward_model

  ```bash
  # 使用匹配分数（默认）
  python -m train.train_grpo --model_path ./save/official_humanml_enc_512_50steps/model000750000.pt --save_dir ./save/grpo_finetuned_humanml_enc_512_50steps_750000_mdm_matching --dataset humanml --batch_size 1 --group_size 4 --learning_rate 1e-6 --num_steps 15000 --reward_model_type mdm --reward_type matching --device 3
  
  # 使用 R-Precision
  python -m train.train_grpo --model_path ./save/official_humanml_enc_512_50steps/model000750000.pt --save_dir ./save/grpo_finetuned_humanml_enc_512_50steps_750000_r_precision --dataset humanml --batch_size 1 --group_size 4 --learning_rate 1e-6 --num_steps 15000 --reward_model_type mdm --reward_type r_precision --device 4  
  
  # 使用组合奖励
  python -m train.train_grpo --model_path ./save/official_humanml_enc_512_50steps/model000750000.pt --save_dir ./save/grpo_finetuned_humanml_enc_512_50steps_750000_mdm_combined --dataset humanml --batch_size 1 --group_size 4 --learning_rate 1e-6 --num_steps 15000 --reward_model_type mdm  --reward_type combined  --device 1  
  ```

- 使用 TMR 预训练模型奖励函数

  余弦相似度

  ```bash
  python -m train.train_grpo --model_path ./save/official_humanml_enc_512_50steps/model000750000.pt --save_dir ./save/grpo_finetuned_humanml_enc_512_50steps_750000_tmr_cosine --dataset humanml  --batch_size 1 --group_size 4 --learning_rate 1e-6  --num_steps 15000 --reward_model_type tmr --reward_type cosine --tmr_text_encoder_path ./model/GRPO/tmr_weights/text_encoder.pt --tmr_motion_encoder_path ./model/GRPO/tmr_weights/motion_encoder.pt --tmr_movement_encoder_path ./model/GRPO/tmr_weights/motion_decoder.pt --device 2
  ```

  匹配分数

  ```bash
  # 使用余弦相似度 + 线性归一化
  python -m train.train_grpo --model_path ./save/official_humanml_enc_512_50steps/model000750000.pt --save_dir ./save/grpo_finetuned_humanml_enc_512_50steps_750000_tmr_matching_cosine --dataset humanml --batch_size 1 --group_size 4 --num_steps 10000 --reward_model_type tmr  --reward_type matching  --tmr_text_encoder_path ./model/GRPO/tmr_weights/text_encoder.pt --tmr_motion_encoder_path ./model/GRPO/tmr_weights/motion_encoder.pt --tmr_movement_encoder_path ./model/GRPO/tmr_weights/motion_decoder.pt  --tmr_similarity_type cosine --device 1
  
  # 使用欧氏距离 + 线性归一化
  python -m train.train_grpo --model_path ./save/official_humanml_enc_512_50steps/model000750000.pt  --save_dir ./save/grpo_finetuned_humanml_enc_512_50steps_750000_tmr_matching_euclidean_linear --dataset humanml --batch_size 1 --group_size 4 --num_steps 10000 --reward_model_type tmr --reward_type matching --tmr_text_encoder_path ./model/GRPO/tmr_weights/text_encoder.pt --tmr_motion_encoder_path ./model/GRPO/tmr_weights/motion_encoder.pt --tmr_movement_encoder_path ./model/GRPO/tmr_weights/motion_decoder.pt --tmr_similarity_type euclidean --tmr_normalization linear --tmr_max_distance 10.0 --device 0
  
  # 使用欧氏距离 + 指数归一化
  python -m train.train_grpo --model_path ./save/official_humanml_enc_512_50steps/model000750000.pt --save_dir ./save/grpo_finetuned_humanml_enc_512_50steps_750000_tmr_matching_euclidean_exp --dataset humanml --batch_size 1 --group_size 4 --num_steps 10000 --reward_model_type tmr --reward_type matching 
  --tmr_text_encoder_path ./model/GRPO/tmr_weights/text_encoder.pt --tmr_motion_encoder_path ./model/GRPO/tmr_weights/motion_encoder.pt --tmr_movement_encoder_path ./model/GRPO/tmr_weights/motion_decoder.pt --tmr_similarity_type euclidean --tmr_normalization exponential --tmr_scale 2.0 --device 1
  
  # 使用欧氏距离 + Sigmoid 归一化
  python -m train.train_grpo --model_path ./save/official_humanml_enc_512_50steps/model000750000.pt --save_dir /save/grpo_finetuned_humanml_enc_512_50steps_750000_tmr_matching_euclidean_sigmoid --dataset humanml --batch_size 1 --group_size 4 --num_steps 10000 --reward_model_type tmr --reward_type matching 
  --tmr_text_encoder_path ./model/GRPO/tmr_weights/text_encoder.pt --tmr_motion_encoder_path ./model/GRPO/tmr_weights/motion_encoder.pt --tmr_movement_encoder_path ./model/GRPO/tmr_weights/motion_decoder.pt --tmr_similarity_type euclidean --tmr_normalization sigmoid --tmr_scale 2.0 --device 2
  ```
  
  

## GRPO 核心参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model_path` | str | **必需** | 预训练模型检查点路径 |
| `--ref_model_path` | str | None | 参考模型路径（如果与 model_path 不同） |
| `--group_size` | int | 4 | GRPO 组大小 G（每个 prompt 的采样数量） |
| `--learning_rate` | float | 5e-7 | GRPO 训练的学习率 |
| `--clip_epsilon` | float | 0.2 | PPO 裁剪参数 ε |
| `--kl_penalty` | float | 1.0 | KL 散度惩罚权重 β |
| `--grpo_type` | str | `normal_grpo` | GRPO 训练器类型：`normal_grpo`（标准 GRPO）或 `flow_grpo`（基于 Flow 的 GRPO） |

### 参数说明

- **`--group_size`**: 每个文本提示生成的样本数量。较大的组大小（如 8）可以提供更稳定的优势估计，但会增加计算成本。
- **`--learning_rate`**: GRPO 训练的学习率，通常比预训练阶段的学习率小 1-2 个数量级（推荐 1e-6 到 5e-7）。
- **`--clip_epsilon`**: PPO 风格的裁剪参数，控制策略更新的幅度。典型值：0.1-0.3。
- **`--kl_penalty`**: KL 散度惩罚权重，防止策略偏离参考模型太远。较大的值（如 1.0）更保守，较小的值（如 0.1）允许更大的更新。

## 奖励模型参数

### 奖励模型类型选择

| 参数 | 选项 | 说明 |
|------|------|------|
| `--reward_model_type` | `mdm` (默认) | 使用 MDM 评估器奖励函数 |
| | `tmr` | 使用 TMR 预训练模型奖励函数 |

### 奖励类型

**对于 MDM (`--reward_model_type=mdm`)**:
- `matching`: 基于文本-动作匹配分数（欧氏距离）
- `r_precision`: 基于 R-Precision 检索精度
- `combined`: 组合匹配分数和 R-Precision

**对于 TMR (`--reward_model_type=tmr`)**:
- `matching`: 匹配分数（可配置相似度和归一化）
- `cosine`: 余弦相似度（最简单，推荐）

### MDM 奖励函数参数

| 参数                  | 选项          | 说明                              |
| --------------------- | ------------- | --------------------------------- |
| `--reward_model_type` | `mdm`         | 使用 MDM 评估器                   |
| `--reward_type`       | `matching`    | 基于文本-动作匹配分数（欧氏距离） |
|                       | `r_precision` | 基于 R-Precision 检索精度         |
|                       | `combined`    | 组合匹配分数和 R-Precision        |

### TMR 奖励函数参数

| 参数                          | 选项          | 说明                                             |
| ----------------------------- | ------------- | ------------------------------------------------ |
| `--reward_model_type`         | `tmr`         | 使用 TMR 预训练模型                              |
| `--reward_type`               | `cosine`      | 余弦相似度（最简单，推荐）                       |
|                               | `matching`    | 匹配分数（可配置相似度和归一化）                 |
| `--tmr_text_encoder_path`     | 路径          | TMR 文本编码器权重路径 (text_encoder.pt，必需)   |
| `--tmr_motion_encoder_path`   | 路径          | TMR 动作编码器权重路径 (motion_encoder.pt，必需) |
| `--tmr_movement_encoder_path` | 路径          | TMR 动作解码器权重路径 (motion_decoder.pt，必需) |
| `--tmr_similarity_type`       | `cosine`      | 余弦相似度（推荐）                               |
|                               | `euclidean`   | 欧氏距离                                         |
| `--tmr_normalization`         | `linear`      | 线性归一化                                       |
|                               | `exponential` | 指数衰减归一化                                   |
|                               | `sigmoid`     | Sigmoid 归一化                                   |
| `--tmr_max_distance`          | 浮点数        | 最大距离（用于线性归一化，默认: 10.0）           |
| `--tmr_scale`                 | 浮点数        | 缩放因子（用于指数/Sigmoid，默认: 2.0）          |

## 奖励函数高级参数

这些参数适用于 MDM 和 TMR 奖励模型，用于控制奖励计算的详细行为。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--use_dense_reward` | flag | False | 启用分段密集打分模式（Segment-Dense）。False=整体打分（Global） |
| `--use_physics_reward` | flag | False | 启用物理正则化奖励 |
| `--k_segments` | int | 1 | 文本拼接数量（用于校验或默认处理） |
| `--max_motion_length` | int | 196 | 动作最大帧数限制，超过此长度的动作将被截断 |
| `--alpha` | float | 0.5 | 负向惩罚权重（用于 Segment-Dense 模式） |
| `--beta_s` | float | 1.0 | 语义奖励权重 |
| `--beta_p` | float | 0.1 | 物理奖励权重 |
| `--lambda_skate` | float | 1.0 | 滑行惩罚权重（用于物理奖励） |
| `--lambda_jerk` | float | 1.0 | 加速度突变惩罚权重（用于物理奖励） |
| `--fps` | float | 20.0 | 数据集帧率（HumanML=20.0, KIT=12.5） |
| `--disable_random_crop` | flag | False | 禁用随机 Crop 和偏移增强。使用复合数据集时必须启用 |

## 复合数据集参数

这些参数用于使用预构造的复合数据集（包含 K=3/4/5 个动作组合）进行训练和评估。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--use_composite_dataset` | flag | False | 启用复合数据集（由 construct_composite_dataset.py 构造） |
| `--composite_data_path` | str | None | 复合数据集 .npy 文件路径（使用复合数据集时必需） |
| `--composite_k_segments` | int | 3 | 复合数据集的分段数 K（3, 4, 或 5） |

### 参数说明

**复合数据集**：
- 复合数据集是通过 `scripts/construct_composite_dataset.py` 预先构造的数据集
- 每个样本包含 K 个动作的组合，以及对应的文本描述、durations（秒数）、B_matrix 等
- 使用复合数据集时，`--disable_random_crop` 会自动启用，因为 durations 是固定的
- `--composite_k_segments` 应该与构造数据集时的 `--k_segments` 一致
- `--k_segments`（奖励函数参数）应该与 `--composite_k_segments` 一致

**构造复合数据集**：
```bash
python scripts/construct_composite_dataset.py \
    --dataset humanml \
    --split train \
    --k_segments 3 \
    --output_dir dataset/HumanML3D/composite \
    --target_length 196 \
    --tolerance 20 \
    --fps 20.0 \
    --max_samples 1000 \
    --compute_b_matrix \
    --device cuda
```

### 参数说明

**分段密集打分模式 (`--use_dense_reward`)**:
- **False (Global 模式)**: 将多个子文本拼接后，计算整体动作与拼接文本的相似度
- **True (Segment-Dense 模式)**: 对每个动作片段与对应子文本分别计算相似度，并应用负向惩罚防止片段与错误文本对齐

**物理正则化 (`--use_physics_reward`)**:
- 计算滑行惩罚：脚部接触地面时的水平速度应该接近 0
- 计算加速度突变惩罚：关节加速度的平滑度
- 物理奖励公式：`R_phy = exp(-lambda_skate * L_skate - lambda_jerk * L_jerk)`

**总分聚合**:
- `R_total = beta_s * R_sem + beta_p * R_phy`
- 默认情况下，语义奖励权重为 1.0，物理奖励权重为 0.1

## Flow-GRPO 参数

这些参数仅在 `--grpo_type=flow_grpo` 时使用。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--noise_scale` | float | 0.7 | SDE 噪声缩放系数 a，用于 Flow-GRPO |
| `--train_timesteps` | int | 10 | 训练时的推理步数（降噪缩减策略） |
| `--inference_timesteps` | int | 40 | 推理时的步数（更高质量） |

### 参数说明

- **`--noise_scale`**: SDE 噪声缩放系数，控制探索强度。典型值：0.5-1.0
- **`--train_timesteps`**: 训练时使用较少的推理步数以加速训练（默认 10 步）
- **`--inference_timesteps`**: 推理时使用更多步数以获得更高质量（默认 40 步）

## 完整参数说明表格

### GRPO 核心参数

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--model_path` | str | ✅ | - | 预训练模型检查点路径 |
| `--ref_model_path` | str | ❌ | None | 参考模型路径（如果与 model_path 不同） |
| `--group_size` | int | ❌ | 4 | GRPO 组大小 G（每个 prompt 的采样数量） |
| `--learning_rate` | float | ❌ | 5e-7 | GRPO 训练的学习率 |
| `--clip_epsilon` | float | ❌ | 0.2 | PPO 裁剪参数 ε |
| `--kl_penalty` | float | ❌ | 1.0 | KL 散度惩罚权重 β |
| `--grpo_type` | str | ❌ | `normal_grpo` | GRPO 训练器类型：`normal_grpo` 或 `flow_grpo` |

### 复合数据集参数

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--use_composite_dataset` | flag | ❌ | False | 启用复合数据集（由 construct_composite_dataset.py 构造） |
| `--composite_data_path` | str | ✅* | None | 复合数据集 .npy 文件路径 |
| `--composite_k_segments` | int | ❌ | 3 | 复合数据集的分段数 K（3, 4, 或 5） |

*仅在 `--use_composite_dataset` 时必需

### TMR 特定参数

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--tmr_text_encoder_path` | str | ✅* | None | TMR 文本编码器权重路径 (text_encoder.pt) |
| `--tmr_motion_encoder_path` | str | ✅* | None | TMR 动作编码器权重路径 (motion_encoder.pt) |
| `--tmr_movement_encoder_path` | str | ✅* | None | TMR 动作解码器权重路径 (motion_decoder.pt) |
| `--tmr_similarity_type` | str | ❌ | `cosine` | 相似度类型：`cosine` 或 `euclidean` |
| `--tmr_normalization` | str | ❌ | `linear` | 归一化方式：`linear`, `exponential`, `sigmoid` |
| `--tmr_max_distance` | float | ❌ | 10.0 | 最大距离（用于线性归一化） |
| `--tmr_scale` | float | ❌ | 2.0 | 缩放因子（用于指数/Sigmoid 归一化） |

*仅在 `--reward_model_type=tmr` 时必需

## 训练示例

### 示例 1: 基础 GRPO 训练（使用 MDM 奖励，无 LoRA）

```bash
python -m train.train_grpo \
    --model_path ./save/official_humanml_enc_512_50steps/model000750000.pt \
    --save_dir ./save/grpo_finetuned \
    --dataset humanml \
    --batch_size 1 \
    --group_size 4 \
    --num_steps 10000 \
    --learning_rate 1e-6 \
    --reward_model_type mdm \
    --reward_type matching \
    --device 0
```

### 示例 2: 使用 LoRA 的 GRPO 训练

```bash
python -m train.train_grpo \
    --model_path ./save/pretrained_model/model000200000.pt \
    --save_dir ./save/grpo_finetuned \
    --dataset humanml \
    --batch_size 1 \
    --group_size 4 \
    --num_steps 10000 \
    --learning_rate 5e-7 \
    --use_lora \
    --lora_r 8 \
    --lora_alpha 16 \
    --reward_model_type mdm \
    --reward_type matching \
    --device 0
```

### 示例 3: 使用分段密集奖励和物理正则化

```bash
python -m train.train_grpo \
    --model_path ./save/official_humanml_enc_512_50steps/model000750000.pt \
    --save_dir ./save/grpo_dense_physics \
    --dataset humanml \
    --batch_size 1 \
    --group_size 4 \
    --num_steps 15000 \
    --learning_rate 1e-6 \
    --reward_model_type mdm \
    --reward_type matching \
    --use_dense_reward \
    --use_physics_reward \
    --k_segments 3 \
    --alpha 0.5 \
    --beta_s 1.0 \
    --beta_p 0.1 \
    --lambda_skate 1.0 \
    --lambda_jerk 1.0 \
    --device 0
```

### 示例 4: Flow-GRPO 训练

```bash
python -m train.train_grpo \
    --model_path ./save/official_humanml_enc_512_50steps/model000750000.pt \
    --save_dir ./save/grpo_flow \
    --dataset humanml \
    --batch_size 1 \
    --group_size 4 \
    --num_steps 15000 \
    --learning_rate 1e-6 \
    --grpo_type flow_grpo \
    --noise_scale 0.7 \
    --train_timesteps 10 \
    --inference_timesteps 40 \
    --reward_model_type mdm \
    --reward_type matching \
    --device 0
```

### 示例 5: 使用 TMR 奖励模型（余弦相似度）

```bash
python -m train.train_grpo \
    --model_path ./save/official_humanml_enc_512_50steps/model000750000.pt \
    --save_dir ./save/grpo_tmr_cosine \
    --dataset humanml \
    --batch_size 1 \
    --group_size 4 \
    --num_steps 15000 \
    --learning_rate 1e-6 \
    --reward_model_type tmr \
    --reward_type cosine \
    --tmr_text_encoder_path ./model/GRPO/tmr_weights/text_encoder.pt \
    --tmr_motion_encoder_path ./model/GRPO/tmr_weights/motion_encoder.pt \
    --tmr_movement_encoder_path ./model/GRPO/tmr_weights/motion_decoder.pt \
    --device 2
```

### 示例 6: 使用 TMR 奖励模型（匹配分数，分段密集模式）

```bash
python -m train.train_grpo \
    --model_path ./save/official_humanml_enc_512_50steps/model000750000.pt \
    --save_dir ./save/grpo_tmr_dense \
    --dataset humanml \
    --batch_size 1 \
    --group_size 4 \
    --num_steps 15000 \
    --learning_rate 1e-6 \
    --reward_model_type tmr \
    --reward_type matching \
    --tmr_text_encoder_path ./model/GRPO/tmr_weights/text_encoder.pt \
    --tmr_motion_encoder_path ./model/GRPO/tmr_weights/motion_encoder.pt \
    --tmr_movement_encoder_path ./model/GRPO/tmr_weights/motion_decoder.pt \
    --tmr_similarity_type cosine \
    --tmr_normalization linear \
    --use_dense_reward \
    --k_segments 3 \
    --alpha 0.5 \
    --device 1
```

### 示例 7: 使用复合数据集进行 GRPO 训练

```bash
# 首先构造复合数据集
python scripts/construct_composite_dataset.py \
    --dataset humanml \
    --split train \
    --k_segments 3 \
    --output_dir dataset/HumanML3D/composite \
    --max_samples 1000 \
    --compute_b_matrix \
    --device cuda

# 使用复合数据集训练
python -m train.train_grpo \
    --model_path ./save/official_humanml_enc_512_50steps/model000750000.pt \
    --save_dir ./save/grpo_composite_k3 \
    --dataset humanml \
    --batch_size 1 \
    --group_size 4 \
    --num_steps 1000 \
    --learning_rate 1e-6 \
    --reward_model_type tmr \
    --reward_type cosine \
    --tmr_text_encoder_path ./model/GRPO/tmr_weights/text_encoder.pt \
    --tmr_motion_encoder_path ./model/GRPO/tmr_weights/motion_encoder.pt \
    --tmr_movement_encoder_path ./model/GRPO/tmr_weights/motion_decoder.pt \
    --use_composite_dataset \
    --composite_data_path dataset/HumanML3D/composite/composite_k3_train.npy \
    --composite_k_segments 3 \
    --use_dense_reward \
    --k_segments 3 \
    --fps 20.0 \
    --disable_random_crop \
    --device cuda
```

### 示例 8: 使用复合数据集进行评估

```bash
# 首先构造测试集的复合数据集
python scripts/construct_composite_dataset.py \
    --dataset humanml \
    --split test \
    --k_segments 3 \
    --output_dir dataset/HumanML3D/composite \
    --max_samples 1000 \
    --compute_b_matrix \
    --device cuda

# 使用复合数据集评估
python -m eval.eval_humanml \
    --model_path ./save/grpo_composite_k3/model_final.pt \
    --dataset humanml \
    --eval_mode wo_mm \
    --use_composite_dataset \
    --composite_data_path dataset/HumanML3D/composite/composite_k3_test.npy \
    --composite_k_segments 3 \
    --device cuda
```

## 注意事项

1. **TMR 权重文件**: 使用 TMR 奖励函数时，必须提供三个独立的权重文件路径：
   - `--tmr_text_encoder_path`: 文本编码器权重 (text_encoder.pt)
   - `--tmr_motion_encoder_path`: 动作编码器权重 (motion_encoder.pt)
   - `--tmr_movement_encoder_path`: 动作解码器权重 (motion_decoder.pt)

2. **参数兼容性**: 
   - `--tmr_similarity_type`, `--tmr_normalization` 等参数仅在 `--reward_type=matching` 时生效
   - 当 `--reward_type=cosine` 时，这些参数会被忽略
   - Flow-GRPO 参数（`--noise_scale`, `--train_timesteps`, `--inference_timesteps`）仅在 `--grpo_type=flow_grpo` 时使用

3. **数据集支持**: 两种奖励函数都支持 `humanml` 和 `kit` 数据集

4. **性能建议**: 
   - MDM 评估器：使用项目内置的评估器，无需额外下载，适合快速实验
   - TMR：需要下载三个预训练权重文件，但可能提供更好的文本-动作对齐
   - 使用 LoRA 可以大幅降低显存占用，推荐用于显存受限的情况

5. **分段密集奖励模式**:
   - 适用于多段文本拼接的场景（如 "walk forward then jump up"）
   - 需要提供正确的 `--k_segments` 参数
   - 会增加计算成本，但可以提供更细粒度的奖励信号

6. **物理正则化**:
   - 有助于生成更符合物理规律的动作
   - 会增加计算成本
   - 建议与语义奖励结合使用（`--beta_s=1.0`, `--beta_p=0.1`）

7. **复合数据集**:
   - 使用预构造的复合数据集可以确保所有实验使用同一套指令
   - 复合数据集包含 K 个动作的组合，以及预计算的 B_matrix
   - 使用复合数据集时，`--disable_random_crop` 会自动启用
   - `--composite_k_segments` 应该与构造数据集时的 `--k_segments` 一致
   - `--k_segments`（奖励函数参数）应该与 `--composite_k_segments` 一致
   - 构造复合数据集的方法见 [COMPOSITE_DATASET_USAGE.md](./COMPOSITE_DATASET_USAGE.md)

**可视化平台选项**：

- `--train_platform_type NoPlatform`: 不使用可视化（默认）
- `--train_platform_type TensorboardPlatform`: 使用 TensorBoard
- `--train_platform_type WandBPlatform`: 使用 Weights & Biases
- `--train_platform_type ClearmlPlatform`: 使用 ClearML

**使用 TensorBoard 查看训练进度**：

```bash
# 训练时使用 TensorBoard
python -m train.train_grpo ... --train_platform_type TensorboardPlatform

# 在另一个终端启动 TensorBoard
tensorboard --logdir ./save/grpo_finetuned
```

**记录的训练指标**：

- **Loss**: `loss`, `policy_loss`, `kl_penalty`
- **Reward**: `mean_reward`, `std_reward`, `min_reward`, `max_reward`
- **Advantage**: `mean_advantage`, `std_advantage`
- **LogProb**: `mean_log_prob_current`, `mean_log_prob_ref`, `mean_ratio`
- **Training**: `grad_norm`, `learning_rate`



## 实验配置：MDM + TMR 奖励（1000 步训练）

本节提供了5个对比实验的训练命令，用于评估不同奖励配置对 GRPO 训练效果的影响。所有实验均使用：
- **基础模型**: MDM 预训练模型
- **奖励模型**: TMR 预训练模型
- **训练步数**: 1000 步
- **学习率**: 1e-6
- **组大小**: 4

### 实验配置对比表

| 实验 | 分段密集 | 物理约束 | 负向惩罚 | alpha | beta_s | beta_p | 说明 |
|------|---------|---------|---------|-------|--------|--------|------|
| 实验1 | ✅ | ❌ | ❌ | 0.0 | 1.0 | 0.0 | 分段密集TMR（仅正向奖励） |
| 实验2 | ❌ | ❌ | ❌ | - | 1.0 | 0.0 | 整体TMR（Global模式） |
| 实验3 | ✅ | ✅ | ❌ | 0.0 | 1.0 | 0.1 | 分段密集TMR + 物理约束（仅正向） |
| 实验4 | ❌ | ✅ | ❌ | - | 1.0 | 0.1 | 整体TMR + 物理约束 |
| 实验5 | ✅ | ✅ | ✅ | 0.5 | 1.0 | 0.1 | 分段密集TMR（正向+负向）+ 物理约束 |

**说明**:
- **分段密集**: 使用 `--use_dense_reward` 启用分段密集打分模式
- **物理约束**: 使用 `--use_physics_reward` 启用物理正则化
- **负向惩罚**: 在分段密集模式下，`alpha > 0` 表示启用负向惩罚
- **alpha**: 负向惩罚权重，0.0 表示只计算正向奖励，0.5 表示同时考虑正向和负向奖励

### 训练命令

```bash
# ============================================
# 实验 1: 分段密集TMR（正向）
# 使用分段密集模式，只计算正向奖励（alpha=0，无负向惩罚）
# ============================================
python -m train.train_grpo \
    --model_path ./save/official_humanml_enc_512_50steps/model000750000.pt \
    --save_dir ./save/grpo_finetuned_humanml_enc_512_50steps_750000_tmr_dense_pos_only \
    --dataset humanml \
    --batch_size 1 \
    --group_size 4 \
    --num_steps 1000 \
    --learning_rate 1e-6 \
    --reward_model_type tmr \
    --reward_type matching \
    --tmr_text_encoder_path ./model/GRPO/tmr_weights/text_encoder.pt \
    --tmr_motion_encoder_path ./model/GRPO/tmr_weights/motion_encoder.pt \
    --tmr_movement_encoder_path ./model/GRPO/tmr_weights/motion_decoder.pt \
    --tmr_similarity_type cosine \
    --tmr_normalization linear \
    --use_dense_reward \
    --k_segments 3 \
    --alpha 0.0 \
    --beta_s 1.0 \
    --disable_random_crop \
    --device 3


python -m train.train_grpo \
    --model_path ./save/official_humanml_enc_512_50steps/model000750000.pt \
    --save_dir ./save/grpo_finetuned_humanml_enc_512_50steps_750000_tmr_dense_pos_only \
    --dataset humanml \
    --batch_size 1 \
    --group_size 4 \
    --num_steps 1000 \
    --learning_rate 1e-6 \
    --reward_model_type tmr \
    --reward_type matching \
    --tmr_text_encoder_path ./model/GRPO/tmr_weights/text_encoder.pt \
    --tmr_motion_encoder_path ./model/GRPO/tmr_weights/motion_encoder.pt \
    --tmr_movement_encoder_path ./model/GRPO/tmr_weights/motion_decoder.pt \
    --tmr_similarity_type cosine \
    --tmr_normalization linear \
    --use_dense_reward \
    --k_segments 3 \
    --alpha 0.0 \
    --beta_s 1.0 \
    --disable_random_crop \
    --device mps


#  训练10000步
python -m train.train_grpo \
    --model_path ./save/official_humanml_enc_512_50steps/model000750000.pt \
    --save_dir ./save/grpo_finetuned_humanml_enc_512_50steps_750000_tmr_dense_pos_only_10000steps \
    --dataset humanml \
    --batch_size 1 \
    --group_size 4 \
    --num_steps 10000 \
    --learning_rate 1e-6 \
    --reward_model_type tmr \
    --reward_type matching \
    --tmr_text_encoder_path ./model/GRPO/tmr_weights/text_encoder.pt \
    --tmr_motion_encoder_path ./model/GRPO/tmr_weights/motion_encoder.pt \
    --tmr_movement_encoder_path ./model/GRPO/tmr_weights/motion_decoder.pt \
    --tmr_similarity_type cosine \
    --tmr_normalization linear \
    --use_dense_reward \
    --k_segments 3 \
    --alpha 0.0 \
    --beta_s 1.0 \
    --disable_random_crop \
    --device 3
#  训练10000步 用余炫相似度
python -m train.train_grpo \
    --model_path ./save/official_humanml_enc_512_50steps/model000750000.pt \
    --save_dir ./save/grpo_finetuned_humanml_enc_512_50steps_750000_tmr_dense_pos_only_10000steps \
    --dataset humanml \
    --batch_size 1 \
    --group_size 4 \
    --num_steps 10000 \
    --learning_rate 1e-6 \
    --reward_model_type tmr \
    --reward_type cosine \
    --tmr_text_encoder_path ./model/GRPO/tmr_weights/text_encoder.pt \
    --tmr_motion_encoder_path ./model/GRPO/tmr_weights/motion_encoder.pt \
    --tmr_movement_encoder_path ./model/GRPO/tmr_weights/motion_decoder.pt \
    --use_dense_reward \
    --k_segments 3 \
    --alpha 0.0 \
    --beta_s 1.0 \
    --disable_random_crop \
    --device 4

python -m train.train_grpo \
    --model_path ./save/official_humanml_enc_512_50steps/model000750000.pt \
    --save_dir ./save/grpo_finetuned_humanml_enc_512_50steps_750000_tmr_dense_pos_only_10000steps \
    --dataset humanml \
    --batch_size 1 \
    --group_size 4 \
    --num_steps 10000 \
    --learning_rate 1e-6 \
    --reward_model_type tmr \
    --reward_type cosine \
    --tmr_text_encoder_path ./model/GRPO/tmr_weights/text_encoder.pt \
    --tmr_motion_encoder_path ./model/GRPO/tmr_weights/motion_encoder.pt \
    --tmr_movement_encoder_path ./model/GRPO/tmr_weights/motion_decoder.pt \
    --use_dense_reward \
    --k_segments 3 \
    --alpha 0.0 \
    --beta_s 1.0 \
    --disable_random_crop \
    --device 4

# ============================================
# 实验 2: 整体TMR
# 使用整体模式（Global），不启用分段密集和物理约束
# ============================================
python -m train.train_grpo \
    --model_path ./save/official_humanml_enc_512_50steps/model000750000.pt \
    --save_dir ./save/grpo_finetuned_humanml_enc_512_50steps_750000_tmr_global \
    --dataset humanml \
    --batch_size 1 \
    --group_size 4 \
    --num_steps 1000 \
    --learning_rate 1e-6 \
    --reward_model_type tmr \
    --reward_type matching \
    --tmr_text_encoder_path ./model/GRPO/tmr_weights/text_encoder.pt \
    --tmr_motion_encoder_path ./model/GRPO/tmr_weights/motion_encoder.pt \
    --tmr_movement_encoder_path ./model/GRPO/tmr_weights/motion_decoder.pt \
    --tmr_similarity_type cosine \
    --tmr_normalization linear \
    --beta_s 1.0 \
    --disable_random_crop \
    --device 4

# ============================================
# 实验 3: 分段密集TMR + 物理约束
# 使用分段密集模式，启用物理正则化，只计算正向奖励（alpha=0）
# ============================================
python -m train.train_grpo \
    --model_path ./save/official_humanml_enc_512_50steps/model000750000.pt \
    --save_dir ./save/grpo_finetuned_humanml_enc_512_50steps_750000_tmr_dense_physics \
    --dataset humanml \
    --batch_size 1 \
    --group_size 4 \
    --num_steps 1000 \
    --learning_rate 1e-6 \
    --reward_model_type tmr \
    --reward_type matching \
    --tmr_text_encoder_path ./model/GRPO/tmr_weights/text_encoder.pt \
    --tmr_motion_encoder_path ./model/GRPO/tmr_weights/motion_encoder.pt \
    --tmr_movement_encoder_path ./model/GRPO/tmr_weights/motion_decoder.pt \
    --tmr_similarity_type cosine \
    --tmr_normalization linear \
    --use_dense_reward \
    --use_physics_reward \
    --k_segments 3 \
    --alpha 0.0 \
    --beta_s 1.0 \
    --beta_p 0.1 \
    --lambda_skate 1.0 \
    --lambda_jerk 1.0 \
    --disable_random_crop \
    --device 2

# ============================================
# 实验 4: 整体TMR + 物理约束
# 使用整体模式，启用物理正则化
# ============================================
python -m train.train_grpo \
    --model_path ./save/official_humanml_enc_512_50steps/model000750000.pt \
    --save_dir ./save/grpo_finetuned_humanml_enc_512_50steps_750000_tmr_global_physics \
    --dataset humanml \
    --batch_size 1 \
    --group_size 4 \
    --num_steps 1000 \
    --learning_rate 1e-6 \
    --reward_model_type tmr \
    --reward_type matching \
    --tmr_text_encoder_path ./model/GRPO/tmr_weights/text_encoder.pt \
    --tmr_motion_encoder_path ./model/GRPO/tmr_weights/motion_encoder.pt \
    --tmr_movement_encoder_path ./model/GRPO/tmr_weights/motion_decoder.pt \
    --tmr_similarity_type cosine \
    --tmr_normalization linear \
    --use_physics_reward \
    --beta_s 1.0 \
    --beta_p 0.1 \
    --lambda_skate 1.0 \
    --lambda_jerk 1.0 \
    --disable_random_crop \
    --device 3

# ============================================
# 实验 5: 分段密集TMR（正向+负向）+ 物理约束
# 使用分段密集模式，包含负向惩罚（alpha>0），启用物理正则化
# ============================================
python -m train.train_grpo \
    --model_path ./save/official_humanml_enc_512_50steps/model000750000.pt \
    --save_dir ./save/grpo_finetuned_humanml_enc_512_50steps_750000_tmr_dense_posneg_physics \
    --dataset humanml \
    --batch_size 1 \
    --group_size 4 \
    --num_steps 1000 \
    --learning_rate 1e-6 \
    --reward_model_type tmr \
    --reward_type matching \
    --tmr_text_encoder_path ./model/GRPO/tmr_weights/text_encoder.pt \
    --tmr_motion_encoder_path ./model/GRPO/tmr_weights/motion_encoder.pt \
    --tmr_movement_encoder_path ./model/GRPO/tmr_weights/motion_decoder.pt \
    --tmr_similarity_type cosine \
    --tmr_normalization linear \
    --use_dense_reward \
    --use_physics_reward \
    --k_segments 3 \
    --alpha 0.5 \
    --beta_s 1.0 \
    --beta_p 0.1 \
    --lambda_skate 1.0 \
    --lambda_jerk 1.0 \
    --disable_random_crop \
    --device 4


python -m train.train_grpo \
    --model_path ./save/official_humanml_enc_512_50steps/model000750000.pt \
    --save_dir ./save/grpo_finetuned_humanml_enc_512_50steps_750000_tmr_dense_posneg_physics \
    --dataset humanml \
    --batch_size 1 \
    --group_size 4 \
    --num_steps 1000 \
    --learning_rate 1e-6 \
    --reward_model_type tmr \
    --reward_type matching \
    --tmr_text_encoder_path ./model/GRPO/tmr_weights/text_encoder.pt \
    --tmr_motion_encoder_path ./model/GRPO/tmr_weights/motion_encoder.pt \
    --tmr_movement_encoder_path ./model/GRPO/tmr_weights/motion_decoder.pt \
    --tmr_similarity_type cosine \
    --tmr_normalization linear \
    --use_dense_reward \
    --use_physics_reward \
    --k_segments 3 \
    --alpha 0.5 \
    --beta_s 1.0 \
    --beta_p 0.1 \
    --lambda_skate 1.0 \
    --lambda_jerk 1.0 \
    --disable_random_crop \
    --device 4
```

### 实验说明

#### 实验 1: 分段密集TMR（正向）
- **目标**: 评估分段密集模式仅使用正向奖励的效果
- **特点**: 
  - 对每个动作片段与对应子文本分别计算相似度
  - 不计算负向惩罚（`alpha=0.0`）
  - 适用于评估分段对齐的基线效果

#### 实验 2: 整体TMR
- **目标**: 评估整体模式（Global）的效果，作为对比基线
- **特点**:
  - 将多个子文本拼接后，计算整体动作与拼接文本的相似度
  - 计算效率最高
  - 适用于单段文本或不需要细粒度对齐的场景

#### 实验 3: 分段密集TMR + 物理约束
- **目标**: 评估物理正则化对分段密集模式的影响
- **特点**:
  - 结合分段密集奖励和物理约束
  - 仅使用正向奖励（`alpha=0.0`）
  - 评估物理约束是否能改善动作质量

#### 实验 4: 整体TMR + 物理约束
- **目标**: 评估物理正则化对整体模式的影响
- **特点**:
  - 结合整体奖励和物理约束
  - 计算效率较高
  - 评估物理约束在整体模式下的效果

#### 实验 5: 分段密集TMR（正向+负向）+ 物理约束
- **目标**: 评估完整配置（包含负向惩罚和物理约束）的效果
- **特点**:
  - 最完整的配置，包含所有高级特性
  - 负向惩罚（`alpha=0.5`）防止动作片段与错误文本对齐
  - 物理约束确保动作符合物理规律
  - 预期效果最好，但计算成本最高

### 使用建议

1. **并行运行**: 5个实验可以同时在不同GPU上运行（device 0-4）
2. **路径修改**: 请根据实际情况修改以下路径：
   - `--model_path`: 你的MDM预训练模型路径
   - `--tmr_*_path`: TMR权重文件路径
   - `--save_dir`: 保存目录路径
3. **LoRA支持**: 如需使用LoRA，在所有命令中添加：
   ```bash
   --use_lora --lora_r 8 --lora_alpha 16
   ```
4. **监控训练**: 建议使用TensorBoard监控训练进度：
   ```bash
   tensorboard --logdir ./save --port 6006
   ```

### 预期结果分析

- **实验1 vs 实验2**: 对比分段密集模式与整体模式的效果
- **实验3 vs 实验1**: 评估物理约束对分段密集模式的影响
- **实验4 vs 实验2**: 评估物理约束对整体模式的影响
- **实验5 vs 实验3**: 评估负向惩罚的作用
- **实验5**: 预期效果最好，可作为最终配置






**评估模型**

### 标准数据集评估

- 50步模型
```bash
python -m eval.eval_humanml --model_path ./save/humanml_trans_enc_512/model000475000.pt
```

- lora_attnffn
```bash
python -m eval.eval_humanml --model_path ./save/test_lora_lorar128_loraalpha16_attnffn/model000600000.pt
```

- 分段密集TMR（正向）
```bash
python -m eval.eval_humanml --model_path ./save/grpo_finetuned_humanml_enc_512_50steps_750000_tmr_dense_pos_only/model_final.pt
```

- 整体TMR
```bash
python -m eval.eval_humanml --model_path ./save/grpo_finetuned_humanml_enc_512_50steps_750000_tmr_global/model_final.pt
```

- 分段密集TMR + 物理约束
```bash
python -m eval.eval_humanml --model_path ./save/grpo_finetuned_humanml_enc_512_50steps_750000_tmr_dense_physics/model_final.pt
```
- 整体TMR + 物理约束
```bash
python -m eval.eval_humanml --model_path ./save/grpo_finetuned_humanml_enc_512_50steps_750000_tmr_global_physics/model_final.pt
```
- 分段密集TMR（正向+负向）+ 物理约束
```bash
python -m eval.eval_humanml --model_path ./save/grpo_finetuned_humanml_enc_512_50steps_750000_tmr_dense_posneg_physics/model_final.pt
```


- 官方mdm扩散50步
```bash
python -m eval.eval_humanml --model_path ./save/official_humanml_enc_512_50steps/model000750000.pt
```

- mdm 扩散1000步
```bash
python -m eval.eval_humanml --model_path ./save/my_humanml_trans_enc_512/model000600000.pt
```


- mdm 奖励函数用的matching score
```bash
python -m eval.eval_humanml --model_path ./save/grpo_finetuned_humanml_enc_512_50steps_750000_mdm_matching/model_final.pt/model_final.pt
```

- mdm 奖励函数用的r_precision
```bash
python -m eval.eval_humanml --model_path ./save/grpo_finetuned_humanml_enc_512_50steps_750000_r_precision/model_final.pt
```

- mdm 奖励函数用的combined
```bash
python -m eval.eval_humanml --model_path ./save/grpo_finetuned_humanml_enc_512_50steps_750000_mdm_combined/model_final.pt
```

### 复合数据集评估

使用复合数据集进行评估时，需要先构造测试集的复合数据集，然后使用 `--use_composite_dataset` 参数：

```bash
# 构造测试集复合数据集
python scripts/construct_composite_dataset.py \
    --dataset humanml \
    --split test \
    --k_segments 3 \
    --output_dir dataset/HumanML3D/composite \
    --max_samples 1000 \
    --compute_b_matrix \
    --device cuda

# 使用复合数据集评估
python -m eval.eval_humanml \
    --model_path ./save/grpo_composite_k3/model_final.pt \
    --dataset humanml \
    --eval_mode wo_mm \
    --use_composite_dataset \
    --composite_data_path dataset/HumanML3D/composite/composite_k3_test.npy \
    --composite_k_segments 3 \
    --device cuda
```

**注意**：
- 评估时使用复合数据集作为生成数据源
- FID 等指标仍基于标准数据集（ground truth）计算
- 确保评估时使用的 `--composite_k_segments` 与训练时一致
