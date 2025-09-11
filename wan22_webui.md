# Wan2.2 Web UI 使用说明

## 概述

Wan2.2 Web UI 是一个现代化的 Wan2.2 视频生成模型训练 Web 界面，提供了直观的图形化界面来进行 Wan2.2 模型的 LoRA 训练。该界面集成了完整的训练流程，包括数据预处理、模型训练和监控功能，支持文生视频（T2V）和图生视频（I2V）两种任务模式。

## 主要功能

- **双任务支持**：支持 T2V-A14B（文生视频）和 I2V-A14B（图生视频）训练
- **双模型架构**：支持 Wan2.2 的高低噪声双模型训练机制
- **智能参数推荐**：根据数据集大小和任务类型自动推荐最优训练参数
- **配置管理**：支持训练配置的保存和加载，便于复用和分享
- **训练进度管理**：支持训练进度的保存和恢复，断点续训功能
- **实时日志监控**：基于 WebSocket 的实时训练日志和进度监控
- **TensorBoard 集成**：自动启动 TensorBoard 进行训练可视化
- **进程管理**：支持训练进程的启动、停止和状态监控
- **参数配置**：提供丰富的训练参数配置选项和验证
- **显存优化**：多种显存优化选项，支持不同规格显卡
- **训练时间预估**：根据数据集大小和参数配置预估训练时间
- **预计文件生成**：显示训练过程中将生成的检查点文件列表
- **Web界面优化**：基于Flask框架的现代化Web界面，支持响应式设计

## 核心脚本说明

### 1. wan_cache_latents.py

**功能**：预缓存 VAE Latents，将视频数据预处理为潜在空间表示

**主要参数**：
- `--dataset_config`：数据集配置文件路径（TOML 格式）
- `--vae`：VAE 模型路径（使用 Wan2.1 VAE）
- `--i2v`：启用图生视频模式（I2V 任务时使用）
- `--vae_cache_cpu`：将 VAE 缓存到 CPU 以节省显存

**使用场景**：
- 在训练前预处理视频数据
- 减少训练时的计算开销
- 支持 T2V 和 I2V 两种任务模式

### 2. wan_cache_text_encoder_outputs.py

**功能**：预缓存文本编码器输出，将文本提示预处理为嵌入向量

**主要参数**：
- `--dataset_config`：数据集配置文件路径
- `--t5`：T5-XXL 文本编码器模型路径
- `--batch_size`：批处理大小
- `--fp8_t5`：启用 FP8 模式以节省显存

**使用场景**：
- 预处理文本提示数据
- 支持 T5-XXL 大模型的高效编码
- 显存优化（FP8 模式）

### 3. wan_train_network.py

**功能**：LoRA 网络训练的核心脚本，支持 Wan2.2 的双模型训练

**主要参数**：

#### 模型路径参数
- `--dit`：DiT 低噪声模型路径（必需）
- `--dit_high_noise`：DiT 高噪声模型路径（双模型训练时使用）
- `--vae`：VAE 模型路径（必需）
- `--t5`：T5-XXL 文本编码器路径（必需）
- `--dataset_config`：数据集配置文件（必需）

#### 任务配置参数
- `--task`：训练任务类型
  - 选项：`t2v-A14B`, `i2v-A14B`
  - 推荐：`t2v-A14B`（文生视频）

#### 训练核心参数
- `--mixed_precision`：混合精度类型
  - 选项：`no`, `fp16`, `bf16`
  - 推荐：`bf16`（现代 GPU，节省显存且保持精度）

- `--timestep_sampling`：时间步采样策略
  - 选项：`uniform`, `shift`
  - 推荐：`shift`（官方推荐，提升训练效果）

- `--weighting_scheme`：损失权重方案
  - 选项：`none`, `logit_normal`, `mode`, `cosmap`, `sigma_sqrt`
  - 推荐：`none`（官方推荐的默认方案）

- `--discrete_flow_shift`：离散流偏移参数
  - 范围：1.0-20.0
  - 推荐：T2V 使用 12.0，I2V 使用 5.0

- `--timestep_boundary`：时间步边界（双模型训练）
  - 范围：0-1000
  - 推荐：T2V 使用 875，I2V 使用 900

- `--min_timestep` / `--max_timestep`：时间步范围限制
  - 范围：0-1000
  - 默认：0-1000（全范围）

- `--optimizer_type`：优化器类型
  - 选项：`adamw`, `adamw8bit`, `lion`, `sgd`
  - 推荐：`adamw8bit`（8位 AdamW，显著节省显存）

- `--learning_rate`：学习率
  - 推荐：2e-4（视频模型推荐值）

#### 网络配置参数
- `--network_module`：网络模块
  - 固定值：`networks.lora_wan`

- `--network_dim`：LoRA 维度
  - 推荐：32（平衡模型大小和表现力）

- `--network_alpha`：LoRA Alpha 参数
  - 推荐：32（通常与 network_dim 相等）

#### 训练控制参数
- `--max_train_epochs`：最大训练轮数
  - 推荐：16（视频模型需要更多轮次）

- `--save_every_n_epochs`：保存间隔
  - 推荐：1（每轮保存，便于选择最佳检查点）

- `--seed`：随机种子
  - 推荐：42（确保结果可复现）

#### 性能优化参数
- `--sdpa`：启用 SDPA 优化（推荐启用）
- `--gradient_checkpointing`：梯度检查点（推荐启用，节省显存）
- `--fp8_base`：启用 FP8 基础模型量化
- `--fp8_t5`：启用 FP8 T5 编码器量化
- `--fp8_scaled`：启用 FP8 缩放量化
- `--vae_cache_cpu`：VAE 缓存到 CPU
- `--offload_inactive_dit`：卸载非活跃 DiT 模型（双模型训练时推荐）
- `--blocks_to_swap`：模型块交换数量（显存不足时使用）

#### 数据加载参数
- `--max_data_loader_n_workers`：数据加载器工作进程数
  - 推荐：2（平衡加载速度和内存使用）
- `--persistent_data_loader_workers`：持久化数据加载器
  - 推荐：禁用（避免内存泄漏）

#### 日志配置参数
- `--logging_dir`：日志目录（默认：./logs）
- `--log_with`：日志工具（推荐：tensorboard）

#### 输出配置参数
- `--output_dir`：输出目录（必需）
- `--output_name`：输出文件名前缀（必需）

## Web UI 参数详细配置说明

### 核心模型路径配置

| 参数 | 说明 | 默认值 | 必需 | 备注 |
|------|------|--------|------|------|
| `dit` | DiT 低噪声模型路径 | ./model/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors | ✓ | Wan2.2的主要扩散模型，处理低噪声时间步 |
| `dit_high_noise` | DiT 高噪声模型路径 | ./model/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors | ○ | 双模型训练时使用，处理高噪声时间步 |
| `vae` | VAE 模型路径 | ./model/vae/wan_2.1_vae.safetensors | ✓ | 视频编码解码器，与Wan2.1兼容 |
| `t5` | T5 文本编码器路径 | ./model/text_encoders/models_t5_umt5-xxl-enc-bf16.pth | ✓ | T5-XXL文本编码器，用于文本理解 |
| `dataset_config` | 数据集配置文件 | ./ai_data/datasets/lovemf_config_t2v.toml | ✓ | TOML格式的数据集配置文件 |

### 任务配置

| 参数 | 说明 | 默认值 | 选项 | 备注 |
|------|------|--------|------|------|
| `task` | 训练任务类型 | t2v-A14B | t2v-A14B, i2v-A14B | t2v为文生视频，i2v为图生视频 |

### 训练核心参数

| 参数 | 说明 | 默认值 | 选项/范围 | 作用详解 |
|------|------|--------|----------|----------|
| `mixed_precision` | 混合精度训练类型 | fp16 | no, fp16, bf16 | bf16推荐用于现代GPU，节省显存且保持精度 |
| `timestep_sampling` | 时间步采样策略 | shift | uniform, shift | shift为官方推荐，提升训练效果 |
| `weighting_scheme` | 损失权重方案 | none | none, logit_normal, mode, cosmap, sigma_sqrt | none为官方推荐的默认方案 |
| `discrete_flow_shift` | 离散流偏移参数 | 12.0 | 1.0-20.0 | T2V推荐12.0，I2V推荐5.0，影响生成质量 |
| `timestep_boundary` | 时间步边界 | 875 | 0-1000 | 双模型训练时的切换点，T2V默认875，I2V默认900 |
| `min_timestep` | 最小时间步 | 0 | 0-1000 | 限制训练的时间步范围下限 |
| `max_timestep` | 最大时间步 | 1000 | 0-1000 | 限制训练的时间步范围上限 |
| `optimizer_type` | 优化器类型 | adamw8bit | adamw, adamw8bit, lion, sgd | adamw8bit为8位AdamW，显著节省显存 |
| `learning_rate` | 学习率 | 2e-4 | 1e-6 到 1e-3 | 控制模型参数更新幅度，影响训练稳定性 |

### LoRA 网络配置

| 参数 | 说明 | 默认值 | 范围 | 作用详解 |
|------|------|--------|------|----------|
| `network_module` | LoRA网络模块 | networks.lora_wan | 固定值 | 专为Wan模型优化的LoRA实现 |
| `network_dim` | LoRA维度(rank) | 32 | 4-128 | 控制LoRA适配器的容量，影响模型表现力和文件大小 |
| `network_alpha` | LoRA Alpha参数 | 32 | 1-128 | 控制LoRA强度，通常与network_dim相等 |

### 训练控制参数

| 参数 | 说明 | 默认值 | 范围 | 作用详解 |
|------|------|--------|------|----------|
| `max_train_epochs` | 最大训练轮数 | 16 | 1-100 | 训练数据集的完整遍历次数，视频模型需要更多轮次 |
| `save_every_n_epochs` | 保存间隔 | 1 | 1-10 | 每N轮保存一次模型，便于选择最佳检查点 |
| `seed` | 随机种子 | 42 | 0-2147483647 | 固定种子确保结果可复现 |


### 数据加载配置

| 参数 | 说明 | 默认值 | 范围 | 作用详解 |
|------|------|--------|------|----------|
| `max_data_loader_n_workers` | 数据加载器工作进程数 | 2 | 0-8 | 平衡加载速度和内存使用，过多会占用大量内存 |
| `persistent_data_loader_workers` | 持久化数据加载器 | false | true/false | 避免内存泄漏问题，推荐禁用 |

### 性能优化选项

| 参数 | 说明 | 默认值 | 作用详解 |
|------|------|--------|----------|
| `sdpa` | SDPA优化 | true | 启用Scaled Dot-Product Attention优化，显著提升效率 |
| `gradient_checkpointing` | 梯度检查点 | true | 以计算时间换取显存节省，推荐启用 |
| `fp8_base` | FP8基础模型量化 | true | 显著节省显存，轻微影响精度 |
| `fp8_t5` | FP8 T5编码器量化 | false | 节省T5模型显存使用 |
| `fp8_scaled` | FP8缩放量化 | false | 进一步节省显存，需要支持的硬件 |
| `vae_cache_cpu` | VAE缓存到CPU | false | 减少显存使用，略微降低速度 |
| `offload_inactive_dit` | 卸载非活跃DiT | true | 双模型训练时节省显存 |
| `blocks_to_swap` | 交换到CPU的块数量 | 0 | 在VRAM和内存间交换模型块以节省显存 |

### 输出配置

| 参数 | 说明 | 默认值 | 作用详解 |
|------|------|--------|----------|
| `output_dir` | 输出目录 | ./output | 训练结果保存路径 |
| `output_name` | 输出文件名前缀 | wan22-lora | LoRA模型文件的命名前缀 |
| `logging_dir` | 日志目录 | ./logs | TensorBoard日志保存路径 |
| `log_with` | 日志记录工具 | tensorboard | 可视化工具，用于监控训练过程 |

## 根据训练文件数量的参数推荐配置

### 20个训练文件（小数据集）

**特点**：数据量少，容易过拟合，需要更保守的参数设置

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `max_train_epochs` | 8-12 | 减少训练轮数，避免过拟合 |
| `learning_rate` | 1e-4 | 降低学习率，更稳定的训练 |
| `network_dim` | 16-24 | 降低LoRA维度，减少参数量 |
| `network_alpha` | 16-24 | 与network_dim保持一致 |
| `save_every_n_epochs` | 1 | 每轮保存，便于选择最佳点 |
| `gradient_checkpointing` | true | 启用梯度检查点 |
| `mixed_precision` | bf16 | 使用混合精度节省显存 |
| `optimizer_type` | adamw8bit | 8位优化器节省显存 |

### 40个训练文件（小-中数据集）

**特点**：数据量适中，可以使用标准参数配置

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `max_train_epochs` | 12-16 | 适中的训练轮数 |
| `learning_rate` | 1.5e-4 | 略微提高学习率 |
| `network_dim` | 24-32 | 标准LoRA维度 |
| `network_alpha` | 24-32 | 与network_dim保持一致 |
| `save_every_n_epochs` | 1 | 每轮保存 |
| `gradient_checkpointing` | true | 启用梯度检查点 |
| `mixed_precision` | bf16 | 使用混合精度 |
| `optimizer_type` | adamw8bit | 8位优化器 |

### 60个训练文件（中数据集）

**特点**：数据量较好，可以使用推荐的标准配置

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `max_train_epochs` | 16-20 | 标准训练轮数 |
| `learning_rate` | 2e-4 | 推荐学习率 |
| `network_dim` | 32 | 标准LoRA维度 |
| `network_alpha` | 32 | 与network_dim保持一致 |
| `save_every_n_epochs` | 1 | 每轮保存 |
| `gradient_checkpointing` | true | 启用梯度检查点 |
| `mixed_precision` | bf16 | 使用混合精度 |
| `optimizer_type` | adamw8bit | 8位优化器 |

### 80个训练文件（中-大数据集）

**特点**：数据量充足，可以使用更积极的参数设置

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `max_train_epochs` | 16-24 | 可以增加训练轮数 |
| `learning_rate` | 2e-4 到 2.5e-4 | 可以略微提高学习率 |
| `network_dim` | 32-48 | 可以增加LoRA维度 |
| `network_alpha` | 32-48 | 与network_dim保持一致 |
| `save_every_n_epochs` | 1-2 | 可以每1-2轮保存 |
| `gradient_checkpointing` | true | 启用梯度检查点 |
| `mixed_precision` | bf16 | 使用混合精度 |
| `optimizer_type` | adamw8bit | 8位优化器 |

### 100个训练文件（大数据集）

**特点**：数据量丰富，可以充分发挥模型潜力

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `max_train_epochs` | 20-30 | 增加训练轮数充分学习 |
| `learning_rate` | 2e-4 到 3e-4 | 可以使用更高学习率 |
| `network_dim` | 48-64 | 增加LoRA维度提升表现力 |
| `network_alpha` | 48-64 | 与network_dim保持一致 |
| `save_every_n_epochs` | 2 | 每2轮保存节省空间 |
| `gradient_checkpointing` | true | 启用梯度检查点 |
| `mixed_precision` | bf16 | 使用混合精度 |
| `optimizer_type` | adamw8bit | 8位优化器 |

### ≤1000个训练文件（超大数据集）

**特点**：数据量非常丰富，需要更长时间和更大模型容量

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `max_train_epochs` | 15-25 | 数据多时可以减少轮数 |
| `learning_rate` | 1.5e-4 到 2.5e-4 | 稳定的学习率 |
| `network_dim` | 64-96 | 大幅增加LoRA维度 |
| `network_alpha` | 64-96 | 与network_dim保持一致 |
| `save_every_n_epochs` | 2-3 | 每2-3轮保存 |
| `max_data_loader_n_workers` | 4-6 | 增加数据加载进程 |
| `gradient_checkpointing` | true | 必须启用 |
| `mixed_precision` | bf16 | 使用混合精度 |
| `optimizer_type` | adamw8bit | 8位优化器 |
| `blocks_to_swap` | 10-20 | 可能需要模型块交换 |

### ≥1000个训练文件（海量数据集）

**特点**：海量数据，需要专业级配置和更多显存优化

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `max_train_epochs` | 10-20 | 数据海量时减少轮数 |
| `learning_rate` | 1e-4 到 2e-4 | 保守的学习率确保稳定 |
| `network_dim` | 96-128 | 最大LoRA维度 |
| `network_alpha` | 96-128 | 与network_dim保持一致 |
| `save_every_n_epochs` | 3-5 | 每3-5轮保存 |
| `max_data_loader_n_workers` | 6-8 | 最大数据加载进程 |
| `gradient_checkpointing` | true | 必须启用 |
| `mixed_precision` | bf16 | 使用混合精度 |
| `optimizer_type` | adamw8bit | 8位优化器 |
| `blocks_to_swap` | 20-40 | 需要大量模型块交换 |
| `fp8_base` | true | 启用FP8量化 |
| `fp8_t5` | true | 启用T5的FP8量化 |
| `vae_cache_cpu` | true | VAE缓存到CPU |
| `offload_inactive_dit` | true | 卸载非活跃模型 |

### 通用优化建议

#### 显存不足时的优化策略

1. **启用所有FP8选项**：
   - `fp8_base: true`
   - `fp8_t5: true`
   - `fp8_scaled: true`

2. **使用模型块交换**：
   - `blocks_to_swap: 20-40`（需要64GB+内存）

3. **降低LoRA参数**：
   - `network_dim: 16-24`
   - `network_alpha: 16-24`

4. **启用CPU缓存**：
   - `vae_cache_cpu: true`
   - `offload_inactive_dit: true`

#### 训练速度优化

1. **数据加载优化**：
   - 小数据集：`max_data_loader_n_workers: 2`
   - 大数据集：`max_data_loader_n_workers: 4-8`
   - `persistent_data_loader_workers: false`（避免内存泄漏）

2. **计算优化**：
   - `sdpa: true`（必须启用）
   - `gradient_checkpointing: true`
   - `mixed_precision: bf16`

#### 训练质量优化

1. **学习率调度**：
   - 小数据集：较低学习率（1e-4）
   - 大数据集：标准学习率（2e-4）
   - 海量数据：保守学习率（1.5e-4）

2. **LoRA配置**：
   - 数据少：低维度（16-24）
   - 数据多：高维度（64-128）
   - `network_alpha = network_dim`

3. **训练轮数**：
   - 小数据集：少轮数（8-12）避免过拟合
   - 大数据集：多轮数（20-30）充分学习
   - 海量数据：适中轮数（10-20）

## 显存使用估算与硬件配置建议

### 基础显存需求（Wan2.2 14B模型，批大小=1）

| 配置类型 | 显存使用量 | 适用显卡 | 推荐内存 | 适用数据集规模 |
|----------|------------|----------|----------|----------------|
| **基础双模型配置** | 48GB+ | A100 80GB, H100 | 128GB+ | 所有规模 |
| **优化双模型配置** | 32-40GB | RTX 4090 24GB, A6000 48GB | 64GB+ | ≤1000个文件 |
| **高度优化配置** | 24-28GB | RTX 3090 24GB, RTX 4080 16GB | 64GB+ | ≤500个文件 |
| **极限优化配置** | 16-20GB | RTX 4060Ti 16GB, RTX 3080Ti | 32GB+ | ≤100个文件 |
| **单模型配置** | 20-24GB | RTX 4070Ti, RTX 3080Ti | 32GB+ | 小数据集 |

### 详细配置方案

#### 专业级配置（A100/H100）

**适用场景**：海量数据集（≥1000个文件），追求最佳训练效果

```yaml
# 基础配置
mixed_precision: bf16
gradient_checkpointing: true
sdpa: true

# 双模型训练
dit: wan2.2_t2v_low_noise_14B_fp16.safetensors
dit_high_noise: wan2.2_t2v_high_noise_14B_fp16.safetensors
offload_inactive_dit: true

# LoRA配置
network_dim: 96-128
network_alpha: 96-128

# 优化选项
fp8_base: false  # A100可以不启用
fp8_t5: false
blocks_to_swap: 0
```

**预期显存使用**：45-50GB

#### 高端消费级配置（RTX 4090/A6000）

**适用场景**：大数据集（100-1000个文件），平衡性能和成本

```yaml
# 基础配置
mixed_precision: bf16
gradient_checkpointing: true
sdpa: true

# 双模型训练
dit: wan2.2_t2v_low_noise_14B_fp16.safetensors
dit_high_noise: wan2.2_t2v_high_noise_14B_fp16.safetensors
offload_inactive_dit: true

# LoRA配置
network_dim: 48-64
network_alpha: 48-64

# 显存优化
fp8_base: true
fp8_t5: true
vae_cache_cpu: true
blocks_to_swap: 10-20
```

**预期显存使用**：20-24GB（RTX 4090），30-35GB（A6000）

#### 中端配置（RTX 3090/4080）

**适用场景**：中等数据集（40-200个文件），需要较多优化

```yaml
# 基础配置
mixed_precision: bf16
gradient_checkpointing: true
sdpa: true

# 双模型训练（可选单模型）
dit: wan2.2_t2v_low_noise_14B_fp16.safetensors
# dit_high_noise: 可以注释掉使用单模型
offload_inactive_dit: true

# LoRA配置
network_dim: 32-48
network_alpha: 32-48

# 显存优化
fp8_base: true
fp8_t5: true
fp8_scaled: true
vae_cache_cpu: true
blocks_to_swap: 20-30
```

**预期显存使用**：18-22GB（需要64GB+内存）

#### 入门级配置（RTX 4060Ti 16GB/3080Ti）

**适用场景**：小数据集（20-100个文件），极限优化

```yaml
# 基础配置
mixed_precision: bf16
gradient_checkpointing: true
sdpa: true

# 单模型训练
dit: wan2.2_t2v_low_noise_14B_fp16.safetensors
# 不使用dit_high_noise

# LoRA配置
network_dim: 16-32
network_alpha: 16-32

# 极限显存优化
fp8_base: true
fp8_t5: true
fp8_scaled: true
vae_cache_cpu: true
blocks_to_swap: 30-40
max_data_loader_n_workers: 1
```

**预期显存使用**：14-18GB（需要64GB+内存）

### 内存需求说明

| blocks_to_swap | 推荐内存 | 说明 |
|----------------|----------|------|
| 0 | 16GB+ | 无模型交换 |
| 10-20 | 32GB+ | 轻度模型交换 |
| 20-30 | 64GB+ | 中度模型交换 |
| 30-40 | 128GB+ | 重度模型交换 |

### 特殊情况说明

1. **I2V vs T2V**：
   - I2V任务比T2V多需要2-4GB显存
   - I2V推荐使用更保守的参数设置

2. **双模型 vs 单模型**：
   - 双模型训练效果更好，但显存需求高20-30%
   - 显存不足时可以只使用低噪声模型

3. **FP8量化效果**：
   - `fp8_base`：节省30-40%显存
   - `fp8_t5`：节省10-15%显存
   - `fp8_scaled`：额外节省5-10%显存

4. **blocks_to_swap性能影响**：
   - 每增加10个交换块，训练速度降低约10-15%
   - 需要高速SSD和大内存支持

### 硬件采购建议

#### 预算充足（专业用途）
- **GPU**：A100 80GB 或 H100
- **内存**：128GB+ DDR4/DDR5
- **存储**：2TB+ NVMe SSD
- **适用**：商业项目，大规模数据集

#### 预算适中（个人/小团队）
- **GPU**：RTX 4090 24GB
- **内存**：64GB DDR4/DDR5
- **存储**：1TB+ NVMe SSD
- **适用**：个人项目，中等数据集

#### 预算有限（学习/实验）
- **GPU**：RTX 4060Ti 16GB 或 RTX 3090
- **内存**：32-64GB DDR4
- **存储**：500GB+ NVMe SSD
- **适用**：学习实验，小数据集

## 使用流程

### 智能功能特色

#### 1. 一键推荐参数系统

基于数据集分析的智能参数推荐：
- **自动数据集分析**：扫描训练数据文件数量和类型
- **任务类型识别**：自动识别T2V或I2V训练任务
- **参数智能推荐**：根据数据集大小推荐最优的LoRA维度和Alpha值
- **显存适配**：根据GPU显存自动调整batch size和优化参数
- **训练策略建议**：提供学习率、训练轮数等关键参数建议

#### 2. 配置管理系统

- **配置保存**：将当前所有训练参数保存为JSON配置文件
- **配置加载**：一键加载之前保存的训练配置
- **配置验证**：自动验证配置参数的有效性和兼容性
- **配置分享**：支持配置文件的导入导出，便于团队协作
- **历史记录**：保存最近使用的配置，快速切换

#### 3. 训练进度管理

- **断点续训**：支持从任意检查点恢复训练
- **进度保存**：自动保存训练状态和优化器状态
- **进度监控**：实时显示训练进度、剩余时间和完成度
- **检查点管理**：智能管理检查点文件，避免磁盘空间浪费

#### 4. 训练时间预估系统

- **精确预估**：基于硬件性能和数据集大小预估训练时间
- **文件预览**：显示将要生成的所有检查点文件列表
- **资源评估**：预估显存、内存和存储空间需求
- **性能分析**：分析当前配置的训练效率

### 使用流程

### 1. 准备工作

1. **下载模型文件**：
   - DiT 低噪声模型：`wan2.2_t2v_low_noise_14B_fp16.safetensors`
   - DiT 高噪声模型：`wan2.2_t2v_high_noise_14B_fp16.safetensors`（可选）
   - VAE 模型：`wan_2.1_vae.safetensors`
   - T5 编码器：`models_t5_umt5-xxl-enc-bf16.pth`

2. **准备数据集配置**：
   - 创建 `dataset_config.toml` 文件
   - 配置训练数据路径和标注信息
   - T2V 任务：文本-视频对
   - I2V 任务：图像-视频对

### 2. 启动 Web UI

```bash
python wan22_webui.py
```

访问：http://localhost:7860

### 3. 训练流程

1. **配置参数**：在 Web 界面中设置训练参数
   - 根据数据集大小选择合适的参数配置
   - 参考本文档的推荐配置表格

2. **选择任务类型**：T2V-A14B 或 I2V-A14B
   - T2V：文本生成视频
   - I2V：图像生成视频

3. **预估训练时间**：
   - 点击"开始预算训练时间"按钮
   - 系统会分析数据集并显示：
     - 预计训练总时间
     - 每轮训练时间
     - 预计生成的检查点文件列表
     - 每个文件对应的训练步数范围

4. **预缓存数据**：
   - 缓存 VAE Latents（视频编码）
   - 缓存 T5 文本编码器输出（文本编码）

5. **开始训练**：启动 LoRA 训练
   - 自动保存检查点文件

6. **监控进度**：通过实时日志和 TensorBoard 监控训练
   - 实时显示训练损失
   - 监控学习率变化
   - 查看生成的检查点文件

### 4. 完整流程（推荐）

使用 "完整训练流程" 功能，自动执行：
1. 预缓存 VAE Latents
2. 预缓存 T5 文本编码器输出
3. LoRA 训练

## TensorBoard 监控

Web UI 会自动启动 TensorBoard 服务：
- 访问地址：http://localhost:6006
- 监控训练损失、学习率等指标
- 查看训练进度和模型性能
- 支持多实验对比

## 任务类型详解

### T2V-A14B（文生视频）

**特点**：
- 从文本描述生成视频
- 使用单一 DiT 模型或双模型架构
- 推荐参数：`discrete_flow_shift=12.0`, `timestep_boundary=875`

**数据格式**：
```toml
[[datasets]]
resolution = [720, 1280]  # 高度, 宽度
video_dir = "./videos"
caption_extension = ".txt"
```

### I2V-A14B（图生视频）

**特点**：
- 从静态图像生成视频
- 需要图像-视频对数据
- 推荐参数：`discrete_flow_shift=5.0`, `timestep_boundary=900`

**数据格式**：
```toml
[[datasets]]
resolution = [720, 1280]
video_dir = "./videos"
image_dir = "./images"  # 对应的起始帧
caption_extension = ".txt"
```

## 故障排除

### 常见问题与解决方案

#### 1. 显存不足问题

**症状**：CUDA out of memory 错误

**解决方案**（按优先级排序）：
1. **启用FP8量化**：
   ```yaml
   fp8_base: true
   fp8_t5: true
   fp8_scaled: true  # 如果硬件支持
   ```

2. **使用模型块交换**：
   ```yaml
   blocks_to_swap: 20-40  # 根据内存大小调整
   ```
   注意：需要64GB+内存支持

3. **降低LoRA参数**：
   ```yaml
   network_dim: 16-24
   network_alpha: 16-24
   ```

4. **启用CPU缓存**：
   ```yaml
   vae_cache_cpu: true
   offload_inactive_dit: true
   ```

5. **考虑单模型训练**：
   - 注释掉 `dit_high_noise` 参数
   - 节省20-30%显存

#### 2. 训练速度慢问题

**症状**：每步训练时间过长

**解决方案**：
1. **计算优化**：
   ```yaml
   sdpa: true  # 必须启用
   mixed_precision: bf16
   optimizer_type: adamw8bit
   ```

2. **数据加载优化**：
   ```yaml
   max_data_loader_n_workers: 2-4  # 根据CPU核心数调整
   persistent_data_loader_workers: false  # 避免内存泄漏
   ```

3. **避免过度交换**：
   - 如果使用 `blocks_to_swap`，确保有足够内存
   - 考虑升级到更大显存的显卡

#### 3. 训练不稳定问题

**症状**：损失值震荡、NaN值、训练崩溃

**解决方案**：
1. **降低学习率**：
   ```yaml
   learning_rate: 1e-4  # 从2e-4降低到1e-4
   ```

2. **检查数据质量**：
   - 确保视频分辨率一致
   - 检查标注文件格式正确
   - 验证视频文件完整性

3. **稳定性设置**：
   ```yaml
   gradient_checkpointing: true
   mixed_precision: bf16  # 比fp16更稳定
   seed: 42  # 固定随机种子
   ```

#### 4. 双模型训练问题

**症状**：双模型加载失败、切换异常

**解决方案**：
1. **检查模型路径**：
   - 确保两个DiT模型文件存在
   - 路径使用正确的分隔符

2. **调整切换参数**：
   ```yaml
   timestep_boundary: 875  # T2V推荐值
   timestep_boundary: 900  # I2V推荐值
   offload_inactive_dit: true  # 必须启用
   ```

#### 5. 预估时间功能问题

**症状**：预估时间显示异常、文件列表错误

**解决方案**：
1. **检查数据集配置**：
   - 确保TOML文件格式正确
   - 验证数据集路径存在

2. **参数设置检查**：
   ```yaml
   max_train_epochs: 16  # 确保大于0
   save_every_n_epochs: 1  # 确保大于0且小于max_train_epochs
   output_name: "wan22-lora"  # 确保有效的文件名
   ```

#### 6. 内存不足问题

**症状**：系统内存耗尽、交换文件过大

**解决方案**：
1. **减少数据加载进程**：
   ```yaml
   max_data_loader_n_workers: 1-2
   ```

2. **降低blocks_to_swap**：
   ```yaml
   blocks_to_swap: 10-20  # 从40降低到20
   ```

3. **系统优化**：
   - 关闭不必要的程序
   - 增加虚拟内存大小
   - 考虑升级物理内存

#### 8. TensorBoard无法访问

**症状**：无法打开TensorBoard界面

**解决方案**：
1. **检查端口占用**：
   - 默认端口6006是否被占用
   - 尝试重启Web UI

2. **防火墙设置**：
   - 允许端口6006通过防火墙
   - 检查杀毒软件拦截

### 日志分析

- **成功标志**：✅ 表示步骤完成
- **错误标志**：❌ 表示步骤失败
- **警告标志**：⏰ 表示超时或其他警告
- **实时输出**：通过 WebSocket 实时显示训练进度

## 高级配置

### 双模型训练优化

1. **模型选择**：
   - 低噪声模型：处理清晰细节
   - 高噪声模型：处理粗糙结构
   - 通过 `timestep_boundary` 控制切换点

2. **显存优化**：
   - 启用 `offload_inactive_dit`
   - 使用 `blocks_to_swap`
   - 考虑 FP8 量化

### 自定义网络参数

可通过配置文件自定义 LoRA 参数：
- `network_dim`：控制 LoRA 容量
- `network_alpha`：控制 LoRA 强度
- 建议保持 `network_dim = network_alpha`

### 数据集优化

1. **视频质量**：
   - 推荐 720p 或 1080p 分辨率
   - 保持宽高比一致
   - 帧率建议 24-30 FPS

2. **标注质量**：
   - 详细准确的文本描述
   - 避免过长或过短的描述
   - I2V 任务需要准确的起始帧

## 技术支持

- 查看实时日志了解训练状态
- 使用 TensorBoard 分析训练指标
- 参考官方文档获取更多信息
- 根据显存情况调整优化参数
- 监控 CPU 和内存使用情况

## 与 Qwen-Image 的区别

| 特性 | Wan2.2 | Qwen-Image |
|------|--------|------------|
| 模态 | 视频生成 | 图像生成 |
| 模型架构 | 双 DiT 模型 | 单 DiT 模型 |
| 文本编码器 | T5-XXL | Qwen2.5-VL |
| 任务类型 | T2V/I2V | T2I/Edit |
| 训练复杂度 | 更高 | 较低 |
| 显存需求 | 更大 | 较小 |
| 训练时间 | 更长 | 较短 |

## 更新日志

### v2.0 (最新)
- ✅ 新增训练时间预估功能
- ✅ 新增预计文件生成列表显示
- ✅ 完善参数配置说明和推荐值
- ✅ 根据数据集大小提供详细的参数推荐
- ✅ 增强故障排除指南
- ✅ 优化硬件配置建议

### v1.0
- ✅ 基础Web UI界面
- ✅ LoRA训练功能
- ✅ 双模型训练支持
- ✅ TensorBoard监控
- ✅ 数据预缓存功能

## 总结

本文档详细介绍了Wan2.2 Web UI的使用方法，包括：

1. **完整的参数配置指南**：每个参数都有详细说明、默认值和推荐配置
2. **基于数据集大小的推荐配置**：从20个文件到1000+文件的不同配置方案
3. **硬件配置建议**：从入门级到专业级的完整硬件方案
4. **训练时间预估**：帮助用户提前了解训练时间和生成文件
5. **详细的故障排除**：涵盖常见问题和解决方案

通过本文档，用户可以：
- 快速上手Wan2.2视频生成模型训练
- 根据自己的硬件配置选择合适的参数
- 预估训练时间和资源需求
- 解决训练过程中遇到的问题
- 优化训练效果和速度

建议用户在开始训练前仔细阅读相关章节，特别是参数配置和硬件要求部分，以确保获得最佳的训练体验。

---

**注意**：本文档基于 Musubi Tuner 框架的 Wan2.2 实现，功能仍在持续开发中。建议在使用前备份重要数据，并根据实际硬件配置调整参数。Wan2.2 作为视频生成模型，对硬件要求较高，请确保有足够的显存和内存资源。