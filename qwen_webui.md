# Qwen-Image Web UI 使用说明

## 概述

Qwen-Image Web UI 是一个现代化的 Qwen-Image 训练 Web 界面，提供了直观的图形化界面来进行 Qwen-Image 模型的 LoRA 训练。该界面集成了完整的训练流程，包括数据预处理、模型训练和监控功能。

## 主要功能

- **完整训练流程**：支持从数据预处理到模型训练的完整流程
- **Web界面**：基于Flask框架的现代化Web界面，支持响应式设计
- **配置管理**：支持训练配置的保存和加载，便于复用和分享
- **实时日志监控**：基于WebSocket的实时训练日志和进度监控
- **TensorBoard 集成**：自动启动 TensorBoard 进行训练可视化
- **进程管理**：支持训练进程的启动、停止和状态监控
- **参数配置**：提供丰富的训练参数配置选项和验证
- **图像编辑支持**：支持Qwen-Image-Edit模式的图像编辑训练
- **显存优化**：多种显存优化选项，支持不同规格显卡
- **FP8量化**：支持FP8量化以节省显存，适合中低端显卡

## 核心脚本说明

### 1. qwen_image_cache_latents.py

**功能**：预缓存 VAE Latents，将图像数据预处理为潜在空间表示

**主要参数**：
- `--dataset_config`：数据集配置文件路径（TOML 格式）
- `--vae`：VAE 模型路径

**使用场景**：
- 在训练前预处理图像数据
- 减少训练时的计算开销
- 支持 Qwen-Image-Edit 的控制图像缓存

### 2. qwen_image_cache_text_encoder_outputs.py

**功能**：预缓存文本编码器输出，将文本提示预处理为嵌入向量

**主要参数**：
- `--dataset_config`：数据集配置文件路径
- `--text_encoder`：文本编码器模型路径（Qwen2.5-VL）
- `--batch_size`：批处理大小
- `--fp8_vl`：启用 FP8 模式以节省显存（推荐 <16GB 显卡使用）
- `--edit`：启用 Qwen-Image-Edit 模式

**使用场景**：
- 预处理文本提示数据
- 支持图像编辑模式的文本-图像联合编码
- 显存优化（FP8 模式）

### 3. qwen_image_train_network.py

**功能**：LoRA 网络训练的核心脚本

**主要参数**：

#### 模型路径参数
- `--dit`：DiT 模型路径（必需）
- `--vae`：VAE 模型路径（必需）
- `--text_encoder`：文本编码器路径（必需）
- `--dataset_config`：数据集配置文件（必需）

#### 训练核心参数
- `--mixed_precision`：混合精度类型
  - 选项：`no`, `fp16`, `bf16`
  - 推荐：`bf16`（现代 GPU，节省显存且保持精度）

- `--timestep_sampling`：时间步采样策略
  - 选项：`uniform`, `shift`
  - 推荐：`shift`（官方推荐，提升训练效果）

- `--weighting_scheme`：损失权重方案
  - 选项：`none`, `sigma_sqrt`, `logit_normal`
  - 推荐：`none`（适合大多数场景）

- `--discrete_flow_shift`：离散流偏移参数
  - 范围：1.0-5.0
  - 推荐：2.2（官方调优的最佳值）

- `--optimizer_type`：优化器类型
  - 选项：`adamw`, `adamw8bit`, `lion`, `sgd`
  - 推荐：`adamw8bit`（8位 AdamW，显著节省显存）

- `--learning_rate`：学习率
  - 推荐：5e-5（平衡训练速度和稳定性）

#### 网络配置参数
- `--network_module`：网络模块
  - 固定值：`networks.lora_qwen_image`

- `--network_dim`：LoRA 维度
  - 推荐：8-16（平衡模型大小和表现力）

- `--network_args`：网络额外参数
  - 推荐：`loraplus_lr_ratio=4`（LoRA A/B 矩阵学习率比例）

#### 训练控制参数
- `--max_train_epochs`：最大训练轮数
  - 推荐：8（数据少时可提高）

- `--save_every_n_epochs`：保存间隔
  - 推荐：1（每轮保存，便于选择最佳检查点）

- `--seed`：随机种子
  - 推荐：42（确保结果可复现）

#### 性能优化参数
- `--sdpa`：启用 SDPA 优化（推荐启用）
- `--gradient_checkpointing`：梯度检查点（推荐启用，节省显存）
- `--persistent_data_loader_workers`：持久化数据加载器（推荐启用）
- `--fp8_base`：启用 FP8 基础模型量化（12GB 显卡推荐）
- `--fp8_scaled`：启用 FP8 缩放量化
- `--blocks_to_swap`：模型块交换（显存不足时使用）

#### 日志配置参数
- `--logging_dir`：日志目录（默认：./logs）
- `--log_with`：日志工具（推荐：tensorboard）

#### 输出配置参数
- `--output_dir`：输出目录（必需）
- `--output_name`：输出文件名前缀（必需）

## Web UI 参数配置说明

### 核心模型路径

| 参数 | 说明 | 建议值 | 必需 |
|------|------|--------|------|
| `dit_path` | DiT 模型路径 | qwen_image_bf16.safetensors | ✓ |
| `vae_path` | VAE 模型路径 | qwen_image_vae.safetensors | ✓ |
| `text_encoder_path` | 文本编码器路径 | qwen_2.5_vl_7b.safetensors | ✓ |
| `dataset_config` | 数据集配置文件 | dataset_config.toml | ✓ |

### 训练参数

| 参数 | 说明 | 推荐值 | 范围/选项 |
|------|------|--------|----------|
| `mixed_precision` | 混合精度训练 | bf16 | no/fp16/bf16 |
| `timestep_sampling` | 时间步采样 | shift | uniform/shift |
| `weighting_scheme` | 损失权重方案 | none | none/sigma_sqrt/logit_normal |
| `discrete_flow_shift` | 离散流偏移 | 2.2 | 1.0-5.0 |
| `optimizer_type` | 优化器类型 | adamw8bit | adamw/adamw8bit/lion/sgd |
| `learning_rate` | 学习率 | 5e-5 | 1e-6 到 1e-3 |
| `max_train_epochs` | 最大训练轮数 | 8 | 1-100 |
| `save_every_n_epochs` | 保存间隔 | 1 | 1-10 |
| `seed` | 随机种子 | 42 | 0-2147483647 |

### 网络配置

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `network_module` | 网络模块 | networks.lora_qwen_image |
| `network_dim` | LoRA 维度 | 8 |
| `network_args` | 网络参数 | loraplus_lr_ratio=4 |

### 优化选项

| 参数 | 说明 | 推荐 |
|------|------|------|
| `sdpa` | SDPA 优化 | 启用 |
| `gradient_checkpointing` | 梯度检查点 | 启用 |
| `persistent_data_loader_workers` | 持久化数据加载器 | 启用 |
| `fp8_base` | FP8 基础量化 | 启用（12GB 显卡） |
| `fp8_scaled` | FP8 缩放量化 | 启用 |

## 显存使用估算

基于 1024x1024 训练，批大小为 1，启用 `--mixed_precision bf16` 和 `--gradient_checkpointing`：

| 配置 | 显存使用量 |
|------|------------|
| 基础配置 | 42GB |
| + fp8_base + fp8_scaled | 30GB |
| + blocks_to_swap 16 | 24GB |
| + blocks_to_swap 45 | 12GB |

**注意**：使用 `blocks_to_swap` 时推荐 64GB 主内存。

## 功能特色

### 1. 配置管理系统

- **配置保存**：将当前所有训练参数保存为JSON配置文件
- **配置加载**：一键加载之前保存的训练配置
- **配置验证**：自动验证配置参数的有效性和兼容性
- **配置分享**：支持配置文件的导入导出，便于团队协作
- **默认配置**：提供经过优化的默认参数配置

### 2. 图像编辑模式

- **Edit模式支持**：支持Qwen-Image-Edit模型的训练
- **控制图像处理**：自动处理控制图像的缓存和编码
- **联合编码**：支持文本-图像联合编码，提升编辑效果
- **专用参数**：为图像编辑任务优化的参数配置

### 3. 显存优化策略

- **FP8量化**：支持fp8_base和fp8_scaled量化，显著节省显存
- **模型块交换**：通过blocks_to_swap参数实现显存-内存交换
- **梯度检查点**：启用gradient_checkpointing减少显存占用
- **混合精度**：支持bf16/fp16混合精度训练
- **优化器选择**：提供adamw8bit等显存友好的优化器

### 4. 实时监控系统

- **WebSocket通信**：实时推送训练日志和进度信息
- **进程状态监控**：实时显示训练进程的运行状态
- **TensorBoard集成**：自动启动并管理TensorBoard服务
- **错误处理**：智能错误检测和用户友好的错误提示

## 使用流程

### 1. 准备工作

1. **下载模型文件**：
   - DiT 模型：`qwen_image_bf16.safetensors`
   - VAE 模型：`diffusion_pytorch_model.safetensors`
   - 文本编码器：`qwen_2.5_vl_7b.safetensors`

2. **准备数据集配置**：
   - 创建 `dataset_config.toml` 文件
   - 配置训练数据路径和标注信息

### 2. 启动 Web UI

```bash
python qwen_webui.py
```

访问：http://localhost:5000

### 3. 训练流程

1. **配置参数**：在 Web 界面中设置训练参数
2. **预缓存数据**：
   - 缓存 VAE Latents
   - 缓存文本编码器输出
3. **开始训练**：启动 LoRA 训练
4. **监控进度**：通过日志和 TensorBoard 监控训练

### 4. 完整流程（推荐）

使用 "完整训练流程" 功能，自动执行：
1. 预缓存 VAE Latents
2. 预缓存文本编码器输出
3. LoRA 训练

## TensorBoard 监控

Web UI 会自动启动 TensorBoard 服务：
- 访问地址：http://localhost:6006
- 监控训练损失、学习率等指标
- 查看训练进度和模型性能

## 故障排除

### 常见问题

1. **显存不足**：
   - 启用 `fp8_base` 和 `fp8_scaled`
   - 使用 `blocks_to_swap`
   - 降低 `network_dim`

2. **训练速度慢**：
   - 启用 `sdpa`
   - 使用 `adamw8bit` 优化器
   - 启用 `persistent_data_loader_workers`

3. **训练不稳定**：
   - 降低学习率
   - 启用 `gradient_checkpointing`
   - 检查数据集质量

### 日志分析

- **成功标志**：✅ 表示步骤完成
- **错误标志**：❌ 表示步骤失败
- **警告标志**：⏰ 表示超时或其他警告

## 高级配置

### Qwen-Image-Edit 训练

如需训练图像编辑模型：
1. 使用 `qwen_image_edit_bf16.safetensors` 作为 DiT 模型
2. 在缓存和训练时添加 `--edit` 参数
3. 数据集配置中包含控制图像路径

### 自定义网络参数

可通过 `network_args` 参数自定义 LoRA 配置：
- `loraplus_lr_ratio`：A/B 矩阵学习率比例
- 其他 LoRA 特定参数

## 技术支持

- 查看实时日志了解训练状态
- 使用 TensorBoard 分析训练指标
- 参考官方文档获取更多信息
- 根据显存情况调整优化参数

---

**注意**：本文档基于 Musubi Tuner 框架的 Qwen-Image 实现，功能仍在持续开发中。建议在使用前备份重要数据，并根据实际硬件配置调整参数。