#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wan2.2 WebUI - 基于Flask的Wan2.2 LoRA训练Web界面
参考qwen_webui.py实现，专门为Wan2.2视频生成模型优化
"""

import os
import sys
import json
import threading
import subprocess
import time
import logging
import signal
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
import psutil

def recommend_wan22_lora_params(num_files, total_video_seconds, task_type, batch_size):
    """豆包推荐算法：使用数学建模方法计算最优训练参数"""
    import math
    
    # 基础参数设定
    base_lr = 2e-4
    base_dim = 32
    base_epochs = 16
    base_repeats = 1
    
    # 数据复杂度因子计算
    data_complexity = math.log(num_files + 1) / math.log(100)  # 归一化到[0,1]
    
    # 视频时长影响因子
    duration_factor = min(total_video_seconds / 300, 2.0)  # 5分钟为基准，最大2倍
    
    # 任务类型影响因子
    task_factor = 1.2 if task_type == 'i2v' else 1.0  # i2v任务更复杂
    
    # 批次大小影响因子
    batch_factor = math.sqrt(batch_size) / math.sqrt(4)  # 批次4为基准
    
    # 学习率计算：基于数据量和复杂度动态调整
    learning_rate = base_lr * (1 - 0.3 * data_complexity) * (1 + 0.2 * duration_factor) / batch_factor
    learning_rate = max(1e-5, min(5e-4, learning_rate))  # 限制范围
    
    # LoRA维度计算：基于数据复杂度和任务类型
    network_dim = int(base_dim * (1 + 0.5 * data_complexity) * task_factor)
    network_dim = max(16, min(128, network_dim))  # 限制范围
    
    # LoRA Alpha：通常等于维度
    network_alpha = network_dim
    
    # 训练轮数计算：基于数据量反比调整
    max_train_epochs = int(base_epochs * (1 + 0.5 / (data_complexity + 0.1)))
    max_train_epochs = max(8, min(32, max_train_epochs))  # 限制范围
    
    # 重复次数计算：小数据集需要更多重复
    num_repeats = int(base_repeats * (2 - data_complexity) * duration_factor)
    num_repeats = max(1, min(8, num_repeats))  # 限制范围
    
    return {
        'learning_rate': f"{learning_rate:.1e}",
        'network_dim': network_dim,
        'network_alpha': network_alpha,
        'max_train_epochs': max_train_epochs,
        'num_repeats': num_repeats,
        'batch_size': batch_size
    }


def wan22_lora_params(num_files, total_video_seconds, avg_video_len, mode, batch_size):
    """
    Wan2.2 LoRA参数推荐算法（batch_size自由输入版本）
    
    参数:
    - num_files: 数据集文件数量
    - total_video_seconds: 总视频秒数
    - avg_video_len: 平均视频长度（秒）
    - mode: 任务模式 ('i2v' 或 't2v')
    - batch_size: 批次大小 (1-8)
    
    返回: 包含所有推荐参数的字典
    """
    # 1. 计算有效样本数
    # 有效样本数就是txt文件数量（每个txt对应一个训练样本）
    N_eff = num_files
    
    # 2. 数据重复次数
    if N_eff <= 50:
        repeat_base = 2
    elif N_eff <= 200:
        repeat_base = 2
    else:
        repeat_base = 1
    
    # T2V任务增强
    mode_mult_repeat = 1.2 if mode.lower() == "t2v" and N_eff <= 200 else 1.0
    
    # 按batch_size缩放
    num_repeats = max(1, int(round(repeat_base * mode_mult_repeat / math.sqrt(batch_size))))
    
    # 3. 学习率
    if mode.lower() == "i2v":
        if N_eff <= 50:
            lr_base = 1e-5
        elif N_eff <= 200:
            lr_base = 1.5e-5
        else:
            lr_base = 2e-5
    else:  # t2v
        if N_eff <= 50:
            lr_base = 2e-5
        elif N_eff <= 200:
            lr_base = 3e-5
        else:
            lr_base = 4e-5
    
    # 按batch_size缩放学习率
    learning_rate = lr_base * math.sqrt(batch_size)
    
    # 4. LoRA维度
    if N_eff <= 50:
        dim_base = 48
    elif N_eff <= 200:
        dim_base = 64
    else:
        dim_base = 96
    
    mode_mult_dim = 0.95 if mode.lower() == "i2v" else 1.15
    dim_adj = dim_base * mode_mult_dim * (1 + 0.1 * math.log2(batch_size))
    allowed_dims = [8, 12, 16, 24, 32, 48, 64, 96, 128]
    network_dim = min(allowed_dims, key=lambda x: abs(x - dim_adj))
    
    # 5. network_alpha
    network_alpha = max(1, int(round(network_dim * (0.95 if batch_size >= 4 else 1.0))))
    
    # 6. 最大训练轮数
    if N_eff <= 50:
        epochs_base = 30
    elif N_eff <= 200:
        epochs_base = 20
    else:
        epochs_base = 10
    
    mode_mult_epoch = 1.2 if mode.lower() == "t2v" and N_eff <= 200 else 1.0
    max_train_epochs = max(1, int(round(epochs_base * mode_mult_epoch / math.sqrt(batch_size))))
    
    return {
        "num_repeats": num_repeats,
        "learning_rate": learning_rate,
        "network_dim": network_dim,
        "network_alpha": network_alpha,
        "max_train_epochs": max_train_epochs,
        "N_eff": N_eff,
        "batch_size": batch_size
    }


def recommend_software_lora_params(num_files, total_video_seconds, lora_type, task_type, batch_size):
    """
    软件推荐算法：基于写实/动漫类型和数据量推荐最优参数
    
    参数:
    - num_files: 数据集文件数量
    - total_video_seconds: 总视频秒数（视频训练时使用）
    - lora_type: LoRA类型 ('realistic' 写实 或 'anime' 动漫)
    - task_type: 任务类型 ('t2v' 或 'i2v')
    - batch_size: 批次大小
    
    返回: 包含推荐参数的字典
    """
    import math
    
    # 判断是否为视频训练
    is_video_training = total_video_seconds > 0
    
    # 确定数据量级别
    data_count = num_files
    if is_video_training:
        # 视频训练时，将视频总时长换算成等效图片数
        # 假设抽帧率为4fps（每秒取4帧）
        fps_sample = 4
        equivalent_frames = int(total_video_seconds * fps_sample)
        data_count = equivalent_frames  # 使用等效帧数作为数据量
    
    # 根据LoRA类型和数据量确定参数
    if lora_type == 'realistic':  # 写实类型
        if is_video_training:
            # 写实视频训练 - 按等效图片数推荐
            if data_count <= 1200:  # ≤1200帧（相当于≤60图）
                num_repeats = 5
                learning_rate = 2.5e-4
                network_dim = 64
                network_alpha = 32
                max_train_epochs = 25  # 20-30范围取中值
            elif data_count <= 3000:  # 1200-3000帧（相当于≤100图）
                num_repeats = 3
                learning_rate = 3e-4
                network_dim = 64
                network_alpha = 64
                max_train_epochs = 20
            else:  # >3000帧
                num_repeats = 2  # 1-2范围取2
                learning_rate = 3e-4
                network_dim = 48  # 32-64范围取中值
                network_alpha = 48  # 32-64范围取中值
                max_train_epochs = 15  # 10-20范围取中值
        else:
            # 写实图片训练
            if data_count <= 20:
                num_repeats = 10
                learning_rate = 2.5e-4
                network_dim = 64
                network_alpha = 32
                max_train_epochs = 35
            elif data_count <= 60:
                num_repeats = 5
                learning_rate = 2.5e-4
                network_dim = 64
                network_alpha = 48
                max_train_epochs = 30
            elif data_count <= 100:
                num_repeats = 3
                learning_rate = 3e-4
                network_dim = 64
                network_alpha = 64
                max_train_epochs = 25
            elif data_count <= 500:
                num_repeats = 2
                learning_rate = 3e-4
                network_dim = 48
                network_alpha = 48
                max_train_epochs = 20
            else:  # >= 500
                num_repeats = 1
                learning_rate = 3e-4
                network_dim = 32
                network_alpha = 32
                max_train_epochs = 12
    
    else:  # 动漫类型
        if is_video_training:
            # 动漫视频训练 - 按等效图片数推荐
            if data_count <= 1200:  # ≤1200帧
                num_repeats = 8
                learning_rate = 3e-4
                network_dim = 32
                network_alpha = 32
                max_train_epochs = 28  # 25-30范围取中值
            elif data_count <= 3000:  # 1200-3000帧
                num_repeats = 5
                learning_rate = 3e-4
                network_dim = 32
                network_alpha = 32
                max_train_epochs = 23  # 20-25范围取中值
            else:  # >3000帧
                num_repeats = 2  # 1-2范围取2
                learning_rate = 3e-4
                network_dim = 24  # 16-32范围取中值
                network_alpha = 24  # alpha=dim
                max_train_epochs = 13  # 10-15范围取中值
        else:
            # 动漫图片训练
            if data_count <= 20:
                num_repeats = 15
                learning_rate = 3e-4
                network_dim = 24
                network_alpha = 24
                max_train_epochs = 35
            elif data_count <= 60:
                num_repeats = 8
                learning_rate = 3e-4
                network_dim = 32
                network_alpha = 32
                max_train_epochs = 30
            elif data_count <= 100:
                num_repeats = 5
                learning_rate = 3e-4
                network_dim = 32
                network_alpha = 32
                max_train_epochs = 25
            elif data_count <= 500:
                num_repeats = 2
                learning_rate = 3.5e-4
                network_dim = 32
                network_alpha = 32
                max_train_epochs = 17
            else:  # >= 500
                num_repeats = 1
                learning_rate = 3.5e-4
                network_dim = 24
                network_alpha = 24
                max_train_epochs = 12
    
    # 限制参数范围
    learning_rate = max(1e-5, min(5e-4, learning_rate))
    network_dim = max(8, min(128, network_dim))
    network_alpha = max(1, min(128, network_alpha))
    max_train_epochs = max(5, min(50, max_train_epochs))
    num_repeats = max(1, min(20, num_repeats))
    
    return {
        'learning_rate': f"{learning_rate:.1e}",
        'network_dim': network_dim,
        'network_alpha': network_alpha,
        'max_train_epochs': max_train_epochs,
        'num_repeats': num_repeats,
        'batch_size': batch_size,  # 返回传入的批次大小参数
        'lora_type': lora_type,
        'data_count': data_count,
        'is_video_training': is_video_training
    }


# 导入tkinter用于文件对话框
try:
    import tkinter as tk
    from tkinter import filedialog
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    logger.warning("tkinter不可用，文件选择对话框功能将被禁用")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('wan22_webui.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 全局变量
app = Flask(__name__, template_folder='templates')
socketio = SocketIO(app, cors_allowed_origins="*")
training_process = None
tensorboard_process = None
cache_process = None
cache_type_running = None  # 'vae' 或 'text_encoder'
training_logs = []
config_file = 'wan_webui_config.json'
process_lock = threading.Lock()

# 进程管理

# Wan2.2参数描述和配置
PARAM_DESCRIPTIONS = {
    # 核心模型路径
    'dit': {
        'description': 'DiT模型路径 (低噪声模型)',
        'suggestion': 'Wan2.2的主要DiT模型文件，支持t2v-A14B和i2v-A14B任务',
        'required': True,
        'type': 'file'
    },
    'dit_high_noise': {
        'description': 'DiT高噪声模型路径 (可选)',
        'suggestion': 'Wan2.2的高噪声DiT模型，用于双模型训练',
        'required': False,
        'type': 'file'
    },
    'vae': {
        'description': 'VAE模型路径',
        'suggestion': 'Wan2.2使用与Wan2.1相同的VAE模型',
        'required': True,
        'type': 'file'
    },
    't5': {
        'description': 'T5文本编码器路径',
        'suggestion': 'T5-XXL编码器模型，用于文本理解',
        'required': True,
        'type': 'file'
    },
    'dataset_config': {
        'description': '数据集配置文件路径 (TOML格式)',
        'suggestion': '包含训练数据路径和标注信息的配置文件',
        'required': True,
        'type': 'file'
    },
    
    # 任务配置
    'task': {
        'description': '训练任务类型',
        'suggestion': 't2v-A14B用于文生视频，i2v-A14B用于图生视频',
        'options': ['t2v-A14B', 'i2v-A14B'],
        'default': 't2v-A14B'
    },
    'dit_high_noise': {
        'description': '高噪声DiT模型路径（Wan2.2双模型训练）',
        'suggestion': '用于高噪声时间步的DiT模型，与dit配合使用',
        'required': False,
        'type': 'file'
    },
    
    # 训练核心参数
    'mixed_precision': {
        'description': '混合精度训练类型',
        'suggestion': 'bf16 - 推荐用于现代GPU，节省显存且保持精度',
        'options': ['no', 'fp16', 'bf16'],
        'default': 'bf16'
    },
    'timestep_sampling': {
        'description': '时间步采样策略',
        'suggestion': 'shift - 官方推荐的采样方法，提升训练效果',
        'options': ['uniform', 'shift'],
        'default': 'shift'
    },
    'weighting_scheme': {
        'description': '损失权重方案',
        'suggestion': 'none - 官方推荐的默认权重方案',
        'options': ['none', 'logit_normal', 'mode', 'cosmap', 'sigma_sqrt'],
        'default': 'none'
    },
    'discrete_flow_shift': {
        'description': '离散流偏移参数',
        'suggestion': 'T2V推荐12.0，I2V推荐5.0，训练时可适当调整',
        'range': [1.0, 20.0],
        'default': 12.0
    },
    'timestep_boundary': {
        'description': '时间步边界 (双模型训练)',
        'suggestion': 'T2V默认875，I2V默认900，用于切换高低噪声模型',
        'range': [0, 1000],
        'default': 875
    },
    'fp8_scaled': {
        'description': '使用缩放FP8精度',
        'suggestion': '进一步节省显存，需要支持的硬件',
        'default': False
    },
    'fp8_t5': {
        'description': 'T5编码器使用FP8精度',
        'suggestion': '节省T5编码器显存占用',
        'default': False
    },
    'fp8_base': {
        'description': '使用基础FP8精度',
        'suggestion': '节省显存的FP8模式',
        'default': False
    },
    'min_timestep': {
        'description': '最小时间步',
        'suggestion': '限制训练的时间步范围，0-1000',
        'range': [0, 1000],
        'default': 0
    },
    'max_timestep': {
        'description': '最大时间步',
        'suggestion': '限制训练的时间步范围，0-1000',
        'range': [0, 1000],
        'default': 1000
    },
    'optimizer_type': {
        'description': '优化器类型',
        'suggestion': 'adamw8bit - 8位AdamW，显著节省显存',
        'options': ['adamw', 'adamw8bit', 'lion', 'sgd'],
        'default': 'adamw8bit'
    },
    'learning_rate': {
        'description': '学习率',
        'suggestion': '推荐2e-4，根据数据量和任务复杂度调整',
        'range': [1e-6, 1e-3],
        'default': 2e-4
    },
    
    # 网络配置
    'network_module': {
        'description': 'LoRA网络模块',
        'suggestion': 'networks.lora_wan - 专为Wan模型优化的LoRA实现',
        'default': 'networks.lora_wan',
        'readonly': True
    },
    'network_dim': {
        'description': 'LoRA维度 (rank)',
        'suggestion': '32 - 推荐值，平衡模型大小和表现力',
        'range': [4, 128],
        'default': 32
    },
    'network_alpha': {
        'description': 'LoRA Alpha参数',
        'suggestion': '32 - 通常与network_dim相等，控制LoRA强度',
        'range': [1, 128],
        'default': 32
    },
    
    # 训练控制
    'max_train_epochs': {
        'description': '最大训练轮数',
        'suggestion': '16 - 视频模型需要更多轮次训练',
        'range': [1, 100],
        'default': 16
    },
    'save_every_n_epochs': {
        'description': '每N轮保存一次模型',
        'suggestion': '1 - 每轮都保存，便于选择最佳检查点',
        'range': [1, 10],
        'default': 1
    },
    'seed': {
        'description': '随机种子',
        'suggestion': '42 - 固定种子确保结果可复现',
        'range': [0, 2147483647],
        'default': 42
    },
    
    # 数据加载
    'max_data_loader_n_workers': {
        'description': '数据加载器工作进程数',
        'suggestion': '2 - 平衡加载速度和内存使用',
        'range': [0, 8],
        'default': 2
    },
    'persistent_data_loader_workers': {
        'description': '持久化数据加载器工作进程',
        'suggestion': '禁用 - 避免内存泄漏问题',
        'default': False
    },
    
    # 优化选项
    'sdpa': {
        'description': '启用SDPA优化',
        'suggestion': '启用 - 显著提升注意力计算效率',
        'default': True
    },
    'gradient_checkpointing': {
        'description': '梯度检查点',
        'suggestion': '启用 - 以计算时间换取显存节省',
        'default': True
    },
    'fp8_base': {
        'description': '启用FP8基础模型',
        'suggestion': '启用 - 显著节省显存，轻微影响精度',
        'default': False
    },
    'fp8_t5': {
        'description': '启用FP8 T5编码器',
        'suggestion': '启用 - 节省T5模型显存使用',
        'default': False
    },
    'vae_cache_cpu': {
        'description': 'VAE缓存到CPU',
        'suggestion': '启用 - 减少显存使用，略微降低速度',
        'default': False
    },
    'offload_inactive_dit': {
        'description': '卸载非活跃DiT模型',
        'suggestion': '启用 - 双模型训练时节省显存',
        'default': False
    },
    'blocks_to_swap': {
        'description': '交换到CPU的块数量',
        'suggestion': '设置为20 - 在VRAM和内存间交换模型块以节省显存',
        'default': 20,
        'type': 'number'
    },
    
    # 输出配置
    'output_dir': {
        'description': '输出目录',
        'suggestion': '训练结果保存路径',
        'default': './output',
        'type': 'directory'
    },
    'output_name': {
        'description': '输出文件名前缀',
        'suggestion': 'LoRA模型文件的命名前缀',
        'default': 'wan22-lora'
    },
    'logging_dir': {
        'description': '日志目录',
        'suggestion': 'TensorBoard日志保存路径',
        'default': './logs'
    },
    'log_with': {
        'description': '日志记录工具',
        'suggestion': 'tensorboard - 官方推荐的可视化工具，用于监控训练过程',
        'options': ['tensorboard', 'wandb', 'all'],
        'default': 'tensorboard'
    },
    
    # 断点续训相关参数
    'save_state': {
        'description': '保存训练状态',
        'suggestion': '启用后会保存模型权重、优化器状态、调度器状态等，用于断点续训',
        'default': False
    },
    'resume': {
        'description': '恢复训练状态路径',
        'suggestion': '指定之前保存的状态目录，实现真正的断点续训',
        'required': False,
        'type': 'directory'
    },

}

# 默认配置
DEFAULT_CONFIG = {
    # 环境配置
    'enable_venv': True,
    'venv_python_path': './venv/Scripts/',
    
    # 核心模型路径
    'dit': './model/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors',
    'dit_high_noise': './model/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors',
    'vae': './model/vae/wan_2.1_vae.safetensors',
    't5': './model/text_encoders/models_t5_umt5-xxl-enc-bf16.pth',
    'dataset_config': './ai_data/datasets/lovemf_config_t2v.toml',
    
    # 任务配置
    'task': 't2v-A14B',
    
    # 训练核心参数
    'mixed_precision': 'fp16',
    'timestep_sampling': 'shift',
    'weighting_scheme': 'none',
    'discrete_flow_shift': 12.0,
    'timestep_boundary': 875,
    'min_timestep': 0,
    'max_timestep': 1000,
    
    # 优化器配置
    'optimizer_type': 'adamw8bit',
    'learning_rate': '2e-4',
    
    # LoRA网络配置
    'network_module': 'networks.lora_wan',
    'network_dim': 32,
    'network_alpha': 32,
    
    # 训练控制
    'max_train_epochs': 16,
    'save_every_n_epochs': 1,
    'seed': 42,
    'batch_size': 1,
    'num_repeats': 1,
    
    # 视频训练参数
    'video_duration': 300,
    
    # 数据加载
    'max_data_loader_n_workers': 2,
    'persistent_data_loader_workers': False,
    
    # 优化选项
    'sdpa': True,
    'gradient_checkpointing': True,
    'fp8_base': True,
    'fp8_t5': False,
    'fp8_scaled': False,
    'vae_cache_cpu': False,
    'offload_inactive_dit': True,
    'blocks_to_swap': 0,
    
    # 输出配置
    'output_dir': './output',
    'output_name': 'wan22-lora',
    'logging_dir': './logs',
    'log_with': 'tensorboard',
    
    # 断点续训相关参数
    'save_state': False,
    'resume_training': False,
    'resume': '',
    
    # 其他参数
    'num_cpu_threads_per_process': 1,
    'avg_step_time': 2.92,

}

def load_config() -> Dict[str, Any]:
    """加载配置文件"""
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                # 合并默认配置
                merged_config = DEFAULT_CONFIG.copy()
                merged_config.update(config)
                return merged_config
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
    return DEFAULT_CONFIG.copy()

def save_config(config: Dict[str, Any]) -> bool:
    """保存配置文件"""
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"保存配置文件失败: {e}")
        return False

def add_log(message: str, level: str = 'info'):
    """添加日志消息并通过WebSocket广播"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = {
        'timestamp': timestamp,
        'level': level,
        'message': message
    }
    training_logs.append(log_entry)
    
    # 保持日志数量在合理范围内
    if len(training_logs) > 1000:
        training_logs[:] = training_logs[-500:]
    
    # 通过WebSocket广播日志
    socketio.emit('log_message', log_entry)
    
    # 同时写入Python日志
    if level == 'error':
        logger.error(message)
    elif level == 'warning':
        logger.warning(message)
    else:
        logger.info(message)


# WebSocket事件处理
@socketio.on('connect')
def handle_connect():
    """处理客户端连接"""
    logger.info('WebSocket客户端已连接')
    # 发送最近的日志给新连接的客户端
    recent_logs = training_logs[-50:] if len(training_logs) > 50 else training_logs
    for log_entry in recent_logs:
        emit('log_message', log_entry)

@socketio.on('disconnect')
def handle_disconnect():
    """处理客户端断开连接"""
    logger.info('WebSocket客户端已断开连接')


def get_venv_python_path(config: Dict[str, Any]):
    """获取虚拟环境Python路径"""
    enable_venv = config.get('enable_venv', True)
    venv_python_path = config.get('venv_python_path', './venv/Scripts/')
    
    if enable_venv:
        # 处理路径
        if venv_python_path.startswith('./'):
            # 相对路径，转换为绝对路径
            venv_python_path = venv_python_path[2:]  # 移除 './'
            venv_dir = os.path.join(os.getcwd(), venv_python_path)
        elif os.path.isabs(venv_python_path):
            # 绝对路径
            venv_dir = venv_python_path
        else:
            # 相对路径（不以./开头）
            venv_dir = os.path.join(os.getcwd(), venv_python_path)
        
        # 确保路径以正确的分隔符结尾
        if not venv_dir.endswith(('/', '\\')):
            venv_dir = venv_dir + os.sep
        
        # 构建Python可执行文件路径
        venv_python = os.path.join(venv_dir, 'python.exe')
        venv_python = os.path.normpath(venv_python)
        
        # 检查Python可执行文件是否存在
        if os.path.exists(venv_python):
            return venv_python, True
        else:
            add_log(f"虚拟环境Python不存在: {venv_python}，使用系统Python", 'warning')
            return 'python', False
    else:
        return 'python', False

def generate_training_record_file(config: Dict[str, Any], cmd: List[str], avg_step_time: float = 2.92):
    """生成训练记录文件"""
    try:
        output_dir = config.get('output_dir', './output')
        output_name = config.get('output_name', 'wan22-lora')
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成记录文件路径
        record_file = os.path.join(output_dir, f"{output_name}.md")
        
        # 获取当前时间
        start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 构建执行命令字符串
        cmd_str = ' '.join(cmd)
        
        # 计算数据集信息
        dataset_config_path = config.get('dataset_config', '')
        video_duration = config.get('video_duration', 30.0)  # 获取用户设置的视频时长
        dataset_info = get_dataset_info(dataset_config_path, video_duration)
        
        # 计算训练参数
        batch_size = config.get('batch_size', 1)
        num_repeats = config.get('num_repeats', 1)
        max_epochs = config.get('max_train_epochs', 16)
        save_every_n_epochs = config.get('save_every_n_epochs', 1)
        
        # 检查是否为视频训练模式
        # 只使用前端传递的is_video_training参数，不再根据task_type自动判断
        is_video_training = config.get('is_video_training', False)
        task_type = config.get('task', 't2v')
        
        # 初始化步数变量
        steps_per_epoch = 0
        total_steps = 0
        
        # 计算总步数
        if dataset_info['file_count'] > 0:
            if is_video_training:
                # 视频训练模式：从TOML配置文件读取视频训练参数
                try:
                    import toml
                    config_data = toml.load(dataset_config_path)
                    datasets = config_data.get('datasets', [])
                    
                    if datasets:
                        dataset = datasets[0]  # 使用第一个数据集的配置
                        target_frames = dataset.get('target_frames', [1, 25, 49, 73])
                        frame_sample = dataset.get('frame_sample', 8)
                        # 优先使用前端传递的num_repeats，而不是TOML文件中的硬编码值
                        # 计算有效训练样本数（视频片段数）
                        # 正确公式：视频文件数 × target_frames长度 × frame_sample × num_repeats
                        effective_samples = dataset_info['file_count'] * len(target_frames) * frame_sample * num_repeats
                        steps_per_epoch = math.ceil(effective_samples / batch_size)
                        total_steps = steps_per_epoch * max_epochs
                        
                        print(f"[视频训练记录] target_frames: {target_frames}, frame_sample: {frame_sample}, num_repeats: {num_repeats}")
                        print(f"[视频训练记录] 有效样本数: {effective_samples}, 每轮步数: {steps_per_epoch}, 总步数: {total_steps}")
                    else:
                        # 如果无法读取配置，使用默认计算
                        steps_per_epoch = math.ceil(dataset_info['file_count'] * num_repeats / batch_size)
                        total_steps = steps_per_epoch * max_epochs
                except Exception as e:
                    print(f"[视频训练记录] 读取TOML配置失败: {e}，使用默认计算")
                    steps_per_epoch = math.ceil(dataset_info['file_count'] * num_repeats / batch_size)
                    total_steps = steps_per_epoch * max_epochs
            else:
                # 图片训练模式：使用原有逻辑
                steps_per_epoch = math.ceil(dataset_info['file_count'] * num_repeats / batch_size)
                total_steps = steps_per_epoch * max_epochs
        
        # 生成预计文件列表
        # 对于视频训练模式，使用重新计算的步数
        if is_video_training:
            try:
                import toml
                config_data = toml.load(dataset_config_path)
                datasets = config_data.get('datasets', [])
                
                if datasets:
                    dataset = datasets[0]
                    target_frames = dataset.get('target_frames', [1, 25, 49, 73])
                    frame_sample = dataset.get('frame_sample', 8)
                    # 使用前端传递的num_repeats值，不再从TOML文件读取
                    # 正确公式：视频文件数 × target_frames长度 × frame_sample × num_repeats
                    effective_samples = dataset_info['file_count'] * len(target_frames) * frame_sample * num_repeats
                    expected_steps_per_epoch = math.ceil(effective_samples / batch_size)
                    expected_files = generate_expected_files_list(output_dir, output_name, max_epochs, save_every_n_epochs, config.get('save_state', False), expected_steps_per_epoch)
                else:
                    expected_files = generate_expected_files_list(output_dir, output_name, max_epochs, save_every_n_epochs, config.get('save_state', False), steps_per_epoch)
            except Exception as e:
                expected_files = generate_expected_files_list(output_dir, output_name, max_epochs, save_every_n_epochs, config.get('save_state', False), steps_per_epoch)
        else:
            expected_files = generate_expected_files_list(output_dir, output_name, max_epochs, save_every_n_epochs, config.get('save_state', False), steps_per_epoch)
        
        # 生成记录文件内容
        if is_video_training:
            # 视频训练模式的记录内容
            try:
                import toml
                config_data = toml.load(dataset_config_path)
                datasets = config_data.get('datasets', [])
                
                if datasets:
                    dataset = datasets[0]
                    target_frames = dataset.get('target_frames', [1, 25, 49, 73])
                    frame_sample = dataset.get('frame_sample', 8)
                    # 使用前端传递的num_repeats值，不再从TOML文件读取
                    # 正确公式：视频文件数 × target_frames长度 × frame_sample × num_repeats
                    effective_samples = dataset_info['file_count'] * len(target_frames) * frame_sample * num_repeats
                    # 重新计算步数用于.md文件显示
                    md_steps_per_epoch = math.ceil(effective_samples / batch_size)
                    md_total_steps = md_steps_per_epoch * max_epochs
                    
                    content = f"""# 训练记录 - {output_name}

## 基本信息
**开始执行时间：** {start_time}
**训练模式：** 视频训练 ({task_type})

**执行命令：**
```bash
{cmd_str}
```

## 训练参数
**数据集文件数量：** {dataset_info['file_count']}个
**批次大小：** {batch_size}
**重复次数：** {num_repeats}
**训练轮次：** {max_epochs}
**每N轮保存：** {save_every_n_epochs}

## 视频训练特殊参数
**目标帧数配置：** {target_frames}
**每个长度桶采样数：** {frame_sample}
**有效训练样本数：** {effective_samples} (视频片段数)
**计算公式：** {dataset_info['file_count']} × {len(target_frames)} × {frame_sample} × {num_repeats} = {effective_samples}
**总步数：** {md_total_steps}
**每轮步数：** {md_steps_per_epoch}

## 数据集详情
**数据集配置文件：** {dataset_config_path}
**总视频时长：** {dataset_info['total_duration']:.1f}秒
**平均视频长度：** {dataset_info['avg_duration']:.1f}秒

## 预计生成文件列表
{expected_files}

---
*此文件由 Wan2.2 WebUI 自动生成于 {start_time}*
"""
                else:
                    # 无法读取配置时的默认内容
                    content = f"""# 训练记录 - {output_name}

## 基本信息
**开始执行时间：** {start_time}
**训练模式：** 视频训练 ({task_type})

**执行命令：**
```bash
{cmd_str}
```

## 训练参数
**数据集文件数量：** {dataset_info['file_count']}个
**批次大小：** {batch_size}
**重复次数：** {num_repeats}
**训练轮次：** {max_epochs}
**每N轮保存：** {save_every_n_epochs}
**总步数：** {total_steps}
**每轮步数：** {steps_per_epoch}

## 数据集详情
**数据集配置文件：** {dataset_config_path}
**总视频时长：** {dataset_info['total_duration']:.1f}秒
**平均视频长度：** {dataset_info['avg_duration']:.1f}秒

## 预计生成文件列表
{expected_files}

---
*此文件由 Wan2.2 WebUI 自动生成于 {start_time}*
"""
            except Exception as e:
                # 异常时的默认内容
                content = f"""# 训练记录 - {output_name}

## 基本信息
**开始执行时间：** {start_time}
**训练模式：** 视频训练 ({task_type})

**执行命令：**
```bash
{cmd_str}
```

## 训练参数
**数据集文件数量：** {dataset_info['file_count']}个
**批次大小：** {batch_size}
**重复次数：** {num_repeats}
**训练轮次：** {max_epochs}
**每N轮保存：** {save_every_n_epochs}
**总步数：** {total_steps}
**每轮步数：** {steps_per_epoch}

## 数据集详情
**数据集配置文件：** {dataset_config_path}
**总视频时长：** {dataset_info['total_duration']:.1f}秒
**平均视频长度：** {dataset_info['avg_duration']:.1f}秒

## 预计生成文件列表
{expected_files}

---
*此文件由 Wan2.2 WebUI 自动生成于 {start_time}*
"""
        else:
            # 图片训练模式的记录内容
            # 计算预估时间
            total_seconds = total_steps * avg_step_time
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            seconds = int(total_seconds % 60)
            
            if hours > 0:
                estimated_time = f"{hours} 小时 {minutes} 分钟 {seconds} 秒"
            elif minutes > 0:
                estimated_time = f"{minutes} 分钟 {seconds} 秒"
            else:
                estimated_time = f"{seconds} 秒"
            
            content = f"""# 训练记录 - {output_name}

## 基本信息
**开始执行时间：** {start_time}
**训练模式：** 图片训练

**执行命令：**
```bash
{cmd_str}
```

## 训练参数
**数据集文件数量：** {dataset_info['file_count']}个
**批次大小：** {batch_size}
**重复次数：** {num_repeats}
**训练轮次：** {max_epochs}
**每N轮保存：** {save_every_n_epochs}

## 训练步数计算
**第一步：计算总步数**
总步数 = (数据集文件数 × 重复次数 ÷ 批次大小) × 训练轮次
总步数 = ({dataset_info['file_count']} × {num_repeats} ÷ {batch_size}) × {max_epochs} = {total_steps} 步

**第二步：计算预估时间**
预估时间 = 总步数 × 每步耗时
预估时间 = {total_steps} × {avg_step_time} = {total_seconds:.1f} 秒 ({estimated_time})

## 数据集详情
**数据集配置文件：** {dataset_config_path}

## 预计生成文件列表
{expected_files}

---
*此文件由 Wan2.2 WebUI 自动生成于 {start_time}*
"""
        
        # 写入记录文件
        with open(record_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        add_log(f"✅ 训练记录文件已生成: {record_file}", 'info')
        return record_file
        
    except Exception as e:
        add_log(f"❌ 生成训练记录文件失败: {str(e)}", 'error')
        return None

def get_dataset_info(dataset_config_path: str, video_duration: float = 30.0) -> Dict[str, Any]:
    """获取数据集信息"""
    info = {
        'file_count': 0,
        'total_duration': 0.0,
        'avg_duration': 0.0
    }
    
    try:
        if not dataset_config_path or not os.path.exists(dataset_config_path):
            return info
        
        # 读取TOML配置文件
        import toml
        config = toml.load(dataset_config_path)
        
        # 获取数据集路径
        datasets = config.get('datasets', [])
        if not datasets:
            return info
        
        total_files = 0
        total_duration = 0.0
        
        for dataset in datasets:
            # 支持视频训练和图片训练的不同字段名
            data_dir = dataset.get('video_directory', '') or dataset.get('image_directory', '')
            if os.path.exists(data_dir):
                # 统计txt文件数量（训练数据的标注文件）
                files = [f for f in os.listdir(data_dir) 
                        if f.lower().endswith('.txt')]
                total_files += len(files)
                
                # 检查是否为视频训练（有video_directory字段）
                if dataset.get('video_directory'):
                    # 视频训练：统计对应的视频文件并使用传入的视频时长
                    video_files = [f for f in os.listdir(data_dir) 
                                 if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))]
                    # 使用传入的video_duration参数
                    total_duration += len(video_files) * video_duration
                else:
                    # 图片训练：假设每个图片对应5秒
                    total_duration += len(files) * 5.0
        
        info['file_count'] = total_files
        info['total_duration'] = total_duration
        info['avg_duration'] = total_duration / total_files if total_files > 0 else 0.0
        
    except Exception as e:
        add_log(f"获取数据集信息失败: {str(e)}", 'warning')
    
    return info

def generate_expected_files_list(output_dir: str, output_name: str, max_epochs: int, save_every_n_epochs: int, save_state: bool, steps_per_epoch: int = 0) -> str:
    """生成预计文件列表"""
    files_list = []
    
    # LoRA模型文件
    save_count = 0
    for epoch in range(save_every_n_epochs, max_epochs + 1, save_every_n_epochs):
        save_count += 1
        lora_file = f"{output_name}-{epoch:06d}.safetensors"
        
        # 计算步数范围
        if steps_per_epoch > 0:
            start_step = (epoch - save_every_n_epochs) * steps_per_epoch
            end_step = epoch * steps_per_epoch - 1
            # 如果是第一次保存，起始步数为0
            if save_count == 1:
                start_step = 0
                end_step = epoch * steps_per_epoch - 1
            
            files_list.append(f"{lora_file}")
            files_list.append(f"第{save_count}次保存")
            files_list.append(f"轮次: {epoch} 步数: {start_step}~{end_step}")
        else:
            files_list.append(f"{lora_file}")
            files_list.append(f"第{save_count}次保存")
            files_list.append(f"轮次: {epoch}")
        
        # 如果启用了save_state，还会生成状态文件夹
        if save_state:
            state_dir = f"{output_name}-{epoch:06d}-state/"
            files_list.append(f"{state_dir} (训练状态目录)")
        
        files_list.append("")  # 添加空行分隔
    
    # 最终模型文件
    final_lora = f"{output_name}.safetensors"
    files_list.append(f"{final_lora}")
    files_list.append("最终合并模型")
    files_list.append("")
    
    # 日志文件
    files_list.append("训练日志文件 (在 logs/ 目录下)")
    
    return "\n".join(files_list)

def run_training_command(config: Dict[str, Any]):
    """运行训练命令"""
    global training_process
    
    try:
        # 获取虚拟环境Python路径
        venv_python, use_venv = get_venv_python_path(config)
        
        # 显示虚拟环境信息
        if use_venv:
            add_log(f"启用虚拟环境，Python路径: {venv_python}", 'info')
        else:
            add_log("未启用虚拟环境，使用系统Python", 'info')
        
        # 构建训练命令
        if use_venv:
            # 使用虚拟环境的accelerate-launch.exe
            venv_accelerate_launch = os.path.join(os.path.dirname(venv_python), 'accelerate-launch.exe')
            if os.path.exists(venv_accelerate_launch):
                cmd = [venv_accelerate_launch, '--config_file', './accelerate_config.yaml']
                add_log(f"使用虚拟环境accelerate-launch: {venv_accelerate_launch}", 'debug')
            else:
                # 如果accelerate-launch.exe不存在，使用python -m accelerate
                cmd = [venv_python, '-m', 'accelerate', 'launch', '--config_file', './accelerate_config.yaml']
                add_log(f"使用虚拟环境Python执行accelerate模块: {venv_python}", 'debug')
        else:
            cmd = ['accelerate', 'launch', '--config_file', './accelerate_config.yaml']
            add_log("使用系统accelerate", 'debug')
        
        # 添加accelerate参数
        cmd.extend([
            '--num_cpu_threads_per_process', str(config.get('num_cpu_threads_per_process', 1)),
            'src/musubi_tuner/wan_train_network.py'
        ])
        
        # 添加必需参数
        required_params = ['task', 'dit', 'vae', 't5', 'dataset_config']
        for param in required_params:
            if param in config and config[param]:
                cmd.extend([f'--{param}', str(config[param])])
            else:
                add_log(f"缺少必需参数: {param}", 'error')
                return
        
        # 添加可选参数
        optional_params = [
            'dit_high_noise', 'timestep_sampling', 'weighting_scheme', 'discrete_flow_shift',
            'timestep_boundary', 'min_timestep', 'max_timestep',
            'optimizer_type', 'learning_rate', 'network_module', 'network_dim', 'network_alpha',
            'max_train_epochs', 'save_every_n_epochs', 'seed',
            'max_data_loader_n_workers', 'output_dir', 'output_name', 'logging_dir', 'log_with',
            'mixed_precision'
        ]
        
        for param in optional_params:
            if param in config and config[param] != '':
                cmd.extend([f'--{param}', str(config[param])])
        
        # 处理resume参数 - 只有当resume_training为true且resume路径不为空时才添加
        if config.get('resume_training', False) and config.get('resume', '').strip():
            cmd.extend(['--resume', str(config['resume'])])
        
        # 处理互斥的内存优化参数
        if config.get('offload_inactive_dit', False):
            cmd.append('--offload_inactive_dit')
        elif config.get('blocks_to_swap', 0) > 0:
            cmd.extend(['--blocks_to_swap', str(config['blocks_to_swap'])])
        
        # 添加FP8相关布尔参数
        fp8_params = ['fp8_scaled', 'fp8_t5', 'fp8_base']
        for param in fp8_params:
            if config.get(param, False):
                cmd.append(f'--{param}')
        
        # 添加其他布尔参数
        bool_params = [
            'sdpa', 'gradient_checkpointing', 'persistent_data_loader_workers',
            'vae_cache_cpu', 'save_state'
        ]
        
        for param in bool_params:
            if config.get(param, False):
                cmd.append(f'--{param}')
        

        
        # 显示完整的执行命令
        add_log("开始训练")
        cmd_str = ' '.join(cmd)
        if use_venv:
            add_log(f"执行命令 (虚拟环境): {cmd_str}", 'info')
        else:
            add_log(f"执行命令 (系统环境): {cmd_str}", 'info')
        
        # 生成训练记录文件
        try:
            # 从config中获取avg_step_time参数，默认值为2.92
            avg_step_time = config.get('avg_step_time', 2.92)
            generate_training_record_file(config, cmd, avg_step_time)
            add_log("训练记录文件已生成", 'info')
        except Exception as e:
            add_log(f"生成训练记录文件失败: {e}", 'warning')
        
        # 启动训练进程
        with process_lock:
            if os.name == 'nt':  # Windows
                training_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='gbk',
                    errors='replace',
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                )
            else:  # Unix/Linux
                training_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    preexec_fn=os.setsid
                )
        
        # 在新线程中读取输出并通过WebSocket实时发送
        def read_output():
            try:
                for line in iter(training_process.stdout.readline, ''):
                    if line:
                        stripped_line = line.strip()
                        # 添加到日志列表
                        log_entry = {
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'level': 'info',
                            'message': stripped_line
                        }
                        training_logs.append(log_entry)
                        
                        # 限制日志数量
                        if len(training_logs) > 1000:
                            training_logs.pop(0)
                        
                        # 写入Python日志
                        logger.info(stripped_line)
                        
                        # 通过WebSocket实时广播
                        socketio.emit('log_message', log_entry)
                        
                training_process.stdout.close()
            except Exception as e:
                error_msg = f"读取训练输出时出错: {str(e)}"
                add_log(error_msg, 'error')
        
        output_thread = threading.Thread(target=read_output)
        output_thread.daemon = True
        output_thread.start()
        
        # 等待进程完成
        training_process.wait()
        
        if training_process.returncode == 0:
            add_log("✅ 训练完成！", 'success')
        else:
            add_log(f"❌ 训练失败，退出代码: {training_process.returncode}", 'error')
            
    except Exception as e:
        add_log(f"❌ 训练过程出错: {str(e)}", 'error')
    finally:
        training_process = None

def run_cache_command(cache_type: str, config: Dict[str, Any]):
    """运行缓存命令"""
    global cache_process, cache_type_running
    
    try:
        # 获取虚拟环境Python路径
        venv_python, use_venv = get_venv_python_path(config)
        
        # 显示虚拟环境信息
        if use_venv:
            add_log(f"启用虚拟环境，Python路径: {venv_python}", 'info')
        else:
            add_log("未启用虚拟环境，使用系统Python", 'info')
        
        if cache_type == 'vae':
            cmd = [
                venv_python, 'src/musubi_tuner/wan_cache_latents.py',
                '--dataset_config', config.get('dataset_config', ''),
                '--vae', config.get('vae', '')
            ]
            
            # 添加I2V选项
            if config.get('task', '').startswith('i2v'):
                cmd.append('--i2v')
            
            # 添加可选参数
            if config.get('vae_cache_cpu', False):
                cmd.append('--vae_cache_cpu')
            
            # Wan2.2不需要CLIP，仅Wan2.1的I2V任务需要
            # if config.get('clip') and not config.get('task', '').endswith('A14B'):
            #     cmd.extend(['--clip', config.get('clip')])
                
        elif cache_type == 'text_encoder':
            cmd = [
                venv_python, 'src/musubi_tuner/wan_cache_text_encoder_outputs.py',
                '--dataset_config', config.get('dataset_config', ''),
                '--t5', config.get('t5', ''),
                '--batch_size', str(config.get('batch_size', 16))
            ]
            
            # 添加FP8选项
            if config.get('fp8_t5', False):
                cmd.append('--fp8_t5')
        
        # 验证必要参数
        if cache_type == 'vae' and not config.get('vae'):
            add_log("VAE缓存失败: VAE模型路径未设置", 'error')
            cache_process = None
            cache_type_running = None
            return
        elif cache_type == 'text_encoder' and not config.get('t5'):
            add_log("文本编码器缓存失败: T5模型路径未设置", 'error')
            cache_process = None
            cache_type_running = None
            return
            
        # 显示完整的执行命令
        add_log(f"开始{cache_type}缓存")
        cmd_str = ' '.join(cmd)
        if use_venv:
            add_log(f"执行命令 (虚拟环境): {cmd_str}", 'info')
        else:
            add_log(f"执行命令 (系统环境): {cmd_str}", 'info')
            
        # 运行缓存命令 - 启动进程但不等待
        import os
        if os.name == 'nt':  # Windows
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='gbk',
                errors='replace',
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )
        else:  # Unix/Linux
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                preexec_fn=os.setsid
            )
        
        # 设置全局变量
        cache_process = process
        cache_type_running = cache_type
        
        # 启动输出读取和监控线程
        import threading
        
        def read_and_monitor():
            global cache_process, cache_type_running
            try:
                # 读取输出并通过WebSocket实时发送
                for line in iter(process.stdout.readline, ''):
                    if line:
                        stripped_line = line.strip()
                        # 添加到日志列表
                        log_entry = {
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'level': 'info',
                            'message': stripped_line
                        }
                        training_logs.append(log_entry)
                        
                        # 限制日志数量
                        if len(training_logs) > 1000:
                            training_logs.pop(0)
                        
                        # 写入Python日志
                        logger.info(stripped_line)
                        
                        # 通过WebSocket实时广播
                        socketio.emit('log_message', log_entry)
                        
                process.stdout.close()
                
                # 等待进程完成
                process.wait()
                
                if process.returncode == 0:
                    add_log(f"✅ {cache_type}缓存完成！", 'success')
                else:
                    add_log(f"❌ {cache_type}缓存失败，返回码: {process.returncode}", 'error')
                    
            except Exception as e:
                error_msg = f"读取缓存输出时出错: {str(e)}"
                add_log(error_msg, 'error')
            finally:
                # 清理全局变量
                cache_process = None
                cache_type_running = None
        
        # 启动读取和监控线程
        monitor_thread = threading.Thread(target=read_and_monitor)
        monitor_thread.daemon = True
        monitor_thread.start()
            
    except Exception as e:
        add_log(f"❌ {cache_type}缓存过程出错: {str(e)}", 'error')
        cache_process = None
        cache_type_running = None

def run_convert_lora_command(input_file: str, output_file: str, config: Dict[str, Any]):
    """运行LoRA转换命令"""
    global cache_process
    
    try:
        with process_lock:
            if cache_process and cache_process.poll() is None:
                add_log("已有转换进程在运行，请等待完成", 'warning')
                return
        
        # 显示开始转换提示
        add_log("开始LoRA转换...", 'info')
        
        # 获取Python路径
        python_path_info = get_venv_python_path(config)
        use_venv = config.get('enable_venv', True) and python_path_info
        
        if use_venv:
            python_path = python_path_info[0] if isinstance(python_path_info, tuple) else python_path_info
            add_log(f"启用虚拟环境，Python路径: {python_path}", 'info')
        else:
            python_path = sys.executable
            add_log(f"使用系统环境，Python路径: {python_path}", 'info')
        
        # 如果输出文件已存在，先删除它以避免编码错误
        if os.path.exists(output_file):
            try:
                os.remove(output_file)
                add_log(f"删除已存在的输出文件: {output_file}", 'info')
            except Exception as e:
                add_log(f"删除输出文件失败: {e}", 'warning')
        
        # 构建转换命令
        cmd = [
            python_path,
            '-m', 'musubi_tuner.convert_lora',
            '--input', input_file,
            '--output', output_file,
            '--target', 'other'  # 转换为ComfyUI等其他环境可用的格式
        ]
        
        # 显示详细的执行命令信息
        add_log("开始lora转换", 'info')
        cmd_str = ' '.join(cmd)
        if use_venv:
            add_log(f"执行命令 (虚拟环境): {cmd_str}", 'info')
        else:
            add_log(f"执行命令 (系统环境): {cmd_str}", 'info')
        
        # 启动转换进程
        with process_lock:
            cache_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                bufsize=1,
                universal_newlines=True
            )
        
        add_log("LoRA转换启动成功", 'success')
        
        # 实时读取输出
        while True:
            output = cache_process.stdout.readline()
            if output == '' and cache_process.poll() is not None:
                break
            if output:
                line = output.strip()
                if line:
                    add_log(f"[转换] {line}", 'info')
                    socketio.emit('log_update', {
                        'message': f"[转换] {line}",
                        'level': 'info',
                        'timestamp': datetime.now().strftime('%H:%M:%S')
                    })
        
        # 等待进程完成
        cache_process.wait()
        
        if cache_process.returncode == 0:
            add_log("LoRA转换完成", 'success')
        else:
            add_log(f"LoRA转换失败，退出代码: {cache_process.returncode}", 'error')
            
    except Exception as e:
        add_log(f"LoRA转换过程出错: {str(e)}", 'error')
        cache_process = None


@app.route('/api/calculate_doubao_recommended_params', methods=['POST'])
def calculate_doubao_recommended_params():
    """豆包智能推荐所有训练参数"""
    try:
        data = request.get_json()
        dataset_config = data.get('dataset_config')
        task_type = data.get('task_type', 't2v')
        is_video_training = data.get('is_video_training', True)
        video_duration = data.get('video_duration', 5)
        batch_size = data.get('batch_size', 1)
        
        if not dataset_config or not os.path.exists(dataset_config):
            return jsonify({'success': False, 'message': '数据集配置文件不存在'})
        
        # 读取TOML文件获取数据集信息
        import toml
        import math
        
        config = toml.load(dataset_config)
        datasets = config.get('datasets', [])
        if not datasets:
            return jsonify({'success': False, 'message': 'TOML文件中未找到datasets配置'})
        
        # 根据训练类型获取正确的数据集目录
        if is_video_training:
            data_directory = datasets[0].get('video_directory')
            directory_type = 'video_directory'
        else:
            data_directory = datasets[0].get('image_directory')
            directory_type = 'image_directory'
            
        if not data_directory or not os.path.exists(data_directory):
            return jsonify({'success': False, 'message': '数据集目录不存在'})
        
        # 统计txt文件数量
        txt_files = [f for f in os.listdir(data_directory) if f.endswith('.txt')]
        dataset_files = len(txt_files)
        
        if dataset_files == 0:
            return jsonify({'success': False, 'message': '数据集目录中未找到txt文件'})
        
        # 根据是否为视频训练调整参数
        if is_video_training:
            total_video_seconds = video_duration
        else:
            total_video_seconds = 0  # 非视频训练时不考虑视频时长
        
        # 使用豆包推荐算法
        recommended_params = recommend_wan22_lora_params(
            dataset_files, total_video_seconds, task_type, batch_size
        )
        
        # 生成推荐理由
        training_mode = "视频训练" if is_video_training else "图片训练"
        duration_info = f"• 视频总时长：{video_duration} 秒" if is_video_training else "• 训练模式：图片训练（不考虑视频时长）"
        
        reasoning = f"""🧠 豆包智能推荐分析：
        
📊 数据集分析：
• 训练文件数量：{dataset_files} 个
{duration_info}
• 任务类型：{task_type}
• 训练模式：{training_mode}
• 批次大小：{batch_size}

🎯 推荐参数说明：
• 学习率：{recommended_params['learning_rate']} - 基于数据量和任务复杂度优化
• LoRA维度：{recommended_params['network_dim']} - 平衡模型容量和训练效率
• LoRA Alpha：{recommended_params['network_alpha']} - 控制LoRA适配强度
• 训练轮数：{recommended_params['max_train_epochs']} - 确保充分学习而不过拟合
• 重复次数：{recommended_params['num_repeats']} - 优化数据利用效率

💡 豆包算法特点：
• 采用数学建模方法精确计算最优参数
• 考虑视频时长对训练复杂度的影响
• 针对不同任务类型进行专门优化
• 平衡训练效果与计算资源消耗"""
        
        return jsonify({
            'success': True,
            'recommended_params': recommended_params,
            'reasoning': reasoning
        })
        
    except Exception as e:
        logger.error(f"豆包推荐参数计算失败: {e}")
        return jsonify({'success': False, 'message': f'计算失败: {str(e)}'})


@app.route('/api/calculate_software_recommended_params', methods=['POST'])
def calculate_software_recommended_params():
    """软件智能推荐所有训练参数"""
    try:
        data = request.get_json()
        dataset_config = data.get('dataset_config')
        task_type = data.get('task_type', 't2v')
        is_video_training = data.get('is_video_training', False)
        video_duration = data.get('video_duration', 0)
        batch_size = data.get('batch_size', 1)
        lora_type = data.get('lora_type', 'realistic')  # 'realistic' 或 'anime'
        
        if not dataset_config or not os.path.exists(dataset_config):
            return jsonify({'success': False, 'message': '数据集配置文件不存在'})
        
        # 读取TOML文件获取数据集信息
        import toml
        
        config = toml.load(dataset_config)
        datasets = config.get('datasets', [])
        if not datasets:
            return jsonify({'success': False, 'message': 'TOML文件中未找到datasets配置'})
        
        # 根据训练类型获取正确的数据集目录
        if is_video_training:
            data_directory = datasets[0].get('video_directory')
            directory_type = 'video_directory'
        else:
            data_directory = datasets[0].get('image_directory')
            directory_type = 'image_directory'
            
        if not data_directory or not os.path.exists(data_directory):
            return jsonify({'success': False, 'message': '数据集目录不存在'})
        
        # 统计txt文件数量
        txt_files = [f for f in os.listdir(data_directory) if f.endswith('.txt')]
        dataset_files = len(txt_files)
        
        if dataset_files == 0:
            return jsonify({'success': False, 'message': '数据集目录中未找到txt文件'})
        
        # 根据是否为视频训练调整参数
        total_video_seconds = video_duration if is_video_training else 0
        
        # 使用软件推荐算法
        recommended_params = recommend_software_lora_params(
            dataset_files, total_video_seconds, lora_type, task_type, batch_size
        )
        
        # 生成推荐理由
        training_mode = "视频训练" if is_video_training else "图片训练"
        lora_type_name = "写实风格" if lora_type == 'realistic' else "动漫风格"
        
        if is_video_training:
            equivalent_frames = int(video_duration * 4)  # 4fps抽帧
            duration_info = f"• 视频总时长：{video_duration} 秒\n• 等效帧数：{equivalent_frames} 帧（按4fps抽帧计算）"
        else:
            duration_info = "• 训练模式：图片训练（不考虑视频时长）"
        
        # 根据数据量级别给出说明
        data_count = recommended_params['data_count']
        
        if is_video_training:
            # 视频训练按等效帧数分级
            if data_count <= 1200:
                data_level = "小规模视频数据（≤1200帧）"
                strategy = "高重复次数训练，确保充分学习视频特征"
            elif data_count <= 3000:
                data_level = "中等规模视频数据（1200-3000帧）"
                strategy = "平衡重复次数和维度，优化视频训练效果"
            else:
                data_level = "大规模视频数据（>3000帧）"
                strategy = "降低重复次数，防止过拟合大量视频数据"
        else:
            # 图片训练按原有逻辑分级
            if data_count <= 20:
                data_level = "极小数据集"
                strategy = "高重复次数 + 适中维度，防止过拟合"
            elif data_count <= 60:
                data_level = "小数据集"
                strategy = "平衡重复次数和学习率，确保充分学习"
            elif data_count <= 100:
                data_level = "中等数据集"
                strategy = "降低重复次数，提高学习效率"
            elif data_count <= 500:
                data_level = "大数据集"
                strategy = "减少重复和轮数，避免过拟合"
            else:
                data_level = "超大数据集"
                strategy = "最小重复次数，快速收敛"
        
        reasoning = f"""⚙️ 软件智能推荐分析：
        
📊 数据集分析：
• 训练文件数量：{dataset_files} 个
{duration_info}
• 任务类型：{task_type}
• 训练模式：{training_mode}
• LoRA类型：{lora_type_name}
• 数据量级别：{data_level}
• 批次大小：{batch_size}

🎯 推荐参数说明：
• 学习率：{recommended_params['learning_rate']} - 基于{lora_type_name}优化的学习率
• LoRA维度：{recommended_params['network_dim']} - {lora_type_name}推荐维度配置
• LoRA Alpha：{recommended_params['network_alpha']} - 匹配维度的Alpha值
• 训练轮数：{recommended_params['max_train_epochs']} - 根据数据量调整的轮数
• 重复次数：{recommended_params['num_repeats']} - {data_level}优化的重复策略

💡 推荐策略：
• {strategy}
• {lora_type_name}专用参数优化
• 考虑批次大小对学习率的影响
• 平衡训练效果与计算效率"""
        
        return jsonify({
            'success': True,
            'recommended_params': recommended_params,
            'reasoning': reasoning
        })
        
    except Exception as e:
        logger.error(f"软件推荐参数计算失败: {e}")
        return jsonify({'success': False, 'message': f'计算失败: {str(e)}'})


        # 检查转换结果
        return_code = cache_process.poll()
        if return_code == 0:
            success_msg = f"LoRA转换完成: {Path(output_file).name}"
            add_log(success_msg, 'success')
            socketio.emit('log_update', {
                'message': success_msg,
                'level': 'success',
                'timestamp': datetime.now().strftime('%H:%M:%S')
            })
        else:
            error_msg = f"LoRA转换失败，退出码: {return_code}"
            add_log(error_msg, 'error')
            socketio.emit('log_update', {
                'message': error_msg,
                'level': 'error',
                'timestamp': datetime.now().strftime('%H:%M:%S')
            })
            
    except Exception as e:
        error_msg = f"LoRA转换过程中出错: {str(e)}"
        add_log(error_msg, 'error')
        socketio.emit('log_update', {
            'message': error_msg,
            'level': 'error',
            'timestamp': datetime.now().strftime('%H:%M:%S')
        })
    finally:
        with process_lock:
            cache_process = None

def start_tensorboard_process(log_dir: str, config: Dict[str, Any] = None):
    """启动TensorBoard进程"""
    global tensorboard_process
    
    try:
        # 检查日志目录
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            add_log(f"创建日志目录: {log_dir}")
        
        # 检查是否已有TensorBoard运行
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'tensorboard' in proc.info['name'].lower():
                    if any('--logdir' in arg for arg in proc.info['cmdline']):
                        add_log("检测到已运行的TensorBoard进程")
                        return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # 获取虚拟环境Python路径
        if config:
            venv_python, use_venv = get_venv_python_path(config)
            if use_venv:
                # 使用虚拟环境的tensorboard
                venv_tensorboard = os.path.join(os.path.dirname(venv_python), 'tensorboard.exe')
                if os.path.exists(venv_tensorboard):
                    cmd = [venv_tensorboard, '--logdir', log_dir, '--port', '6006']
                    add_log(f"使用虚拟环境tensorboard: {venv_tensorboard}")
                else:
                    # 如果tensorboard.exe不存在，使用python -m tensorboard
                    cmd = [venv_python, '-m', 'tensorboard', '--logdir', log_dir, '--port', '6006']
                    add_log(f"使用虚拟环境Python执行tensorboard模块: {venv_python}")
            else:
                cmd = ['tensorboard', '--logdir', log_dir, '--port', '6006']
                add_log("使用系统tensorboard")
        else:
            cmd = ['tensorboard', '--logdir', log_dir, '--port', '6006']
            add_log("使用系统tensorboard")
        
        add_log(f"启动TensorBoard: {' '.join(cmd)}")
        
        if os.name == 'nt':  # Windows
            tensorboard_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='gbk',
                errors='ignore',
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )
        else:  # Unix/Linux
            tensorboard_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='ignore'
            )
        
        # 等待启动
        time.sleep(3)
        
        if tensorboard_process.poll() is None:
            add_log("TensorBoard启动成功，访问地址: http://localhost:6006")
            return True
        else:
            add_log("TensorBoard启动失败", 'error')
            return False
            
    except Exception as e:
        add_log(f"启动TensorBoard失败: {str(e)}", 'error')
        return False

# Flask路由
@app.route('/')
def index():
    """主页"""
    return render_template('wan22.html')

@app.route('/api/config', methods=['GET'])
def get_config():
    """获取配置"""
    config = load_config()
    return jsonify({
        'success': True,
        'config': config,
        'param_descriptions': PARAM_DESCRIPTIONS
    })

@app.route('/api/load_config', methods=['GET'])
def load_config_api():
    """加载配置文件"""
    try:
        config = load_config()
        
        # 尝试从TOML文件读取batch_size和num_repeats（优先使用配置中的路径，否则使用默认路径）
        dataset_config_path = None
        if 'dataset_config' in config and config['dataset_config']:
            dataset_config_path = Path(config['dataset_config'])
        else:
            # 使用默认的TOML文件路径
            dataset_config_path = Path(DEFAULT_CONFIG['dataset_config'])
        
        if dataset_config_path and dataset_config_path.exists():
            try:
                import toml
                with open(dataset_config_path, 'r', encoding='utf-8') as f:
                    toml_config = toml.load(f)
                
                # 从TOML文件中读取batch_size（从[general]部分）
                if 'general' in toml_config and 'batch_size' in toml_config['general']:
                    config['batch_size'] = toml_config['general']['batch_size']
                    add_log(f"从 {dataset_config_path} 读取 batch_size: {config['batch_size']}", 'info')
                
                # 从TOML文件中读取num_repeats（先检查[general]，再检查[[datasets]]）
                if 'general' in toml_config and 'num_repeats' in toml_config['general']:
                    config['num_repeats'] = toml_config['general']['num_repeats']
                    add_log(f"从 {dataset_config_path} [general] 读取 num_repeats: {config['num_repeats']}", 'info')
                elif 'datasets' in toml_config and isinstance(toml_config['datasets'], list) and len(toml_config['datasets']) > 0:
                    if 'num_repeats' in toml_config['datasets'][0]:
                        config['num_repeats'] = toml_config['datasets'][0]['num_repeats']
                        add_log(f"从 {dataset_config_path} [[datasets]] 读取 num_repeats: {config['num_repeats']}", 'info')
            except Exception as e:
                add_log(f"读取数据集配置文件失败: {e}", 'warning')
        
        return jsonify({'success': True, 'config': config})
    except Exception as e:
        add_log(f"加载配置失败: {e}", 'error')
        return jsonify({'success': False, 'message': f'加载配置失败: {e}', 'config': DEFAULT_CONFIG.copy()})

@app.route('/api/save_config', methods=['POST'])
def save_config_api():
    """保存配置"""
    try:
        config = request.json
        
        # 保存到JSON文件
        if not save_config(config):
            return jsonify({'success': False, 'message': '配置保存失败'})
        
        # 保存批次大小和重复次数到TOML文件
        if 'dataset_config' in config and ('batch_size' in config or 'num_repeats' in config):
            # 使用用户指定的数据集配置文件路径
            toml_file = Path(config['dataset_config'])
            toml_content = ""
            
            # 如果TOML文件存在，读取现有内容
            if toml_file.exists():
                with open(toml_file, 'r', encoding='utf-8') as f:
                    toml_content = f.read()
            
            import re
            updated_params = []
            
            # 处理batch_size
            if 'batch_size' in config:
                batch_size_pattern = r'batch_size\s*=\s*\d+'
                new_batch_size = f"batch_size = {config['batch_size']}"
                
                if '[general]' in toml_content:
                    if re.search(batch_size_pattern, toml_content):
                        toml_content = re.sub(batch_size_pattern, new_batch_size, toml_content)
                    else:
                        toml_content = re.sub(r'(\[general\])', r'\1\n' + new_batch_size, toml_content)
                else:
                    if toml_content and not toml_content.endswith('\n'):
                        toml_content += '\n'
                    toml_content += f"\n[general]\n{new_batch_size}\n"
                updated_params.append(f"batch_size={config['batch_size']}")
            
            # 处理num_repeats - 保存到[[datasets]]部分
            if 'num_repeats' in config:
                new_num_repeats = f"num_repeats = {config['num_repeats']}   # 提高重复次数，少量数据也能收敛"
                
                # 查找[[datasets]]部分到下一个section或文件结尾
                datasets_pattern = r'(\[\[datasets\]\].*?)(?=\n\[|$)'
                datasets_match = re.search(datasets_pattern, toml_content, re.DOTALL)
                
                if datasets_match:
                    # 如果[[datasets]]部分存在
                    datasets_section = datasets_match.group(1)
                    # 删除所有现有的num_repeats行（包括注释）
                    lines = datasets_section.split('\n')
                    filtered_lines = []
                    for line in lines:
                        if not re.match(r'^\s*num_repeats\s*=', line.strip()):
                            filtered_lines.append(line)
                    
                    # 重新构建[[datasets]]部分，在开头添加num_repeats
                    if filtered_lines and filtered_lines[0].strip() == '[[datasets]]':
                        new_section = filtered_lines[0] + '\n' + new_num_repeats
                        if len(filtered_lines) > 1:
                            new_section += '\n' + '\n'.join(filtered_lines[1:])
                    else:
                        new_section = '\n'.join(filtered_lines) + '\n' + new_num_repeats
                    
                    toml_content = toml_content.replace(datasets_section, new_section)
                else:
                    # 如果[[datasets]]部分不存在，创建它
                    if toml_content and not toml_content.endswith('\n'):
                        toml_content += '\n'
                    toml_content += f"\n[[datasets]]\n{new_num_repeats}\n"
                updated_params.append(f"num_repeats={config['num_repeats']}")
            
            # 写入TOML文件
            with open(toml_file, 'w', encoding='utf-8') as f:
                f.write(toml_content)
            
            if updated_params:
                params_str = ', '.join(updated_params)
                add_log(f"配置已保存到 wan_webui_config.json 和 {config['dataset_config']} ({params_str})", 'info')
                return jsonify({'success': True, 'message': f'配置保存成功，已同步到TOML文件 ({params_str})'})
            else:
                add_log("配置已保存到 wan_webui_config.json", 'info')
                return jsonify({'success': True, 'message': '配置保存成功'})
        else:
            add_log("配置已保存到 wan_webui_config.json", 'info')
            return jsonify({'success': True, 'message': '配置保存成功'})
        
    except Exception as e:
        add_log(f"保存配置失败: {e}", 'error')
        return jsonify({'success': False, 'message': f'保存配置失败: {e}'})

@app.route('/api/reset_config', methods=['POST'])
def reset_config_api():
    """重置配置为默认值并保存"""
    try:
        # 获取默认配置的副本
        reset_config = DEFAULT_CONFIG.copy()
        
        # 确保learning_rate使用科学计数法格式
        reset_config['learning_rate'] = '2e-4'
        
        # 保存重置后的配置
        if save_config(reset_config):
            add_log("配置已重置为默认值并保存", 'info')
            return jsonify({'success': True, 'config': reset_config, 'message': '配置已重置为默认值'})
        else:
            return jsonify({'success': False, 'message': '重置配置保存失败'})
    except Exception as e:
        add_log(f"重置配置失败: {str(e)}", 'error')
        return jsonify({'success': False, 'message': f'重置配置失败: {str(e)}'})

@app.route('/api/start_training', methods=['POST'])
def start_training():
    """开始训练"""
    global training_process
    
    if training_process and training_process.poll() is None:
        return jsonify({'success': False, 'message': '训练正在进行中'})
    
    try:
        config = request.json
        # 保存配置
        save_config(config)
        
        # 在新线程中启动训练
        training_thread = threading.Thread(
            target=run_training_command,
            args=(config,)
        )
        training_thread.daemon = True
        training_thread.start()
        
        return jsonify({'success': True, 'message': '训练已开始'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'启动训练失败: {str(e)}'})

def stop_current_process():
    """停止当前运行的进程"""
    global training_process
    
    with process_lock:
        if training_process and training_process.poll() is None:
            try:
                add_log("正在停止训练进程...", 'warning')
                
                if os.name == 'nt':  # Windows
                    # 在Windows上，发送CTRL_BREAK_EVENT信号到进程组
                    try:
                        import signal
                        training_process.send_signal(signal.CTRL_BREAK_EVENT)
                        add_log("已发送停止信号到进程组", 'info')
                    except Exception as e:
                        add_log(f"发送停止信号失败: {str(e)}", 'warning')
                        training_process.terminate()
                else:  # Unix/Linux
                    # 在Unix/Linux上，终止整个进程组
                    import signal
                    os.killpg(os.getpgid(training_process.pid), signal.SIGTERM)
                
                # 等待进程结束，最多等待10秒
                try:
                    training_process.wait(timeout=10)
                    add_log("进程已停止", 'info')
                except subprocess.TimeoutExpired:
                    add_log("进程未在10秒内停止，强制终止", 'warning')
                    
                    if os.name == 'nt':  # Windows
                        # 强制终止进程组
                        try:
                            import subprocess
                            subprocess.run(['taskkill', '/F', '/T', '/PID', str(training_process.pid)], 
                                         capture_output=True, text=True)
                            add_log("已强制终止进程组", 'info')
                        except Exception as e:
                            add_log(f"强制终止进程组失败: {str(e)}", 'error')
                            training_process.kill()
                    else:  # Unix/Linux
                        os.killpg(os.getpgid(training_process.pid), signal.SIGKILL)
                    
                    training_process.wait()
                    add_log("进程已强制终止", 'info')
                    
            except Exception as e:
                add_log(f"停止进程时出错: {str(e)}", 'error')
                return False
            finally:
                training_process = None
            return True
        else:
            add_log("没有正在运行的进程", 'info')
            return True

@app.route('/api/stop_training', methods=['POST'])
def stop_training():
    """停止训练"""
    success = stop_current_process()
    return jsonify({'success': success})

@app.route('/api/cache_vae', methods=['POST'])
def cache_vae():
    """缓存VAE latents"""
    try:
        config = request.json
        # 在新线程中运行缓存
        cache_thread = threading.Thread(
            target=run_cache_command,
            args=('vae', config)
        )
        cache_thread.daemon = True
        cache_thread.start()
        
        return jsonify({'success': True, 'message': 'VAE缓存已开始'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'启动VAE缓存失败: {str(e)}'})

@app.route('/api/cache_text_encoder', methods=['POST'])
def cache_text_encoder():
    """缓存文本编码器输出"""
    try:
        config = request.json
        # 在新线程中运行缓存
        cache_thread = threading.Thread(
            target=run_cache_command,
            args=('text_encoder', config)
        )
        cache_thread.daemon = True
        cache_thread.start()
        
        return jsonify({'success': True, 'message': '文本编码器缓存已开始'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'启动文本编码器缓存失败: {str(e)}'})

@app.route('/api/start_tensorboard', methods=['POST'])
def start_tensorboard():
    """启动TensorBoard"""
    try:
        config = request.json
        log_dir = config.get('logging_dir', './logs')
        
        if start_tensorboard_process(log_dir, config):
            tensorboard_url = 'http://localhost:6006'
            return jsonify({
                'success': True, 
                'message': 'TensorBoard启动成功',
                'url': tensorboard_url
            })
        else:
            return jsonify({'success': False, 'message': 'TensorBoard启动失败'})
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'启动TensorBoard失败: {str(e)}'})

@app.route('/api/logs', methods=['GET'])
def get_logs():
    """获取日志"""
    return jsonify({
        'success': True,
        'logs': training_logs[-100:]  # 返回最近100条日志
    })

@app.route('/api/clear_logs', methods=['POST'])
def clear_logs():
    """清空日志"""
    global training_logs
    training_logs.clear()
    return jsonify({'success': True, 'message': '日志已清空'})



@app.route('/api/convert_lora', methods=['POST'])
def convert_lora():
    """转换LoRA为ComfyUI格式"""
    try:
        config = request.json
        input_file = config.get('input_file')
        output_dir = config.get('output_dir')
        
        if not input_file or not output_dir:
            return jsonify({'success': False, 'message': '请提供输入文件路径和输出目录路径'})
        
        # 验证输入文件存在
        input_path = Path(input_file)
        if not input_path.exists():
            return jsonify({'success': False, 'message': f'输入文件不存在: {input_file}'})
        
        # 验证输出目录存在
        output_path = Path(output_dir)
        if not output_path.exists():
            return jsonify({'success': False, 'message': f'输出目录不存在: {output_dir}'})
        
        # 构建输出文件路径，在原文件名前添加Comfyui_前缀
        original_filename = input_path.name
        new_filename = f'Comfyui_{original_filename}'
        output_file = output_path / new_filename
        
        add_log(f'开始转换LoRA: {original_filename} -> {new_filename}', 'info')
        
        # 在新线程中运行转换
        convert_thread = threading.Thread(
            target=run_convert_lora_command,
            args=(str(input_path), str(output_file), config)
        )
        convert_thread.daemon = True
        convert_thread.start()
        
        return jsonify({
            'success': True, 
            'message': f'LoRA转换已开始: {original_filename} -> {new_filename}'
        })
        
    except Exception as e:
        error_msg = f'启动LoRA转换失败: {str(e)}'
        add_log(error_msg, 'error')
        return jsonify({'success': False, 'message': error_msg})

@app.route('/api/read_batch_size_from_toml', methods=['POST'])
def api_read_batch_size_from_toml():
    """从TOML文件读取batch_size"""
    try:
        data = request.get_json()
        toml_path = data.get('toml_path')
        
        if not toml_path:
            return jsonify({'success': False, 'message': 'TOML文件路径不能为空'})
        
        toml_path = Path(toml_path)
        if not toml_path.exists():
            return jsonify({'success': False, 'message': f'TOML文件不存在: {toml_path}'})
        
        import toml
        with open(toml_path, 'r', encoding='utf-8') as f:
            toml_config = toml.load(f)
        
        batch_size = None
        
        # 从 [general] 部分读取 batch_size
        if 'general' in toml_config and 'batch_size' in toml_config['general']:
            batch_size = toml_config['general']['batch_size']
        
        if batch_size is not None:
            add_log(f"从 {toml_path} 读取 batch_size: {batch_size}", 'info')
            return jsonify({'success': True, 'batch_size': batch_size})
        else:
            return jsonify({'success': False, 'message': 'TOML文件中未找到 batch_size 参数（检查了 [general] 部分）', 'batch_size': None})
    
    except Exception as e:
        add_log(f"读取TOML文件失败: {e}", 'error')
        return jsonify({'success': False, 'message': f'读取TOML文件失败: {e}', 'batch_size': None})

@app.route('/api/read_num_repeats_from_toml', methods=['POST'])
def api_read_num_repeats_from_toml():
    """从TOML文件读取num_repeats"""
    try:
        data = request.get_json()
        toml_path = data.get('toml_path')
        
        if not toml_path:
            return jsonify({'success': False, 'message': 'TOML文件路径不能为空'})
        
        toml_path = Path(toml_path)
        if not toml_path.exists():
            return jsonify({'success': False, 'message': f'TOML文件不存在: {toml_path}'})
        
        import toml
        with open(toml_path, 'r', encoding='utf-8') as f:
            toml_config = toml.load(f)
        
        num_repeats = None
        
        # 首先检查 [general] 部分
        if 'general' in toml_config and 'num_repeats' in toml_config['general']:
            num_repeats = toml_config['general']['num_repeats']
        
        # 如果 [general] 中没有，检查 [[datasets]] 部分
        if num_repeats is None and 'datasets' in toml_config:
            for dataset in toml_config['datasets']:
                if 'num_repeats' in dataset:
                    num_repeats = dataset['num_repeats']
                    break  # 使用第一个找到的 num_repeats
        
        if num_repeats is not None:
            add_log(f"从 {toml_path} 读取 num_repeats: {num_repeats}", 'info')
            return jsonify({'success': True, 'num_repeats': num_repeats})
        else:
            return jsonify({'success': False, 'message': 'TOML文件中未找到 num_repeats 参数（检查了 [general] 和 [[datasets]] 部分）', 'num_repeats': None})
    
    except Exception as e:
        add_log(f"读取TOML文件失败: {e}", 'error')
        return jsonify({'success': False, 'message': f'读取TOML文件失败: {e}', 'num_repeats': None})

@app.route('/api/select_file', methods=['POST'])
def select_file():
    """使用系统文件选择对话框选择文件"""
    if not TKINTER_AVAILABLE:
        return jsonify({'success': False, 'message': 'tkinter不可用，无法使用文件选择对话框'})
    
    try:
        data = request.get_json()
        file_type = data.get('file_type', 'all')  # 文件类型: 'safetensors', 'toml', 'all'
        title = data.get('title', '选择文件')
        initial_dir = data.get('initial_dir', 'C:\\Users\\oskey\\ai\\musubi-tuner')
        
        # 根据文件类型设置文件过滤器 <mcreference link="https://docs.python.org/3/library/dialog.html" index="1">1</mcreference>
        if file_type == 'safetensors':
            filetypes = [('SafeTensors文件', '*.safetensors'), ('所有文件', '*.*')]
        elif file_type == 'toml':
            filetypes = [('TOML配置文件', '*.toml'), ('所有文件', '*.*')]
        else:
            filetypes = [('所有文件', '*.*')]
        
        # 创建隐藏的tkinter根窗口 <mcreference link="https://stackoverflow.com/questions/23775211/flask-desktop-application-file-chooser" index="4">4</mcreference>
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        root.attributes('-topmost', True)  # 确保对话框在最前面
        
        # 打开文件选择对话框 <mcreference link="https://www.pythontutorial.net/tkinter/tkinter-open-file-dialog/" index="3">3</mcreference> <mcreference link="https://pythonspot.com/tk-file-dialogs/" index="5">5</mcreference>
        filename = filedialog.askopenfilename(
            title=title,
            initialdir=initial_dir,
            filetypes=filetypes
        )
        
        # 销毁tkinter根窗口
        root.destroy()
        
        if filename:
            add_log(f"用户选择文件: {filename}", 'info')
            return jsonify({'success': True, 'filename': filename})
        else:
            return jsonify({'success': False, 'message': '用户取消了文件选择'})
            
    except Exception as e:
        add_log(f"文件选择对话框出错: {e}", 'error')
        return jsonify({'success': False, 'message': f'文件选择失败: {str(e)}'})

@app.route('/api/select_folder', methods=['POST'])
def select_folder():
    """使用系统文件夹选择对话框选择文件夹"""
    if not TKINTER_AVAILABLE:
        return jsonify({'success': False, 'message': 'tkinter不可用，无法使用文件夹选择对话框'})
    
    try:
        data = request.get_json()
        title = data.get('title', '选择文件夹')
        initial_dir = data.get('initial_dir', 'C:\\Users\\oskey\\ai\\musubi-tuner')
        
        # 创建隐藏的tkinter根窗口
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        root.attributes('-topmost', True)  # 确保对话框在最前面
        
        # 打开文件夹选择对话框
        folder_path = filedialog.askdirectory(
            title=title,
            initialdir=initial_dir
        )
        
        # 销毁tkinter根窗口
        root.destroy()
        
        if folder_path:
            add_log(f"用户选择文件夹: {folder_path}", 'info')
            return jsonify({'success': True, 'folder_path': folder_path})
        else:
            return jsonify({'success': False, 'message': '用户取消了文件夹选择'})
            
    except Exception as e:
        add_log(f"文件夹选择对话框出错: {e}", 'error')
        return jsonify({'success': False, 'message': f'文件夹选择失败: {str(e)}'})

@app.route('/api/calculate_recommended_learning_rate', methods=['POST'])
def calculate_recommended_learning_rate():
    """计算推荐学习率（支持T2V/I2V差异化推荐）"""
    try:
        data = request.get_json()
        dataset_config = data.get('dataset_config')
        batch_size = data.get('batch_size', 1)
        optimizer_type = data.get('optimizer_type', 'adamw8bit')  # 获取优化器类型
        task_type = data.get('task_type', 't2v')  # T2V或I2V
        is_video_training = data.get('is_video_training', True)  # 是否为视频训练
        
        if not dataset_config or not os.path.exists(dataset_config):
            return jsonify({'success': False, 'message': '数据集配置文件不存在'})
        
        # 根据优化器类型设置基础学习率
        optimizer_base_lr = {
            'adamw8bit': 1e-4,
            'AdamW': 1e-4, 
            'Lion': 5e-5,
            'SGDNesterov': 1e-2,
            'Adafactor': 1e-3
        }
        base_lr = optimizer_base_lr.get(optimizer_type, 1e-4)
        
        # 读取TOML文件获取image_directory路径
        import toml
        try:
            config = toml.load(dataset_config)
            datasets = config.get('datasets', [])
            if not datasets:
                return jsonify({'success': False, 'message': 'TOML文件中未找到datasets配置'})
            
            # 根据训练类型获取正确的数据集目录
            if is_video_training:
                data_directory = datasets[0].get('video_directory')
                directory_type = 'video_directory'
            else:
                data_directory = datasets[0].get('image_directory')
                directory_type = 'image_directory'
                
            if not data_directory:
                return jsonify({'success': False, 'message': f'数据集配置中未找到{directory_type}'})
            
            # 统计txt文件数量
            dataset_files = 0
            if os.path.exists(data_directory):
                for file in os.listdir(data_directory):
                    if file.lower().endswith('.txt'):
                        dataset_files += 1
            
            if dataset_files == 0:
                return jsonify({'success': False, 'message': f'在 {data_directory} 中未找到txt文件'})
            
            # 计算视频时长（用于视频训练模式显示）
            video_duration = 0
            if is_video_training and os.path.exists(data_directory):
                # 假设每个视频片段为5秒，根据txt文件数量估算总时长
                video_duration = dataset_files * 5 / 60  # 转换为分钟
            else:
                video_duration = 0  # 图像训练模式下为0
            
            # 根据训练类型、任务类型和优化器类型使用不同的计算公式
            import math
            
            if is_video_training:
                # 视频训练：考虑T2V/I2V差异
                if task_type == 'i2v':
                    # I2V：更保守的学习率
                    if optimizer_type in ['adamw8bit', 'AdamW']:
                        batch_factor = batch_size / 64  # I2V更保守的batch缩放
                        data_factor = math.log10(dataset_files / 20 + 1) * 0.5  # I2V数据缩放更保守
                        recommended_lr = base_lr * batch_factor * data_factor * 0.3  # I2V总体缩放因子
                    elif optimizer_type == 'Lion':
                        batch_factor = math.sqrt(batch_size / 32)
                        data_factor = math.sqrt(dataset_files / 100) * 0.5
                        recommended_lr = base_lr * batch_factor * data_factor * 0.2
                    else:
                        batch_factor = batch_size / 32
                        data_factor = math.sqrt(dataset_files / 100)
                        recommended_lr = base_lr * batch_factor * data_factor * 0.3
                    calculation_detail = f"I2V视频训练，{optimizer_type}优化器，保守缩放策略"
                else:
                    # T2V：相对更高的学习率
                    if optimizer_type in ['adamw8bit', 'AdamW']:
                        batch_factor = batch_size / 32  # T2V标准batch缩放
                        data_factor = math.log10(dataset_files / 10 + 1)  # T2V标准数据缩放
                        recommended_lr = base_lr * batch_factor * data_factor * 0.5  # T2V适中缩放因子
                    elif optimizer_type == 'Lion':
                        batch_factor = math.sqrt(batch_size / 16)
                        data_factor = math.sqrt(dataset_files / 50)
                        recommended_lr = base_lr * batch_factor * data_factor * 0.4
                    else:
                        batch_factor = batch_size / 16
                        data_factor = math.sqrt(dataset_files / 100)
                        recommended_lr = base_lr * batch_factor * data_factor * 0.5
                    calculation_detail = f"T2V视频训练，{optimizer_type}优化器，标准缩放策略"
            else:
                # 图像训练：使用原有逻辑
                if optimizer_type in ['adamw8bit', 'AdamW']:
                    batch_factor = batch_size / 32
                    data_factor = math.log10(dataset_files / 10 + 1)
                    recommended_lr = base_lr * batch_factor * data_factor
                elif optimizer_type == 'Lion':
                    batch_factor = math.sqrt(batch_size / 16)
                    data_factor = math.sqrt(dataset_files / 50)
                    recommended_lr = base_lr * batch_factor * data_factor
                elif optimizer_type == 'SGDNesterov':
                    batch_factor = batch_size / 16
                    data_factor = math.sqrt(dataset_files / 100)
                    recommended_lr = base_lr * batch_factor * data_factor
                else:  # Adafactor
                    batch_factor = math.sqrt(batch_size)
                    data_factor = math.log10(dataset_files / 20 + 1)
                    recommended_lr = base_lr * batch_factor * data_factor
                calculation_detail = f"图像训练，{optimizer_type}优化器，传统缩放策略"
            
            # 格式化学习率为科学计数法
            lr_str = f"{recommended_lr:.1e}"
            
            # 获取优化器描述
            optimizer_descriptions = {
                'adamw8bit': 'Adam系列，内存优化版本，适合大模型训练',
                'AdamW': 'Adam系列，标准版本，稳定可靠',
                'Lion': 'Lion优化器，内存效率高，需要较小学习率',
                'SGDNesterov': 'SGD动量优化器，需要较大学习率',
                'Adafactor': '自适应优化器，学习率自动调整'
            }
            
            task_name = "图生视频(I2V)" if task_type == 'i2v' else "文生视频(T2V)"
            training_type = "视频训练" if is_video_training else "图像训练"
            
            add_log(f"推荐学习率计算完成: {lr_str} ({training_type}的{task_name}, 优化器: {optimizer_type}, 数据集: {dataset_files}个文件, batch_size: {batch_size})", 'info')
            
            return jsonify({
                'success': True,
                'recommended_lr': lr_str,
                'dataset_files': dataset_files,
                'batch_size': batch_size,
                'optimizer_type': optimizer_type,
                'optimizer_description': optimizer_descriptions.get(optimizer_type, ''),
                'base_lr': base_lr,
                'task_type': task_type,
                'task_name': task_name,
                'training_type': training_type,
                'is_video_training': is_video_training,
                'video_duration': video_duration,
                'calculation': f"基于{training_type}的{task_name}，{calculation_detail}: {base_lr:.1e} × 缩放因子 = {lr_str}"
            })
            
        except Exception as e:
            return jsonify({'success': False, 'message': f'解析TOML文件失败: {e}'})
            
    except Exception as e:
        add_log(f"推荐学习率计算失败: {e}", 'error')
        return jsonify({'success': False, 'message': f'推荐学习率计算失败: {e}'})

@app.route('/api/calculate_gpt_recommended_learning_rate', methods=['POST'])
def calculate_gpt_recommended_learning_rate():
    """基于ChatGPT专业建议的智能学习率推荐算法"""
    try:
        data = request.get_json()
        dataset_config = data.get('dataset_config')
        task_type = data.get('task_type', 't2v')  # T2V或I2V
        is_video_training = data.get('is_video_training', True)  # 是否为视频训练
        video_duration = data.get('video_duration', 5)  # 视频时长（秒）
        optimizer = data.get('optimizer', 'AdamW')  # 优化器类型
        lora_rank = data.get('lora_rank', 8)  # LoRA rank
        
        if not dataset_config or not os.path.exists(dataset_config):
            return jsonify({'success': False, 'message': '数据集配置文件不存在'})
        
        # 读取TOML文件获取数据集信息
        import toml
        import math
        try:
            config = toml.load(dataset_config)
            datasets = config.get('datasets', [])
            if not datasets:
                return jsonify({'success': False, 'message': 'TOML文件中未找到datasets配置'})
            
            # 根据训练类型获取正确的数据集目录
            if is_video_training:
                data_directory = datasets[0].get('video_directory')
                directory_type = 'video_directory'
            else:
                data_directory = datasets[0].get('image_directory')
                directory_type = 'image_directory'
                
            if not data_directory or not os.path.exists(data_directory):
                return jsonify({'success': False, 'message': '数据集目录不存在'})
            
            # 统计txt文件数量
            txt_files = [f for f in os.listdir(data_directory) if f.endswith('.txt')]
            txt_count = len(txt_files)
            
            if txt_count == 0:
                return jsonify({'success': False, 'message': '数据集目录中未找到txt文件'})
            
            # ChatGPT专业建议：基于公式的智能推荐算法
            # 基线参数
            base_lr = 1e-4
            N_ref = 1000
            
            # 计算有效样本数N_eff
            if is_video_training:
                # 视频训练：考虑帧数
                fps = 30  # 假设30fps
                frames_per_video = int(video_duration * fps)
                N_eff = txt_count * frames_per_video  # 拆帧训练
            else:
                # 图像训练
                N_eff = txt_count
            
            # 核心公式：lr = base_lr * sqrt(N_eff / N_ref)
            lr_raw = base_lr * math.sqrt(N_eff / N_ref)
            
            # 任务类型调整因子
            if task_type == 'i2v':
                task_mult = 0.8  # I2V更保守
                task_desc = "I2V(图生视频)"
            else:
                task_mult = 1.0  # T2V标准
                task_desc = "T2V(文生视频)"
            
            # LoRA rank调整因子
            rank_mult = 1.0 / math.sqrt(lora_rank / 8)
            
            # 优化器调整因子
            opt_mult_map = {
                'AdamW': 1.0,
                'AdamW8bit': 0.85,
                'Lion': 1.2,
                'Adafactor': 0.6
            }
            opt_mult = opt_mult_map.get(optimizer, 1.0)
            
            # 应用所有调整因子
            lr_adjusted = lr_raw * task_mult * rank_mult * opt_mult
            
            # 安全边界裁剪
            lr_min, lr_max = 5e-6, 5e-4
            recommended_lr = max(lr_min, min(lr_max, lr_adjusted))
            
            # 推荐训练轮数（基于数据规模）
            if N_eff <= 500:
                recommended_epochs = 15
                epoch_reason = "小数据集：15轮训练，防止过拟合"
            elif N_eff <= 2000:
                recommended_epochs = 20
                epoch_reason = "中等数据集：20轮训练，充分学习"
            elif N_eff <= 5000:
                recommended_epochs = 25
                epoch_reason = "大数据集：25轮训练，平衡效果与时间"
            else:
                recommended_epochs = 30
                epoch_reason = "超大数据集：30轮训练，最大化数据利用"
            
            # 生成详细说明
            lr_reason = f"ChatGPT智能推荐：基线{base_lr:.0e}×√({N_eff}/{N_ref})×{task_mult}({task_desc})×{rank_mult:.2f}(rank{lora_rank})×{opt_mult}({optimizer}) = {recommended_lr:.2e}"
            
            # 数据集分级描述
            if N_eff <= 500:
                data_level = "小数据集"
            elif N_eff <= 2000:
                data_level = "中等数据集"
            elif N_eff <= 5000:
                data_level = "大数据集"
            else:
                data_level = "超大数据集"
            
            # 格式化学习率
            lr_str = f"{recommended_lr:.2e}"
            
            # 生成详细描述
            training_type = "视频训练" if is_video_training else "图像训练"
            description = f"ChatGPT智能推荐：{data_level}({N_eff}样本)，{task_desc}，{optimizer}优化器，rank{lora_rank}"
            
            # 生成推荐说明
            recommendation_details = {
                'formula': f"lr = {base_lr:.0e} × √({N_eff}/{N_ref}) × {task_mult} × {rank_mult:.2f} × {opt_mult} = {recommended_lr:.2e}",
                'factors': {
                    'base_lr': base_lr,
                    'N_eff': N_eff,
                    'N_ref': N_ref,
                    'task_mult': task_mult,
                    'rank_mult': rank_mult,
                    'opt_mult': opt_mult,
                    'final_lr': recommended_lr
                },
                'safety_bounds': f"安全边界：[{lr_min:.0e}, {lr_max:.0e}]",
                'warmup_suggestion': f"建议warmup：{max(100, int(0.01 * recommended_epochs * 100))}步"
            }
            
            return jsonify({
                'success': True,
                'recommended_lr': lr_str,
                'recommended_epochs': recommended_epochs,
                'txt_count': txt_count,
                'N_eff': N_eff,
                'data_level': data_level,
                'task_type': task_type,
                'task_desc': task_desc,
                'training_type': training_type,
                'is_video_training': is_video_training,
                'video_duration': video_duration,
                'optimizer': optimizer,
                'lora_rank': lora_rank,
                'lr_reason': lr_reason,
                'epoch_reason': epoch_reason,
                'description': description,
                'recommendation_details': recommendation_details
            })
            
        except Exception as e:
            return jsonify({'success': False, 'message': f'解析TOML文件失败: {e}'})
            
    except Exception as e:
        add_log(f"GPT推荐学习率计算失败: {e}", 'error')
        return jsonify({'success': False, 'message': f'GPT推荐学习率计算失败: {e}'})

@app.route('/api/calculate_recommended_lora_params', methods=['POST'])
def calculate_recommended_lora_params():
    """计算推荐LoRA参数（支持任务类型和训练类型）"""
    try:
        data = request.get_json()
        dataset_config = data.get('dataset_config')
        param_type = data.get('param_type')  # 'network_dim' 或 'network_alpha'
        current_dim = data.get('current_dim', 8)  # 当前维度值，用于计算alpha
        task_type = data.get('task_type', 't2v')  # 任务类型
        is_video_training = data.get('is_video_training', True)  # 是否为视频训练
        
        if not dataset_config or not os.path.exists(dataset_config):
            return jsonify({'success': False, 'message': '数据集配置文件不存在'})
        
        # 读取TOML文件获取数据集信息
        import toml
        try:
            config = toml.load(dataset_config)
            datasets = config.get('datasets', [])
            if not datasets:
                return jsonify({'success': False, 'message': 'TOML文件中未找到datasets配置'})
            
            # 获取训练类型信息
            task_type = config.get('task_type', 't2v')
            is_video_training = config.get('is_video_training', False)
            
            # 根据训练类型获取数据集目录
            first_dataset = datasets[0]
            if is_video_training:
                data_directory = first_dataset.get('video_directory')
                directory_type = 'video_directory'
            else:
                data_directory = first_dataset.get('image_dir') or first_dataset.get('image_directory')
                directory_type = 'image_directory'
            
            if not data_directory:
                return jsonify({'success': False, 'message': f'数据集配置中未找到{directory_type}'})
            
            if not os.path.exists(data_directory):
                return jsonify({'success': False, 'message': f'数据集目录不存在: {data_directory}'})
            
            # 统计txt文件数量
            dataset_files = 0
            for file in os.listdir(data_directory):
                if file.lower().endswith('.txt'):
                    dataset_files += 1
            
            if dataset_files == 0:
                return jsonify({'success': False, 'message': f'在 {data_directory} 中未找到txt文件'})
            
            if param_type == 'network_dim':
                # LoRA维度推荐逻辑（根据训练类型和任务类型差异化推荐）
                if not is_video_training:
                    # 图像训练模式 - 使用相对保守的维度
                    if dataset_files <= 50:
                        recommended_dim = 8
                        reason = "图像训练小数据集(≤50张)，使用小维度避免过拟合"
                    elif dataset_files <= 200:
                        recommended_dim = 16
                        reason = "图像训练中小数据集(51-200张)，使用中等维度"
                    elif dataset_files <= 500:
                        recommended_dim = 24
                        reason = "图像训练中等数据集(201-500张)，使用适中维度"
                    elif dataset_files <= 1000:
                        recommended_dim = 32
                        reason = "图像训练大数据集(501-1000张)，使用标准维度"
                    else:
                        recommended_dim = 48
                        reason = "图像训练超大数据集(>1000张)，使用较大维度"
                    task_name = "图像训练"
                elif task_type == 'i2v':
                    # I2V视频训练 - 相对保守的维度设置
                    if dataset_files <= 50:
                        recommended_dim = 16
                        reason = "I2V小数据集(≤50张)，使用中等维度学习图像到视频转换"
                    elif dataset_files <= 200:
                        recommended_dim = 24
                        reason = "I2V中小数据集(51-200张)，平衡学习能力和稳定性"
                    elif dataset_files <= 500:
                        recommended_dim = 32
                        reason = "I2V中等数据集(201-500张)，使用标准维度获得良好效果"
                    elif dataset_files <= 1000:
                        recommended_dim = 48
                        reason = "I2V大数据集(501-1000张)，使用较大维度充分学习特征"
                    else:
                        recommended_dim = 64
                        reason = "I2V超大数据集(>1000张)，使用大维度捕获复杂时序特征"
                    task_name = "图生视频(I2V)"
                else:
                    # T2V视频训练 - 需要更大的维度来处理复杂的文本到视频生成
                    if dataset_files <= 50:
                        recommended_dim = 24
                        reason = "T2V小数据集(≤50张)，使用较大维度学习文本到视频生成"
                    elif dataset_files <= 200:
                        recommended_dim = 32
                        reason = "T2V中小数据集(51-200张)，使用标准维度平衡效果与稳定性"
                    elif dataset_files <= 500:
                        recommended_dim = 48
                        reason = "T2V中等数据集(201-500张)，使用较大维度获得良好效果"
                    elif dataset_files <= 1000:
                        recommended_dim = 64
                        reason = "T2V大数据集(501-1000张)，使用大维度充分学习复杂特征"
                    else:
                        recommended_dim = 96
                        reason = "T2V超大数据集(>1000张)，使用很大维度捕获复杂的文本-视频映射"
                    task_name = "文生视频(T2V)"
                
                return jsonify({
                    'success': True,
                    'recommended_value': recommended_dim,
                    'dataset_files': dataset_files,
                    'reason': reason,
                    'task_name': task_name,
                    'description': f'LoRA维度控制模型容量，{task_name}模式下根据数据量智能推荐'
                })
                
            elif param_type == 'network_alpha':
                # LoRA Alpha推荐逻辑 - 根据训练类型和任务类型差异化推荐
                if is_video_training:
                    if task_type == 't2v':
                        # T2V视频训练：需要更强的学习能力
                        if current_dim <= 16:
                            recommended_alpha = current_dim * 2
                            reason = f"T2V小维度({current_dim})时，Alpha设为2倍增强文本到视频的学习能力"
                        elif current_dim <= 64:
                            recommended_alpha = int(current_dim * 1.5)
                            reason = f"T2V中等维度({current_dim})时，Alpha设为1.5倍平衡学习强度"
                        else:
                            recommended_alpha = current_dim
                            reason = f"T2V大维度({current_dim})时，Alpha与维度相等防止过拟合"
                        task_name = "T2V视频训练"
                    else:
                        # I2V视频训练：相对保守的设置
                        if current_dim <= 16:
                            recommended_alpha = int(current_dim * 1.5)
                            reason = f"I2V小维度({current_dim})时，Alpha设为1.5倍保持稳定学习"
                        elif current_dim <= 64:
                            recommended_alpha = current_dim
                            reason = f"I2V中等维度({current_dim})时，Alpha与维度相等保持平衡"
                        else:
                            recommended_alpha = current_dim // 2
                            reason = f"I2V大维度({current_dim})时，Alpha设为一半防止对输入图像过拟合"
                        task_name = "I2V视频训练"
                else:
                    # 图像训练：传统策略
                    if current_dim <= 16:
                        recommended_alpha = current_dim * 2
                        reason = f"图像训练小维度({current_dim})时，Alpha设为2倍增强学习能力"
                    elif current_dim <= 64:
                        recommended_alpha = current_dim
                        reason = f"图像训练中等维度({current_dim})时，Alpha与维度相等保持平衡"
                    else:
                        recommended_alpha = current_dim // 2
                        reason = f"图像训练大维度({current_dim})时，Alpha设为一半防止过拟合"
                    task_name = "图像训练"
                
                return jsonify({
                    'success': True,
                    'recommended_value': recommended_alpha,
                    'current_dim': current_dim,
                    'reason': reason,
                    'task_name': task_name,
                    'description': f'LoRA Alpha控制学习强度，{task_name}模式下与维度协调设置'
                })
            
            else:
                return jsonify({'success': False, 'message': '未知的参数类型'})
                
        except Exception as e:
            return jsonify({'success': False, 'message': f'解析TOML文件失败: {e}'})
            
    except Exception as e:
        add_log(f"推荐LoRA参数计算失败: {e}", 'error')
        return jsonify({'success': False, 'message': f'推荐LoRA参数计算失败: {e}'})

@app.route('/api/calculate_recommended_lora_joint', methods=['POST'])
def calculate_recommended_lora_joint():
    """联合推荐LoRA维度和Alpha（根据任务类型和训练类型差异化推荐）"""
    try:
        data = request.get_json()
        dataset_config = data.get('dataset_config')
        task_type = data.get('task_type', 't2v')  # 获取任务类型，默认T2V
        is_video_training = data.get('is_video_training', True)  # 获取是否为视频训练，默认True
        
        if not dataset_config or not os.path.exists(dataset_config):
            return jsonify({'success': False, 'message': '数据集配置文件不存在'})
        
        # 读取TOML文件获取数据集信息
        import toml
        try:
            config = toml.load(dataset_config)
            datasets = config.get('datasets', [])
            if not datasets:
                return jsonify({'success': False, 'message': 'TOML文件中未找到datasets配置'})
            
            # 根据训练类型获取正确的数据集目录
            if is_video_training:
                data_directory = datasets[0].get('video_directory')
                directory_type = 'video_directory'
            else:
                data_directory = datasets[0].get('image_directory')
                directory_type = 'image_directory'
                
            if not data_directory or not os.path.exists(data_directory):
                return jsonify({'success': False, 'message': '数据集目录不存在'})
            
            # 统计txt文件数量
            txt_files = [f for f in os.listdir(data_directory) if f.endswith('.txt')]
            txt_count = len(txt_files)
            
            if txt_count == 0:
                return jsonify({'success': False, 'message': '数据集目录中未找到txt文件'})
            
            # 根据训练类型、任务类型和数据集大小的专业实战经验推荐策略
            if not is_video_training:
                # 图像训练模式：使用传统图像LoRA推荐策略
                if txt_count <= 50:
                    recommended_dim = 16
                    recommended_alpha = 16
                    dim_reason = "图像训练小数据集(≤50张)：16维度避免过拟合"
                    alpha_reason = "Alpha=Dim，图像训练的经典安全配置"
                elif txt_count <= 200:
                    recommended_dim = 32
                    recommended_alpha = 32
                    dim_reason = "图像训练中等数据集(51-200张)：32维度平衡表现力和泛化"
                    alpha_reason = "Alpha=Dim，适合图像训练的标准配置"
                elif txt_count <= 500:
                    recommended_dim = 64
                    recommended_alpha = 64
                    dim_reason = "图像训练大数据集(201-500张)：64维度处理复杂图像特征"
                    alpha_reason = "Alpha=Dim，图像训练大数据集的推荐配置"
                else:
                    recommended_dim = 128
                    recommended_alpha = 128
                    dim_reason = "图像训练超大数据集(>500张)：128维度捕获丰富图像特征"
                    alpha_reason = "Alpha=Dim，图像训练超大数据集的最优配置"
                task_name = "图像训练"
            elif task_type == 'i2v':
                # I2V（图生视频）：更注重细节保持和运动连贯性
                if txt_count <= 50:
                    recommended_dim = 24  # I2V需要更高维度保持图像细节
                    recommended_alpha = 24
                    dim_reason = "I2V小数据集(≤50张)：24维度保持图像细节和运动连贯性"
                    alpha_reason = "Alpha=Dim，I2V任务需要平衡图像保真度和运动生成"
                elif txt_count <= 200:
                    recommended_dim = 48  # I2V中等数据集需要更高容量
                    recommended_alpha = 48
                    dim_reason = "I2V中等数据集(51-200张)：48维度处理图像到视频的复杂映射"
                    alpha_reason = "Alpha=Dim，适合I2V的图像条件引导和运动生成平衡"
                elif txt_count <= 500:
                    recommended_dim = 80
                    recommended_alpha = 120  # dim*1.5
                    dim_reason = "I2V大数据集(201-500张)：80维度处理复杂图像条件和多样运动"
                    alpha_reason = "Alpha=Dim*1.5(120)，增强I2V的条件引导能力"
                else:
                    recommended_dim = 128
                    recommended_alpha = 192  # dim*1.5，I2V不需要过高alpha
                    dim_reason = "I2V超大数据集(>500张)：128维度处理复杂图像到视频生成"
                    alpha_reason = "Alpha=Dim*1.5(192)，I2V任务的最优权重配置"
            else:
                # T2V（文生视频）：更注重文本理解和创意生成
                if txt_count <= 50:
                    recommended_dim = 16
                    recommended_alpha = 16
                    dim_reason = "T2V小数据集(≤50张)：16维度适合文本到视频的基础映射"
                    alpha_reason = "Alpha=Dim，T2V任务的经典安全配置"
                elif txt_count <= 200:
                    recommended_dim = 32
                    recommended_alpha = 32
                    dim_reason = "T2V中等数据集(51-200张)：32维度平衡文本理解和视频生成"
                    alpha_reason = "Alpha=Dim，适合T2V的文本条件和视觉生成平衡"
                elif txt_count <= 500:
                    recommended_dim = 64
                    recommended_alpha = 96  # dim*1.5
                    dim_reason = "T2V大数据集(201-500张)：64维度处理复杂文本描述和视频生成"
                    alpha_reason = "Alpha=Dim*1.5(96)，增强T2V的文本理解和创意生成能力"
                else:
                    recommended_dim = 128
                    recommended_alpha = 256  # dim*2，T2V需要更高alpha支持创意
                    dim_reason = "T2V超大数据集(>500张)：128维度支持复杂文本到视频生成"
                    alpha_reason = "Alpha=Dim*2(256)，T2V任务的最大创意生成权重"
                task_name = "文生视频(T2V)"
            
            # 如果是视频训练，根据任务类型设置task_name
            if is_video_training and task_type == 'i2v':
                task_name = "图生视频(I2V)"
            
            return jsonify({
                'success': True,
                'recommended_dim': recommended_dim,
                'recommended_alpha': recommended_alpha,
                'txt_count': txt_count,
                'task_type': task_type,
                'task_name': task_name,
                'dim_reason': dim_reason,
                'alpha_reason': alpha_reason,
                'description': f'基于{txt_count}张图片的{task_name}数据集，专业推荐维度和Alpha参数'
            })
            
        except Exception as e:
            return jsonify({'success': False, 'message': f'解析TOML文件失败: {e}'})
            
    except Exception as e:
        add_log(f"联合推荐LoRA参数计算失败: {e}", 'error')
        return jsonify({'success': False, 'message': f'联合推荐LoRA参数计算失败: {e}'})

@app.route('/api/calculate_recommended_epochs', methods=['POST'])
def calculate_recommended_epochs():
    """计算推荐训练轮数"""
    try:
        data = request.get_json()
        dataset_config = data.get('dataset_config')
        batch_size = data.get('batch_size', 1)
        
        if not dataset_config or not os.path.exists(dataset_config):
            return jsonify({'success': False, 'message': '数据集配置文件不存在'})
        
        # 读取TOML文件获取数据集信息
        import toml
        try:
            config = toml.load(dataset_config)
            datasets = config.get('datasets', [])
            if not datasets:
                return jsonify({'success': False, 'message': 'TOML文件中未找到datasets配置'})
            
            # 获取第一个数据集的image_directory
            image_directory = datasets[0].get('image_directory')
            if not image_directory:
                return jsonify({'success': False, 'message': '数据集配置中未找到image_directory'})
            
            # 统计txt文件数量
            dataset_files = 0
            if os.path.exists(image_directory):
                for file in os.listdir(image_directory):
                    if file.lower().endswith('.txt'):
                        dataset_files += 1
            
            if dataset_files == 0:
                return jsonify({'success': False, 'message': f'在 {image_directory} 中未找到txt文件'})
            
            # 计算每轮步数
            steps_per_epoch = max(1, dataset_files // batch_size)
            
            # 根据数据集大小推荐训练轮数
            if dataset_files <= 20:
                # 小数据集：需要更多轮数但要防止过拟合
                recommended_epochs = min(30, max(15, 300 // steps_per_epoch))
                reason = f"小数据集({dataset_files}张)，需要足够轮数学习但防止过拟合"
            elif dataset_files <= 50:
                # 中小数据集：平衡学习效果和训练时间
                recommended_epochs = min(20, max(10, 500 // steps_per_epoch))
                reason = f"中小数据集({dataset_files}张)，平衡学习效果和训练效率"
            elif dataset_files <= 100:
                # 中等数据集：适中的轮数
                recommended_epochs = min(15, max(8, 800 // steps_per_epoch))
                reason = f"中等数据集({dataset_files}张)，适中轮数确保收敛"
            elif dataset_files <= 200:
                # 较大数据集：较少轮数即可收敛
                recommended_epochs = min(10, max(5, 1000 // steps_per_epoch))
                reason = f"较大数据集({dataset_files}张)，较少轮数即可有效学习"
            else:
                # 超大数据集：最少轮数防止过拟合
                recommended_epochs = min(8, max(3, 1200 // steps_per_epoch))
                reason = f"超大数据集({dataset_files}张)，少量轮数防止过拟合"
            
            # 计算预估训练步数和时间
            estimated_steps = recommended_epochs * steps_per_epoch
            estimated_time_minutes = estimated_steps * 2.5  # 假设每步2.5秒
            
            return jsonify({
                'success': True,
                'recommended_value': recommended_epochs,
                'dataset_files': dataset_files,
                'batch_size': batch_size,
                'steps_per_epoch': steps_per_epoch,
                'estimated_steps': estimated_steps,
                'estimated_time_minutes': int(estimated_time_minutes),
                'reason': reason,
                'description': '训练轮数需要平衡学习效果和过拟合风险'
            })
            
        except Exception as e:
            return jsonify({'success': False, 'message': f'解析TOML文件失败: {e}'})
            
    except Exception as e:
        add_log(f"推荐训练轮数计算失败: {e}", 'error')
        return jsonify({'success': False, 'message': f'推荐训练轮数计算失败: {e}'})

@app.route('/api/estimate_training_time', methods=['POST'])
def estimate_training_time():
    """估算训练时间"""
    try:
        data = request.get_json()
        dataset_config = data.get('dataset_config')
        batch_size = data.get('batch_size', 1)
        num_repeats = data.get('num_repeats', 1)
        max_train_epochs = data.get('max_train_epochs', 1)
        save_every_n_epochs = data.get('save_every_n_epochs', 1)
        output_name = data.get('output_name', 'wan22-lora')
        avg_step_time = data.get('avg_step_time', 2.92)
        
        # 获取前端传递的视频训练参数
        is_video_training_frontend = data.get('is_video_training', False)
        video_duration = data.get('video_duration', 0)
        
        if not dataset_config or not os.path.exists(dataset_config):
            return jsonify({'success': False, 'message': '数据集配置文件不存在'})
        
        # 读取TOML文件获取数据集目录路径
        import toml
        try:
            config = toml.load(dataset_config)
            datasets = config.get('datasets', [])
            if not datasets:
                return jsonify({'success': False, 'message': 'TOML文件中未找到datasets配置'})
            
            # 获取训练类型信息，优先使用前端传递的参数
            task_type = config.get('task_type', 't2v')
            is_video_training = is_video_training_frontend or config.get('is_video_training', False)
            
            # 根据训练类型获取数据集目录
            first_dataset = datasets[0]
            if is_video_training:
                data_directory = first_dataset.get('video_directory')
                directory_type = 'video_directory'
            else:
                data_directory = first_dataset.get('image_dir') or first_dataset.get('image_directory')
                directory_type = 'image_directory'
            
            if not data_directory:
                return jsonify({'success': False, 'message': f'数据集配置中未找到{directory_type}'})
            
            if not os.path.exists(data_directory):
                return jsonify({'success': False, 'message': f'数据集目录不存在: {data_directory}'})
            
            # 统计txt文件数量
            dataset_files = 0
            for file in os.listdir(data_directory):
                if file.endswith('.txt'):
                    dataset_files += 1
            
            if dataset_files == 0:
                return jsonify({'success': False, 'message': f'在 {data_directory} 中未找到任何.txt文件'})
            
            # 计算实际训练样本数
            if is_video_training:
                # 视频训练模式：需要考虑视频片段拆分
                # 从TOML配置文件读取视频训练参数
                target_frames = first_dataset.get('target_frames', [1, 25, 49, 73])
                frame_sample = first_dataset.get('frame_sample', 8)
                dataset_num_repeats = first_dataset.get('num_repeats', num_repeats)
                
                # 计算总片段数：视频文件数 × 目标帧数种类数 × 每种帧数的采样数
                total_segments = dataset_files * len(target_frames) * frame_sample
                
                # 应用数据重复次数
                effective_samples = total_segments * dataset_num_repeats
                
                add_log(f"视频训练模式计算详情:", 'info')
                add_log(f"- 视频文件数: {dataset_files}", 'info')
                add_log(f"- 目标帧数配置: {target_frames} (共{len(target_frames)}种)", 'info')
                add_log(f"- 每种帧数采样数: {frame_sample}", 'info')
                add_log(f"- 总片段数: {dataset_files} × {len(target_frames)} × {frame_sample} = {total_segments}", 'info')
                add_log(f"- 数据重复次数: {dataset_num_repeats}", 'info')
                add_log(f"- 有效训练样本数: {total_segments} × {dataset_num_repeats} = {effective_samples}", 'info')
            else:
                # 图片训练模式：使用原有逻辑
                effective_samples = dataset_files * num_repeats
            
            # 计算总步数和每轮步数
            steps_per_epoch = int(effective_samples / batch_size)
            total_steps = steps_per_epoch * max_train_epochs
            
            # 生成预计文件列表
            predicted_files = []
            for epoch in range(save_every_n_epochs, max_train_epochs + 1, save_every_n_epochs):
                # 计算文件名（6位数字格式）
                filename = f"{output_name}-{epoch:06d}.safetensors"
                
                # 计算步数范围
                start_step = (epoch - save_every_n_epochs) * steps_per_epoch + 1
                end_step = epoch * steps_per_epoch
                
                # 如果是第一次保存，起始步数为0
                if epoch == save_every_n_epochs:
                    start_step = 0
                
                save_count = epoch // save_every_n_epochs
                
                predicted_files.append({
                    'filename': filename,
                    'epoch': epoch,
                    'save_count': save_count,
                    'step_range': f"{start_step}~{end_step}",
                    'start_step': start_step,
                    'end_step': end_step
                })
            
            # 计算总时间（秒）
            total_seconds = total_steps * avg_step_time
            
            # 转换为小时分钟格式
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            seconds = int(total_seconds % 60)
            
            if hours > 0:
                estimated_time = f"{hours}小时{minutes}分钟{seconds}秒"
            elif minutes > 0:
                estimated_time = f"{minutes}分钟{seconds}秒"
            else:
                estimated_time = f"{seconds}秒"
            
            add_log(f"训练时间估算完成: {estimated_time} (总步数: {total_steps}, 数据集文件数: {dataset_files}, 预计生成{len(predicted_files)}个文件)", 'info')
            
            # 准备返回数据
            result_data = {
                'success': True,
                'estimated_time': estimated_time,
                'total_steps': total_steps,
                'dataset_files': dataset_files,
                'total_seconds': total_seconds,
                'steps_per_epoch': steps_per_epoch,
                'predicted_files': predicted_files
            }
            
            # 如果是视频训练模式，添加视频片段信息
            if is_video_training_frontend:
                result_data['video_segments'] = effective_samples
                print(f"[视频训练] 返回视频片段数: {effective_samples}")
            
            return jsonify(result_data)
            
        except Exception as e:
            return jsonify({'success': False, 'message': f'解析TOML文件失败: {e}'})
            
    except Exception as e:
        add_log(f"训练时间估算失败: {e}", 'error')
        return jsonify({'success': False, 'message': f'训练时间估算失败: {e}'})

@app.route('/api/calculate_recommended_num_repeats', methods=['POST'])
def calculate_recommended_num_repeats():
    """计算推荐的数据重复次数"""
    try:
        data = request.get_json()
        dataset_config = data.get('dataset_config')
        task_type = data.get('task_type', 't2v')
        is_video_training = data.get('is_video_training', False)
        video_duration = data.get('video_duration', 0)
        
        if not dataset_config or not os.path.exists(dataset_config):
            return jsonify({'success': False, 'message': '数据集配置文件不存在'})
        
        # 读取TOML文件获取数据集信息
        import toml
        try:
            config = toml.load(dataset_config)
            datasets = config.get('datasets', [])
            if not datasets:
                return jsonify({'success': False, 'message': '数据集配置文件中没有找到datasets配置'})
            
            # 计算数据集文件数量
            dataset_files = 0
            for dataset in datasets:
                # 根据训练类型获取正确的目录字段
                if is_video_training:
                    data_dir = dataset.get('video_directory')
                    directory_type = 'video_directory'
                    file_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
                else:
                    data_dir = dataset.get('image_dir') or dataset.get('image_directory')
                    directory_type = 'image_dir或image_directory'
                    file_extensions = ('.jpg', '.jpeg', '.png', '.webp')
                    
                if data_dir and os.path.exists(data_dir):
                    try:
                        files = [f for f in os.listdir(data_dir) if f.lower().endswith(file_extensions)]
                        dataset_files += len(files)
                        add_log(f"数据集目录 {data_dir} 找到 {len(files)} 个文件", 'info')
                    except Exception as e:
                        add_log(f"读取数据集目录 {data_dir} 失败: {e}", 'warning')
                elif data_dir:
                    add_log(f"数据集目录不存在: {data_dir}", 'warning')
                else:
                    add_log(f"数据集配置中缺少{directory_type}字段", 'warning')
            
            # 根据训练类型计算推荐重复次数
            if is_video_training:
                # 视频训练模式：基于视频时长计算
                if video_duration <= 0:
                    return jsonify({'success': False, 'message': '视频训练模式需要设置有效的视频时长'})
                
                # 视频时长越短，需要更多重复次数
                if video_duration <= 5:
                    recommended_repeats = 20
                    reason = f"短视频({video_duration}秒)，需要大量重复以充分学习时序特征"
                elif video_duration <= 10:
                    recommended_repeats = 15
                    reason = f"中短视频({video_duration}秒)，适当增加重复次数"
                elif video_duration <= 20:
                    recommended_repeats = 10
                    reason = f"中等视频({video_duration}秒)，平衡重复次数"
                elif video_duration <= 30:
                    recommended_repeats = 8
                    reason = f"较长视频({video_duration}秒)，适度重复"
                else:
                    recommended_repeats = 5
                    reason = f"长视频({video_duration}秒)，较少重复次数即可"
                    
                # 考虑数据集大小进行微调
                if dataset_files < 50:
                    recommended_repeats = min(recommended_repeats + 5, 30)
                    reason += "，小数据集增加重复"
                elif dataset_files > 200:
                    recommended_repeats = max(recommended_repeats - 2, 3)
                    reason += "，大数据集减少重复"
            else:
                # 非视频训练模式：基于数据集文件数量计算
                if dataset_files <= 50:
                    recommended_repeats = 10
                    reason = f"小数据集({dataset_files}个文件)，建议较高重复次数以充分学习"
                elif dataset_files <= 200:
                    recommended_repeats = 5
                    reason = f"中等数据集({dataset_files}个文件)，适中重复次数平衡训练效果"
                elif dataset_files <= 500:
                    recommended_repeats = 3
                    reason = f"较大数据集({dataset_files}个文件)，适当降低重复次数"
                else:
                    recommended_repeats = 1
                    reason = f"大数据集({dataset_files}个文件)，单次重复即可获得良好效果"
            
            add_log(f"推荐数据重复次数: {recommended_repeats} ({reason})", 'info')
            
            return jsonify({
                'success': True,
                'recommended_repeats': recommended_repeats,
                'dataset_files': dataset_files,
                'reason': reason,
                'task_type': task_type,
                'is_video_training': is_video_training,
                'video_duration': video_duration
            })
            
        except Exception as e:
            return jsonify({'success': False, 'message': f'解析TOML文件失败: {e}'})
            
    except Exception as e:
        add_log(f"推荐数据重复次数计算失败: {e}", 'error')
        return jsonify({'success': False, 'message': f'推荐数据重复次数计算失败: {e}'})

@app.route('/api/calculate_chatgpt_recommended_params', methods=['POST'])
def calculate_chatgpt_recommended_params():
    """ChatGPT智能推荐所有训练参数"""
    try:
        data = request.get_json()
        dataset_config = data.get('dataset_config')
        task_type = data.get('task_type', 't2v')
        is_video_training = data.get('is_video_training', True)
        video_duration = data.get('video_duration', 5)
        batch_size = data.get('batch_size', 1)  # 获取批次大小参数
        
        if not dataset_config or not os.path.exists(dataset_config):
            return jsonify({'success': False, 'message': '数据集配置文件不存在'})
        
        # 读取TOML文件获取数据集信息
        import toml
        import math
        
        config = toml.load(dataset_config)
        datasets = config.get('datasets', [])
        if not datasets:
            return jsonify({'success': False, 'message': 'TOML文件中未找到datasets配置'})
        
        # 根据训练类型获取正确的数据集目录
        if is_video_training:
            data_directory = datasets[0].get('video_directory')
            directory_type = 'video_directory'
        else:
            data_directory = datasets[0].get('image_directory')
            directory_type = 'image_directory'
            
        if not data_directory or not os.path.exists(data_directory):
            return jsonify({'success': False, 'message': '数据集目录不存在'})
        
        # 统计txt文件数量
        txt_files = [f for f in os.listdir(data_directory) if f.endswith('.txt')]
        dataset_files = len(txt_files)
        
        if dataset_files == 0:
            return jsonify({'success': False, 'message': '数据集目录中未找到txt文件'})
        
        # 统一推荐公式（Wan2.2 LoRA I2V/T2V）实现
        # 设定变量：N = 数据集文件数，L = 单个视频平均时长，M = 任务类型，B = batch_size
        N_files = dataset_files
        
        # 按总时长计算方案：video_duration 现在表示所有视频的总时长（秒）
        if is_video_training:
            total_video_seconds = video_duration  # 直接使用用户设置的总时长
            avg_video_len = video_duration / N_files if N_files > 0 else 0  # 计算平均单个视频时长
            # 有效样本数就是txt文件数量（每个txt对应一个训练样本）
            N_eff = N_files
            # 视频时长保持秒数显示
            video_duration_calc = total_video_seconds
        else:
            total_video_seconds = 0
            avg_video_len = 0
            N_eff = N_files  # 图片训练时，有效样本数等于文件数
            video_duration_calc = 0
        
        # 使用新的wan22_lora_params算法（batch_size自由输入，训练充分）
        
        # 任务类型参数
        mode = "i2v" if task_type.lower().startswith('i2v') else "t2v"
        
        # 调用新的wan22_lora_params函数
        params = wan22_lora_params(
            num_files=N_files,
            total_video_seconds=total_video_seconds,
            avg_video_len=avg_video_len,
            mode=mode,
            batch_size=batch_size
        )
        
        # 提取计算结果
        num_repeats = params["num_repeats"]
        learning_rate = f"{params['learning_rate']:.1e}"
        network_dim = params["network_dim"]
        network_alpha = params["network_alpha"]
        max_train_epochs = params["max_train_epochs"]
        N_eff = params["N_eff"]
        
        # 生成推荐理由
        data_scale = "小规模" if N_eff <= 50 else "中等规模" if N_eff <= 200 else "大规模"
        task_desc = "图像到视频" if mode.lower() == "i2v" else "文本到视频"
        
        if N_eff <= 50:
            repeat_reason = f"小规模数据集({N_eff}样本)基础重复次数2，T2V任务额外增强1.2倍，按批次大小√{batch_size}缩放"
        elif N_eff <= 200:
            repeat_reason = f"中等规模数据集({N_eff}样本)基础重复次数2，T2V任务额外增强1.2倍，按批次大小√{batch_size}缩放"
        else:
            repeat_reason = f"大规模数据集({N_eff}样本)基础重复次数1，按批次大小√{batch_size}缩放避免过拟合"
        
        repeat_reason += f"。最终重复次数{num_repeats}，充分训练算法根据batch_size动态调整"
        
        lr_reason = f"{'I2V' if mode.lower() == 'i2v' else 'T2V'}任务基础学习率{params['learning_rate']/math.sqrt(batch_size):.1e}，按批次大小√{batch_size}缩放至{learning_rate}，保证训练稳定性"
        
        dim_reason = f"基于数据规模选择基础维度{48 if N_eff <= 50 else 64 if N_eff <= 200 else 96}，{'I2V任务0.95倍调整' if mode.lower() == 'i2v' else 'T2V任务1.15倍增强'}，按批次大小动态调整至最接近的标准维度{network_dim}"
        
        alpha_reason = f"设为LoRA维度的{'95%' if batch_size >= 4 else '100%'}({network_alpha})，大批次时适当降低以提升稳定性"
        
        epoch_reason = f"基于数据规模设定基础轮数{30 if N_eff <= 50 else 20 if N_eff <= 200 else 10}，T2V小数据集额外增强1.2倍，按批次大小√{batch_size}缩放至{max_train_epochs}轮，充分训练算法确保收敛"
        
        reasons = {
            "数据重复次数": f"推荐值: {num_repeats} - {repeat_reason}",
            "学习率": f"推荐值: {learning_rate} - {lr_reason}",
            "LoRA维度": f"推荐值: {network_dim} - {dim_reason}",
            "网络Alpha": f"推荐值: {network_alpha} - {alpha_reason}",
            "最大训练轮数": f"推荐值: {max_train_epochs} - {epoch_reason}"
        }
        
        # 生成简化的计算说明
        formula = f"""Wan2.2智能推荐算法：
• 数据集文件数：{N_files}
• 有效样本数：{N_eff}
• 任务类型：{mode.upper()}
• 批次大小：{batch_size}
• 视频训练：{'是' if is_video_training else '否'}
{f'• 视频时长：{video_duration}秒' if is_video_training else ''}

📊 推荐结果基于Wan2.2优化算法，综合考虑数据规模、任务类型和批次大小。"""
        
        return jsonify({
            'success': True,
            'num_repeats': num_repeats,
            'learning_rate': learning_rate,
            'network_dim': network_dim,
            'network_alpha': network_alpha,
            'max_train_epochs': max_train_epochs,
            'dataset_files': N_files,  # 原始文件数
            'effective_samples': N_eff,     # 有效样本数
            'video_duration': video_duration_calc,
            'task_type': task_type,
            'reasons': reasons,
            'formula': formula,
            'batch_size': batch_size
        })
        
    except Exception as e:
        logger.error(f"ChatGPT参数推荐计算失败: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'计算失败: {str(e)}'
        })

@app.route('/api/calculate_chatgpt_recommended_num_repeats', methods=['POST'])
def calculate_chatgpt_recommended_num_repeats():
    """基于ChatGPT算法计算推荐的数据重复次数"""
    try:
        data = request.get_json()
        dataset_config = data.get('dataset_config')
        task_type = data.get('task_type', 't2v')
        is_video_training = data.get('is_video_training', False)
        video_duration = data.get('video_duration', 0)
        batch_size = data.get('batch_size', 8)
        epochs = data.get('epochs', 10)
        
        if not dataset_config or not os.path.exists(dataset_config):
            return jsonify({'success': False, 'message': '数据集配置文件不存在'})
        
        # 读取TOML文件获取数据集信息
        import toml
        import math
        try:
            config = toml.load(dataset_config)
            datasets = config.get('datasets', [])
            if not datasets:
                return jsonify({'success': False, 'message': '数据集配置文件中没有找到datasets配置'})
            
            # 计算数据集文件数量
            dataset_files = 0
            for dataset in datasets:
                # 根据训练类型获取正确的目录字段
                if is_video_training:
                    data_dir = dataset.get('video_directory')
                    directory_type = 'video_directory'
                    file_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
                else:
                    data_dir = dataset.get('image_dir') or dataset.get('image_directory')
                    directory_type = 'image_dir或image_directory'
                    file_extensions = ('.jpg', '.jpeg', '.png', '.webp')
                    
                if data_dir and os.path.exists(data_dir):
                    try:
                        files = [f for f in os.listdir(data_dir) if f.lower().endswith(file_extensions)]
                        dataset_files += len(files)
                        add_log(f"ChatGPT算法 - 数据集目录 {data_dir} 找到 {len(files)} 个文件", 'info')
                    except Exception as e:
                        add_log(f"ChatGPT算法 - 读取数据集目录 {data_dir} 失败: {e}", 'warning')
                elif data_dir:
                    add_log(f"ChatGPT算法 - 数据集目录不存在: {data_dir}", 'warning')
                else:
                    add_log(f"ChatGPT算法 - 数据集配置中缺少{directory_type}字段", 'warning')
            
            # ChatGPT推荐算法：基于目标总步数计算
            # 目标总步数范围选择
            if dataset_files <= 50:
                target_steps = 5000  # 小数据集使用中等质量目标
            elif dataset_files <= 200:
                target_steps = 10000  # 中等数据集使用较高质量目标
            else:
                target_steps = 15000  # 大数据集使用高质量目标
            
            # 根据I2V vs T2V调整目标步数
            if task_type == 'i2v':
                target_steps = int(target_steps * 0.8)  # I2V通常需要较少步数
            elif task_type == 't2v':
                target_steps = int(target_steps * 1.0)  # T2V保持标准步数
            
            # 根据视频时长调整（如果是视频训练）
            if is_video_training and video_duration > 0:
                if video_duration <= 10:
                    target_steps = int(target_steps * 1.2)  # 短视频需要更多步数
                elif video_duration > 30:
                    target_steps = int(target_steps * 0.8)  # 长视频可以减少步数
            
            # 使用ChatGPT公式计算推荐repeats
            # total_steps = ceil((N_images × repeats) / batch_size) × epochs
            # 反推: repeats = (target_steps / epochs) * batch_size / N_images
            if dataset_files > 0 and epochs > 0:
                steps_per_epoch_needed = target_steps / epochs
                repeats_needed = (steps_per_epoch_needed * batch_size) / dataset_files
                
                # 根据ChatGPT建议的区间约束
                if dataset_files <= 50:
                    # 小数据集：50-200
                    recommended_repeats = max(50, min(200, int(repeats_needed)))
                elif dataset_files <= 200:
                    # 中等数据集：10-50
                    recommended_repeats = max(10, min(50, int(repeats_needed)))
                else:
                    # 大数据集：1-5
                    recommended_repeats = max(1, min(5, int(repeats_needed)))
            else:
                # 默认值
                if dataset_files <= 50:
                    recommended_repeats = 100
                elif dataset_files <= 200:
                    recommended_repeats = 25
                else:
                    recommended_repeats = 3
            
            # 计算实际的训练步数
            steps_per_epoch = math.ceil((dataset_files * recommended_repeats) / batch_size)
            actual_total_steps = steps_per_epoch * epochs
            
            # 生成推荐理由
            reason = f"ChatGPT算法: 目标{target_steps}步数，数据集{dataset_files}文件，批次{batch_size}，轮数{epochs}，计算得出{recommended_repeats}次重复"
            if task_type == 'i2v':
                reason += "，I2V任务已优化"
            elif task_type == 't2v':
                reason += "，T2V任务标准配置"
            
            add_log(f"ChatGPT推荐数据重复次数: {recommended_repeats} ({reason})", 'info')
            add_log(f"ChatGPT算法计算: 每轮{steps_per_epoch}步，总计{actual_total_steps}步", 'info')
            
            return jsonify({
                'success': True,
                'recommended_repeats': recommended_repeats,
                'dataset_files': dataset_files,
                'target_steps': target_steps,
                'steps_per_epoch': steps_per_epoch,
                'actual_total_steps': actual_total_steps,
                'reason': reason,
                'task_type': task_type,
                'is_video_training': is_video_training,
                'video_duration': video_duration,
                'batch_size': batch_size,
                'epochs': epochs
            })
            
        except Exception as e:
            return jsonify({'success': False, 'message': f'ChatGPT算法解析TOML文件失败: {e}'})
            
    except Exception as e:
        add_log(f"ChatGPT推荐数据重复次数计算失败: {e}", 'error')
        return jsonify({'success': False, 'message': f'ChatGPT推荐数据重复次数计算失败: {e}'})

@app.route('/api/calculate_recommended_network_dim', methods=['POST'])
def calculate_recommended_network_dim():
    """计算推荐的LoRA维度"""
    try:
        data = request.get_json()
        dataset_config = data.get('dataset_config')
        task_type = data.get('task_type', 't2v')
        is_video_training = data.get('is_video_training', True)
        
        if not dataset_config or not os.path.exists(dataset_config):
            return jsonify({'success': False, 'message': '数据集配置文件不存在'})
        
        # 读取TOML文件获取数据集信息
        import toml
        try:
            config = toml.load(dataset_config)
            datasets = config.get('datasets', [])
            if not datasets:
                return jsonify({'success': False, 'message': '数据集配置文件中没有找到datasets配置'})
            
            # 计算数据集文件数量
            dataset_files = 0
            for dataset in datasets:
                if 'image_dir' in dataset:
                    image_dir = dataset['image_dir']
                    if os.path.exists(image_dir):
                        dataset_files += len([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.mp4', '.avi', '.mov'))])
            
            # 基于数据集大小和任务类型推荐LoRA维度
            if dataset_files <= 100:
                if task_type == 'i2v':
                    recommended_dim = 16
                    reason = "小数据集I2V任务，使用较小维度避免过拟合"
                else:
                    recommended_dim = 24
                    reason = "小数据集T2V任务，适中维度保证学习能力"
            elif dataset_files <= 500:
                if task_type == 'i2v':
                    recommended_dim = 32
                    reason = "中等数据集I2V任务，标准维度平衡效果和效率"
                else:
                    recommended_dim = 48
                    reason = "中等数据集T2V任务，较高维度提升表现力"
            else:
                if task_type == 'i2v':
                    recommended_dim = 48
                    reason = "大数据集I2V任务，较高维度充分利用数据"
                else:
                    recommended_dim = 64
                    reason = "大数据集T2V任务，高维度获得最佳效果"
            
            add_log(f"推荐LoRA维度: {recommended_dim} ({reason}, 数据集文件数: {dataset_files})", 'info')
            
            return jsonify({
                'success': True,
                'recommended_dim': recommended_dim,
                'dataset_files': dataset_files,
                'reason': reason,
                'task_type': task_type,
                'is_video_training': is_video_training
            })
            
        except Exception as e:
            return jsonify({'success': False, 'message': f'解析TOML文件失败: {e}'})
            
    except Exception as e:
        add_log(f"推荐LoRA维度计算失败: {e}", 'error')
        return jsonify({'success': False, 'message': f'推荐LoRA维度计算失败: {e}'})

@app.route('/api/calculate_recommended_network_alpha', methods=['POST'])
def calculate_recommended_network_alpha():
    """计算推荐的网络Alpha值"""
    try:
        data = request.get_json()
        dataset_config = data.get('dataset_config')
        task_type = data.get('task_type', 't2v')
        is_video_training = data.get('is_video_training', True)
        network_dim = data.get('network_dim', 32)
        
        if not dataset_config or not os.path.exists(dataset_config):
            return jsonify({'success': False, 'message': '数据集配置文件不存在'})
        
        # 读取TOML文件获取数据集信息
        import toml
        try:
            config = toml.load(dataset_config)
            datasets = config.get('datasets', [])
            if not datasets:
                return jsonify({'success': False, 'message': '数据集配置文件中没有找到datasets配置'})
            
            # 计算数据集文件数量
            dataset_files = 0
            for dataset in datasets:
                if 'image_dir' in dataset:
                    image_dir = dataset['image_dir']
                    if os.path.exists(image_dir):
                        dataset_files += len([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.mp4', '.avi', '.mov'))])
            
            # 基于network_dim和数据集大小推荐Alpha值
            if dataset_files <= 100:
                # 小数据集，使用较小的Alpha避免过拟合
                recommended_alpha = max(network_dim // 2, 8)
                reason = f"小数据集，Alpha设为维度的一半({network_dim}//2={recommended_alpha})避免过拟合"
            elif dataset_files <= 500:
                # 中等数据集，Alpha等于维度
                recommended_alpha = network_dim
                reason = f"中等数据集，Alpha等于维度({network_dim})获得标准强度"
            else:
                # 大数据集，可以使用更高的Alpha
                recommended_alpha = min(network_dim * 2, 128)
                reason = f"大数据集，Alpha设为维度的两倍({network_dim}*2={recommended_alpha})增强学习能力"
            
            # 确保Alpha在合理范围内
            recommended_alpha = max(1, min(recommended_alpha, 128))
            
            add_log(f"推荐网络Alpha: {recommended_alpha} ({reason}, 数据集文件数: {dataset_files})", 'info')
            
            return jsonify({
                'success': True,
                'recommended_alpha': recommended_alpha,
                'dataset_files': dataset_files,
                'network_dim': network_dim,
                'reason': reason,
                'task_type': task_type,
                'is_video_training': is_video_training
            })
            
        except Exception as e:
            return jsonify({'success': False, 'message': f'解析TOML文件失败: {e}'})
            
    except Exception as e:
        add_log(f"推荐网络Alpha计算失败: {e}", 'error')
        return jsonify({'success': False, 'message': f'推荐网络Alpha计算失败: {e}'})

@app.route('/api/status', methods=['GET'])
def get_status():
    """获取状态"""
    global training_process, tensorboard_process, cache_process, cache_type_running
    
    training_running = training_process is not None and training_process.poll() is None
    tensorboard_running = tensorboard_process is not None and tensorboard_process.poll() is None
    cache_running = cache_process is not None and cache_process.poll() is None
    
    # 检查缓存进程状态
    if cache_process is not None and cache_process.poll() is not None:
        # 缓存进程已结束，清理变量
        cache_process = None
        cache_type_running = None
        cache_running = False
    
    # 检查训练是否完成
    if training_process is not None and training_process.poll() is not None:
        # 训练已完成
        success = training_process.returncode == 0
        error = None if success else f"训练失败，退出代码: {training_process.returncode}"
        
        return jsonify({
            'running': False,
            'success': success,
            'error': error,
            'training_running': False,
            'tensorboard_running': tensorboard_running,
            'cache_running': cache_running,
            'cache_type': cache_type_running
        })
    
    return jsonify({
        'running': training_running,
        'status': '训练进行中...' if training_running else '就绪',
        'training_running': training_running,
        'tensorboard_running': tensorboard_running,
        'cache_running': cache_running,
        'cache_type': cache_type_running
    })

def signal_handler(sig, frame):
    """信号处理器，处理Ctrl+C等信号"""
    add_log("收到停止信号，正在关闭所有进程...", 'info')
    stop_current_process()
    sys.exit(0)

if __name__ == '__main__':
    # 设置控制台编码为UTF-8
    import sys
    if sys.platform == 'win32':
        import subprocess
        try:
            subprocess.run(['chcp', '65001'], shell=True, check=True, capture_output=True)
        except:
            pass
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)
    
    # 创建必要的目录
    os.makedirs('./output', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)
    
    add_log("Wan2.2 WebUI 启动")
    
    # 自动启动TensorBoard
    try:
        default_config = load_config()
        log_dir = default_config.get('logging_dir', './logs')
        start_tensorboard_process(log_dir, default_config)
        add_log("TensorBoard进程已自动启动", 'success')
    except Exception as e:
        add_log(f"TensorBoard进程自动启动失败: {e}", 'warning')
    
    print("🚀 启动 Wan2.2 Web UI")
    print("📍 访问地址: http://localhost:7860")
    print("📊 TensorBoard: 已自动启动，点击界面按钮打开")
    print("⏹️  按 Ctrl+C 停止服务器")
    
    # 禁用Flask的访问日志
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    try:
        # 启动Flask-SocketIO应用
        socketio.run(
            app,
            host='0.0.0.0',
            port=7860,
            debug=False,
            allow_unsafe_werkzeug=True
        )
    except KeyboardInterrupt:
        add_log("收到键盘中断信号，正在关闭...", 'info')
        stop_current_process()
        sys.exit(0)