#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen-Image Web UI
现代化的Qwen-Image训练Web界面
"""

import os
import sys
import json
import subprocess
import threading
import time
import webbrowser
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template_string, request, jsonify, send_from_directory, render_template
from werkzeug.utils import secure_filename
import queue
import signal
import logging

app = Flask(__name__)
app.secret_key = 'qwen_webui_secret_key'

# 全局变量
log_queue = queue.Queue()
log_history = []  # 存储所有日志的持久列表
current_process = None
process_lock = threading.Lock()

# 参数说明和建议值
PARAM_DESCRIPTIONS = {
    # 核心模型路径
    'dit_path': {
        'description': 'DiT (Diffusion Transformer) 模型路径',
        'suggestion': '使用官方预训练的 qwen_image_bf16.safetensors 模型',
        'required': True,
        'type': 'file'
    },
    'vae_path': {
        'description': 'VAE (变分自编码器) 模型路径',
        'suggestion': '使用官方预训练的 qwen_image_vae.safetensors 模型',
        'required': True,
        'type': 'file'
    },
    'text_encoder_path': {
        'description': '文本编码器模型路径',
        'suggestion': '使用官方预训练的 qwen_2.5_vl_7b.safetensors 模型',
        'required': True,
        'type': 'file'
    },
    'dataset_config': {
        'description': '数据集配置文件路径 (TOML格式)',
        'suggestion': '包含训练数据路径和标注信息的配置文件',
        'required': True,
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
        'suggestion': 'none - 使用默认权重，适合大多数场景',
        'options': ['none', 'sigma_sqrt', 'logit_normal'],
        'default': 'none'
    },
    'discrete_flow_shift': {
        'description': '离散流偏移参数',
        'suggestion': '2.2 - 官方调优的最佳值，影响生成质量',
        'range': [1.0, 5.0],
        'default': 2.2
    },
    'optimizer_type': {
        'description': '优化器类型',
        'suggestion': 'adamw8bit - 8位AdamW，显著节省显存',
        'options': ['adamw', 'adamw8bit', 'lion', 'sgd'],
        'default': 'adamw8bit'
    },
    'learning_rate': {
        'description': '学习率',
        'suggestion': '推荐区间：2e-4 ~ 5e-05（触发容易，效果立竿见影）。数据少时，过高会导致过拟合，过低又难学到',
        'range': [1e-6, 1e-3],
        'default': 5e-05
    },
    'max_data_loader_n_workers': {
        'description': '数据加载器工作进程数',
        'suggestion': '2 - 平衡加载速度和内存使用，可根据CPU核心数调整',
        'range': [0, 8],
        'default': 2
    },
    'train_batch_size': {
        'description': '训练批次大小',
        'suggestion': '从TOML配置文件的[general]部分读取，影响显存使用和训练速度',
        'range': [1, 32],
        'default': 1,
        'toml_source': True
    },
    'num_repeats': {
        'description': '数据重复次数',
        'suggestion': '从TOML文件的[[datasets]]部分读取。推荐值20~30（图片较少时），数据少时提高重复次数可有效提升训练效果',
        'range': [1, 100],
        'default': 5,
        'toml_source': True
    },
    'network_module': {
        'description': 'LoRA网络模块',
        'suggestion': 'networks.lora_qwen_image - 专为Qwen-Image优化的LoRA实现',
        'default': 'networks.lora_qwen_image',
        'readonly': True
    },
    'network_dim': {
        'description': 'LoRA维度 (rank)',
        'suggestion': '8 - 推荐值。如果只训练单一角色/风格，32更合适',
        'range': [4, 128],
        'default': 8
    },
    'max_train_epochs': {
        'description': '最大训练轮数',
        'suggestion': '8 - 数据少时，提高轮数可以有效提升训练效果',
        'range': [1, 100],
        'default': 8
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
    
    # 加速器配置
    'num_cpu_threads_per_process': {
        'description': '每进程CPU线程数',
        'suggestion': '1 - 避免线程竞争，推荐值',
        'range': [1, 16],
        'default': 1
    },
    
    # 优化选项
    'sdpa': {
        'description': '启用SDPA (Scaled Dot-Product Attention) 优化',
        'suggestion': '启用 - 显著提升注意力计算效率',
        'default': True
    },
    'gradient_checkpointing': {
        'description': '梯度检查点',
        'suggestion': '启用 - 以计算时间换取显存节省，推荐启用',
        'default': True
    },
    'persistent_data_loader_workers': {
        'description': '持久化数据加载器工作进程',
        'suggestion': '启用 - 避免重复创建进程，提升性能',
        'default': True
    },
    
    # 日志配置
    'logging_dir': {
        'description': '日志目录路径',
        'suggestion': './logs - 存储训练日志和TensorBoard数据的目录',
        'default': './logs',
        'type': 'directory'
    },
    'log_with': {
        'description': '日志记录工具',
        'suggestion': 'tensorboard - 官方推荐的可视化工具，用于监控训练过程',
        'options': ['tensorboard', 'wandb', 'all'],
        'default': 'tensorboard'
    },
    
    # 输出配置
    'output_dir': {
        'description': '输出目录',
        'suggestion': '训练结果保存路径',
        'required': True,
        'type': 'directory'
    },
    'output_name': {
        'description': '输出文件名前缀',
        'suggestion': '生成的LoRA模型文件名',
        'required': True,
        'type': 'string'
    },
    
    # 新增参数 - 基于用户最新配置
    'network_args': {
        'description': 'LoRA网络额外参数',
        'suggestion': 'loraplus_lr_ratio=4 - 如果你想让LoRA更容易触发，提高该值。为LoRA的A和B矩阵设置不同学习率比例',
        'default': 'loraplus_lr_ratio=4',
        'type': 'string'
    },
    'fp8_base': {
        'description': '启用FP8基础模型量化',
        'suggestion': '启用 - 显著节省显存，适合显存不足的情况，对12GB显卡推荐启用',
        'default': True
    },
    'fp8_scaled': {
        'description': '启用FP8缩放量化',
        'suggestion': '启用 - 进一步优化显存使用，与fp8_base配合使用效果更佳',
        'default': True
    },
    'blocks_to_swap': {
        'description': '交换到CPU的Transformer块数量',
        'suggestion': '20 - 将部分模型层交换到CPU以节省显存，数值越大节省显存越多但速度越慢',
        'range': [0, 50],
        'default': 20
    }
}

# 默认配置 - 基于官方默认命令行参数
DEFAULT_CONFIG = {
    # 环境配置
    'enable_venv': True,
    'venv_python_path': './venv/Scripts/',
    
    # 核心模型路径 (必需参数)
    'dit_path': './model/diffusion_models/qwen_image_bf16.safetensors',  # DiT模型路径
    'vae_path': './model/vae/qwen_image_vae.safetensors',  # VAE模型路径
    'text_encoder_path': './model/text_encoders/qwen_2.5_vl_7b.safetensors',  # 文本编码器路径
    'dataset_config': './ai_data/datasets/lovemf_config.toml',  # 数据集配置文件路径
    
    # 输出配置
    'output_dir': './output',  # 输出目录
    'output_name': 'lovemf_lora',  # 输出文件名
    
    # 训练参数 - 基于官方默认命令行参数
    'mixed_precision': 'bf16',  # 混合精度训练，建议值：bf16
    'timestep_sampling': 'shift',  # 时间步采样方法，建议值：shift
    'weighting_scheme': 'none',  # 权重方案，建议值：none
    'discrete_flow_shift': 2.2,  # 离散流偏移，建议值：2.2
    'optimizer_type': 'adamw8bit',  # 优化器类型，建议值：adamw8bit (节省显存)
    'learning_rate': 5e-05,  # 学习率，推荐值：5e-05
    'max_data_loader_n_workers': 2,  # 数据加载器工作进程数，建议值：2
    'train_batch_size': 1,  # 训练批次大小，从TOML文件读取
    'num_repeats': 5,  # 数据重复次数，从TOML文件读取
    'network_module': 'networks.lora_qwen_image',  # 网络模块，固定值
    'network_dim': 8,  # LoRA维度，推荐值：8
    'network_args': 'loraplus_lr_ratio=4',  # LoRA网络额外参数
    'max_train_epochs': 8,  # 最大训练轮数，用户配置值：8
    'save_every_n_epochs': 1,  # 每N轮保存一次，官方默认值：1
    'seed': 42,  # 随机种子，官方默认值：42
    
    # 加速器配置
    'num_cpu_threads_per_process': 1,  # 每进程CPU线程数，建议值：1
    
    # 优化选项 (官方默认启用)
    'sdpa': True,  # 启用SDPA注意力优化，建议启用
    'gradient_checkpointing': True,  # 梯度检查点，节省显存，建议启用
    'persistent_data_loader_workers': True,  # 持久化数据加载器，提升性能，建议启用
    
    # 新增FP8和内存优化选项 - 基于用户最新配置
    'fp8_base': True,  # 启用FP8基础模型量化，节省显存
    'fp8_scaled': True,  # 启用FP8缩放量化，进一步优化显存
    'blocks_to_swap': 20,  # 交换到CPU的Transformer块数量，节省显存
    
    # 缓存配置 (用于预处理)
    'cache_dir': './ai_data/cache',
    'batch_size': 1,  # 用于缓存操作的batch_size
    
    # 日志配置
    'logging_dir': './logs',
    'log_with': 'tensorboard'
}

def log_message(message, level='info'):
    """添加日志消息到队列"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = {
        'timestamp': timestamp,
        'level': level,
        'message': message
    }
    log_queue.put(log_entry)
    log_history.append(log_entry)  # 同时添加到持久列表
    print(f"[{timestamp}] {level.upper()}: {message}")

def run_command(cmd, step_name, config=None):
    """运行命令并实时输出日志"""
    global current_process
    
    log_message(f"开始执行: {step_name}", 'info')
    
    # 检查是否启用虚拟环境
    enable_venv = config.get('enable_venv', True) if config else True
    venv_python_path = config.get('venv_python_path', './venv/Scripts/') if config else './venv/Scripts/'
    
    if enable_venv:
        # 构建虚拟环境Python路径
        if not venv_python_path.endswith(('/', '\\')):
            venv_python_path += '/'
        # 如果是相对路径，转换为绝对路径
        if venv_python_path.startswith('./'):
            venv_python_path = venv_python_path[2:]  # 移除 './'
        venv_python = os.path.join(os.getcwd(), venv_python_path, 'python.exe')
        venv_python = os.path.normpath(venv_python)
        
        log_message(f"启用虚拟环境，Python路径: {venv_python}", 'info')
        
        # 处理不同类型的命令
        if cmd[0] == sys.executable or cmd[0] == 'python':
            # 替换Python命令
            actual_cmd = [venv_python] + cmd[1:]
            log_message(f"使用虚拟环境Python执行: {' '.join(actual_cmd)}", 'debug')
        elif cmd[0] == 'accelerate':
            # 对于accelerate命令，直接使用accelerate可执行文件
            venv_accelerate = os.path.join(os.path.dirname(venv_python), 'accelerate.exe')
            if os.path.exists(venv_accelerate):
                actual_cmd = [venv_accelerate] + cmd[1:]
                log_message(f"使用虚拟环境accelerate执行: {' '.join(actual_cmd)}", 'debug')
            else:
                # 如果accelerate.exe不存在，尝试使用python -m accelerate
                actual_cmd = [venv_python, '-m', 'accelerate'] + cmd[1:]
                log_message(f"使用虚拟环境Python执行accelerate模块: {' '.join(actual_cmd)}", 'debug')
        else:
            actual_cmd = cmd
            log_message("非Python命令，直接执行", 'debug')
    else:
        # 不使用虚拟环境，直接执行原命令
        actual_cmd = cmd
        log_message("未启用虚拟环境，直接执行命令", 'info')
    
    try:
        with process_lock:
            # 在Windows上创建新的进程组，以便能够终止所有子进程
            import subprocess
            if os.name == 'nt':  # Windows
                current_process = subprocess.Popen(
                    actual_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='gbk',
                    errors='ignore',
                    bufsize=1,
                    universal_newlines=True,
                    cwd=os.getcwd(),
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                )
            else:  # Unix/Linux
                current_process = subprocess.Popen(
                    actual_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',
                    errors='ignore',
                    bufsize=1,
                    universal_newlines=True,
                    cwd=os.getcwd(),
                    preexec_fn=os.setsid
                )
        
        # 实时读取输出
        while True:
            output = current_process.stdout.readline()
            if output == '' and current_process.poll() is not None:
                break
            if output:
                log_message(output.strip(), 'output')
        
        return_code = current_process.wait()
        
        # 进程完成后清除引用
        with process_lock:
            current_process = None
        
        if return_code == 0:
            log_message(f"✅ {step_name} 完成", 'success')
            return True
        else:
            log_message(f"❌ {step_name} 失败 (返回码: {return_code})", 'error')
            return False
            
    except subprocess.TimeoutExpired:
        log_message(f"⏰ {step_name} 执行超时", 'error')
        with process_lock:
            if current_process:
                current_process.kill()
            current_process = None
        return False
    except KeyboardInterrupt:
        log_message(f"⏹️ {step_name} 被用户中断", 'warning')
        with process_lock:
            if current_process:
                current_process.terminate()
            current_process = None
        return False
    except Exception as e:
        log_message(f"❌ {step_name} 执行异常: {str(e)}", 'error')
        with process_lock:
            current_process = None
        return False

def stop_current_process():
    """停止当前运行的进程"""
    global current_process
    
    with process_lock:
        if current_process and current_process.poll() is None:
            try:
                log_message("正在停止训练进程...", 'warning')
                
                if os.name == 'nt':  # Windows
                    # 在Windows上，发送CTRL_BREAK_EVENT信号到进程组
                    try:
                        import signal
                        current_process.send_signal(signal.CTRL_BREAK_EVENT)
                        log_message("已发送停止信号到进程组", 'info')
                    except Exception as e:
                        log_message(f"发送停止信号失败: {str(e)}", 'warning')
                        current_process.terminate()
                else:  # Unix/Linux
                    # 在Unix/Linux上，终止整个进程组
                    import signal
                    os.killpg(os.getpgid(current_process.pid), signal.SIGTERM)
                
                # 等待进程结束，最多等待10秒
                try:
                    current_process.wait(timeout=10)
                    log_message("进程已停止", 'info')
                except subprocess.TimeoutExpired:
                    log_message("进程未在10秒内停止，强制终止", 'warning')
                    
                    if os.name == 'nt':  # Windows
                        # 强制终止进程组
                        try:
                            import subprocess
                            subprocess.run(['taskkill', '/F', '/T', '/PID', str(current_process.pid)], 
                                         capture_output=True, text=True)
                            log_message("已强制终止进程组", 'info')
                        except Exception as e:
                            log_message(f"强制终止进程组失败: {str(e)}", 'error')
                            current_process.kill()
                    else:  # Unix/Linux
                        os.killpg(os.getpgid(current_process.pid), signal.SIGKILL)
                    
                    current_process.wait()
                    log_message("进程已强制终止", 'info')
                    
            except Exception as e:
                log_message(f"停止进程时出错: {str(e)}", 'error')
                return False
            finally:
                current_process = None
            return True
        else:
            log_message("没有正在运行的进程", 'info')
            return True

def cache_vae_latents(config):
    """预缓存VAE Latents"""
    cmd = [
        'python', 'src/musubi_tuner/qwen_image_cache_latents.py',
        '--dataset_config', config['dataset_config'],
        '--vae', config['vae_path']
    ]
    return run_command(cmd, "预缓存 VAE Latents", config)

def cache_text_encoder_outputs(config):
    """预缓存文本编码器输出"""
    cmd = [
        'python', 'src/musubi_tuner/qwen_image_cache_text_encoder_outputs.py',
        '--dataset_config', config['dataset_config'],
        '--text_encoder', config['text_encoder_path'],
        '--batch_size', str(config.get('train_batch_size', 1))
    ]
    return run_command(cmd, "预缓存文本编码器输出", config)

def train_lora(config):
    """LoRA训练 - 基于官方默认命令行参数"""
    # 创建输出目录和日志目录
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config.get('logging_dir', './logs'), exist_ok=True)
    
    # 构建官方默认训练命令
    # accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 src/musubi_tuner/qwen_image_train_network.py \
    #     --dit path/to/dit_model \
    #     --vae path/to/vae_model \
    #     --text_encoder path/to/text_encoder \
    #     --dataset_config path/to/toml \
    #     --sdpa --mixed_precision bf16 \
    #     --timestep_sampling shift \
    #     --weighting_scheme none --discrete_flow_shift 2.2 \
    #     --optimizer_type adamw8bit --learning_rate 5e-5 --gradient_checkpointing \
    #     --max_data_loader_n_workers 2 --persistent_data_loader_workers \
    #     --network_module networks.lora_qwen_image \
    #     --network_dim 16 \
    #     --max_train_epochs 16 --save_every_n_epochs 1 --seed 42 \
    #     --output_dir path/to/output_dir --output_name name-of-lora
    
    cmd = [
        'accelerate', 'launch',
        '--num_cpu_threads_per_process', str(config.get('num_cpu_threads_per_process', 1)),
        '--mixed_precision', config.get('mixed_precision', 'bf16'),
        'src/musubi_tuner/qwen_image_train_network.py',
        '--dit', config['dit_path'],
        '--vae', config['vae_path'],
        '--text_encoder', config['text_encoder_path'],
        '--dataset_config', config['dataset_config'],
        '--mixed_precision', config.get('mixed_precision', 'bf16'),
        '--timestep_sampling', config.get('timestep_sampling', 'shift'),
        '--weighting_scheme', config.get('weighting_scheme', 'none'),
        '--discrete_flow_shift', str(config.get('discrete_flow_shift', 2.2)),
        '--optimizer_type', config.get('optimizer_type', 'adamw8bit'),
        '--learning_rate', str(config.get('learning_rate', 5e-5)),
        '--max_data_loader_n_workers', str(config.get('max_data_loader_n_workers', 2)),
        '--network_module', config.get('network_module', 'networks.lora_qwen_image'),
        '--network_dim', str(config.get('network_dim', 8)),
        '--network_args', config.get('network_args', 'loraplus_lr_ratio=4'),
        '--max_train_epochs', str(config.get('max_train_epochs', 8)),
        '--save_every_n_epochs', str(config.get('save_every_n_epochs', 1)),
        '--seed', str(config.get('seed', 42)),
        '--logging_dir', config.get('logging_dir', './logs'),
        '--log_with', config.get('log_with', 'tensorboard'),
        '--output_dir', config['output_dir'],
        '--output_name', config['output_name']
    ]
    
    # 添加官方默认的布尔选项
    if config.get('sdpa'):
        cmd.append('--sdpa')
    
    if config.get('gradient_checkpointing'):
        cmd.append('--gradient_checkpointing')
    
    if config.get('persistent_data_loader_workers'):
        cmd.append('--persistent_data_loader_workers')
    
    # 添加FP8和内存优化选项
    if config.get('fp8_base', True):
        cmd.append('--fp8_base')
    if config.get('fp8_scaled', True):
        cmd.append('--fp8_scaled')
    if config.get('blocks_to_swap'):
        cmd.extend(['--blocks_to_swap', str(config.get('blocks_to_swap', 20))])
    
    return run_command(cmd, "LoRA训练", config)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/cache_vae', methods=['POST'])
def api_cache_vae():
    # 合并默认配置和用户配置
    config = {**DEFAULT_CONFIG, **(request.json or {})}
    
    def run_task():
        success = cache_vae_latents(config)
        return success
    
    thread = threading.Thread(target=run_task)
    thread.daemon = True
    thread.start()
    
    return jsonify({'success': True, 'message': '开始预缓存 VAE Latents'})

@app.route('/api/cache_text_encoder', methods=['POST'])
def api_cache_text_encoder():
    # 合并默认配置和用户配置
    config = {**DEFAULT_CONFIG, **(request.json or {})}
    
    def run_task():
        success = cache_text_encoder_outputs(config)
        return success
    
    thread = threading.Thread(target=run_task)
    thread.daemon = True
    thread.start()
    
    return jsonify({'success': True, 'message': '开始预缓存文本编码器输出'})

@app.route('/api/start_training', methods=['POST'])
def api_start_training():
    global current_process
    
    # 检查是否已有训练进程在运行
    with process_lock:
        if current_process and current_process.poll() is None:
            return jsonify({'success': False, 'message': '训练进程已在运行中'})
    
    # 合并默认配置和用户配置
    config = {**DEFAULT_CONFIG, **(request.json or {})}
    
    def run_task():
        success = train_lora(config)
        return success
    
    thread = threading.Thread(target=run_task)
    thread.daemon = True
    thread.start()
    
    return jsonify({'success': True, 'message': '开始 LoRA 训练'})

@app.route('/api/full_pipeline', methods=['POST'])
def api_full_pipeline():
    # 合并默认配置和用户配置
    config = {**DEFAULT_CONFIG, **(request.json or {})}
    
    def run_task():
        log_message("🚀 开始完整训练流程", 'info')
        
        # 步骤1: 预缓存 VAE Latents
        log_message("📦 步骤 1/3: 预缓存 VAE Latents", 'info')
        if not cache_vae_latents(config):
            log_message("❌ 完整流程失败: VAE Latents 预缓存失败", 'error')
            return False
        
        # 步骤2: 预缓存文本编码器输出
        log_message("📝 步骤 2/3: 预缓存文本编码器输出", 'info')
        if not cache_text_encoder_outputs(config):
            log_message("❌ 完整流程失败: 文本编码器输出预缓存失败", 'error')
            return False
        
        # 步骤3: LoRA训练
        log_message("🎯 步骤 3/3: LoRA 训练", 'info')
        # TensorBoard已在Web UI启动时自动启动，无需重复启动
        if not train_lora(config):
            log_message("❌ 完整流程失败: LoRA 训练失败", 'error')
            return False
        
        log_message("🎉 完整训练流程成功完成！", 'success')
        return True
    
    thread = threading.Thread(target=run_task)
    thread.daemon = True
    thread.start()
    
    return jsonify({'success': True, 'message': '开始完整训练流程'})

@app.route('/api/stop_training', methods=['POST'])
def api_stop_training():
    success = stop_current_process()
    return jsonify({'success': success})

@app.route('/api/start_tensorboard', methods=['POST'])
def api_start_tensorboard():
    """启动TensorBoard"""
    try:
        start_tensorboard()
        return jsonify({'success': True, 'message': 'TensorBoard启动中...'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/logs', methods=['GET'])
def api_logs():
    # 处理队列中的新日志（如果有的话）
    try:
        while True:
            log_entry = log_queue.get_nowait()
            # 日志已经在log_message中添加到log_history了
    except queue.Empty:
        pass
    
    # 返回所有历史日志
    return jsonify({'logs': log_history})



@app.route('/api/save_config', methods=['POST'])
def api_save_config():
    """保存当前配置到JSON文件和TOML文件"""
    try:
        config = request.json
        # 添加调试日志
        log_message(f"收到保存配置请求，包含参数: {list(config.keys())}", 'info')
        if 'num_repeats' in config:
            log_message(f"num_repeats值: {config['num_repeats']}", 'info')
        if 'dataset_config' in config:
            log_message(f"dataset_config值: {config['dataset_config']}", 'info')
        if 'network_args' in config:
            log_message(f"network_args值: {config['network_args']}", 'info')
        if 'seed' in config:
            log_message(f"seed值: {config['seed']}", 'info')
        if 'blocks_to_swap' in config:
            log_message(f"blocks_to_swap值: {config['blocks_to_swap']}", 'info')
        
        config_file = Path('./webui_config.json')
        
        # 特殊处理learning_rate，确保以科学计数法格式保存
        if 'learning_rate' in config:
            lr_value = config['learning_rate']
            if isinstance(lr_value, (int, float)) and lr_value == 0.00005:
                config['learning_rate'] = '5e-05'
            elif isinstance(lr_value, str):
                try:
                    lr_float = float(lr_value)
                    if lr_float == 0.00005:
                        config['learning_rate'] = '5e-05'
                except ValueError:
                    pass
        
        # 保存到JSON文件
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # 保存批次大小和重复次数到TOML文件
        if 'dataset_config' in config and ('train_batch_size' in config or 'num_repeats' in config):
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
            if 'train_batch_size' in config:
                batch_size_pattern = r'batch_size\s*=\s*\d+'
                new_batch_size = f"batch_size = {config['train_batch_size']}"
                
                if '[general]' in toml_content:
                    if re.search(batch_size_pattern, toml_content):
                        toml_content = re.sub(batch_size_pattern, new_batch_size, toml_content)
                    else:
                        toml_content = re.sub(r'(\[general\])', r'\1\n' + new_batch_size, toml_content)
                else:
                    if toml_content and not toml_content.endswith('\n'):
                        toml_content += '\n'
                    toml_content += f"\n[general]\n{new_batch_size}\n"
                updated_params.append(f"batch_size={config['train_batch_size']}")
            
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
                log_message(f"配置已保存到 webui_config.json 和 {config['dataset_config']} ({params_str})", 'success')
            else:
                log_message("配置已保存到 webui_config.json", 'success')
        else:
            log_message("配置已保存到 webui_config.json", 'success')
        
        return jsonify({'success': True, 'message': '配置保存成功'})
    except Exception as e:
        log_message(f"保存配置失败: {e}", 'error')
        return jsonify({'success': False, 'message': f'保存配置失败: {e}'})

@app.route('/api/load_config', methods=['GET'])
def api_load_config():
    """从JSON文件和数据集配置文件加载配置"""
    try:
        config_file = Path('./webui_config.json')
        
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 特殊处理learning_rate，确保以科学计数法格式返回
            if 'learning_rate' in config:
                lr_value = config['learning_rate']
                if isinstance(lr_value, (int, float)) and lr_value == 0.00005:
                    config['learning_rate'] = '5e-05'
                elif isinstance(lr_value, str):
                    try:
                        lr_float = float(lr_value)
                        if lr_float == 0.00005:
                            config['learning_rate'] = '5e-05'
                    except ValueError:
                        pass
            
            log_message(f"已加载保存的配置 - learning_rate: {config.get('learning_rate', 'N/A')}, network_dim: {config.get('network_dim', 'N/A')}", 'success')
        else:
            # 使用默认配置
            config = DEFAULT_CONFIG.copy()
            log_message(f"使用默认配置 - learning_rate: {config['learning_rate']}, network_dim: {config['network_dim']}", 'info')
            log_message(f"DEFAULT_CONFIG原始值 - learning_rate: {DEFAULT_CONFIG['learning_rate']}, network_dim: {DEFAULT_CONFIG['network_dim']}", 'info')
        
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
                    config['train_batch_size'] = toml_config['general']['batch_size']
                    log_message(f"从 {dataset_config_path} 读取 batch_size: {config['train_batch_size']}", 'info')
                
                # 从TOML文件中读取num_repeats（先检查[general]，再检查[[datasets]]）
                if 'general' in toml_config and 'num_repeats' in toml_config['general']:
                    config['num_repeats'] = toml_config['general']['num_repeats']
                    log_message(f"从 {dataset_config_path} [general] 读取 num_repeats: {config['num_repeats']}", 'info')
                elif 'datasets' in toml_config and isinstance(toml_config['datasets'], list) and len(toml_config['datasets']) > 0:
                    if 'num_repeats' in toml_config['datasets'][0]:
                        config['num_repeats'] = toml_config['datasets'][0]['num_repeats']
                        log_message(f"从 {dataset_config_path} [[datasets]] 读取 num_repeats: {config['num_repeats']}", 'info')
            except Exception as e:
                log_message(f"读取数据集配置文件失败: {e}", 'warning')
        
        return jsonify({'success': True, 'config': config})
    except Exception as e:
        log_message(f"加载配置失败: {e}", 'error')
        return jsonify({'success': False, 'message': f'加载配置失败: {e}', 'config': DEFAULT_CONFIG})

@app.route('/api/test_log', methods=['POST'])
def api_test_log():
    """添加测试日志消息"""
    try:
        data = request.get_json()
        message = data.get('message', '测试日志消息')
        log_message(message, 'info')
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/clear_logs', methods=['POST'])
def api_clear_logs():
    """清空所有日志历史"""
    try:
        global log_history
        log_history.clear()
        # 清空队列中的日志
        while not log_queue.empty():
            try:
                log_queue.get_nowait()
            except queue.Empty:
                break
        return jsonify({'success': True, 'message': '日志已清空'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/reset_config', methods=['POST'])
def api_reset_config():
    """重置配置为默认值（保留最大训练步数）"""
    try:
        config_file = 'webui_config.json'
        current_config = DEFAULT_CONFIG.copy()
        
        # 确保关键参数使用正确的默认值，learning_rate使用科学计数法格式
        current_config['learning_rate'] = '5e-05'
        current_config['network_dim'] = 8
        
        log_message(f"重置配置 - learning_rate: {current_config['learning_rate']}, network_dim: {current_config['network_dim']}", 'info')
        log_message(f"DEFAULT_CONFIG原始值 - learning_rate: {DEFAULT_CONFIG['learning_rate']}, network_dim: {DEFAULT_CONFIG['network_dim']}", 'info')
        log_message("配置已重置为默认值", 'success')
        return jsonify(current_config)
    except Exception as e:
        log_message(f"重置配置失败: {e}", 'error')
        return jsonify({'success': False, 'message': f'重置配置失败: {e}'})

@app.route('/api/check_files', methods=['GET'])
def api_check_files():
    """检查必要文件是否存在"""
    try:
        # 检查数据集配置文件
        dataset_exists = os.path.exists('dataset_config.toml')
        
        # 检查模型文件（从默认配置获取路径）
        model_path = DEFAULT_CONFIG.get('pretrained_model_name_or_path', '')
        model_exists = os.path.exists(model_path) if model_path else False
        
        # 检查VAE模型文件
        vae_path = DEFAULT_CONFIG.get('vae', '')
        vae_exists = os.path.exists(vae_path) if vae_path else False
        
        return jsonify({
            'dataset_exists': dataset_exists,
            'model_exists': model_exists,
            'vae_exists': vae_exists
        })
    except Exception as e:
        log_message(f"检查文件失败: {e}", 'error')
        return jsonify({'success': False, 'message': f'检查文件失败: {e}'})

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
            log_message(f"从 {toml_path} 读取 batch_size: {batch_size}", 'info')
            return jsonify({'success': True, 'batch_size': batch_size})
        else:
            return jsonify({'success': False, 'message': 'TOML文件中未找到 batch_size 参数（检查了 [general] 部分）', 'batch_size': None})
    
    except Exception as e:
        log_message(f"读取TOML文件失败: {e}", 'error')
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
            log_message(f"从 {toml_path} 读取 num_repeats: {num_repeats}", 'info')
            return jsonify({'success': True, 'num_repeats': num_repeats})
        else:
            return jsonify({'success': False, 'message': 'TOML文件中未找到 num_repeats 参数（检查了 [general] 和 [[datasets]] 部分）', 'num_repeats': None})
    
    except Exception as e:
        log_message(f"读取TOML文件失败: {e}", 'error')
        return jsonify({'success': False, 'message': f'读取TOML文件失败: {e}', 'num_repeats': None})

def start_tensorboard_process():
    """仅启动TensorBoard进程，不打开浏览器"""
    def run_tensorboard():
        try:
            # 检查TensorBoard是否已经在运行
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('localhost', 6006))
            sock.close()
            
            if result != 0:  # 端口未被占用，启动TensorBoard
                log_message("启动TensorBoard进程...", 'info')
                
                # 读取虚拟环境配置
                config_file = 'webui_config.json'
                config = {}
                if os.path.exists(config_file):
                    try:
                        with open(config_file, 'r', encoding='utf-8') as f:
                            config = json.load(f)
                    except Exception as e:
                        log_message(f"读取配置文件失败: {e}", 'warning')
                
                # 获取虚拟环境设置
                enable_venv = config.get('enable_venv', True)
                venv_python_path = config.get('venv_python_path', './venv/Scripts/')
                
                if enable_venv:
                    # 构建虚拟环境Python路径
                    if not venv_python_path.endswith(('/', '\\')):
                        venv_python_path += '/'
                    # 如果是相对路径，转换为绝对路径
                    if venv_python_path.startswith('./'):
                        venv_python_path = venv_python_path[2:]  # 移除 './'
                    venv_python = os.path.join(os.getcwd(), venv_python_path, 'python.exe')
                    venv_python = os.path.normpath(venv_python)
                    
                    log_message(f"使用虚拟环境Python启动TensorBoard: {venv_python}", 'info')
                    python_cmd = venv_python
                else:
                    log_message("使用系统Python启动TensorBoard", 'info')
                    python_cmd = sys.executable
                
                subprocess.Popen([
                    python_cmd, "-m", "tensorboard.main", 
                    "--logdir", "./logs", "--port", "6006", "--host", "0.0.0.0"
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # 等待TensorBoard启动
                time.sleep(3)
                log_message("TensorBoard进程已启动: http://localhost:6006", 'success')
            else:
                log_message("TensorBoard已在运行中", 'info')
            
        except Exception as e:
            log_message(f"启动TensorBoard失败: {e}", 'error')
    
    # 在后台线程中启动TensorBoard
    threading.Thread(target=run_tensorboard, daemon=True).start()

def start_tensorboard():
    """启动TensorBoard并打开浏览器"""
    def open_tensorboard():
        try:
            # 检查TensorBoard是否在运行
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('localhost', 6006))
            sock.close()
            
            if result == 0:  # TensorBoard正在运行
                # 直接打开浏览器
                webbrowser.open('http://localhost:6006')
                log_message("已打开TensorBoard页面: http://localhost:6006", 'success')
            else:
                # TensorBoard未运行，先启动再打开
                start_tensorboard_process()
                time.sleep(3)  # 等待启动
                webbrowser.open('http://localhost:6006')
                log_message("TensorBoard已启动并打开: http://localhost:6006", 'success')
            
        except Exception as e:
            log_message(f"打开TensorBoard失败: {e}", 'error')
    
    # 在后台线程中处理
    threading.Thread(target=open_tensorboard, daemon=True).start()

def clear_logs_directory():
    """清空logs目录下的所有文件和子目录"""
    logs_dir = Path('./logs')
    try:
        if logs_dir.exists():
            import shutil
            # 删除整个logs目录及其内容
            shutil.rmtree(logs_dir)
            log_message("已删除logs目录及其所有内容", 'info')
            # 重新创建空的logs目录
            logs_dir.mkdir(exist_ok=True)
            log_message("已重新创建logs目录", 'info')
        else:
            # 如果logs目录不存在，创建它
            logs_dir.mkdir(exist_ok=True)
            log_message("已创建logs目录", 'info')
        log_message("logs目录已清空", 'success')
    except Exception as e:
        log_message(f"清空logs目录失败: {e}", 'error')

def signal_handler(sig, frame):
    """处理Ctrl+C信号"""
    print("\n正在关闭Web服务器...")
    stop_current_process()
    sys.exit(0)

if __name__ == '__main__':
    # 设置环境变量以抑制MSVC警告
    os.environ['DISTUTILS_USE_SDK'] = '1'
    os.environ['MSSdk'] = '1'
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    
    # 确保logs目录存在（不清理）
    logs_dir = Path('./logs')
    logs_dir.mkdir(exist_ok=True)
    
    # 自动启动TensorBoard进程
    try:
        start_tensorboard_process()
        log_message("TensorBoard进程已自动启动", 'success')
    except Exception as e:
        log_message(f"TensorBoard进程自动启动失败: {e}", 'warning')
    
    print("🚀 启动 Qwen-Image Web UI")
    print("📍 访问地址: http://localhost:5000")
    print("📊 TensorBoard: 已自动启动，点击界面按钮打开")
    print("⏹️  按 Ctrl+C 停止服务器")
    
    log_message("Web UI 服务器启动", 'info')
    
    try:
        # 禁用Flask的访问日志
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n服务器已停止")
    except Exception as e:
        print(f"服务器启动失败: {e}")