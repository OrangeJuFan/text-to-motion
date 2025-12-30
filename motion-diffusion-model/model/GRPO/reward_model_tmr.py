"""
基于 TMR (Text-to-Motion Retrieval) 预训练模型的奖励模型实现

TMR 是一个用于文本-动作检索的模型，通过对比学习将文本和动作映射到共同的嵌入空间。
本文件提供基于 TMR 预训练权重的奖励函数实现，可用于 GRPO 训练。

TMR 模型通常包含：
1. 文本编码器（Text Encoder）- 将文本编码为嵌入向量
2. 动作编码器（Motion Encoder）- 将动作序列编码为嵌入向量
3. 可能包含 Movement Encoder - 用于动作的预处理

奖励计算方式：
- 使用文本和动作嵌入之间的相似度（余弦相似度或欧氏距离）作为奖励
- 相似度越高，奖励越大
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Dict, Union
from os.path import join as pjoin
import os

# 尝试导入 TMR 相关的模块
try:
    from data_loaders.humanml.networks.modules import (
        TextEncoderBiGRUCo,
        MotionEncoderBiGRUCo,
        MovementConvEncoder,
    )
    from data_loaders.humanml.utils.word_vectorizer import WordVectorizer, POS_enumerator
    _tmr_modules_available = True
except ImportError:
    _tmr_modules_available = False
    print("警告: TMR 相关模块未找到，请确保已正确安装依赖")

# 尝试导入 spacy，如果不可用则使用简单处理
try:
    import spacy
    _spacy_available = True
except ImportError:
    _spacy_available = False
    print("警告: spacy 未安装，将使用简单的文本处理方式。建议安装 spacy: pip install spacy && python -m spacy download en_core_web_sm")


class TMRModelWrapper:
    """
    TMR 模型包装器，用于加载和使用 TMR 预训练权重
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        dataset_name: str = 'humanml',
        device: str = 'cuda',
    ):
        """
        初始化 TMR 模型包装器
        
        参数:
            checkpoint_path: TMR 预训练权重路径（.pth 或 .tar 文件）
            dataset_name: 数据集名称 ('humanml' 或 'kit')
            device: 设备
        """
        if not _tmr_modules_available:
            raise ImportError("TMR 相关模块未找到，无法加载 TMR 模型")
        
        self.device = device
        self.dataset_name = dataset_name
        
        # 根据数据集设置维度
        if dataset_name == 'humanml' or dataset_name == 't2m':
            dim_pose = 263
        elif dataset_name == 'kit':
            dim_pose = 251
        else:
            raise ValueError(f"不支持的数据集: {dataset_name}")
        
        # 构建模型
        self.text_encoder = TextEncoderBiGRUCo(
            word_size=300,
            pos_size=len(POS_enumerator),
            hidden_size=512,
            output_size=512,
            device=device
        )
        
        self.motion_encoder = MotionEncoderBiGRUCo(
            input_size=512,
            hidden_size=1024,
            output_size=512,
            device=device
        )
        
        self.movement_encoder = MovementConvEncoder(
            dim_pose - 4,  # 排除根位置和旋转
            512,  # hidden size
            512   # latent size
        )
        
        # 加载预训练权重
        self._load_checkpoint(checkpoint_path)
        
        # 移动到设备并设置为评估模式
        self.text_encoder.to(device)
        self.motion_encoder.to(device)
        self.movement_encoder.to(device)
        
        self.text_encoder.eval()
        self.motion_encoder.eval()
        self.movement_encoder.eval()
    
    def _load_checkpoint(self, checkpoint_path: str):
        """
        加载预训练权重
        
        参数:
            checkpoint_path: 权重文件路径
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"TMR 权重文件不存在: {checkpoint_path}")
        
        print(f"加载 TMR 预训练权重: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 尝试不同的权重键名（兼容不同的保存格式）
        if isinstance(checkpoint, dict):
            # 如果 checkpoint 是字典，尝试加载各个组件
            if 'text_encoder' in checkpoint:
                self.text_encoder.load_state_dict(checkpoint['text_encoder'])
            elif 'text_encoder_state_dict' in checkpoint:
                self.text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])
            
            if 'motion_encoder' in checkpoint:
                self.motion_encoder.load_state_dict(checkpoint['motion_encoder'])
            elif 'motion_encoder_state_dict' in checkpoint:
                self.motion_encoder.load_state_dict(checkpoint['motion_encoder_state_dict'])
            
            if 'movement_encoder' in checkpoint:
                self.movement_encoder.load_state_dict(checkpoint['movement_encoder'])
            elif 'movement_encoder_state_dict' in checkpoint:
                self.movement_encoder.load_state_dict(checkpoint['movement_encoder_state_dict'])
            
            # 如果包含完整的模型状态
            if 'model' in checkpoint:
                model_state = checkpoint['model']
                # 尝试提取各个组件的权重
                text_state = {k.replace('text_encoder.', ''): v 
                             for k, v in model_state.items() 
                             if k.startswith('text_encoder.')}
                motion_state = {k.replace('motion_encoder.', ''): v 
                               for k, v in model_state.items() 
                               if k.startswith('motion_encoder.')}
                movement_state = {k.replace('movement_encoder.', ''): v 
                                 for k, v in model_state.items() 
                                 if k.startswith('movement_encoder.')}
                
                if text_state:
                    self.text_encoder.load_state_dict(text_state, strict=False)
                if motion_state:
                    self.motion_encoder.load_state_dict(motion_state, strict=False)
                if movement_state:
                    self.movement_encoder.load_state_dict(movement_state, strict=False)
        else:
            # 如果 checkpoint 直接是模型状态
            print("警告: checkpoint 格式可能不正确，尝试直接加载...")
            try:
                self.text_encoder.load_state_dict(checkpoint, strict=False)
            except:
                print("无法直接加载 checkpoint，请检查格式")
        
        print("TMR 模型加载完成")
    
    def encode_text(
        self,
        word_embs: torch.Tensor,
        pos_ohot: torch.Tensor,
        cap_lens: torch.Tensor,
    ) -> torch.Tensor:
        """
        编码文本
        
        参数:
            word_embs: 词嵌入 [B, max_len, 300]
            pos_ohot: 词性 one-hot [B, max_len, pos_dim]
            cap_lens: 文本长度 [B]
            
        返回:
            text_embeddings: 文本嵌入 [B, 512]
        """
        with torch.no_grad():
            text_embeddings = self.text_encoder(word_embs, pos_ohot, cap_lens)
        return text_embeddings
    
    def encode_motion(
        self,
        motions: torch.Tensor,
        m_lens: torch.Tensor,
    ) -> torch.Tensor:
        """
        编码动作序列
        
        参数:
            motions: 动作序列 [B, nframes, njoints*nfeats] 或 [B, njoints, nfeats, nframes]
            m_lens: 动作长度 [B]
            
        返回:
            motion_embeddings: 动作嵌入 [B, 512]
        """
        with torch.no_grad():
            # 如果输入是 [B, njoints, nfeats, nframes]，需要转换
            if len(motions.shape) == 4:
                B, njoints, nfeats, nframes = motions.shape
                motions = motions.permute(0, 3, 1, 2).reshape(B, nframes, -1)
            
            # Movement encoding
            movements = self.movement_encoder(motions[..., :-4])  # 排除根位置和旋转
            m_lens = m_lens // 4  # unit_length = 4
            
            # Motion encoding
            motion_embeddings = self.motion_encoder(movements, m_lens)
        
        return motion_embeddings
    
    def get_co_embeddings(
        self,
        word_embs: torch.Tensor,
        pos_ohot: torch.Tensor,
        cap_lens: torch.Tensor,
        motions: torch.Tensor,
        m_lens: torch.Tensor,
    ) -> tuple:
        """
        同时获取文本和动作嵌入
        
        参数:
            word_embs: 词嵌入 [B, max_len, 300]
            pos_ohot: 词性 one-hot [B, max_len, pos_dim]
            cap_lens: 文本长度 [B]
            motions: 动作序列 [B, nframes, njoints*nfeats] 或 [B, njoints, nfeats, nframes]
            m_lens: 动作长度 [B]
            
        返回:
            text_embeddings: 文本嵌入 [B, 512]
            motion_embeddings: 动作嵌入 [B, 512]
        """
        text_embeddings = self.encode_text(word_embs, pos_ohot, cap_lens)
        motion_embeddings = self.encode_motion(motions, m_lens)
        return text_embeddings, motion_embeddings


class TMRRewardFunction:
    """
    基于 TMR 模型的奖励函数基类
    """
    
    def __init__(
        self,
        tmr_checkpoint_path: str,
        dataset_name: str = 'humanml',
        device: str = 'cuda',
        word_vectorizer: Optional[WordVectorizer] = None,
    ):
        """
        初始化 TMR 奖励函数
        
        参数:
            tmr_checkpoint_path: TMR 预训练权重路径
            dataset_name: 数据集名称 ('humanml' 或 'kit')
            device: 设备
            word_vectorizer: 词向量化器（如果为 None，会尝试加载）
        """
        self.device = device
        self.dataset_name = dataset_name
        
        # 初始化 TMR 模型
        self.tmr_model = TMRModelWrapper(
            checkpoint_path=tmr_checkpoint_path,
            dataset_name=dataset_name,
            device=device,
        )
        
        # 初始化词向量化器
        if word_vectorizer is None:
            self.word_vectorizer = WordVectorizer('./glove', 'our_vab')
        else:
            self.word_vectorizer = word_vectorizer
        
        # 初始化 spacy（如果可用）
        if _spacy_available:
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except OSError:
                print("警告: 无法加载 spacy 模型 'en_core_web_sm'，将使用简单文本处理")
                self.nlp = None
        else:
            self.nlp = None
    
    def _process_text(self, sentence: str) -> tuple:
        """
        处理文本，提取词和词性
        
        参数:
            sentence: 输入文本
            
        返回:
            word_list: 词列表
            pos_list: 词性列表
        """
        if self.nlp is not None:
            # 使用 spacy 处理（推荐方式）
            sentence = sentence.replace('-', '')
            doc = self.nlp(sentence)
            word_list = []
            pos_list = []
            for token in doc:
                word = token.text
                if not word.isalpha():
                    continue
                if (token.pos_ == 'NOUN' or token.pos_ == 'VERB') and (word != 'left'):
                    word_list.append(token.lemma_)
                else:
                    word_list.append(word)
                pos_list.append(token.pos_)
            return word_list, pos_list
        else:
            # 简单处理方式（如果 spacy 不可用）
            words = sentence.lower().split()
            pos_list = ['NOUN'] * len(words)
            return words, pos_list
    
    def _prepare_text_inputs(self, prompts: List[str]):
        """
        将文本提示转换为 TMR 模型所需的格式
        
        参数:
            prompts: 文本提示列表
            
        返回:
            word_embs: 词嵌入 [B, max_len, 300]
            pos_ohot: 词性 one-hot [B, max_len, pos_dim]
            cap_lens: 文本长度 [B]
        """
        batch_size = len(prompts)
        max_text_len = 20  # 不包括 SOS 和 EOS
        
        word_embs_list = []
        pos_ohot_list = []
        cap_lens_list = []
        
        for prompt in prompts:
            # 处理文本，获取词和词性
            word_list, pos_list = self._process_text(prompt)
            
            # 转换为 WordVectorizer 所需的格式: 'word/POS'
            tokens = ['%s/%s' % (word_list[i], pos_list[i]) for i in range(len(word_list))]
            tokens = tokens[:max_text_len]  # 截断到最大长度
            
            # 添加 SOS 和 EOS tokens
            if len(tokens) < max_text_len:
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
                tokens = tokens + ['unk/OTHER'] * (max_text_len + 2 - sent_len)
            else:
                # 如果太长，裁剪
                tokens = tokens[:max_text_len]
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
            
            # 获取词嵌入和词性 one-hot
            word_embeddings = []
            pos_one_hots = []
            for token in tokens:
                word_emb, pos_oh = self.word_vectorizer[token]
                word_embeddings.append(word_emb[None, :])
                pos_one_hots.append(pos_oh[None, :])
            
            if len(word_embeddings) == 0:
                # 如果处理失败，使用 unk token
                word_emb, pos_oh = self.word_vectorizer['unk/OTHER']
                word_embeddings = [word_emb[None, :]]
                pos_one_hots = [pos_oh[None, :]]
            
            word_emb = np.concatenate(word_embeddings, axis=0)
            pos_ohot = np.concatenate(pos_one_hots, axis=0)
            
            word_embs_list.append(word_emb)
            pos_ohot_list.append(pos_ohot)
            cap_lens_list.append(sent_len)
        
        word_embs = torch.tensor(np.stack(word_embs_list), dtype=torch.float32).to(self.device)
        pos_ohot = torch.tensor(np.stack(pos_ohot_list), dtype=torch.float32).to(self.device)
        cap_lens = torch.tensor(cap_lens_list, dtype=torch.long).to(self.device)
        
        return word_embs, pos_ohot, cap_lens
    
    def _prepare_motion_inputs(self, motions: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        """
        准备动作输入
        
        参数:
            motions: 动作序列 [B, njoints, nfeats, nframes]
            lengths: 动作长度（如果为 None，使用完整长度）
            
        返回:
            motions_processed: 处理后的动作 [B, nframes, njoints*nfeats]
            m_lens: 动作长度 [B]
        """
        batch_size = motions.shape[0]
        
        if lengths is None:
            # 假设使用完整长度
            m_lens = torch.full((batch_size,), motions.shape[-1], dtype=torch.long, device=self.device)
        else:
            m_lens = lengths.to(self.device)
        
        # 转换格式：从 [B, njoints, nfeats, nframes] 到 [B, nframes, njoints*nfeats]
        if len(motions.shape) == 4:
            # [B, njoints, nfeats, nframes] -> [B, nframes, njoints*nfeats]
            motions_processed = motions.permute(0, 3, 1, 2).reshape(batch_size, motions.shape[-1], -1)
        else:
            motions_processed = motions
        
        return motions_processed, m_lens


class TMRMatchingScoreReward(TMRRewardFunction):
    """
    基于 TMR 匹配分数的奖励函数
    
    使用文本和动作嵌入之间的相似度（余弦相似度或欧氏距离）作为奖励。
    """
    
    def __init__(
        self,
        tmr_checkpoint_path: str,
        similarity_type: str = 'cosine',  # 'cosine' 或 'euclidean'
        max_distance: float = 10.0,  # 用于欧氏距离归一化
        scale: float = 2.0,  # 用于指数衰减
        normalization: str = 'linear',  # 'linear', 'exponential', 'sigmoid'
        *args,
        **kwargs,
    ):
        """
        参数:
            tmr_checkpoint_path: TMR 预训练权重路径
            similarity_type: 相似度类型 ('cosine' 或 'euclidean')
            max_distance: 最大距离（用于线性归一化）
            scale: 缩放因子（用于指数衰减）
            normalization: 归一化方式 ('linear', 'exponential', 'sigmoid')
        """
        super().__init__(tmr_checkpoint_path, *args, **kwargs)
        self.similarity_type = similarity_type
        self.max_distance = max_distance
        self.scale = scale
        self.normalization = normalization
    
    def __call__(
        self,
        motions: torch.Tensor,
        prompts: List[str],
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算基于 TMR 匹配分数的奖励
        
        参数:
            motions: 生成的动作序列 [B, njoints, nfeats, nframes]
            prompts: 文本提示列表 [B]
            lengths: 动作长度（可选）
            
        返回:
            rewards: 奖励值 [B]，范围 [0, 1]
        """
        batch_size = motions.shape[0]
        
        # 准备文本输入
        word_embs, pos_ohot, cap_lens = self._prepare_text_inputs(prompts)
        
        # 准备动作输入
        motions_processed, m_lens = self._prepare_motion_inputs(motions, lengths)
        
        # 获取文本和动作嵌入
        text_embeddings, motion_embeddings = self.tmr_model.get_co_embeddings(
            word_embs=word_embs,
            pos_ohot=pos_ohot,
            cap_lens=cap_lens,
            motions=motions_processed,
            m_lens=m_lens,
        )
        
        # 计算相似度
        if self.similarity_type == 'cosine':
            # 余弦相似度：范围 [-1, 1]，越大越好
            text_embeddings_norm = F.normalize(text_embeddings, p=2, dim=-1)
            motion_embeddings_norm = F.normalize(motion_embeddings, p=2, dim=-1)
            similarities = (text_embeddings_norm * motion_embeddings_norm).sum(dim=-1)  # [B]
            
            # 将余弦相似度从 [-1, 1] 归一化到 [0, 1]
            rewards = (similarities + 1.0) / 2.0
            
        elif self.similarity_type == 'euclidean':
            # 欧氏距离：距离越小，相似度越高
            distances = torch.norm(text_embeddings - motion_embeddings, dim=-1)  # [B]
            
            # 归一化距离到奖励
            if self.normalization == 'linear':
                # 线性归一化
                rewards = 1.0 - torch.clamp(distances / self.max_distance, 0, 1)
            elif self.normalization == 'exponential':
                # 指数衰减
                rewards = torch.exp(-distances / self.scale)
                # 可选：归一化到 [0, 1]
                rewards = (rewards - rewards.min()) / (rewards.max() - rewards.min() + 1e-8)
            elif self.normalization == 'sigmoid':
                # Sigmoid 归一化
                rewards = torch.sigmoid(-distances / self.scale)
            else:
                raise ValueError(f"不支持的归一化方式: {self.normalization}")
        else:
            raise ValueError(f"不支持的相似度类型: {self.similarity_type}")
        
        return rewards


class TMRCosineSimilarityReward(TMRRewardFunction):
    """
    基于 TMR 余弦相似度的奖励函数（简化版本）
    """
    
    def __call__(
        self,
        motions: torch.Tensor,
        prompts: List[str],
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算基于余弦相似度的奖励
        
        参数:
            motions: 生成的动作序列 [B, njoints, nfeats, nframes]
            prompts: 文本提示列表 [B]
            lengths: 动作长度（可选）
            
        返回:
            rewards: 奖励值 [B]，范围 [0, 1]
        """
        batch_size = motions.shape[0]
        
        # 准备输入
        word_embs, pos_ohot, cap_lens = self._prepare_text_inputs(prompts)
        motions_processed, m_lens = self._prepare_motion_inputs(motions, lengths)
        
        # 获取嵌入
        text_embeddings, motion_embeddings = self.tmr_model.get_co_embeddings(
            word_embs=word_embs,
            pos_ohot=pos_ohot,
            cap_lens=cap_lens,
            motions=motions_processed,
            m_lens=m_lens,
        )
        
        # 计算余弦相似度
        text_embeddings_norm = F.normalize(text_embeddings, p=2, dim=-1)
        motion_embeddings_norm = F.normalize(motion_embeddings, p=2, dim=-1)
        similarities = (text_embeddings_norm * motion_embeddings_norm).sum(dim=-1)  # [B]
        
        # 归一化到 [0, 1]
        rewards = (similarities + 1.0) / 2.0
        
        return rewards


def create_tmr_reward_function(
    tmr_checkpoint_path: str,
    reward_type: str = 'matching',
    dataset_name: str = 'humanml',
    device: str = 'cuda',
    **kwargs,
) -> TMRRewardFunction:
    """
    创建 TMR 奖励函数的工厂函数
    
    参数:
        tmr_checkpoint_path: TMR 预训练权重路径
        reward_type: 奖励类型 ('matching', 'cosine')
        dataset_name: 数据集名称
        device: 设备
        **kwargs: 其他参数（如 similarity_type, max_distance, scale, normalization）
        
    返回:
        reward_function: 奖励函数实例
    """
    if reward_type == 'matching':
        return TMRMatchingScoreReward(
            tmr_checkpoint_path=tmr_checkpoint_path,
            dataset_name=dataset_name,
            device=device,
            **kwargs,
        )
    elif reward_type == 'cosine':
        return TMRCosineSimilarityReward(
            tmr_checkpoint_path=tmr_checkpoint_path,
            dataset_name=dataset_name,
            device=device,
            **kwargs,
        )
    else:
        raise ValueError(f"未知的奖励类型: {reward_type}")


# 使用示例
if __name__ == '__main__':
    # 示例：创建 TMR 奖励函数
    # 注意：需要提供 TMR 预训练权重路径
    tmr_checkpoint_path = './path/to/tmr/checkpoint.pth'  # 替换为实际路径
    
    # 创建奖励函数（使用余弦相似度）
    reward_fn = create_tmr_reward_function(
        tmr_checkpoint_path=tmr_checkpoint_path,
        reward_type='cosine',
        device='cuda',
    )
    
    # 或者使用匹配分数（可配置）
    reward_fn = create_tmr_reward_function(
        tmr_checkpoint_path=tmr_checkpoint_path,
        reward_type='matching',
        similarity_type='cosine',  # 或 'euclidean'
        normalization='linear',  # 或 'exponential', 'sigmoid'
        device='cuda',
    )
    
    # 示例用法
    batch_size = 4
    motions = torch.randn(batch_size, 263, 1, 196)  # HumanML3D 格式
    prompts = [
        "a person walks forward",
        "someone jumps up",
        "a person sits down",
        "someone runs fast"
    ]
    
    rewards = reward_fn(motions, prompts)
    print(f"Rewards: {rewards}")

