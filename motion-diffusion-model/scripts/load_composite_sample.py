"""
从复合数据集 .npy 文件中加载一条数据

用法:
    python scripts/load_composite_sample.py --composite_data_path dataset/HumanML3D/composite/composite_k3_test.npy --index 0
"""

import argparse
import numpy as np
import os
from os.path import join as pjoin


def load_composite_sample(composite_data_path, index=0):
    """
    从复合数据集文件中加载一条数据
    
    参数:
        composite_data_path: 复合数据集 .npy 文件路径
        index: 要加载的数据索引（默认 0，即第一条）
    
    返回:
        sample: 数据字典
        metadata: 元数据字典
    """
    # 检查文件是否存在
    if not os.path.exists(composite_data_path):
        raise FileNotFoundError(f"文件不存在: {composite_data_path}")
    
    # 加载数据
    print(f"加载复合数据集: {composite_data_path}")
    data_dict = np.load(composite_data_path, allow_pickle=True)[None][0]
    
    samples = data_dict['samples']
    metadata = data_dict.get('metadata', {})
    
    print(f"数据集包含 {len(samples)} 个样本")
    print(f"K={metadata.get('k_segments', 'unknown')}, Split={metadata.get('split', 'unknown')}")
    
    # 检查索引是否有效
    if index < 0 or index >= len(samples):
        raise IndexError(f"索引 {index} 超出范围 [0, {len(samples)-1}]")
    
    # 获取指定索引的数据
    sample = samples[index]
    
    return sample, metadata


def print_sample_info(sample, metadata, index=0):
    """打印样本信息"""
    print(f"\n{'='*60}")
    print(f"样本 #{index}")
    print(f"{'='*60}")
    
    # 基本信息
    print(f"\n【复合提示文本 (Composite Prompt)】")
    print(f"  {sample['composite_prompt']}")
    
    print(f"\n【子提示文本 (Sub Prompts)】")
    sub_prompts = sample['sub_prompts']
    durations = sample['durations']  # 秒数
    durations_frames = sample['durations_frames']  # 帧数
    source_ids = sample['source_ids']
    
    for i, (prompt, duration, duration_frames, source_id) in enumerate(zip(
        sub_prompts, durations, durations_frames, source_ids
    )):
        print(f"  片段 {i+1}:")
        print(f"    - 文本: {prompt}")
        print(f"    - 时长: {duration:.2f} 秒 ({duration_frames} 帧)")
        print(f"    - 来源 ID: {source_id}")
    
    print(f"\n【汇总信息】")
    print(f"  - 总帧数: {sample['total_frames']}")
    print(f"  - 总时长: {sum(durations):.2f} 秒")
    print(f"  - K 值: {metadata.get('k_segments', 'unknown')}")
    print(f"  - 数据集: {metadata.get('dataset_name', 'unknown')}")
    print(f"  - Split: {metadata.get('split', 'unknown')}")
    
    print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='从复合数据集文件中加载一条数据')
    parser.add_argument('--composite_data_path', type=str, required=True,
                       help='复合数据集 .npy 文件路径')
    parser.add_argument('--index', type=int, default=0,
                       help='要加载的数据索引（默认 0，即第一条）')
    parser.add_argument('--save_text', type=str, default=None,
                       help='如果指定，将文本信息保存到文件（可选）')
    
    args = parser.parse_args()
    
    try:
        # 加载数据
        sample, metadata = load_composite_sample(args.composite_data_path, args.index)
        
        # 打印信息
        print_sample_info(sample, metadata, args.index)
        
        # 如果指定了保存文件，保存文本信息
        if args.save_text:
            with open(args.save_text, 'w', encoding='utf-8') as f:
                f.write("="*60 + "\n")
                f.write(f"复合数据集样本 #{args.index}\n")
                f.write("="*60 + "\n\n")
                f.write(f"复合提示文本:\n  {sample['composite_prompt']}\n\n")
                f.write("子提示文本:\n")
                for i, (prompt, duration, duration_frames, source_id) in enumerate(zip(
                    sample['sub_prompts'],
                    sample['durations'],
                    sample['durations_frames'],
                    sample['source_ids']
                )):
                    f.write(f"  片段 {i+1}:\n")
                    f.write(f"    文本: {prompt}\n")
                    f.write(f"    时长: {duration:.2f} 秒 ({duration_frames} 帧)\n")
                    f.write(f"    来源 ID: {source_id}\n\n")
                f.write(f"总帧数: {sample['total_frames']}\n")
                f.write(f"总时长: {sum(sample['durations']):.2f} 秒\n")
            print(f"✓ 文本信息已保存到: {args.save_text}")
        
        return sample, metadata
    
    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == '__main__':
    main()

