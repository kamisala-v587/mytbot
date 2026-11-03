#!/usr/bin/env python3
import os
import sys
import json
from pathlib import Path

def is_lerobot_dataset(dir_path: Path) -> bool:
    """检查是否为 LeRobot 数据集"""
    data_dir = dir_path / "data"
    meta_dir = dir_path / "meta"
    info_file = meta_dir / "info.json"
    
    if data_dir.is_dir() and meta_dir.is_dir() and info_file.is_file():
        try:
            with open(info_file, 'r', encoding='utf-8') as f:
                json.load(f)          # 验证 JSON 格式
            return True
        except:
            return False
    return False

def find_lerobot_datasets(root_path: Path, output_file: str = "lerobot_datasets.txt"):
    """递归查找所有 LeRobot 数据集"""
    found = []
    print(f"正在搜索: {root_path}\n")
    
    for dirpath, _, _ in os.walk(root_path):
        current = Path(dirpath)
        if is_lerobot_dataset(current):
            full_path = str(current.absolute())
            found.append(full_path)
            print(f"✅ 找到 LeRobot 数据集: {full_path}")
    
    if found:
        with open(output_file, 'w', encoding='utf-8') as f:
            for p in sorted(found):
                f.write(p + '\n')
        print(f"\n🎉 共找到 {len(found)} 个 LeRobot 数据集！")
        print(f"路径已保存至: {output_file}")
    else:
        print("\n未找到任何 LeRobot 数据集。")

if __name__ == "__main__":
    root = Path("/share/lerobot_out/agibotworld")
    output_dir = Path("/home/jovyan/workspace/mytbot/configs/ds_ids/B200")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{root.name}.txt"
    find_lerobot_datasets(root, output_file=str(output_file))