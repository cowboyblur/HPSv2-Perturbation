import os
import subprocess

# 定义 meta_file 的四个类别
meta_categories = ["anime", "concept-art", "paintings", "photo"]
# 定义 image_folder 根目录
image_root = "data/perturbation/pgd"

# 获取 benchmark_imgs 下的所有子文件夹（22个）
image_subfolders = [f for f in os.listdir(image_root) if os.path.isdir(os.path.join(image_root, f))]

# 运行 evaluate_hps.py
for subfolder in image_subfolders:
    for meta in meta_categories:
        meta_file = f"data/benchmark/{meta}.json"
        image_folder = os.path.join(image_root, subfolder, meta)
        
        # 确保 image_folder 目录存在，避免错误
        if not os.path.exists(image_folder):
            print(f"Skipping {image_folder}, directory does not exist.")
            continue
        
        print(f"Running evaluation for meta: {meta}, image folder: {image_folder}")

        # 运行 evaluate_hps.py
        cmd = [
            "python", "evaluate_hps.py",
            "--hpc", "checkpoints/hpc.pt",
            "--meta_file", meta_file,
            "--image_folder", image_folder
        ]
        
        # 执行命令并捕获输出
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        # 解析并输出 HPS 结果
        output = process.stdout.strip().split("\n")
        if output:
            last_line = output[-1]
            print(f"HPS Score for {image_folder}: {last_line}")
