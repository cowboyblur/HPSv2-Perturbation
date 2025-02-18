import os
import subprocess

# 设置基本路径
base_image_folder = './data/benchmark/benchmark_imgs/'
base_meta_file_template = './data/benchmark/{}.json'  # (1) 需要替换
base_save_folder = './data/perturbation/mim/{}/{}/'  # (2)/(1) 需要替换
checkpoint_path = 'checkpoints/HPS_v2_compressed.pt'

# (1) 遍历的四个选项
categories = ['anime', 'concept-art', 'paintings', 'photo']

# (2) 遍历 benchmark_imgs 目录下的所有子文件夹
benchmark_folders = [folder for folder in os.listdir(base_image_folder) if os.path.isdir(os.path.join(base_image_folder, folder))]

# 运行攻击并输出 HPS 分数
def run_attack(image_folder, meta_file, save_folder, category):
    command = [
        'python', 'img_score_mim.py', 
        '--image-folder', image_folder,
        '--meta-file', meta_file,
        '--checkpoint', checkpoint_path,
        '--save-folder', save_folder
    ]
    
    # 执行命令并捕获输出
    result = subprocess.run(command, capture_output=True, text=True)

    lines = result.stdout.strip().split('\n')
    if lines:
        # 假设 HPS 分数是最后一行，返回该行内容
        last_line = lines[-1]
        return last_line
    return "No HPS score found"

# 主函数，遍历所有文件夹和类别
def main():
    for folder in benchmark_folders:
        if folder == 'Laf':
            for category in categories:
                # 构造 meta 文件路径和图像文件夹路径

                image_folder = os.path.join(base_image_folder, folder, category)
                meta_file = base_meta_file_template.format(category)  # 使用 (1) 的类别来生成 meta 文件路径
                save_folder = os.path.join(base_save_folder.format(folder, category))  # 保存路径

                # 确保保存文件夹存在
                os.makedirs(save_folder, exist_ok=True)

                # 执行攻击并输出 HPS 分数
                output = run_attack(image_folder, meta_file, save_folder, category)

                # 输出 HPS 分数以及 image_folder 信息
                print(f"Output for {category} in folder {folder}:\n{output}")

if __name__ == '__main__':
    main()
