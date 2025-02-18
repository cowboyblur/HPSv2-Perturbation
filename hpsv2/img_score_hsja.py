import torch
from PIL import Image
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
from hpsv2.utils import root_path, hps_version_map
import warnings
import argparse
import os
import huggingface_hub
import json
from tqdm import tqdm
from torchvision.utils import save_image
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=UserWarning)

model_dict = {}
device = 'cuda' if torch.cuda.is_available() else 'cpu'

import torch
import numpy as np
import random
from torch.nn import functional as F

def hsja_attack(model, images, texts, epsilon, num_iter, step_size=0.2, max_attempts=10):
    """
    HSJA (HopSkipJump Attack) 攻击方法
    :param model: 目标模型
    :param images: 输入图像 (tensor)
    :param texts: 输入文本 (tensor)
    :param epsilon: 扰动范围
    :param num_iter: 迭代步数
    :param step_size: 每次更新的步长
    :param max_attempts: 每次二分查找的最大尝试次数
    :return: 生成的对抗样本
    """
    model.eval()
    ori_images = images.clone().detach().to(device)
    images = images.clone().detach().to(device)
    images.requires_grad = True

    # 随机初始化扰动
    perturbation = torch.zeros_like(images).to(device)
    best_adv_image = images.clone().detach()

    for _ in range(num_iter):
        output = model(best_adv_image, texts)
        image_features, text_features, logit_scale = output["image_features"], output["text_features"], output["logit_scale"]
        logits_per_image = logit_scale * image_features @ text_features.T
        hps_score = torch.diagonal(logits_per_image)

        loss = hps_score[0]
        print(loss)

        for attempt in range(max_attempts):
            perturbation = perturbation + random.uniform(0.01, 0.1) * torch.randn_like(images).to(device)
            perturbed_images = images + perturbation
            perturbed_images = torch.clamp(perturbed_images, 0, 1)

            output = model(perturbed_images, texts)
            image_features, text_features = output["image_features"], output["text_features"]
            logits_per_image = image_features @ text_features.T
            hps_score = torch.diagonal(logits_per_image)

            if hps_score[0] < loss:  
                best_adv_image = perturbed_images
                loss = hps_score[0]
                break
            else:
                perturbation = perturbation - step_size * perturbation 

        perturbation = torch.clamp(perturbation, -epsilon, epsilon)

    return best_adv_image.detach()


def initialize_model():
    if not model_dict:
        model, preprocess_train, preprocess_val = create_model_and_transforms(
            'ViT-H-14',
            'laion2B-s32B-b79K',
            precision='amp',
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False
        )
        model_dict['model'] = model
        model_dict['preprocess_val'] = preprocess_val

def score(image_folder, meta_file, save_folder, cp=None, hps_version="v2.0", epsilon=0.007):
    initialize_model()
    model = model_dict['model']
    preprocess_val = model_dict['preprocess_val']

    if not os.path.exists(root_path):
        os.makedirs(root_path)
    if cp is None:
        cp = huggingface_hub.hf_hub_download("xswu/HPSv2", hps_version_map[hps_version])

    checkpoint = torch.load(cp, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    tokenizer = get_tokenizer('ViT-H-14')
    model = model.to(device)
    model.eval()
    with open(meta_file, 'r') as f:
        meta_data = json.load(f)

    assert len(meta_data) == 800, f"meta_file 中的 prompt 数量 ({len(meta_data)}) 与图像文件数量不匹配！"

    scores = []
    for idx in tqdm(range(100)):
        image_path = os.path.join(image_folder, f"{idx:05d}.jpg")
        caption = meta_data[idx]

        image = preprocess_val(Image.open(image_path)).unsqueeze(0).to(device)
        text = tokenizer([caption]).to(device)

        adv_image = hsja_attack(model, image, text, epsilon=0.1, num_iter=20)

        adv_image_path = os.path.join(save_folder, f"{idx:05d}.jpg") 
        save_image(adv_image.cpu(), adv_image_path)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                outputs = model(adv_image, text)
                image_features, text_features = outputs["image_features"], outputs["text_features"]
                logits_per_image = image_features @ text_features.T
                hps_score = torch.diagonal(logits_per_image).cpu().numpy()
                scores.append(hps_score[0])

    return sum(scores) / len(scores)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-folder', type=str, required=True, help='Path to the image folder')
    parser.add_argument('--meta-file', type=str, required=True, help='Path to the JSON meta file')
    parser.add_argument('--save-folder', type=str, required=True, help='Folder to save adversarial images')
    parser.add_argument('--checkpoint', type=str, default=os.path.join(root_path, 'HPS_v2_compressed.pt'), help='Path to the model checkpoint')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Perturbation magnitude')

    args = parser.parse_args()

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    avg_hps = score(args.image_folder, args.meta_file, args.save_folder, cp=args.checkpoint, epsilon=args.epsilon)
    print(f"Average HPS: {avg_hps}")
