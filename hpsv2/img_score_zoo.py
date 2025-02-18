import torch
from PIL import Image
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
import warnings
import argparse
import os
import huggingface_hub
from hpsv2.utils import root_path, hps_version_map
from torchattacks import FGSM
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

def zoo_attack(model, images, texts, epsilon, num_iter, population_size=10, mutation_strength=0.05):

    model.eval()
    ori_images = images.clone().detach().to(device)
    images = images.clone().detach().to(device)
    images.requires_grad = True

    population = [torch.randn_like(images).to(device) * epsilon for _ in range(population_size)]
    best_adv_image = images.clone().detach()
    best_score = float('inf')

    for _ in range(num_iter):
        population_scores = []
        for perturbation in population:
            perturbed_images = images + perturbation
            perturbed_images = torch.clamp(perturbed_images, 0, 1) 

            with torch.no_grad():
                output = model(perturbed_images, texts)
                image_features, text_features, logit_scale = output["image_features"], output["text_features"], output["logit_scale"]
                logits_per_image = logit_scale * image_features @ text_features.T
                hps_score = torch.diagonal(logits_per_image).cpu().numpy()

            population_scores.append(hps_score[0])

        best_idx = np.argmin(population_scores)
        if population_scores[best_idx] < best_score:
            best_score = population_scores[best_idx]
            best_adv_image = images + population[best_idx]  

        new_population = []
        for i in range(population_size):
            mutation = population[best_idx] + mutation_strength * torch.randn_like(images).to(device)
            new_population.append(mutation)
        print(best_score)
        population = new_population


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

def score(img_path, prompt, cp=None, hps_version="v2.0", epsilon=0.007):
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

    if isinstance(img_path, list):
        result = []
        for one_img_path in img_path:
            if isinstance(one_img_path, str):
                image = preprocess_val(Image.open(one_img_path)).unsqueeze(0).to(device)
            elif isinstance(one_img_path, Image.Image):
                image = preprocess_val(one_img_path).unsqueeze(0).to(device)
            else:
                raise TypeError('The type of parameter img_path is illegal.')
            text = tokenizer([prompt]).to(device)

            # 生成对抗性样本
            adv_image = zoo_attack(model, image, text, epsilon=0.08, num_iter=30)

            # 保存对抗性样本
            adv_image_path = os.path.splitext(one_img_path)[0] + '_zoo.jpg'
            save_image(adv_image.cpu(), adv_image_path)
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    outputs = model(adv_image, text)
                    image_features, text_features = outputs["image_features"], outputs["text_features"]
                    logits_per_image = image_features @ text_features.T
                    hps_score = torch.diagonal(logits_per_image).cpu().numpy()
            result.append((hps_score[0], adv_image_path))
        return result
    else:
        raise TypeError('The type of parameter img_path is illegal.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', nargs='+', type=str, required=True, help='Path to the input image')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt')
    parser.add_argument('--checkpoint', type=str, default=os.path.join(root_path, 'HPS_v2_compressed.pt'), help='Path to the model checkpoint')
    parser.add_argument('--epsilon', type=float, default=0.6, help='Perturbation magnitude')

    args = parser.parse_args()

    results = score(args.image_path, args.prompt, args.checkpoint, epsilon=args.epsilon)
    for hps_score, adv_image_path in results:
        print(f'HPSv2 score: {hps_score}, Adversarial image saved at: {adv_image_path}')
