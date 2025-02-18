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

def mim_attack(model, images0, images1, texts, epsilon, alpha, num_iter, decay=1.0):
    model.eval()
    ori_images0 = images0.clone().detach()
    images0 = images0.clone().detach().to(device)
    images0.requires_grad = True
    momentum0 = torch.zeros_like(images0).to(device)
    ori_images1 = images1.clone().detach()
    images1 = images1.clone().detach().to(device)
    images1.requires_grad = True
    momentum1 = torch.zeros_like(images1).to(device)

    for _ in range(num_iter):
        with torch.cuda.amp.autocast():
            output = model(images0, texts)
            image_features, text_features, logit_scale = output["image_features"], output["text_features"], output["logit_scale"]
            text_0_logits = logit_scale * image_features @ text_features.T
            output = model(images1, texts)
            image_features, text_features, logit_scale = output["image_features"], output["text_features"], output["logit_scale"]
            text_1_logits = logit_scale * image_features @ text_features.T
            label_0 = 0
            label_1 = 1

            index = torch.arange(text_0_logits.shape[0], device=device, dtype=torch.long)
            text_0_logits = text_0_logits[index, index]
            text_1_logits = text_1_logits[index, index]
            text_logits = torch.stack([text_0_logits, text_1_logits], dim=-1)
            text_0_labels = torch.zeros(text_logits.shape[0], device=device, dtype=torch.long)
            text_1_labels = text_0_labels + 1

            text_0_loss = torch.nn.functional.cross_entropy(text_logits, text_0_labels, reduction="none")
            text_1_loss = torch.nn.functional.cross_entropy(text_logits, text_1_labels, reduction="none")

            text_loss = label_0 * text_0_loss + label_1 * text_1_loss

            # absolute_example_weight = 1 / num_per_prompt
            # denominator = absolute_example_weight.sum()
            # weight_per_example = absolute_example_weight / denominator
            # text_loss *= weight_per_example

            text_loss = text_loss.sum()
            print(text_loss)

        grad0 = torch.autograd.grad(text_loss, images0, retain_graph=True, create_graph=False)[0]
        grad1 = torch.autograd.grad(text_loss, images1, retain_graph=True, create_graph=False)[0]

        grad0 = grad0 / torch.mean(torch.abs(grad0), dim=(1, 2, 3), keepdim=True)
        grad1 = grad1 / torch.mean(torch.abs(grad1), dim=(1, 2, 3), keepdim=True)

        momentum0 = decay * momentum0 + grad0
        images0 = images0 + alpha * momentum0.sign()
        images0 = torch.clamp(images0, ori_images0 - epsilon, ori_images0 + epsilon)
        images0 = torch.clamp(images0, 0, 1)
        momentum1 = decay * momentum1 + grad1
        images1 = images1 + alpha * momentum1.sign()
        images1 = torch.clamp(images1, ori_images1 - epsilon, ori_images1 + epsilon)
        images1 = torch.clamp(images1, 0, 1)

    return images0.detach(), images1.detach()



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

def score(img_path1, img_path2, prompt, cp=None, hps_version="v2.0", epsilon=0.007):
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

    if isinstance(img_path1, list):
        result = []
        for one_img_path1 in img_path1:
            for one_img_path2 in img_path2:
                if isinstance(one_img_path1, str):
                    image1 = preprocess_val(Image.open(one_img_path1)).unsqueeze(0).to(device)
                    image2 = preprocess_val(Image.open(one_img_path2)).unsqueeze(0).to(device)
                elif isinstance(one_img_path1, Image.Image):
                    image1 = preprocess_val(one_img_path1).unsqueeze(0).to(device)
                    image2 = preprocess_val(one_img_path2).unsqueeze(0).to(device)
                else:
                    raise TypeError('The type of parameter img_path is illegal.')
                text = tokenizer([prompt]).to(device)

                # 生成对抗性样本
                adv_image1, adv_image2 = mim_attack(model, image1, image2, text, epsilon=0.08, alpha=0.08, num_iter=40)

                # 保存对抗性样本
                adv_image_path1 = os.path.splitext(one_img_path1)[0] + '_mim.jpg'
                save_image(adv_image1.cpu(), adv_image_path1)
                image_path1 = os.path.splitext(one_img_path1)[0] + '_origin.jpg'
                save_image(image1.cpu(), image_path1)
                adv_image_path2 = os.path.splitext(one_img_path2)[0] + '_mim.jpg'
                save_image(adv_image2.cpu(), adv_image_path2)
                image_path2 = os.path.splitext(one_img_path2)[0] + '_origin.jpg'
                save_image(image2.cpu(), image_path2)
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        output1 = model(adv_image1, text)
                        image_features, text_features = output1["image_features"], output1["text_features"]
                        logits_per_image = image_features @ text_features.T
                        hps_pscore1 = torch.diagonal(logits_per_image).cpu().numpy()
                        output1 = model(image1, text)
                        image_features, text_features = output1["image_features"], output1["text_features"]
                        logits_per_image = image_features @ text_features.T
                        hps_score1 = torch.diagonal(logits_per_image).cpu().numpy()
                        output2 = model(adv_image2, text)
                        image_features, text_features = output2["image_features"], output2["text_features"]
                        logits_per_image = image_features @ text_features.T
                        hps_pscore2 = torch.diagonal(logits_per_image).cpu().numpy()
                        output2 = model(image2, text)
                        image_features, text_features = output2["image_features"], output2["text_features"]
                        logits_per_image = image_features @ text_features.T
                        hps_score2 = torch.diagonal(logits_per_image).cpu().numpy()
                result.append((hps_score1[0], hps_score2[0], hps_pscore1[0], hps_pscore2[0]))
            return result
    else:
        raise TypeError('The type of parameter img_path is illegal.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path1', nargs='+', type=str, required=True, help='Path to the input image')
    parser.add_argument('--image-path2', nargs='+', type=str, required=True, help='Path to the input image')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt')
    parser.add_argument('--checkpoint', type=str, default=os.path.join(root_path, 'HPS_v2_compressed.pt'), help='Path to the model checkpoint')
    parser.add_argument('--epsilon', type=float, default=0.6, help='Perturbation magnitude')

    args = parser.parse_args()

    results = score(args.image_path1, args.image_path2, args.prompt, args.checkpoint, epsilon=args.epsilon)
    for hps_score1, hps_score2, hps_pscore1, hps_pscore2 in results:
        print(f'HPSv2 score1: {hps_score1}, HPSv2 score2: {hps_score2}')
        print(f'HPSv2 perturbed score1: {hps_pscore1}, HPSv2 perturbed score2: {hps_pscore2}')
