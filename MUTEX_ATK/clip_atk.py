import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

def text_process(text, processor):
    text_inputs = processor(text=[text], return_tensors="pt", padding=True)
    return text_inputs

def img_process(img, processor):
    inputs = processor(text=[""], images=img, return_tensors="pt", padding=True)
    image_input = inputs['pixel_values'].clone().detach().requires_grad_(True)
    return image_input
    
def text_encode(text_inputs, text_encoder, text_projection):
    text_features = text_encoder(**text_inputs).last_hidden_state
    text_features = text_projection(text_features.mean(dim=1))
    
    return text_features

def img_encode(img_input, image_encoder, image_projection):
    image_features = image_encoder(pixel_values=img_input).pooler_output
    image_features = image_projection(image_features)

    return image_features

def calc_sim(emb1, emb2):
    embedding_sim = torch.mean(torch.sum(emb1 * emb2, dim=1))  # cos. sim
    # embedding_sim = torch.mean(torch.norm(emb1 - emb2, p=2, dim=1))
    return embedding_sim

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
inverse_normalize = transforms.Normalize(mean=[-0.48145466 / 0.26862954, -0.4578275 / 0.26130258, -0.40821073 / 0.27577711], std=[1.0 / 0.26862954, 1.0 / 0.26130258, 1.0 / 0.27577711])

image_encoder = model.vision_model
text_encoder = model.text_model
image_projection = model.visual_projection
text_projection = model.text_projection

image_encoder.eval()

# load image and text
image = Image.open("/tmp2/johnson906/VLMbot-safety/MUTEX_ATK/LIBERO/libero/datasets/libero_10/task_spec/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo/img_271.png")
original_image = transforms.ToTensor()(image).unsqueeze(0).requires_grad_(True)

original_text = "stove is opend, and moka pot is on the stove"
target_text = "stove is closed, and moka pot is on the table"

ori_text_input = text_process(original_text, processor)
tgt_text_input = text_process(target_text, processor)
img_input = img_process(image, processor)

ori_text_emb = text_encode(ori_text_input, text_encoder, text_projection) 
ori_text_emb = ori_text_emb / ori_text_emb.norm(dim=1, keepdim=True)
tgt_text_emb = text_encode(tgt_text_input, text_encoder, text_projection)
tgt_text_emb = tgt_text_emb / tgt_text_emb.norm(dim=1, keepdim=True)


# for normalized image
scaling_tensor = torch.tensor((0.26862954, 0.26130258, 0.27577711))
scaling_tensor = scaling_tensor.reshape((3, 1, 1)).unsqueeze(0)
    
alpha = 1 / 255.0 / scaling_tensor
epsilon = 12 / 255.0 / scaling_tensor

# perturbed_image = fgsm_attack(img_input, epsilon, image_grad)
delta = torch.zeros_like(img_input, requires_grad=True)
for j in range(32):
    adv_image = img_input + delta   # image is normalized to (0.0, 1.0)
    adv_emb = img_encode(adv_image, image_encoder, image_projection)
    
    adv_emb = adv_emb / adv_emb.norm(dim=1, keepdim=True)
    
    # print(calc_sim(adv_emb, ori_text_emb), calc_sim(adv_emb, tgt_text_emb))
    loss = calc_sim(adv_emb, tgt_text_emb) / (calc_sim(adv_emb, ori_text_emb) + calc_sim(adv_emb, tgt_text_emb))
    
    loss.backward(retain_graph=True)
    
    grad = delta.grad.detach()
    
    delta_data = torch.clamp(delta + alpha * torch.sign(grad), min=-epsilon, max=epsilon)
    delta.data = delta_data
    delta.grad.zero_()

    tgt_sim = calc_sim(adv_emb, tgt_text_emb)
    ori_sim = calc_sim(adv_emb, ori_text_emb) 

    print(f"step:{j:3d}, loss={loss.item():.5f}, delta(tgt-ori)={(tgt_sim-ori_sim):.5f}, sim (ori)={ori_sim:.5f}, sim (tgt)={tgt_sim:.5f}, max delta={torch.max(torch.abs(delta_data)).item():.3f}, mean delta={torch.mean(torch.abs(delta_data)).item():.3f}")

# save the perturbed image
adv_image = img_input + delta
adv_image = torch.clamp(inverse_normalize(adv_image), 0.0, 1.0)
torchvision.utils.save_image(adv_image, 'perturbed.png')