import os
import glob
import torch
import imageio
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
import torchvision
import numpy as np
import torch.nn as nn
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

def calc_sim(emb1, emb2, sim_type="dot"):
    if sim_type == "dot":
        embedding_sim = torch.mean(torch.sum(emb1 * emb2, dim=1))
    elif sim_type == "l2":
        embedding_sim = torch.mean(torch.norm(emb1 - emb2, p=2, dim=1))
    else:
        print(sim_type)
        raise NotImplementedError

    return embedding_sim

def get_last_images(path_pattern):
    # Get all file paths matching the pattern
    all_files = glob.glob(path_pattern)

    # Group files by folder and get the last img_* file in each folder
    all_target_img_path = []
    folders = set(os.path.dirname(file) for file in all_files)
    for folder in folders:
        img_files = sorted(glob.glob(os.path.join(folder, "img_*.png")))
        if img_files:
            all_target_img_path.append(img_files[-1])

    return all_target_img_path

def get_video(path_pattern):
    # Get all file paths matching the pattern
    all_files = glob.glob(path_pattern)

    # Group files by folder and get all img_* files sorted from small to big
    all_img_files = []
    folders = set(os.path.dirname(file) for file in all_files)
    for folder in folders:
        img_files = sorted(glob.glob(os.path.join(folder, "img_*.png")))
        all_img_files.extend(img_files)

    return all_img_files

def adjust_length(target, length):
    target_length = target.size(0)
    
    if target_length == length:
        return target
    elif target_length > length:
        # Select frames at intervals to speed up the video
        indices = torch.linspace(0, target_length - 1, steps=length).long()
        return target[indices]
    else:
        indices = []
        for i in range(length):
            indices.append(i * target_length // length)
        
        return target[torch.tensor(indices)]
    
ATK_TYPE = "img2img" # "vid2vid" #  "img2img" "img2txt"
LOSS_TYPE = "dot" # "l2" # "l2" "dot" "ce"
ENV_DIR = "/tmp2/bungeee/VLMbot-safety/MUTEX_ATK/LIBERO/libero/datasets/libero_attack/task_spec/LIVING_ROOM_SCENE1"
ORIGINAL_TASK = "pick_up_the_alphabet_soup_and_put_it_in_the_basket"
TARGET_TASK = "pick_up_the_tomato_sauce_and_put_it_in_the_basket"

ORIGINAL_LAST_IMG_PATH = get_last_images(f"{ENV_DIR}_{ORIGINAL_TASK}_demo/")[0]
TARGET_LAST_IMG_PATH = get_last_images(f"{ENV_DIR}_{TARGET_TASK}_demo/")[0]
ALL_TARGET_LAST_IMG_LIST = get_last_images(f"{ENV_DIR}_*_demo/")

ORIGINAL_DEMO_PATH = get_video(f"{ENV_DIR}_{ORIGINAL_TASK}_demo/")
TARGET_DEMO_PATH = get_video(f"{ENV_DIR}_{TARGET_TASK}_demo/")
# ALL_TARGET_DEMO_LIST = get_video(f"{ENV_DIR}_*_demo/")

ORIGINAL_TASK_TEXT = "stove is opend, and moka pot is on the stove"
TARGET_TASK_TEXT = "stove is closed, and moka pot is on the table"
ALL_TEXT_LIST = ["hello", "bye", "qq", "nihao"]

device="cuda"

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
inverse_normalize = transforms.Normalize(mean=[-0.48145466 / 0.26862954, -0.4578275 / 0.26130258, -0.40821073 / 0.27577711], std=[1.0 / 0.26862954, 1.0 / 0.26130258, 1.0 / 0.27577711])

image_encoder = model.vision_model
text_encoder = model.text_model
image_projection = model.visual_projection
text_projection = model.text_projection

image_encoder.eval()

# load image and text
if "2vid" in ATK_TYPE:
    vid = [Image.open(path) for path in ORIGINAL_DEMO_PATH]
    ori_vid_input = [img_process(image, processor).to(device) for image in vid]
    img_input = torch.cat(ori_vid_input, dim=0)

else:
    image = Image.open(ORIGINAL_LAST_IMG_PATH)
    img_input = img_process(image, processor).to(device)

if ATK_TYPE == "img2txt":
    ori_text_input = text_process(ORIGINAL_TASK_TEXT, processor).to(device)
    ori_emb = text_encode(ori_text_input, text_encoder, text_projection) 

    tgt_text_input = text_process(TARGET_TASK_TEXT, processor).to(device)
    tgt_emb = text_encode(tgt_text_input, text_encoder, text_projection)

elif ATK_TYPE == "img2img":
    original_image = Image.open(ORIGINAL_LAST_IMG_PATH)
    ori_image_input = img_process(original_image, processor).to(device)
    ori_emb = img_encode(ori_image_input, image_encoder, image_projection)

    target_image = Image.open(TARGET_LAST_IMG_PATH)
    tgt_image_input = img_process(target_image, processor).to(device)
    tgt_emb = img_encode(tgt_image_input, image_encoder, image_projection)

elif ATK_TYPE == "vid2vid":
    original_vid = [Image.open(path) for path in ORIGINAL_DEMO_PATH]
    ori_vid_input = [img_process(image, processor).to(device) for image in original_vid]
    ori_vid_input = torch.cat(ori_vid_input, dim=0)
    ori_emb = img_encode(ori_vid_input, image_encoder, image_projection)

    target_vid = [Image.open(path) for path in TARGET_DEMO_PATH]
    tgt_vid_input = [img_process(image, processor).to(device) for image in target_vid]
    tgt_vid_input = torch.cat(tgt_vid_input, dim=0)
    tgt_emb = img_encode(tgt_vid_input, image_encoder, image_projection)

    tgt_emb = adjust_length(tgt_emb, ori_emb.size(0))

else:
    raise NotImplementedError

ori_emb = ori_emb / ori_emb.norm(dim=1, keepdim=True)
tgt_emb = tgt_emb / tgt_emb.norm(dim=1, keepdim=True)

if LOSS_TYPE == "ce":
    if ATK_TYPE == "img2txt":
        all_txt_input = [text_process(txt, processor) for txt in ALL_TEXT_LIST]
        all_embs = [text_encode(text_input, text_encoder, text_projection) for text_input in all_txt_input]

    elif ATK_TYPE == "img2img":
        all_images = [Image.open(path) for path in ALL_TARGET_LAST_IMG_LIST] 
        all_image_input = [img_process(image, processor).to(device) for image in all_images]
        all_embs = [img_encode(input, image_encoder, image_projection) for input in all_image_input]

    all_embs = [emb / emb.norm(dim=1, keepdim=True) for emb in all_embs]

# for normalized image
scaling_tensor = torch.tensor((0.26862954, 0.26130258, 0.27577711)).to(device)
scaling_tensor = scaling_tensor.reshape((3, 1, 1)).unsqueeze(0)
    
alpha = 0.1 / 255.0 / scaling_tensor
epsilon = 12 / 255.0 / scaling_tensor
steps = 256

# perturbed_image = fgsm_attack(img_input, epsilon, image_grad)
delta = torch.zeros_like(img_input, requires_grad=True).to(device)
for j in range(steps):
    adv_image = img_input + delta 
    adv_emb = img_encode(adv_image, image_encoder, image_projection)
    
    adv_emb = adv_emb / adv_emb.norm(dim=1, keepdim=True)
    
    if LOSS_TYPE == "ce":
        target_index =  ALL_TARGET_LAST_IMG_LIST.index(TARGET_LAST_IMG_PATH) # in all_target_img_path find TARGET_LAST_IMG_PATH
        target = torch.tensor([target_index])
        logits = torch.stack([calc_sim(adv_emb, task_emb) for task_emb in all_embs])
        loss_fn = nn.CrossEntropyLoss()
        loss = -loss_fn(logits.unsqueeze(0), target)

    else:
        loss = calc_sim(adv_emb, tgt_emb, sim_type=LOSS_TYPE)
    
    loss.backward(retain_graph=True)
    
    grad = delta.grad.detach()

    delta_data = torch.clamp(delta + alpha * torch.sign(grad), min=-epsilon, max=epsilon)
    delta.data = delta_data
    delta.grad.zero_()

    if LOSS_TYPE == "ce":
        tgt_sim = calc_sim(adv_emb, tgt_emb)
        ori_sim = calc_sim(adv_emb, ori_emb)
    else:
        tgt_sim = calc_sim(adv_emb, tgt_emb, sim_type=LOSS_TYPE)
        ori_sim = calc_sim(adv_emb, ori_emb, sim_type=LOSS_TYPE) 

    print(f"step={j:3d}, loss={loss.item():.5f}, delta(tgt-ori)={(tgt_sim-ori_sim):.5f}, sim (ori)={ori_sim:.5f}, sim (tgt)={tgt_sim:.5f}, max delta={torch.max(torch.abs(delta_data)).item():.3f}, mean delta={torch.mean(torch.abs(delta_data)).item():.3f}")

    del adv_emb, grad, delta_data, tgt_sim, ori_sim, loss
    torch.cuda.empty_cache()

# save the perturbed image
if "2vid" in ATK_TYPE:
    adv_vid = img_input + delta
    adv_vid = torch.clamp(inverse_normalize(adv_vid), 0.0, 1.0)

    # Ensure the tensor is on the CPU and convert it to numpy
    adv_vid = adv_vid.cpu().detach().numpy()

    # Convert the tensor to a format suitable for imageio (N, H, W, C)
    adv_vid = np.transpose(adv_vid, (0, 2, 3, 1))

    # Normalize the tensor to the range [0, 255] and convert to uint8
    tensor = (adv_vid * 255).astype(np.uint8)

    # Save the frames as a GIF
    with imageio.get_writer("perturbed.gif", mode='I', fps=30) as writer:
        for frame in tensor:
            writer.append_data(frame)


else:
    adv_image = img_input + delta
    adv_image = torch.clamp(inverse_normalize(adv_image), 0.0, 1.0)
    torchvision.utils.save_image(adv_image, 'perturbed.png')

    ori_image = img_input
    ori_image = torch.clamp(inverse_normalize(ori_image), 0.0, 1.0)
    torchvision.utils.save_image(ori_image, 'original.png')

    if ATK_TYPE == "img2img":
        tgt_image = torch.clamp(inverse_normalize(tgt_image_input), 0.0, 1.0)
        torchvision.utils.save_image(tgt_image, 'target.png')
