import numpy as np
import torch
import torchvision
from torchvision import transforms
from mutex.models.task_specs import CLIPVisionSliced
from transformers import CLIPFeatureExtractor, AutoTokenizer, CLIPTextModelWithProjection, CLIPProcessor, CLIPModel
from hydra.utils import to_absolute_path
from tqdm import tqdm
from copy import deepcopy
from PIL import Image
import os
import mutex
from mutex.rephrased_gls import rephrase_sentences
import glob

def generate_perturbed_images_clip(cfg, image, target_image=None):
  # Load the CLIP model and processor
  model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cuda()
  processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
  inverse_normalize = transforms.Normalize(mean=[-0.48145466 / 0.26862954, -0.4578275 / 0.26130258, -0.40821073 / 0.27577711], std=[1.0 / 0.26862954, 1.0 / 0.26130258, 1.0 / 0.27577711])

  image_encoder = model.vision_model
  text_encoder = model.text_model
  image_projection = model.visual_projection
  text_projection = model.text_projection

  image_encoder.eval()

  # load image and text
  # original_image = transforms.ToTensor()(image).unsqueeze(0).requires_grad_(True)
  if cfg.attack_method == 'gl2img':
    ori_text_input = clip_preprocess('text', cfg.base_lang_assets["des"], processor).cuda()
    tgt_text_input = clip_preprocess('text', cfg.target_lang_assets["des"], processor).cuda()

    ori_emb = clip_encode('text', ori_text_input, text_encoder, text_projection) 
    tgt_emb = clip_encode('text', tgt_text_input, text_encoder, text_projection)
  elif cfg.attack_method == 'img2img':
    original_image = deepcopy(image)
    ori_image_input = clip_preprocess('img', original_image, processor).cuda()
    ori_emb = clip_encode('img', ori_image_input, image_encoder, image_projection)

    tgt_image_input = clip_preprocess('img', target_image, processor).cuda()
    tgt_emb = clip_encode('img', tgt_image_input, image_encoder, image_projection)

  img_input = clip_preprocess('img', image, processor).cuda()

  ori_emb = ori_emb / ori_emb.norm(dim=1, keepdim=True)
  tgt_emb = tgt_emb / tgt_emb.norm(dim=1, keepdim=True)

  # for normalized image
  scaling_tensor = torch.tensor((0.26862954, 0.26130258, 0.27577711)).cuda()
  scaling_tensor = scaling_tensor.reshape((3, 1, 1)).unsqueeze(0)
      
  alpha = cfg.alpha / 255.0 / scaling_tensor
  epsilon = cfg.epsilon / 255.0 / scaling_tensor

  # perturbed_image = fgsm_attack(img_input, epsilon, image_grad)
  delta = torch.zeros_like(img_input, requires_grad=True).cuda()
  for j in range(cfg.perturb_steps):
      adv_image = img_input + delta   # image is normalized to (0.0, 1.0)
      adv_emb = clip_encode('img', adv_image, image_encoder, image_projection)
      
      adv_emb = adv_emb / adv_emb.norm(dim=1, keepdim=True)
      
      # print(calc_sim(adv_emb, ori_text_emb), calc_sim(adv_emb, tgt_text_emb))
      loss = calc_sim(adv_emb, tgt_emb, cfg.loss_type) # / (calc_sim(adv_emb, ori_text_emb) + calc_sim(adv_emb, tgt_text_emb))
      
      loss.backward(retain_graph=True)
      
      grad = delta.grad.detach()

      delta_data = torch.clamp(delta + alpha * torch.sign(grad), min=-epsilon, max=epsilon)
      delta.data = delta_data
      delta.grad.zero_()

      tgt_sim = calc_sim(adv_emb, tgt_emb, cfg.loss_type)
      ori_sim = calc_sim(adv_emb, ori_emb, cfg.loss_type)

      print(f"step:{j:3d}, loss={loss.item():.5f}, delta(tgt-ori)={(tgt_sim-ori_sim):.5f}, sim (ori)={ori_sim:.5f}, sim (tgt)={tgt_sim:.5f}, max delta={torch.max(torch.abs(delta_data)).item():.3f}, mean delta={torch.mean(torch.abs(delta_data)).item():.3f}")

  # save the perturbed image
  adv_image = img_input + delta
  adv_image = torch.clamp(inverse_normalize(adv_image), 0.0, 1.0)
  # torchvision.utils.save_image(adv_image, 'perturbed.png')
  return transform_tensor_2_pil_image(adv_image)

def generate_misaligned_sentence(cfg, algo, target_type):
  target_emb, _ = get_lang_task_embs(cfg, cfg.target_lang_assets[target_type], target_type, 'eval')
  # print(target_emb.shape)
  target_proj = [project_embs(algo, target_emb[i],  [target_type], 'cuda') for i in range(target_emb.shape[0])]
  origin_emb, _ = get_lang_task_embs(cfg, cfg.base_lang_assets[target_type], target_type, 'eval')
  origin_proj = [project_embs(algo, origin_emb[i],  [target_type], 'cuda') for i in range(origin_emb.shape[0])]
  effective = []
  for i, s in enumerate(rephrase_sentences[cfg.base_task_name]):
      setence_emb, _ = get_lang_task_embs(cfg, [s], target_type, 'eval')
      proj = project_embs(algo, setence_emb[0],  [target_type], 'cuda')
      target_dist = [torch.norm(proj - target_proj[i]).item() for i in range(len(target_proj))]
      origin_dist = [torch.norm(proj - origin_proj[i]).item() for i in range(len(origin_proj))]
      target_dist = sum(target_dist) / len(target_dist)
      origin_dist = sum(origin_dist) / len(origin_dist)
      effective.append((origin_dist - target_dist, i))
  effective.sort()
  effective.reverse()
  return [rephrase_sentences[cfg.base_task_name][i]  for _, i in effective[:11]]

def generate_perturbed_images(cfg, algo, image, target, target_type, epsilon, iters):
  target_embs = None
  # print('attack', target_type)
  if target_type == 'img':
     target_embs = project_embs(algo, encode_image(target), ['img'], 'cuda')
  else:
     # print(target)
     target_embs, _ = get_lang_task_embs(cfg, target, target_type, 'eval')
     # print(target_embs.shape, target_embs)
     target_embs = project_embs(algo, target_embs[0],  [target_type], 'cuda')
     # print(target_embs.shape, target_embs)
  perturbed = ifgsm_attack(algo, image, target_embs, epsilon, iters)
  return perturbed

def clip_preprocess(modality, input, processor):
  if modality == 'text': # preprocess text
    text_inputs = processor(text=[input], return_tensors="pt", padding=True)
    return text_inputs
  else: # preprocess image
    inputs = processor(text=[""], images=input, return_tensors="pt", padding=True)
    image_input = inputs['pixel_values'].clone().detach().requires_grad_(True)
    return image_input
def clip_encode(modality, inputs, encoder, projector):
  if modality == 'text': # encode text
    text_features = encoder(**inputs).last_hidden_state
    text_features = projector(text_features.mean(dim=1))
    return text_features
  else: # encode image
    image_features = encoder(pixel_values=inputs).pooler_output
    image_features = projector(image_features)
    return image_features
  
def calc_sim(emb1, emb2, metric='dot'):
  if metric == 'dot':
    embedding_sim = torch.mean(torch.sum(emb1 * emb2, dim=1))  # cos. sim
  elif metric == 'l2':
    embedding_sim = torch.mean(torch.norm(emb1 - emb2, p=2, dim=1))
  return embedding_sim

def get_all_img_files(cfg, task_list, task_demo_path_list):
  imgs = []
  for _, task_demo_path in zip(task_list, task_demo_path_list):
    vid_dir = os.path.join(cfg.folder, task_demo_path[:-5].split('/')[0], 'task_spec', task_demo_path[:-5].split('/')[1])
    for task_spec_id in range(1): #set as 1 default =>  cfg.n_ts_per_task): # Iterate over the different task specification examples, save everything
      visual_specs_path = os.path.join(vid_dir, f'vid_{task_spec_id:03d}')
      goal_img = Image.open(sorted(glob.glob(f'{visual_specs_path}/*.png'))[-1])
    imgs.append(goal_img)
  return imgs
def get_ce_loss(cfg, target_type, content):
  if target_type == 'gl':
    content = [k[0] for k in content]
    modal = 'text'
  elif target_type == 'img':
    content = get_all_img_files(cfg, content[0], content[1])
    modal = 'img'
  

def transform_tensor_2_pil_image(img_tensor):
  img_arr = img_tensor[0].permute(1, 2, 0).cpu().detach().numpy()
  img_arr = np.round(img_arr * 255).astype(np.uint8)
  return Image.fromarray(img_arr)

def encode_image(image):
  visual_preprocessor = CLIPFeatureExtractor.from_pretrained('openai/clip-vit-large-patch14')
  visual_emb_model = CLIPVisionSliced.from_pretrained('openai/clip-vit-large-patch14', cache_dir=to_absolute_path("./clip"))
  visual_emb_model.eval()
  visual_emb_model.create_precomputable_models_grad(layer_ind=23)
  visual_emb_model = visual_emb_model.to('cuda')
  visual_task_spec = visual_preprocessor([image], return_tensors='pt', padding=True)['pixel_values']
  visual_task_spec = visual_task_spec.to('cuda')
  input_batch = visual_task_spec
  # with torch.no_grad():
  output_batch = visual_emb_model.pre_compute_feats_grad(input_batch)[0] ## [bs, 50, 512] 0th index has hidden embed
  img_task_emb = output_batch
  return img_task_emb

def ifgsm_attack(algo, image, target_emb, eps, iters):
  print("[IFGSM] Start Perturbating")
  x = deepcopy(image)
  transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
  adv_ex = transform(image)
  uimage = adv_ex.clone()
  limage = adv_ex.clone()
  upper_bound = torch.min(torch.ones_like(uimage), uimage + eps)
  lower_bound = torch.max(torch.zeros_like(limage), limage - eps) 
  for i in range(iters):
    adv_ex, loss = fgsm_attack(algo, x, adv_ex.to('cuda'), target_emb, eps)
    adv_ex = torch.max(torch.min(adv_ex, upper_bound), lower_bound)
    adv_ex = (adv_ex * 255).clamp(0, 255) # 0-255 scale
    adv_ex = adv_ex.detach().cpu().data.numpy().round() # round to remove decimal part
    adv_ex = adv_ex.transpose((0, 2, 3, 1))
    x = Image.fromarray(adv_ex.astype(np.uint8)[0])
    print(f"[ITER {i+1}] Loss: {loss}")
  return x
    

def fgsm_attack(algo, image, img_tensor, target_emb, eps):
  img_adv = img_tensor # .clone() # initialize x_adv as original benign image x # .detach()
  img_adv.requires_grad_(True) # need to obtain gradient of x_adv, thus set required grad
  # print(img_adv)
  emb = project_embs(algo, encode_image(img_adv), ['img'], 'cuda')
  # print(img_adv)
  # print(img_adv.is_leaf, torch.is_grad_enabled())
  # print(emb.shape, target_emb.shape)
  loss = torch.norm(emb - target_emb) # calculate loss
  # print(loss, img_adv, img_adv.grad)
  # img_adv.retain_grad()
  loss.backward() # calculate gradient
  # fgsm: use gradient ascent on x_adv to maximize loss
  # print(img_adv)
  grad = img_adv.grad.detach()
  img_adv = img_adv - eps * grad.sign()
  return img_adv, loss

def project_embs(algo, emb, eval_spec_modalities, device):
    new_task_embs = []
    data_dict = {}
    if 'img' in eval_spec_modalities:
        data_dict['img_spec'] = torch.stack([emb], dim=0).to(device)
        img_spec_mask = None
        data_dict['img_spec_mask'] = img_spec_mask

    if 'inst' in eval_spec_modalities:
        data_dict['inst_emb'] = emb.to(device)

        input_mask = torch.ones(data_dict['inst_emb'].shape[:-1])
        input_mask = input_mask.to(device)
        data_dict['inst_emb_mask'] = input_mask

    if 'gl' in eval_spec_modalities:
        gl_emb = emb
        ## adding time dimension
        data_dict['gl_emb'] = gl_emb.unsqueeze(dim=1).to(device)

    # print(data_dict)
    emb, *temp = algo.policy.get_task_embs(data_dict, modalities=eval_spec_modalities)
    new_task_embs.append(emb)

    new_task_embs = torch.stack(new_task_embs, dim=0)  # [num_tasks, num_eval_ts, T, E]
    
    return new_task_embs

def get_lang_task_embs(cfg, descriptions, spec_type, mode='train'):
    # read stop words from file in mutex.__path__[0], english_stopwords.txt
    stop_words = []
    with open(os.path.join(mutex.__path__[0], 'english_stopwords.txt'), 'r') as fi:
        for line in fi:
            stop_words.append(line.strip())

    task_id_range = range(cfg.n_ts_per_task)
    if mode == 'train' or mode == 'eval':
        train_max_ts = int(0.8*cfg.n_ts_per_task)
        task_id_range = range(train_max_ts) if mode == 'train' else range(train_max_ts, cfg.n_ts_per_task, 1)
    else:
        raise NotImplementedError
    benchmark_name = cfg.benchmark_name.lower()
    tokenizer_str = cfg.lang_tokenizer
    tokenizer_str = tokenizer_str.replace('/', '_')
    saved_emb_path = os.path.join(cfg.folder, benchmark_name, 'task_spec', f'{spec_type}_{tokenizer_str}_ts_mode_{mode}_emb.pt')

    if cfg.recalculate_ts_embs or (not os.path.exists(saved_emb_path)):
        print(f"[WARNING]: Calculating {spec_type} embeddings for ts mode {mode}")
        if cfg.lang_embedding_format == "clip":
            tz = AutoTokenizer.from_pretrained(cfg.lang_tokenizer)
            model = CLIPTextModelWithProjection.from_pretrained(cfg.lang_tokenizer, cache_dir=to_absolute_path("./clip")).eval()
        else:
            raise NotImplementedError
        model = model.to(cfg.device)
        stopword_tokens = tz(stop_words, add_special_tokens=True)['input_ids']
        # make it a single list of tokens
        stopword_tokens = [item for sublist in stopword_tokens for item in sublist]
        # remove duplicates
        stopword_tokens = list(set(stopword_tokens))
        if spec_type == 'inst':
            task_embs = torch.empty((len(descriptions), len(task_id_range), cfg.data.max_instructs, 768))
            tokens = {
                        'input_ids': torch.empty((len(descriptions), len(task_id_range), cfg.data.max_instructs, cfg.data.max_word_len)), \
                        'attention_mask': torch.empty((len(descriptions), len(task_id_range), cfg.data.max_instructs, cfg.data.max_word_len)),
            }
            # iterate over tasks
            for task_ind, descriptions_task in enumerate(descriptions):
                # iterate over different descriptions for a task.
                for store_ind, ind in enumerate(task_id_range):
                    description = descriptions_task[ind]
                    token = tz(
                        text=description,                   # the sentence to be encoded
                        add_special_tokens=True,            # Add [CLS] and [SEP]
                        max_length=cfg.data.max_word_len,   # maximum length of a sentence
                        padding="max_length",
                        return_attention_mask=True,         # Generate the attention mask
                        return_tensors='pt',                # ask the function to return PyTorch tensors
                    )
                    # move token dictionary to cfg.device
                    for k, v in token.items():
                        token[k] = v.to(cfg.device)
                    if token['attention_mask'].size(0) > cfg.data.max_instructs:
                        print("[ERROR] Number of instructions are more than maximum allowed for task:", description)
                        raise Exception
                    elif token['attention_mask'].size(0) < cfg.data.max_instructs:
                        pad_len = cfg.data.max_instructs - token['attention_mask'].size(0)
                        for k, v in token.items():
                            token[k] = torch.cat((v,torch.zeros((pad_len, v.size(-1))).to(cfg.device)), dim=0).long()
                    if cfg.lang_embedding_format == "clip":
                        task_emb = model(**token)['text_embeds'].detach()
                    elif cfg.lang_embedding_format == "t5":
                        task_emb = model(**token)['last_hidden_state'].detach()
                        # mean over embeddings along the dimension=1
                        task_emb = torch.mean(task_emb, dim=-2)
                    tokens['input_ids'][task_ind, store_ind] = token['input_ids']
                    tokens['attention_mask'][task_ind, store_ind] = token['attention_mask']
                    task_embs[task_ind, store_ind] = task_emb
        elif spec_type == 'gl':
            task_embs = torch.empty((len(descriptions), len(task_id_range), 768))
            tokens = {
                        'input_ids': torch.empty((len(descriptions), len(task_id_range), cfg.data.max_word_len)), \
                        'attention_mask': torch.empty((len(descriptions), len(task_id_range), cfg.data.max_word_len)),
            }
            # iterate over tasks
            for task_ind, descriptions_task in enumerate(descriptions):
                descriptions_task = [descriptions_task[ind] for ind in task_id_range]
                token = tz(
                    text=descriptions_task,                 # the sentence to be encoded
                    add_special_tokens=True,                # Add [CLS] and [SEP]
                    max_length = cfg.data.max_word_len,     # maximum length of a sentence
                    padding="max_length",
                    return_attention_mask = True,           # Generate the attention mask
                    return_tensors = 'pt',                  # ask the function to return PyTorch tensors
                )
                # move token dictionary to cfg.device
                for k, v in token.items():
                    token[k] = v.to(cfg.device)
                if cfg.lang_embedding_format == "clip":
                    task_emb = model(**token)['text_embeds'].detach()
                elif cfg.lang_embedding_format == "t5":
                    task_emb = model(**token)['last_hidden_state'].detach()
                    # mean over embeddings along the dimension=-2
                    task_emb = torch.mean(task_emb, dim=-2)
                tokens['input_ids'][task_ind] = token['input_ids']
                tokens['attention_mask'][task_ind] = token['attention_mask']
                task_embs[task_ind] = task_emb
        else:
            raise NotImplementedError
        tokens["mask_token_id"] = torch.zeros((task_embs.shape[0],)).long() ## converts it to int64, there's not such thing as masking in clip
        # repeat stopword tokens for each task
        tokens['stopword_tokens'] = torch.tensor(stopword_tokens).long().unsqueeze(0).repeat(task_embs.shape[0], 1)
        tokens = {k: v.cpu().detach() for k,v in tokens.items()}
        task_embs = task_embs.cpu().detach()
    return task_embs, tokens