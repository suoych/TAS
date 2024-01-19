import argparse
import os
import copy

import numpy as np
import json
import torch
from PIL import Image, ImageDraw, ImageFont

# segment anything
from segment_anything import build_sam, SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
import cv2
import matplotlib.pyplot as plt

import time
from tqdm import tqdm
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn.functional as F
#import wandb
from loguru import logger
#from utils.dataset import tokenize
from utils.misc import (AverageMeter, ProgressMeter, concat_all_gather,
                        trainMetricGPU)
import argparse
import warnings
import torch.nn.parallel
import torch.utils.data
from loguru import logger

import utils.config as config
from utils.dataset import RefTestDataset
from utils.misc import setup_logger
import spacy
import pdb
#import clip
import torchvision.utils as vutils

import os
import glob
import torch
import cv2
import argparse
import time
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from clip import build_model
warnings.filterwarnings("ignore")
cv2.setNumThreads(0)
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
from simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
from typing import Any, Union, List
from pkg_resources import packaging
from lavis.models import load_model_and_preprocess

import re

import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from heuristics import Heuristics
from entity_extraction import Entity, expand_chunks

def get_parser():
    parser = argparse.ArgumentParser(
        description='Pytorch Referring Expression Segmentation')
    parser.add_argument('--config',
                        default='path to xxx.yaml',
                        type=str,
                        help='config file')
    parser.add_argument('--opts',
                        default=None,
                        nargs=argparse.REMAINDER,
                        help='override some settings in the config.')
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg

def crop_image(image, mask):
    x, y, w, h = mask["bbox"]
    masked = image * np.expand_dims(mask["segmentation"], -1)
    crop = masked[y : y + h, x : x + w]
    if h > w:
        top, bottom, left, right = 0, 0, (h - w) // 2, (h - w) // 2
    else:
        top, bottom, left, right = (w - h) // 2, (w - h) // 2, 0, 0
    # padding
    crop = cv2.copyMakeBorder(
        crop,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),)
    
    #padding_len = int(crop.shape[0]/8)
    #crop = cv2.copyMakeBorder(crop,padding_len,padding_len,padding_len,padding_len,cv2.BORDER_CONSTANT,value=(0,0,0))
    
    crop = Image.fromarray(crop)
    return crop

def crop_image_with_background_blur(image, mask):
    """
    We mask the image while blurring the background, currently we don't change the image size.
    """
    x, y, w, h = mask["bbox"]
    masked_img = cv2.bitwise_and(image, image, mask=np.uint8(mask["segmentation"]))
    if h > w:
        top, bottom, left, right = 0, 0, (h - w) // 2, (h - w) // 2
    else:
        top, bottom, left, right = (w - h) // 2, (w - h) // 2, 0, 0
    
    # Invert mask and blur background
    blurred_img = cv2.GaussianBlur(image, (75, 75), 0, 0)
    blurred_background = cv2.bitwise_and(blurred_img, blurred_img, mask=np.uint8(1-mask["segmentation"]))

    # Combine masked image and blurred background
    result = cv2.add(masked_img, blurred_background)


    crop = result[y : y + h, x : x + w]
    #crop = blurred_background[y : y + h, x : x + w]
    crop = cv2.copyMakeBorder(
        crop,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    crop = Image.fromarray(crop)
    return crop

def mask_blur_no_crop(image, mask):
    """
    We mask the image while blurring the background, currently we don't change the image size.
    """
    x, y, w, h = mask["bbox"]
    masked_img = cv2.bitwise_and(image, image, mask=np.uint8(mask["segmentation"]))
    if h > w:
        top, bottom, left, right = 0, 0, (h - w) // 2, (h - w) // 2
    else:
        top, bottom, left, right = (w - h) // 2, (w - h) // 2, 0, 0
    
    # Invert mask and blur background
    blurred_img = cv2.GaussianBlur(image, (75, 75), 0, 0)
    blurred_background = cv2.bitwise_and(blurred_img, blurred_img, mask=np.uint8(1-mask["segmentation"]))

    # Combine masked image and blurred background
    result = cv2.add(masked_img, blurred_background)

    crop = Image.fromarray(result)
    return crop


def extract_target_np(sentence,nlp):
    # Parse the sentence with SpaCy
    doc = nlp(sentence)
    
    # Find the root word of the sentence
    root = next((token for token in doc if token.head == token), None)
    if root is None:
        return sentence  # use the whole sentence if no root word is found
    
    # If the root word is a verb, use its children noun as the root word
    if root.pos_ == 'VERB':
        for child in root.children:
            if child.pos_ == 'NOUN':
                root = child
                break
    
    # Find the noun phrase containing the root word
    target_np = next((np for np in doc.noun_chunks if root in np), None)
    if target_np is None:
        # If no noun phrase containing the root word is found, try to use the children of the root word instead
        children_nouns = [child for child in root.children if child.pos_ == 'NOUN']
        if len(children_nouns) == 1:
            target_np = next((np for np in doc.noun_chunks if children_nouns[0] in np), None)
    
    if target_np is None:
        return sentence  # use the whole sentence if still no target noun phrase is found
    
    return target_np.text

def extract_noun_phrase(text, nlp, need_index=False):
    # text = text.lower()

    doc = nlp(text)

    chunks = {}
    chunks_index = {}
    for chunk in doc.noun_chunks:
        for i in range(chunk.start, chunk.end):
            chunks[i] = chunk
            chunks_index[i] = (chunk.start, chunk.end)

    for token in doc:
        if token.head.i == token.i:
            head = token.head

    if head.i not in chunks:
        children = list(head.children)
        if children and children[0].i in chunks:
            head = children[0]
        else:
            if need_index:
                return text, [], text
            else:
                return text

    head_noun = head.text
    head_index = chunks_index[head.i]
    head_index = [i for i in range(head_index[0], head_index[1])]

    sentence_index = [i for i in range(len(doc))]
    not_phrase_index = []
    for i in sentence_index:
        not_phrase_index.append(i) if i not in head_index else None

    head = chunks[head.i]
    if need_index:
        return head.text, not_phrase_index, head_noun
    else:
        return head.text


def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(n_px):
    # preprocess function used in clip
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)
    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize
    context_length : int
        The context length to use; all CLIP models use 77 as the context length
    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length
    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result

def calculate_overlap(masks, min_area=2000,min_iou=0.4):
    """
    There are redundant masks generated by sam, we filter them by keeping the larger masks
    masks: dict type, raw output of sam,
    min_area: int type, keep masks whose area larger than this,
    min_iou: dict type, if the overlap of a pair of masks greater than this, we keep the larger mask
    """
    overlap_areas = np.zeros((len(masks), len(masks)))
    area_large = []
    small_indices = []
    for i in range(len(masks)):
        area_large.append(np.sum(masks[i]["segmentation"]))
        if np.sum(masks[i]["segmentation"]) < min_area:
            small_indices.append(i)

    
    for i in range(len(masks)):
        for j in range(i+1, len(masks)):
            if i not in small_indices and j not in small_indices:
                intersection = np.logical_and(masks[i]["segmentation"], masks[j]["segmentation"])
                #union = np.logical_or(masks[i]["segmentation"], masks[j]["segmentation"]) 
                # we do not use union here, we use the least area large instead
                union = min(np.sum(masks[i]["segmentation"]),np.sum(masks[j]["segmentation"]))
                overlap_area = np.sum(intersection) / union
                overlap_areas[i, j] = overlap_area
                overlap_areas[j, i] = overlap_area

    # Find indices of masks to keep
    keep_indices = set(range(len(masks)))
    for s in small_indices:
        keep_indices.discard(s)
    for i in range(len(masks)):
        for j in range(i+1, len(masks)):
            if i not in small_indices and j not in small_indices:
                if overlap_areas[i, j] > min_iou:
                    if np.sum(masks[i]["segmentation"]) >= np.sum(masks[j]["segmentation"]):
                        keep_indices.discard(j)
                    else:
                        keep_indices.discard(i)

    # Keep only the masks at the selected indices
    final_masks = [masks[i] for i in sorted(list(keep_indices))]
    areas = [area_large[i] for i in sorted(list(keep_indices))]
    return final_masks,sorted(list(keep_indices)), areas


def are_near_synonyms(phrase1, phrase2):
    # Tokenize phrases
    tokens1 = word_tokenize(phrase1)
    tokens2 = word_tokenize(phrase2)

    # Extract nouns from each phrase
    nouns1 = [token for token, pos in nltk.pos_tag(tokens1) if pos.startswith('N')]
    nouns2 = [token for token, pos in nltk.pos_tag(tokens2) if pos.startswith('N')]

    # Check if nouns are present and compare their similarity
    if nouns1 and nouns2:
        synsets1 = wordnet.synsets(nouns1[0])
        synsets2 = wordnet.synsets(nouns2[0])
        """
        for synset1 in synsets1:
            for synset2 in synsets2:
                similarity = synset1.path_similarity(synset2)
                
                #pdb.set_trace()
                if similarity is not None and similarity > 0.9:
                    return True
        """     
        matching_words = 0

        for synset1 in synsets1:
            for synset2 in synsets2:
                similarity = synset1.path_similarity(synset2)
                if similarity is not None and similarity > 0.9:
                    matching_words += 1
                    if matching_words >= 3:
                        return True
        

    return False

def get_head(doc):
        """Return the token that is the head of the dependency parse."""
        for token in doc:
            if token.head.i == token.i:
                return token
        return None

def get_chunks(doc):
    """Return a dictionary mapping sentence indices to their noun chunk."""
    chunks = {}
    for chunk in doc.noun_chunks:
        for idx in range(chunk.start, chunk.end):
            chunks[idx] = chunk
    return chunks

def parse_sentence_entities(nlp,sent):
    counts = {}
    counts["n_0th_noun"] = 0
    doc = nlp(sent)
    head = get_head(doc)
    chunks = get_chunks(doc)
    if expand_chunks:
        chunks = expand_chunks(doc, chunks)
    entity = Entity.extract(head, chunks, heuristics)

    # If no head noun is found, take the first one.
    if entity is None and len(list(doc.noun_chunks)) > 0:
        head = list(doc.noun_chunks)[0]
        entity = Entity.extract(head.root.head, chunks, heuristics)
        counts["n_0th_noun"] += 1
    #pdb.set_trace()
    return entity, counts, chunks



if __name__ == "__main__":
    
    # cfg
    config_file = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"  # change the path of the model config file
    grounded_checkpoint = "groundingdino_swint_ogc.pth"  # change the path of the model
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    text_prompt = "bear"
    output_dir = "outputs"
    box_threshold = 0.3
    text_threshold = 0.25
    device = "cuda"
    spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_trf")
    spacy.tokens.Token.set_extension('wordnet', default=None, force=True)
    
   
    # initialize SAM
    sam = build_sam(checkpoint=sam_checkpoint)
    #sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam,points_per_side=8,
                     pred_iou_thresh=0.7,
                     stability_score_thresh=0.7,
                     crop_n_layers=0,
                     crop_n_points_downscale_factor=1,
                     min_mask_region_area=800,)


    clip_state_dict = torch.jit.load("./ViT-B-32.pt",map_location="cpu").eval()
    #clip_state_dict = torch.jit.load("./RN50.pt",map_location="cpu").eval()
    clip_model = build_model(clip_state_dict.state_dict(),77).float().cuda()
    clip_model.eval()
    preprocess = _transform(clip_model.visual.input_resolution)

    blip_model, vis_processors, _ = load_model_and_preprocess(name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True, device=device)
    
    ref_args = get_parser()
    ref_args.output_dir = os.path.join(ref_args.output_folder, ref_args.exp_name)
    if ref_args.visualize:
        ref_args.vis_dir = os.path.join(ref_args.output_dir, "vis")
        os.makedirs(ref_args.vis_dir, exist_ok=True)

    # build dataset & dataloader
    test_data = RefTestDataset(data_dir=ref_args.data_dir,
                            dataset=ref_args.dataset,
                            split=ref_args.test_split,
                            splitby=ref_args.splitby,
                            #mode='test',
                            input_size=ref_args.input_size,
                            word_length=ref_args.word_len,
                            pretrained_model = ref_args.pretrained_lm)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=1,
                                              pin_memory=True)

    with open(f"./overall_caption_path/overall_caption_{ref_args.dataset}_{ref_args.test_split}_{ref_args.splitby}.json","r") as f:
        overall_caption_dict = json.load(f)

    all_caption_dict = {}
    #with open(f"all_caption_path/caption_dict_{ref_args.dataset}_{ref_args.test_split}_{ref_args.splitby}.json", "r") as f:
    #    all_caption_dict = json.load(f)

    iou_list = []
    cum_I, cum_U = 0, 0
    tbar = tqdm(test_loader, desc='Inference:', ncols=100)
    time.sleep(2)

    heuristics = Heuristics()
    left_relation_words = heuristics.RELATIONS[0].keywords
    right_relation_words = heuristics.RELATIONS[1].keywords
    up_relation_words = heuristics.RELATIONS[2].keywords
    bottom_relation_words = heuristics.RELATIONS[3].keywords
    left_superlative_words = heuristics.SUPERLATIVES[0].keywords
    right_superlative_words = heuristics.SUPERLATIVES[1].keywords
    up_superlative_words = heuristics.SUPERLATIVES[2].keywords
    bottom_superlative_words = heuristics.SUPERLATIVES[3].keywords
    all_length = 0
    for img, param in tbar:
        img = img.cuda(non_blocking=True)
        # multiple sentences
        h, w = param['ori_size'].numpy()[0]
        ori_img_rgb = param['ori_img'][0].cpu().mul_(255).permute(1,2,0).type(torch.uint8).numpy()
        ori_img = cv2.cvtColor(ori_img_rgb,cv2.COLOR_RGB2BGR)
        mask = param["masks"][0]
        mask = mask.cpu().numpy()
        all_masks_list = mask_generator.generate(ori_img_rgb) # a list of dict

        
        all_depth_for_mask_list = []
        all_crop_list = []
        all_mass_list = []
        all_masks_remain_list = []
        input_global_masks = []
        input_ori = []
        ori_img_I = Image.fromarray(ori_img_rgb)

        refined_all_masks_list, kept_indices, areas = calculate_overlap(all_masks_list)
        all_length += len(refined_all_masks_list)
        print(len(refined_all_masks_list))
        #pdb.set_trace()
        #print("kept_indices:", kept_indices)

        captions_for_all_masks = []
        for i, mask_dict in enumerate(refined_all_masks_list):
            crop = crop_image(ori_img_rgb, mask_dict)
            preprocessed_ori = preprocess(ori_img_I).unsqueeze(0).to(device)
            preprocessed = preprocess(crop).unsqueeze(0).to(device)
            
            input_for_blip = crop_image_with_background_blur(cv2.cvtColor(ori_img,cv2.COLOR_BGR2RGB), mask_dict)
            #input_for_blip = crop_image(cv2.cvtColor(ori_img,cv2.COLOR_BGR2RGB), mask_dict)
            image_for_blip = vis_processors["eval"](input_for_blip).unsqueeze(0).to(device)
            caption = blip_model.generate({"image": image_for_blip})[0]
            captions_for_all_masks.append(caption)

            h_list, w_list = np.where(mask_dict["segmentation"])
            center_h = int(np.mean(h_list))
            center_w = int(np.mean(w_list))

            input_ori.append(preprocessed_ori)
            input_global_masks.append(mask_dict["segmentation"])
            all_masks_remain_list.append(mask_dict)
            all_crop_list.append(preprocessed)
            all_mass_list.append(np.array([center_h, center_w]))
        
        input_global_masks = np.stack(input_global_masks)
        all_mass_list = np.stack(all_mass_list)
        stack_crops = torch.cat(all_crop_list,dim=0)
        input_ori = torch.cat(input_ori,dim=0)

        for sent_id, sent in zip(param['sents_id'],param['sents']):
            all_caption_dict[sent_id.item()] = captions_for_all_masks
            #captions_for_all_masks = all_caption_dict[str(sent_id.item())]
            overall_caption = overall_caption_dict[str(sent_id.item())]
            
            #local_sent = extract_target_np(sent[0].lower(),nlp)
            local_sent = extract_noun_phrase(sent[0].lower(),nlp)
        
            local_tokens = tokenize([local_sent]).to(device)
            global_tokens = tokenize([sent[0].lower()]).to(device)

            
            
            with torch.no_grad():
                global_image_features, global_vision_feature_list = clip_model.encode_image(input_ori,input_global_masks)
                local_image_features, local_vision_feature_list = clip_model.encode_image(stack_crops)
                global_text_features, global_state, global_text_feature_list = clip_model.encode_text(global_tokens)
                local_text_features, local_state, local_text_feature_list = clip_model.encode_text(local_tokens)

            global_state = global_state / global_state.norm(dim=-1, keepdim=True)
            local_state = local_state / local_state.norm(dim=-1, keepdim=True)

            local_image_features = local_image_features / local_image_features.norm(dim=-1, keepdim=True)
            global_image_features = global_image_features / global_image_features.norm(dim=-1, keepdim=True)

            # global feature normalized in the clip model
            #text_features = 0.9 * global_state + 0.1 * local_state
            #image_features = 0.7 * global_image_features + 0.3 * local_image_features

            text_features = global_state #global_text_features[1]
            #text_features = local_state #local_text_features[1]
            image_features = local_image_features
            #image_features = global_image_features

            # cosine similarity as logits
            logit_scale = clip_model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            
            refer_tokens = tokenize([sent[0].lower()]).to(device)
            caption_inputs = tokenize(captions_for_all_masks).to(device)
            with torch.no_grad():
                caption_feature, caption_state, caption_feature_list = clip_model.encode_text(caption_inputs)
                refer_feature, refer_state, refer_feature_list = clip_model.encode_text(refer_tokens)

            caption_features = caption_state / caption_state.norm(dim=-1, keepdim=True)
            refer_features = refer_state / refer_state.norm(dim=-1, keepdim=True)

            # cosine similarity as logits
            logit_scale = clip_model.logit_scale.exp()
            logits_per_caption = logit_scale * caption_features @ refer_features.t()

            text_similarity = logits_per_caption.softmax(0).cpu()
            text_index = torch.argmax(text_similarity)
            
            doc = nlp(overall_caption)
            

            local_negative_sents = [("a photo of " + str(chunk)) for chunk in doc.noun_chunks if not are_near_synonyms(str(chunk), local_sent)]
            
            local_negative_tokens = tokenize(local_negative_sents).to(device)
            
            with torch.no_grad():
                local_negative_feature, local_negative_state, local_negative_feature_list = clip_model.encode_text(local_negative_tokens)
            local_negative_features = local_negative_state / local_negative_state.norm(dim=-1, keepdim=True)
            logit_scale = clip_model.logit_scale.exp()

            logits_image_nega = logit_scale * image_features @ local_negative_features.t()
            logits_image_nega = logits_image_nega.mean(-1,keepdim=True)
            
            
            overall_logits_per_image = 1 * logits_per_image + 0.1 * logits_per_caption - 0.7 * logits_image_nega #- 0.05 * logits_text_nega
            overall_logits_per_image = (overall_logits_per_image - overall_logits_per_image.min()) / (overall_logits_per_image.max() - overall_logits_per_image.min())
            
            similarity = overall_logits_per_image.softmax(0).cpu()
            
            index = torch.argmax(similarity)
            
            # Divide points into four parts
            x_coords, y_coords = all_mass_list[:, 0], all_mass_list[:, 1]
            median_x = np.median(x_coords)
            median_y = np.median(y_coords)

            part1_idx = np.where((all_mass_list[:, 0] < median_x) & (all_mass_list[:, 1] < median_y))[0]
            part2_idx = np.where((all_mass_list[:, 0] >= median_x) & (all_mass_list[:, 1] < median_y))[0]
            part3_idx = np.where((all_mass_list[:, 0] < median_x) & (all_mass_list[:, 1] >= median_y))[0]
            part4_idx = np.where((all_mass_list[:, 0] >= median_x) & (all_mass_list[:, 1] >= median_y))[0]

            weights = similarity.squeeze(1).detach().cpu().numpy()
            part1_weights = None if len(part1_idx) == 0 else [weights[i] for i in part1_idx]
            part2_weights = None if len(part2_idx) == 0 else [weights[i] for i in part2_idx]
            part3_weights = None if len(part3_idx) == 0 else [weights[i] for i in part3_idx]
            part4_weights = None if len(part4_idx) == 0 else [weights[i] for i in part4_idx]

            # Calculate maximum weight in each part and its corresponding location and index
            max_weight1, loc1, idx1 = (0, None, index) if len(part1_idx) == 0 else max([(w, p, i) for (w, p, i) in zip(part1_weights, all_mass_list[part1_idx], part1_idx)], key=lambda x: x[0])
            max_weight2, loc2, idx2 = (0, None, index) if len(part2_idx) == 0 else max([(w, p, i) for (w, p, i) in zip(part2_weights, all_mass_list[part2_idx], part2_idx)], key=lambda x: x[0])
            max_weight3, loc3, idx3 = (0, None, index) if len(part3_idx) == 0 else max([(w, p, i) for (w, p, i) in zip(part3_weights, all_mass_list[part3_idx], part3_idx)], key=lambda x: x[0])
            max_weight4, loc4, idx4 = (0, None, index) if len(part4_idx) == 0 else max([(w, p, i) for (w, p, i) in zip(part4_weights, all_mass_list[part4_idx], part4_idx)], key=lambda x: x[0])
            
            
            # spatial rectify
            parse_result, counts, chunks = parse_sentence_entities(nlp,sent[0].lower())
            chunks = list(chunks.values())
            for i in range(len(chunks)):
                chunks[i] = str(chunks[i])
            temp_ori_index = index
            
            
            if parse_result is not None:
                if str(parse_result.head) == local_sent:
                    if len(parse_result.superlatives)!=0 and len(parse_result.relations)==0:
                        if "above" in sent[0].lower() or "top" in sent[0].lower() or "north" in sent[0].lower():
                            if "left" in sent[0].lower() or "west" in sent[0].lower() or "leftmost" in sent[0].lower() or "western" in sent[0].lower():
                                index = idx1
                            elif "right" in sent[0].lower() or "east" in sent[0].lower() or "eastern" in sent[0].lower() or "rightmost" in sent[0].lower():
                                index = idx3
                            else:
                                if max_weight1 > max_weight3:
                                    index = idx1
                                else:
                                    index = idx3
                        elif "below" in sent[0].lower() or "south" in sent[0].lower() or "underneath" in sent[0].lower():
                            if "left" in sent[0].lower() or "west" in sent[0].lower():
                                index = idx2
                            elif "right" in sent[0].lower() or "east" in sent[0].lower():
                                index = idx4
                            else:
                                if max_weight2 > max_weight4:
                                    index = idx2
                                else:
                                    index = idx4
                        else:
                            if "left" in sent[0].lower() or "west" in sent[0].lower() or "leftmost" in sent[0].lower() or "western" in sent[0].lower():
                                if max_weight1 > max_weight2:
                                    index = idx1
                                else:
                                    index = idx2
                            elif "right" in sent[0].lower() or "east" in sent[0].lower() or "eastern" in sent[0].lower() or "rightmost" in sent[0].lower():
                                if max_weight3 > max_weight4:
                                    index = idx3
                                else:
                                    index = idx4
                            else:
                                pass

                    elif len(parse_result.relations)!=0 and len(parse_result.superlatives)==0:
                        for relation in parse_result.relations:
                            if len(relation[0])==0:
                                continue
                            else:
                                if str(relation[1].head) in chunks:
                                    for null_word in heuristics.NULL_KEYWORDS:
                                        if null_word in str(relation[1].head):
                                            break
                                    for word in relation[0]:
                                        if str(word) in left_relation_words:
                                            if str(relation[1].head) == str(parse_result.head) or str(word) in str(relation[1].head):
                                                if "above" in sent[0].lower() or "top" in sent[0].lower() or "north" in sent[0].lower():
                                                    index = idx1
                                                elif "below" in sent[0].lower() or "south" in sent[0].lower() or "under" in sent[0].lower():
                                                    index = idx2
                                                else:
                                                    if max_weight1 > max_weight2:
                                                        index = idx1
                                                    else:
                                                        index = idx2

                                            break
                                            
                                        elif str(word) in right_relation_words:
                                            if str(relation[1].head) == str(parse_result.head) or str(word) in str(relation[1].head):
                                                if "above" in sent[0].lower() or "top" in sent[0].lower() or "north" in sent[0].lower():
                                                    index = idx3
                                                elif "below" in sent[0].lower() or "south" in sent[0].lower() or "under" in sent[0].lower():
                                                    index = idx4
                                                else:
                                                    if max_weight3 > max_weight4:
                                                        index = idx3
                                                    else:
                                                        index = idx4
                                            
                                            break
                                            
                                        elif str(word) in up_relation_words:
                                            if str(relation[1].head) == str(parse_result.head) or str(word) in str(relation[1].head):
                                                if "left" in sent[0].lower() or "west" in sent[0].lower():
                                                    index = idx1
                                                elif "right" in sent[0].lower() or "east" in sent[0].lower():
                                                    index = idx3
                                                else:
                                                    if max_weight1 > max_weight3:
                                                        index = idx1
                                                    else:
                                                        index = idx3
                                            break
                                            
                                        elif str(word) in bottom_relation_words:
                                            if str(relation[1].head) == str(parse_result.head) or str(word) in str(relation[1].head):
                                                if "left" in sent[0].lower() or "west" in sent[0].lower():
                                                    index = idx2
                                                elif "right" in sent[0].lower() or "east" in sent[0].lower():
                                                    index = idx4
                                                else:
                                                    if max_weight2 > max_weight4:
                                                        index = idx2
                                                    else:
                                                        index = idx4
                                            break

                                        else:
                                            pass
                    else:
                        pass
            
            #index = text_index # this line for blip only baseline
            pred = all_masks_remain_list[index]["segmentation"]
            pred = pred.astype(int)
            #pred = pred.int().cpu().numpy()
           
            # iou
            inter = np.logical_and(pred, mask)
            union = np.logical_or(pred, mask)
            iou = np.sum(inter) / (np.sum(union) + 1e-6)
            cum_I += np.sum(inter)
            cum_U += np.sum(union)
            


            iou_list.append(iou)
            # dump image & mask
            if ref_args.visualize:
                img_name = '{}-img.jpg'.format(sent_id)
                mask_name = '{}-mask.png'.format(sent_id)
                cv2.imwrite(filename=os.path.join(ref_args.vis_dir, img_name),
                            img=ori_img)
                cv2.imwrite(filename=os.path.join(ref_args.vis_dir, mask_name),
                            img=mask*255)
            # dump prediction
            if ref_args.visualize:
                pred = np.array(pred*255, dtype=np.uint8)
                sent = "_".join(sent[0].split(" "))
                pred_name = '{}-iou={:.2f}-{}.png'.format(sent_id, iou*100, sent)
                cv2.imwrite(filename=os.path.join(ref_args.vis_dir, pred_name),
                            img=pred)
    logger.info('=> Metric Calculation <=')
    iou_list = np.stack(iou_list)
    iou_list = torch.from_numpy(iou_list).to(img.device)
    prec_list = []
    for thres in torch.arange(0.5, 1.0, 0.1):
        tmp = (iou_list > thres).float().mean()
        prec_list.append(tmp)
    iou = iou_list.mean()
    oiou = cum_I / cum_U
    prec = {}
    for i, thres in enumerate(range(5, 10)):
        key = 'Pr@{}'.format(thres*10)
        value = prec_list[i].item()
        prec[key] = value
    logger.info('mIoU={:.2f}'.format(100.*iou.item()))
    logger.info('oIoU={:.2f}'.format(100.*oiou))
    for k, v in prec.items():
        logger.info('{}: {:.2f}.'.format(k, 100.*v))
    #caption_dict_json = json.dumps(all_caption_dict)
    #with open(f"./all_caption_path/caption_dict_{ref_args.dataset}_{ref_args.test_split}_{ref_args.splitby}_crop.json", "w") as f:
        # Write the JSON string to the file
        #f.write(caption_dict_json)
