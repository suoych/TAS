import os
from typing import List, Union
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.refer import REFER
import utils.transforms as T
#from bert.tokenization_bert import BertTokenizer
from torchvision.transforms import functional as F
#from .simple_tokenizer import SimpleTokenizer as _Tokenizer
import pdb
#import spacy
import json

#_tokenizer = _Tokenizer()



def get_transform(img_size, mode):
    transforms = [T.Resize(img_size, img_size),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ]

    return T.Compose(transforms)


class RefTestDataset(Dataset):
    def __init__(self, data_dir, dataset, split, splitby, input_size,
                 word_length, pretrained_model):
        super(RefTestDataset, self).__init__()
        self.mode = "test"
        self.input_size = (input_size, input_size)
        self.word_length = word_length

        # note that the split means the data partition, mode means program state,
        # splitby is used in G-ref dataset, switching to google version or unc version.
        self.refer = REFER(data_dir, dataset, splitby)
        ref_ids = self.refer.getRefIds(split=split)
        self.split = split
        img_ids = self.refer.getImgIds(ref_ids)

        all_imgs = self.refer.Imgs
        self.imgs = list(all_imgs[i] for i in img_ids)
        # get all image in the train/val/test split
        self.ref_ids = ref_ids
        #self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        self.transform = get_transform(input_size, self.mode)
        #self.nlp = spacy.load("en_core_web_trf")
        #self.subj_dict = json.load(open(os.path.join(data_dir,"subj_"+dataset+"_"+splitby+"_"+split+".json"),'r'))
        #self.attr_dict = json.load(open(os.path.join(data_dir,"attr_"+dataset+"_"+splitby+"_"+split+".json"),'r'))


    def __len__(self):
        # Different form other supervised works, 
        # length of the dataset equals to the number 
        # of images under weakly supervised setting
        return len(self.ref_ids)

    def __getitem__(self, index):
        # To reproduce the method in the weakly supervised setting, the dataloader picks 
        # images instead of referring expressions during training.
        #pdb.set_trace()

        this_ref_id = self.ref_ids[index]
        this_img_id = self.refer.getImgIds(this_ref_id)
        this_img = self.refer.Imgs[this_img_id[0]]

        img = Image.open(os.path.join(self.refer.IMAGE_DIR, this_img['file_name'])).convert("RGB")

        ref = self.refer.loadRefs(this_ref_id)
        ori_img = img.copy()
        ori_img = F.to_tensor(ori_img)

        all_word_vecs = []
        all_attention_masks = []
 
        all_subject_index = []
        all_attribute_index = []

        sents = []
        sents_id = []
        temp_sents = ref[0]["sentences"]
        for temp_sent in temp_sents:
            sents.append(temp_sent['sent'])
            sents_id.append(temp_sent["sent_id"])
            # bert tokenization 
            #attention_mask = [0] * self.word_length
            #padded_input_id = [0] * self.word_length
            #input_id = self.tokenizer.encode(text=temp_sent['sent'], add_special_tokens=False)
            #input_id = input_id[:self.word_length]

            #cur_subj_index = self.subj_dict[str(ref[0]["ref_id"])][str(temp_sent["sent_id"])]
            #all_subject_index.append(torch.tensor(cur_subj_index).unsqueeze(0))

            #cur_attr_index = self.attr_dict[str(ref[0]["ref_id"])][str(temp_sent["sent_id"])]
            #all_attribute_index.append(torch.tensor(cur_attr_index).unsqueeze(0))

            #padded_input_id[:len(input_id)] = input_id
            #attention_mask[:len(input_id)] = [1]*len(input_id)
            #all_word_vecs.append(torch.tensor(padded_input_id).unsqueeze(0))
            #all_attention_masks.append(torch.tensor(attention_mask).unsqueeze(0))

        #word_vecs = torch.cat(all_word_vecs, dim=0)
        #attention_masks = torch.cat(all_attention_masks, dim=0)
        #selected_subj_index = torch.cat(all_subject_index, dim=0)
        #selected_attr_index = torch.cat(all_attribute_index, dim=0)

        ref_mask = np.array(self.refer.getMask(ref[0])['mask'])
        annot = np.zeros(ref_mask.shape)
        annot[ref_mask == 1] = 1

        annot = Image.fromarray(annot.astype(np.uint8), mode="P")

        img_size = ref_mask.shape[:2]
        

        if self.transform is not None:
            # resize, from PIL to tensor, and mean and std normalization
            img, mask = self.transform(img,annot)

        params = {
            #'word_vecs': word_vecs,
            #'attention_masks': attention_masks,
            #'subj_index': selected_subj_index,
            #'attr_index': selected_attr_index,
            'masks': ref_mask, # use the mask in orginal size
            'ori_img': ori_img,
            'sents_id': sents_id,
            'ori_size': np.array(img_size),
            'sents': sents
        }
        return img, params
        

    def __repr__(self):
        return self.__class__.__name__ + "(" + \
            f"dataset={self.dataset}, " + \
            f"split={self.split}, " + \
            f"mode={self.mode}, " + \
            f"input_size={self.input_size}, " + \
            f"word_length={self.word_length}"
            #f"db_path={self.lmdb_dir}, " + \
