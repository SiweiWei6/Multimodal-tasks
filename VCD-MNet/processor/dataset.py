import random
import os
import torch
import json
import ast
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torchvision import transforms
from .dep_parser import DepInstanceParser
import numpy as np
import logging
logger = logging.getLogger(__name__)

class MMREProcessor(object):
    def __init__(self, data_path, bert_name):
        self.data_path = data_path
        self.re_path = data_path['re_path']
        self.tokenizer = BertTokenizer.from_pretrained(bert_name, do_lower_case=True)
        self.tokenizer.add_special_tokens({'additional_special_tokens':['<s>', '</s>', '<o>', '</o>']})

    def load_from_file(self, mode="train",all_dep_info=None, sample_ratio=1.0):
        """
        Args:
            mode (str, optional): dataset mode. Defaults to "train".
            sample_ratio (float, optional): sample ratio in low resouce. Defaults to 1.0.
        """        
        load_file = self.data_path[mode]
        logger.info("Loading data from {}".format(load_file))
        with open(load_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            # lines = lines[:10]
            print(all_dep_info[4110])
            words, relations, heads, tails, imgids, dataid = [], [], [], [], [], []
            ###dep
            all_dep_infos=[]
            ###dep!!
            for i, line in enumerate(lines):
                line = ast.literal_eval(line)   # str to dict
                words.append(line['token'])
                relations.append(line['relation'])
                heads.append(line['h']) # {name, pos}
                tails.append(line['t'])
                imgids.append(line['img_id'])
                ###dep
                all_dep_infos.append(all_dep_info[i])
                ###dep!!
                dataid.append(i)
        ###dep 改 加==(len(all_dep_infos))
        assert len(words) == len(relations) == len(heads) == len(tails) == (len(imgids))==(len(all_dep_infos))
        ###dep!!

        # aux image
        aux_path = self.data_path[mode+"_auximgs"]
        aux_imgs = torch.load(aux_path)

         # sample
        if sample_ratio != 1.0:
            sample_indexes = random.choices(list(range(len(words))), k=int(len(words)*sample_ratio))
            sample_words = [words[idx] for idx in sample_indexes]
            sample_relations = [relations[idx] for idx in sample_indexes]
            sample_heads = [heads[idx] for idx in sample_indexes]
            sample_tails = [tails[idx] for idx in sample_indexes]
            sample_imgids = [imgids[idx] for idx in sample_indexes]
            sample_dep_infos = [all_dep_infos[idx] for idx in sample_indexes]
            sample_dataid = [dataid[idx] for idx in sample_indexes]
            assert len(sample_words) == len(sample_relations) == len(sample_imgids), "{}, {}, {}".format(len(sample_words), len(sample_relations), len(sample_imgids))
            return {'words':sample_words, 'relations':sample_relations, 'heads':sample_heads, 'tails':sample_tails, \
                 'imgids':sample_imgids, 'dataid': sample_dataid, 'aux_imgs':aux_imgs,"all_dep_infos":all_dep_infos}
        
        return {'words':words, 'relations':relations, 'heads':heads, 'tails':tails, 'imgids': imgids, 'dataid': dataid, 'aux_imgs':aux_imgs,"all_dep_infos":all_dep_infos}
       

    def get_relation_dict(self):
        with open(self.re_path, 'r', encoding="utf-8") as f:
            line = f.readlines()[0]
            re_dict = json.loads(line)
        return re_dict

class MMPNERProcessor(object):
    def __init__(self, data_path, bert_name) -> None:
        self.data_path = data_path
        self.tokenizer = BertTokenizer.from_pretrained(bert_name, do_lower_case=True)
    
    def load_from_file(self, mode="train", sample_ratio=1.0):
        """
        Args:
            mode (str, optional): dataset mode. Defaults to "train".
            sample_ratio (float, optional): sample ratio in low resouce. Defaults to 1.0.
        """   
        load_file = self.data_path[mode]
        logger.info("Loading data from {}".format(load_file))
        with open(load_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

            raw_words, raw_targets = [], []
            raw_word, raw_target = [], []
            imgs = []
            for line in lines:
                if line.startswith("IMGID:"):
                    img_id = line.strip().split('IMGID:')[1] + '.jpg'
                    imgs.append(img_id)
                    continue
                if line != "\n":
                    raw_word.append(line.split('\t')[0])
                    label = line.split('\t')[1][:-1]
                    if 'OTHER' in label:
                        label = label[:2] + 'MISC'
                    raw_target.append(label)
                else:
                    raw_words.append(raw_word)
                    raw_targets.append(raw_target)
                    raw_word, raw_target = [], []

        assert len(raw_words) == len(raw_targets) == len(imgs), "{}, {}, {}".format(len(raw_words), len(raw_targets), len(imgs))
        # load aux image
        aux_path = self.data_path[mode+"_auximgs"]
        aux_imgs = torch.load(aux_path)

        # sample data, only for low-resource
        if sample_ratio != 1.0:
            sample_indexes = random.choices(list(range(len(raw_words))), k=int(len(raw_words)*sample_ratio))
            sample_raw_words = [raw_words[idx] for idx in sample_indexes]
            sample_raw_targets = [raw_targets[idx] for idx in sample_indexes]
            sample_imgs = [imgs[idx] for idx in sample_indexes]
            assert len(sample_raw_words) == len(sample_raw_targets) == len(sample_imgs), "{}, {}, {}".format(len(sample_raw_words), len(sample_raw_targets), len(sample_imgs))
            return {"words": sample_raw_words, "targets": sample_raw_targets, "imgs": sample_imgs, "aux_imgs":aux_imgs}

        return {"words": raw_words, "targets": raw_targets, "imgs": imgs, "aux_imgs":aux_imgs}

    def get_label_mapping(self):
        LABEL_LIST = ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]
        label_mapping = {label:idx for idx, label in enumerate(LABEL_LIST, 1)}
        label_mapping["PAD"] = 0
        return label_mapping

class MMREDataset(Dataset):
    ###dep ,all_dep_info=None,dep_direct=True
    def __init__(self, args, caption_data, processor, transform, img_path=None, aux_img_path=None,all_dep_info=None,dep_direct=True, max_seq=40, sample_ratio=1.0, mode="train") -> None:
    ###dep!!
        self.processor = processor
        self.transform = transform
        self.max_seq = max_seq
        self.img_path = img_path[mode]  if img_path is not None else img_path
        self.aux_img_path = aux_img_path[mode] if aux_img_path is not None else aux_img_path
        ###dep
        self.dep_direct=dep_direct
        ###dep!!
        self.mode = mode
        self.args = args
        self.caption_data = caption_data
        self.data_dict = self.processor.load_from_file(mode,all_dep_info, sample_ratio)
        self.re_dict = self.processor.get_relation_dict()
        self.tokenizer = self.processor.tokenizer
        ###dep
        self.prepare_type_dict(r"C:\Users\zhang\Desktop\desktop\VIT-CMNet\dependent")
        ###dep!!


    def __len__(self):
        return len(self.data_dict['words'])

    ###dep
    def get_dep_labels(self, data_dir):
        dep_labels = ["self_loop"]
        dep_type_path = os.path.join(data_dir, "dep_type.json")
        with open(dep_type_path, 'r') as f:
            dep_types = json.load(f)
            for label in dep_types:
                if self.dep_direct:
                    dep_labels.append("{}_in".format(label))
                    dep_labels.append("{}_out".format(label))
                else:
                    dep_labels.append(label)
        return dep_labels

    def prepare_type_dict(self, data_dir):
        dep_type_list = self.get_dep_labels(data_dir)
        types_dict = {"none": 0}
        for dep_type in dep_type_list:
            types_dict[dep_type] = len(types_dict)
        self.dep_label_map = types_dict

    def get_dep_matrix(self,ori_dep_type_matrix):
        dep_type_matrix = np.zeros((self.max_seq, self.max_seq), dtype=np.int)
        max_words_num = len(ori_dep_type_matrix)
        for i in range(max_words_num):
            dep_type_matrix[i][:max_words_num] = ori_dep_type_matrix[i]
        return dep_type_matrix

    ###dep!!
    def __getitem__(self, idx):
        ###dep 加dep_info ,self.data_dict['all_dep_infos']
        word_list, relation, head_d, tail_d, imgid,dep_info = self.data_dict['words'][idx], self.data_dict['relations'][idx], self.data_dict['heads'][idx], self.data_dict['tails'][idx], self.data_dict['imgids'][idx],self.data_dict['all_dep_infos'][idx]
       ###dep!!
        item_id = self.data_dict['dataid'][idx]
        # [CLS] ... <s> head </s> ... <o> tail <o/> .. [SEP]
        head_pos, tail_pos = head_d['pos'], tail_d['pos']
        # insert <s> <s/> <o> <o/>
        extend_word_list = []
        for i in range(len(word_list)):
            if  i == head_pos[0]:
                extend_word_list.append('<s>')
            if i == head_pos[1]:
                extend_word_list.append('</s>')
            if i == tail_pos[0]:
                extend_word_list.append('<o>')
            if i == tail_pos[1]:
                extend_word_list.append('</o>')
            extend_word_list.append(word_list[i])
        extend_word_list0=extend_word_list
        extend_word_list = " ".join(extend_word_list)
        encode_dict = self.tokenizer.encode_plus(text=extend_word_list, max_length=self.max_seq, truncation=True, padding='max_length')
        input_ids, token_type_ids, attention_mask = encode_dict['input_ids'], encode_dict['token_type_ids'], encode_dict['attention_mask']
        input_ids, token_type_ids, attention_mask = torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask)
        re_label = self.re_dict[relation]   # label to id
        ###dep_process
        if self.args.use_dep:
            start_range = list(range(head_pos[0],head_pos[1]))
            end_range = list(range(tail_pos[0], tail_pos[1]))
            dep_instance_parser = DepInstanceParser(basicDependencies=dep_info, tokens=word_list)
            dep_adj_matrix, dep_type_matrix = dep_instance_parser.get_local_global_graph(start_range, end_range,direct=self.dep_direct)
            #怎么弄e1 mask e2 mask 和valid
            b_use_valid_filter = False
            tokens = ["[CLS]"]
            valid = [0]
            e1_mask = []
            e2_mask = []
            e1_mask_val = 0
            e2_mask_val = 0
            entity_start_mark_position = [0, 0]
            for i, word in enumerate(extend_word_list0):
                if len(tokens) >= self.max_seq - 1:
                    break
                if word in ["<s>", "</s>", "<o>", "</o>"]:
                    tokens.append(word)
                    valid.append(0)  ###
                    if word in ["<s>"]:
                        e1_mask_val = 1
                    elif word in ["</s>"]:
                        e1_mask_val = 0
                    if word in ["<o>"]:
                        e2_mask_val = 1
                    elif word in ["</o>"]:
                        e2_mask_val = 0
                    continue

                token = self.tokenizer.tokenize(word)
                if len(tokens) + len(token) > self.max_seq - 1:
                    break
                tokens.extend(token)
                e1_mask.append(e1_mask_val)
                e2_mask.append(e2_mask_val)
                for m in range(len(token)):
                    if m == 0:
                        valid.append(1)
                    else:
                        valid.append(0)
                        b_use_valid_filter = True

            tokens.append("[SEP]")
            valid.append(0)
            e1_mask.append(0)
            e2_mask.append(0)

            # Zero-pad up to the sequence length.
            padding = [0] * (self.max_seq - len(valid))
            valid += padding
            e1_mask += [0] * (self.max_seq - len(e1_mask))
            e2_mask += [0] * (self.max_seq - len(e2_mask))
            assert len(valid) == self.max_seq
            assert len(e1_mask) == self.max_seq
            assert len(e2_mask) == self.max_seq

            max_words_num = sum(valid)

            def get_adj_with_value_matrix(dep_adj_matrix, dep_type_matrix):
                final_dep_adj_matrix = np.zeros((max_words_num, max_words_num), dtype=np.int)
                final_dep_type_matrix = np.zeros((max_words_num, max_words_num), dtype=np.int)
                for pi in range(max_words_num):
                    for pj in range(max_words_num):
                        if dep_adj_matrix[pi][pj] == 0:
                            continue
                        if pi >= self.max_seq or pj >= self.max_seq:
                            continue
                        final_dep_adj_matrix[pi][pj] = dep_adj_matrix[pi][pj]
                        final_dep_type_matrix[pi][pj] = self.dep_label_map[dep_type_matrix[pi][pj]]
                return final_dep_adj_matrix, final_dep_type_matrix

            dep_adj_matrix, dep_type_matrix = get_adj_with_value_matrix(dep_adj_matrix,
                                                                        dep_type_matrix)

            dep_type_matrix = self.get_dep_matrix(dep_type_matrix)
        ####dep!!
        # image process
        if self.img_path is not None:
            try:
                img_path = os.path.join(self.img_path, imgid)
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image)
                sequence = self.caption_data[img_path]['seq']
                seq_dict = self.tokenizer.encode_plus(text=sequence, max_length=13,
                                                      truncation=True, padding='max_length')
                seq_input_ids, seq_token_type_ids, seq_attention_mask = seq_dict['input_ids'], seq_dict[
                    'token_type_ids'], \
                                                                        seq_dict['attention_mask']
                seq_input_ids, seq_token_type_ids, seq_attention_mask = torch.tensor(seq_input_ids), torch.tensor(
                    seq_token_type_ids), torch.tensor(seq_attention_mask)
            except:
                img_path = os.path.join(self.img_path, 'inf.png')
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image)
                sequence = self.caption_data[img_path]['seq']
                seq_dict = self.tokenizer.encode_plus(text=sequence, max_length=13,
                                                      truncation=True, padding='max_length')
                seq_input_ids, seq_token_type_ids, seq_attention_mask = seq_dict['input_ids'], seq_dict[
                    'token_type_ids'], seq_dict['attention_mask']
                seq_input_ids, seq_token_type_ids, seq_attention_mask = torch.tensor(seq_input_ids), torch.tensor(
                    seq_token_type_ids), torch.tensor(seq_attention_mask)
            if self.aux_img_path is not None:
                # process aux image
                aux_imgs = []
                aux_img_paths = []
                imgid = imgid.split(".")[0]
                if item_id in self.data_dict['aux_imgs']:
                    aux_img_paths  = self.data_dict['aux_imgs'][item_id]
                    aux_img_paths = [os.path.join(self.aux_img_path, path) for path in aux_img_paths]
                # discaed more than 3 aux image
                for i in range(min(3, len(aux_img_paths))):
                    aux_img = Image.open(aux_img_paths[i]).convert('RGB')
                    aux_img = self.transform(aux_img)
                    aux_imgs.append(aux_img)

                # padding zero if less than 3
                for i in range(3-len(aux_img_paths)):
                    aux_imgs.append(torch.zeros((3, 224, 224))) 

                aux_imgs = torch.stack(aux_imgs, dim=0)
                assert len(aux_imgs) == 3
        ###dep
            if self.args.use_dep:
                return input_ids, token_type_ids, attention_mask, torch.tensor(re_label), image, aux_imgs,' '.join(self.data_dict['words'][idx]), \
                       self.data_dict['imgids'][idx], relation, \
                       seq_input_ids, seq_token_type_ids, seq_attention_mask,\
                        torch.tensor(valid), torch.tensor(e1_mask), torch.tensor(e2_mask), torch.tensor(dep_type_matrix),torch.tensor(b_use_valid_filter), \
                       self.data_dict['heads'][idx],self.data_dict['tails'][idx]

            else:
                return input_ids, token_type_ids, attention_mask, torch.tensor(re_label), image, aux_imgs,' '.join(self.data_dict['words'][idx]), self.data_dict['imgids'][idx], relation, seq_input_ids, seq_token_type_ids, seq_attention_mask,self.data_dict['heads'][idx],self.data_dict['tails'][idx]
        if self.args.use_dep:
            return input_ids, token_type_ids, attention_mask, torch.tensor(re_label), \
                   torch.tensor(valid), torch.tensor(e1_mask), torch.tensor(e2_mask), torch.tensor(dep_type_matrix),torch.tensor(b_use_valid_filter), \
                   self.data_dict['heads'][idx],self.data_dict['tails'][idx]
        else:
            return input_ids, token_type_ids, attention_mask, torch.tensor(re_label),self.data_dict['heads'][idx],self.data_dict['tails'][idx]
    ###dep!!!
class MMPNERDataset(Dataset):
    def __init__(self, args, caption_data,processor, transform, img_path=None, aux_img_path=None, max_seq=40, sample_ratio=1, mode='train', ignore_idx=0) -> None:
        self.processor = processor
        self.transform = transform
        self.data_dict = processor.load_from_file(mode, sample_ratio)
        self.tokenizer = processor.tokenizer
        self.label_mapping = processor.get_label_mapping()
        self.max_seq = max_seq
        self.ignore_idx = ignore_idx
        self.img_path = img_path
        self.aux_img_path = aux_img_path[mode]  if aux_img_path is not None else None
        self.mode = mode
        self.sample_ratio = sample_ratio
        self.args = args
        self.caption_data = caption_data
    
    def __len__(self):
        return len(self.data_dict['words'])

    def __getitem__(self, idx):
        word_list, label_list, img = self.data_dict['words'][idx], self.data_dict['targets'][idx], self.data_dict['imgs'][idx]
        tokens, labels = [], []

        ######
        for i, word in enumerate(word_list):
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            label = label_list[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(self.label_mapping[label])
                else:
                    labels.append(self.label_mapping["X"])
        if len(tokens) >= self.max_seq - 1:
            tokens = tokens[0:(self.max_seq - 2)]
            labels = labels[0:(self.max_seq - 2)]

        encode_dict = self.tokenizer.encode_plus(tokens, max_length=self.max_seq, truncation=True, padding='max_length')
        input_ids, token_type_ids, attention_mask = encode_dict['input_ids'], encode_dict['token_type_ids'], encode_dict['attention_mask']
        labels = [self.label_mapping["[CLS]"]] + labels + [self.label_mapping["[SEP]"]] + [self.ignore_idx]*(self.max_seq-len(labels)-2)

        if self.img_path is not None:
            # image process
            try:
                img_path = os.path.join(self.img_path, img)
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image)
                sequence = self.caption_data[img_path]['seq']
                seq_dict = self.tokenizer.encode_plus(text=sequence, max_length=13,
                                                      truncation=True, padding='max_length')
                seq_input_ids, seq_token_type_ids, seq_attention_mask = seq_dict['input_ids'], seq_dict[
                    'token_type_ids'], \
                                                                        seq_dict['attention_mask']
                seq_input_ids, seq_token_type_ids, seq_attention_mask = torch.tensor(seq_input_ids), torch.tensor(
                    seq_token_type_ids), torch.tensor(seq_attention_mask)
            except:
                img_path = os.path.join(self.img_path, 'inf.png')
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image)
                sequence = self.caption_data[img_path]['seq']
                seq_dict = self.tokenizer.encode_plus(text=sequence, max_length=13,
                                                      truncation=True, padding='max_length')
                seq_input_ids, seq_token_type_ids, seq_attention_mask = seq_dict['input_ids'], seq_dict[
                    'token_type_ids'], seq_dict['attention_mask']
                seq_input_ids, seq_token_type_ids, seq_attention_mask = torch.tensor(seq_input_ids), torch.tensor(
                    seq_token_type_ids), torch.tensor(seq_attention_mask)

            if self.aux_img_path is not None:
                aux_imgs = []
                aux_img_paths = []
                if img in self.data_dict['aux_imgs']:
                    aux_img_paths  = self.data_dict['aux_imgs'][img]
                    aux_img_paths = [os.path.join(self.aux_img_path, path) for path in aux_img_paths]
                for i in range(min(3, len(aux_img_paths))):
                    aux_img = Image.open(aux_img_paths[i]).convert('RGB')
                    aux_img = self.transform(aux_img)
                    aux_imgs.append(aux_img)

                for i in range(3-len(aux_img_paths)):
                    aux_imgs.append(torch.zeros((3, 224, 224))) 

                aux_imgs = torch.stack(aux_imgs, dim=0)
                assert len(aux_imgs) == 3
                return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask), torch.tensor(labels), image, aux_imgs,seq_input_ids, seq_token_type_ids, seq_attention_mask,' '.join(self.data_dict['words'][idx]), self.data_dict['imgs'][idx]

        assert len(input_ids) == len(token_type_ids) == len(attention_mask) == len(labels)
        return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask), torch.tensor(labels)
