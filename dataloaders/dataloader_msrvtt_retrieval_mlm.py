from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from collections import defaultdict
import json
import random
# from dataloaders.rawvideo_util import RawVideoExtractor
from dataloaders.rawframe_util import RawVideoExtractor
from nltk.corpus import stopwords
import torch
from copy import deepcopy
from tqdm import tqdm

class MSRVTT_DataLoader_MLM(Dataset):
    """MSRVTT dataset loader."""
    def __init__(
            self,
            subset,
            csv_path,
            features_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
    ):
        print("start read csv")
        self.data = pd.read_csv(csv_path)
        print("over")
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        self.word_vocab = self.tokenizer.vocab
        self.word_vocab_length = len(self.tokenizer.vocab)
        self.mask_mode = "fixed"

        self.subset = subset
        assert self.subset in ["test_mlm","val_mlm"]
        # video_id_path_dict = {}
        # video_id_path_dict["test"] = os.path.join(self.data_path, "test_list.txt")
        # sentence = self.data['sentence'].values[idx]
        video_ids  = list(set(self.data['video_id']))
        captions = {}
        for index, row in self.data.iterrows():
            if row['video_id'] not in captions.keys():
                captions[row['video_id']] = []
                captions[row['video_id']].append(row['sentence'])
#         print(captions[row['video_id']])
            else:
                captions[row['video_id']].append(row['sentence'])
        
        self.sample_len = 0
        self.sentences_dict = {}
        self.cut_off_points = []
        for video_id in tqdm(video_ids):
            assert video_id in captions
            for cap in captions[video_id]:
                # cap_txt = " ".join(cap)
                cap_txt = cap
                self.sentences_dict[len(self.sentences_dict)] = (video_id, cap_txt)
            self.cut_off_points.append(len(self.sentences_dict))

        ## below variables are used to multi-sentences retrieval
        # self.cut_off_points: used to tag the label when calculate the metric
        # self.sentence_num: used to cut the sentence representation
        # self.video_num: used to cut the video representation
        self.multi_sentence_per_video = True    # !!! important tag for eval
        if self.subset == "val_mlm" or self.subset == "test_mlm":
            self.sentence_num = len(self.sentences_dict)
            self.video_num = len(video_ids)
            assert len(self.cut_off_points) == self.video_num
            print("For {}, sentence number: {}".format(self.subset, self.sentence_num))
            print("For {}, video number: {}".format(self.subset, self.video_num))


        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    def __len__(self):
        return len(self.data)
    
    def _get_maskinputid_masklabel(self, video_id, caption=None):
        choice_video_ids = [video_id]

        for i, video_id in enumerate(choice_video_ids):
            if caption is not None:
                words = self.tokenizer.tokenize(caption)
            else:
                words = self._get_single_text(video_id)
            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]
            # print(words)
            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)


            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            input_ids = torch.tensor(input_ids)
            input_mask = torch.tensor(input_mask)
            segment_ids = torch.tensor(segment_ids)

            input_label = input_ids.clone()

            # print(input_ids)
            # print(input_label)
            # print(input_mask)

            if self.mask_mode == "random":
                rand = torch.rand(input_ids.shape)
                mask_arr = (rand < 0.15) * (input_ids != 49406) * (input_ids != 49407) * (input_ids != 0)
                not_mask_arr = ~mask_arr

                selection = torch.flatten(torch.nonzero(mask_arr, as_tuple=False)).tolist()
                not_selection = torch.flatten(torch.nonzero(not_mask_arr, as_tuple=False)).tolist()

                input_ids[selection] = 49339
                input_label[not_selection] = -100

                # print(input_ids)
                # print(input_label)
            elif self.mask_mode == "fixed":
                rand = torch.tensor([ 0 if i%2==0 else 1 for i in range(self.max_words)]) #mask掉奇数位置
                mask_arr = (rand < 0.5) * (input_ids != 49406) * (input_ids != 49407) * (input_ids != 0)
                not_mask_arr = ~mask_arr

                selection = torch.flatten(torch.nonzero(mask_arr, as_tuple=False)).tolist()
                not_selection = torch.flatten(torch.nonzero(not_mask_arr, as_tuple=False)).tolist()

                input_ids[selection] = 49339
                input_label[not_selection] = -100

                # print(input_ids)
                # print(input_label)


        input_ids = torch.unsqueeze(input_ids,0)
        input_mask = torch.unsqueeze(input_mask,0)
        segment_ids = torch.unsqueeze(segment_ids,0)

        return input_ids,input_mask,segment_ids,input_label

    def _get_text(self, video_id, sentence):
        choice_video_ids = [video_id]
        n_caption = len(choice_video_ids)

        k = n_caption
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

        for i, video_id in enumerate(choice_video_ids):
            words = self.tokenizer.tokenize(sentence)

            # ##########################
            # # add a cls in the mid #
            # ##########################
            # mid = len(words) // 2
            # words.insert(mid, self.SPECIAL_TOKEN["SEP_TOKEN"])
            # ##########################
            # # add a cls in the mid #
            # ##########################
            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment, choice_video_ids

    def _get_rawvideo(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
        max_video_length = [0] * len(choice_video_ids)

        # Pair x L x T x 3 x H x W
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float)

        for i, video_id in enumerate(choice_video_ids):
            # Individual for YoucokII dataset, due to it video format
            video_path = os.path.join(self.features_path, "{}.mp4".format(video_id))
            if os.path.exists(video_path) is False:
                video_path = video_path.replace(".mp4", ".avi")
                if os.path.exists(video_path) is False:
                    video_path = video_path.replace(".avi", "")
                    if os.path.exists(video_path) is False:
                        print('video path = {} is not exists.'.format(video_path))
                        break

            raw_video_data = self.rawVideoExtractor.get_video_data(video_path)
            raw_video_data = raw_video_data['video']
            if len(raw_video_data.shape) > 3:
                raw_video_data_clip = raw_video_data
                # L x T x 3 x H x W
                raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)
                if self.max_frames < raw_video_slice.shape[0]:
                    if self.slice_framepos == 0:
                        video_slice = raw_video_slice[:self.max_frames, ...]
                    elif self.slice_framepos == 1:
                        video_slice = raw_video_slice[-self.max_frames:, ...]
                    else:
                        sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                        video_slice = raw_video_slice[sample_indx, ...]
                else:
                    video_slice = raw_video_slice

                video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=self.frame_order)

                slice_len = video_slice.shape[0]
                max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
                if slice_len < 1:
                    pass
                else:
                    video[i][:slice_len, ...] = video_slice
            else:
                print("video path: {} error. video id: {}".format(video_path, video_id))

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask

    def __getitem__(self, idx):
        video_id = self.data['video_id'].values[idx]
        sentence = self.data['sentence'].values[idx]

        pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, sentence)
        video, video_mask = self._get_rawvideo(choice_video_ids)

        masked_input_ids,masked_input_mask,masked_segment_ids,masked_input_label = self._get_maskinputid_masklabel(video_id, sentence)

        return pairs_text, pairs_mask, pairs_segment, video, video_mask, masked_input_ids,masked_input_mask,masked_segment_ids,masked_input_label


class MSRVTT_TrainDataLoader_MLM(Dataset):
    """MSRVTT train dataset loader."""
    def __init__(
            self,
            csv_path,
            json_path,
            features_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            unfold_sentences=False,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
    ):
        self.csv = pd.read_csv(csv_path)
        print("loading training json")
        self.data = json.load(open(json_path, 'r'))
        print("loading training json over")
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer

        # self.concept_word_vocab_path = concept_word_vocab_path
        # if self.concept_word_vocab_path is not None:  
        #     self.word_vocab = self._read_concept()
        #     self.word_vocab_length = len(self.word_vocab)
        self.word_vocab = self.tokenizer.vocab
        self.word_vocab_length = len(self.tokenizer.vocab)
        self.mask_mode = "random"
        

        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        self.unfold_sentences = unfold_sentences
        self.sample_len = 0
        if self.unfold_sentences:
            train_video_ids = set(list(self.csv['video_id'].values))
            self.sentences_dict = {}
            for itm in tqdm(self.data['sentences']):
                if itm['video_id'] in train_video_ids:
                    self.sentences_dict[len(self.sentences_dict)] = (itm['video_id'], itm['caption'])
            self.sample_len = len(self.sentences_dict)
        else:
            num_sentences = 0
            self.sentences = defaultdict(list)
            s_video_id_set = set()
            for itm in self.data['sentences']:
                self.sentences[itm['video_id']].append(itm['caption'])
                num_sentences += 1
                s_video_id_set.add(itm['video_id'])

            # Use to find the clips in the same video
            # self.parent_ids = {}
            # self.children_video_ids = defaultdict(list)
            # for itm in self.data['videos']:
            #     vid = itm["video_id"]
            #     url_posfix = itm["url"].split("?v=")[-1]
            #     self.parent_ids[vid] = url_posfix
            #     self.children_video_ids[url_posfix].append(vid)
            self.sample_len = len(self.csv)

        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
        print("init over")

    def __len__(self):
        return self.sample_len

    def _get_maskinputid_masklabel(self, video_id, caption=None):
        choice_video_ids = [video_id]

        for i, video_id in enumerate(choice_video_ids):
            if caption is not None:
                words = self.tokenizer.tokenize(caption)
            else:
                words = self._get_single_text(video_id)
            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]
            # print(words)
            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)


            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            input_ids = torch.tensor(input_ids)
            input_mask = torch.tensor(input_mask)
            segment_ids = torch.tensor(segment_ids)

            input_label = input_ids.clone()

            # print(input_ids)
            # print(input_label)
            # print(input_mask)

            if self.mask_mode == "random":
                rand = torch.rand(input_ids.shape)
                mask_arr = (rand < 0.15) * (input_ids != 49406) * (input_ids != 49407) * (input_ids != 0)
                not_mask_arr = ~mask_arr

                selection = torch.flatten(torch.nonzero(mask_arr, as_tuple=False)).tolist()
                not_selection = torch.flatten(torch.nonzero(not_mask_arr, as_tuple=False)).tolist()

                input_ids[selection] = 49339
                input_label[not_selection] = -100

                # print(input_ids)
                # print(input_label)

        input_ids = torch.unsqueeze(input_ids,0)
        input_mask = torch.unsqueeze(input_mask,0)
        segment_ids = torch.unsqueeze(segment_ids,0)

        return input_ids,input_mask,segment_ids,input_label

    def _get_maskinputid_masklabel_saliency(self, video_id, caption=None):
        choice_video_ids = [video_id]

        for i, video_id in enumerate(choice_video_ids):
            if caption is not None:
                words = self.tokenizer.tokenize(caption)
            else:
                words = self._get_single_text(video_id)
            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]
            # print(words)
            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            saliency_ids = self.tokenizer.convert_tokens_to_saliencyids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)

            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                saliency_ids.append(0)
            
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words
            assert len(saliency_ids) == self.max_words

            input_ids = torch.tensor(input_ids)
            input_mask = torch.tensor(input_mask)
            segment_ids = torch.tensor(segment_ids)
            saliency_ids = torch.tensor(saliency_ids)

            input_label = input_ids.clone()

            # print(input_ids)
            # print(input_label)
            # print(input_mask)

            if self.mask_mode == "random":
                rand = torch.rand(input_ids.shape)
                mask_arr = (rand < 0.15) * (input_ids != 49406) * (input_ids != 49407) * (input_ids != 0) * (saliency_ids!=0)
                not_mask_arr = ~mask_arr

                selection = torch.flatten(torch.nonzero(mask_arr, as_tuple=False)).tolist()
                not_selection = torch.flatten(torch.nonzero(not_mask_arr, as_tuple=False)).tolist()

                input_ids[selection] = 49339
                input_label[not_selection] = -100

                # print(input_ids)
                # print(input_label)

        input_ids = torch.unsqueeze(input_ids,0)
        input_mask = torch.unsqueeze(input_mask,0)
        segment_ids = torch.unsqueeze(segment_ids,0)

        return input_ids,input_mask,segment_ids,input_label


    def _get_text(self, video_id, caption=None):
        k = 1
        choice_video_ids = [video_id]
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

        for i, video_id in enumerate(choice_video_ids):
            if caption is not None:
                words = self.tokenizer.tokenize(caption)
            else:
                words = self._get_single_text(video_id)

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]
            # print(words)

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment, choice_video_ids

    def _get_single_text(self, video_id):
        rind = random.randint(0, len(self.sentences[video_id]) - 1)
        caption = self.sentences[video_id][rind]
        words = self.tokenizer.tokenize(caption)
        return words

    def _get_rawvideo(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
        max_video_length = [0] * len(choice_video_ids)

        # Pair x L x T x 3 x H x W
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float)

        for i, video_id in enumerate(choice_video_ids):
            # Individual for YoucokII dataset, due to it video format
            if "+" in video_id: #webvid
                video_path = os.path.join(self.features_path,video_id.split("+")[0], "{}.mp4".format(video_id.split("+")[1]))
            else:
                video_path = os.path.join(self.features_path, "{}.mp4".format(video_id))

            if os.path.exists(video_path) is False:
                video_path = video_path.replace(".mp4", ".avi")
                if os.path.exists(video_path) is False:
                    video_path = video_path.replace(".avi", "")
                    if os.path.exists(video_path) is False:
                        print('video path = {} is not exists.'.format(video_path))
                        break
            
            raw_video_data = self.rawVideoExtractor.get_video_data(video_path)
            if raw_video_data is None:
                break
            raw_video_data = raw_video_data['video']
            if len(raw_video_data.shape) > 3:
                # print("video path: {} ok. video id: {}".format(video_path, video_id))
                raw_video_data_clip = raw_video_data
                # L x T x 3 x H x W
                raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)
                if self.max_frames < raw_video_slice.shape[0]:
                    if self.slice_framepos == 0:
                        video_slice = raw_video_slice[:self.max_frames, ...]
                    elif self.slice_framepos == 1:
                        video_slice = raw_video_slice[-self.max_frames:, ...]
                    else:
                        sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                        video_slice = raw_video_slice[sample_indx, ...]
                else:
                    video_slice = raw_video_slice

                video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=self.frame_order)

                slice_len = video_slice.shape[0]
                max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
                if slice_len < 1:
                    pass
                else:
                    video[i][:slice_len, ...] = video_slice
            else:
                # print(raw_video_data)
                # _ = self.rawVideoExtractor.get_video_data_check(video_path)
                print("video path: {} error. video id: {}".format(video_path, video_id))

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask

    def __getitem__(self, idx):
        if self.unfold_sentences:
            video_id, caption = self.sentences_dict[idx]
        else:
            video_id, caption = self.csv['video_id'].values[idx], None
        # pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, caption)
        pairs_text, pairs_mask, pairs_segment = np.ones(1),np.ones(1),np.ones(1) #不需要
        choice_video_ids = [video_id]
        video, video_mask = self._get_rawvideo(choice_video_ids)
        
        masked_input_ids,masked_input_mask,masked_segment_ids,masked_input_label = self._get_maskinputid_masklabel_saliency(video_id, caption)

        return pairs_text, pairs_mask, pairs_segment, video, video_mask, masked_input_ids,masked_input_mask,masked_segment_ids,masked_input_label


class MSRVTT_TrainDataLoader_Gen(Dataset):
    """MSRVTT train dataset loader."""
    def __init__(
            self,
            csv_path,
            json_path,
            features_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            unfold_sentences=False,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
            mask_mode="random_add"
    ):
        self.csv = pd.read_csv(csv_path)
        self.data = json.load(open(json_path, 'r'))
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer

        # self.concept_word_vocab_path = concept_word_vocab_path
        # if self.concept_word_vocab_path is not None:  
        #     self.word_vocab = self._read_concept()
        #     self.word_vocab_length = len(self.word_vocab)
        self.word_vocab = self.tokenizer.vocab
        self.word_vocab_length = len(self.tokenizer.vocab)
        self.mask_mode = mask_mode
        # "random_add" "random_replace"


        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        self.unfold_sentences = unfold_sentences
        self.sample_len = 0
        if self.unfold_sentences:
            train_video_ids = list(self.csv['video_id'].values)
            self.sentences_dict = {}
            for itm in self.data['sentences']:
                if itm['video_id'] in train_video_ids:
                    self.sentences_dict[len(self.sentences_dict)] = (itm['video_id'], itm['caption'])
            self.sample_len = len(self.sentences_dict)
        else:
            num_sentences = 0
            self.sentences = defaultdict(list)
            s_video_id_set = set()
            for itm in self.data['sentences']:
                self.sentences[itm['video_id']].append(itm['caption'])
                num_sentences += 1
                s_video_id_set.add(itm['video_id'])

            # Use to find the clips in the same video
            # self.parent_ids = {}
            # self.children_video_ids = defaultdict(list)
            # for itm in self.data['videos']:
            #     vid = itm["video_id"]
            #     url_posfix = itm["url"].split("?v=")[-1]
            #     self.parent_ids[vid] = url_posfix
            #     self.children_video_ids[url_posfix].append(vid)
            self.sample_len = len(self.csv)

        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    def __len__(self):
        return self.sample_len

    def _get_maskinputid_masklabel(self, video_id, caption=None):
        choice_video_ids = [video_id]

        for i, video_id in enumerate(choice_video_ids):
            if caption is not None:
                words = self.tokenizer.tokenize(caption)
            else:
                words = self._get_single_text(video_id)
            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]


            if "random_add" in self.mask_mode:
                input_ids = self.tokenizer.convert_tokens_to_ids(words)
                input_mask = [1] * len(input_ids)
                segment_ids = [0] * len(input_ids)
                input_label = [-100 for i in range(len(input_ids))]

                kk =int(self.mask_mode.strip().split("_")[-1]) 
                for i in range(kk):
                    rand_index = 1+int(torch.rand(1)*(len(input_ids)-1))
                    input_ids.insert(rand_index, 49339)
                    input_mask.insert(rand_index, 1)
                    segment_ids.insert(rand_index, 0)
                    input_label.insert(rand_index, 49339)



                if(len(input_ids)>=self.max_words):
                    input_ids = input_ids[0:self.max_words]
                    input_mask = input_mask[0:self.max_words]
                    segment_ids = segment_ids[0:self.max_words]
                    input_label = input_label[0:self.max_words]


                while len(input_ids) < self.max_words:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)
                    input_label.append(-100)
                
                assert len(input_ids) == self.max_words
                assert len(input_mask) == self.max_words
                assert len(segment_ids) == self.max_words
                assert len(input_label) == self.max_words

                input_ids = torch.tensor(input_ids)
                input_mask = torch.tensor(input_mask)
                segment_ids = torch.tensor(segment_ids)
                input_label = torch.tensor(input_label)

            elif "random_replace" in self.mask_mode:

                
                input_ids = self.tokenizer.convert_tokens_to_ids(words)
                input_mask = [1] * len(input_ids)
                segment_ids = [0] * len(input_ids)
                input_label = [-100 for i in range(len(input_ids))]

                kk =int(self.mask_mode.strip().split("_")[-1]) 
                rand_indexs = []
                for i in range(kk):
                    rand_index = 1+int(torch.rand(1)*(len(input_ids)-2))
                    if rand_index<1:
                        rand_index = 1
                    elif rand_index>=(len(input_ids)-1):
                        rand_index = len(input_ids)-1
                    rand_indexs.append(rand_index)
                

                while len(input_ids) < self.max_words:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)
                    input_label.append(-100)
                
                assert len(input_ids) == self.max_words
                assert len(input_mask) == self.max_words
                assert len(segment_ids) == self.max_words
                assert len(input_label) == self.max_words

                input_ids = torch.tensor(input_ids)
                input_mask = torch.tensor(input_mask)
                segment_ids = torch.tensor(segment_ids)
                input_label = torch.tensor(input_label)


                input_ids[rand_indexs] = 49339
                input_label[rand_indexs] = -100 
            elif "tail_add" in self.mask_mode:
                input_ids = self.tokenizer.convert_tokens_to_ids(words)
                input_mask = [1] * len(input_ids)
                segment_ids = [0] * len(input_ids)
                input_label = [-100 for i in range(len(input_ids))]

                kk =int(self.mask_mode.strip().split("_")[-1]) 
                for i in range(kk):
                    tail_index = len(input_ids)-1
                    input_ids.insert(tail_index, 49339)
                    input_mask.insert(tail_index, 1)
                    segment_ids.insert(tail_index, 0)
                    input_label.insert(tail_index, 49339)



                if(len(input_ids)>=self.max_words):
                    input_ids = input_ids[0:self.max_words]
                    input_mask = input_mask[0:self.max_words]
                    segment_ids = segment_ids[0:self.max_words]
                    input_label = input_label[0:self.max_words]


                while len(input_ids) < self.max_words:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)
                    input_label.append(-100)
                
                assert len(input_ids) == self.max_words
                assert len(input_mask) == self.max_words
                assert len(segment_ids) == self.max_words
                assert len(input_label) == self.max_words

                input_ids = torch.tensor(input_ids)
                input_mask = torch.tensor(input_mask)
                segment_ids = torch.tensor(segment_ids)
                input_label = torch.tensor(input_label)


        input_ids = torch.unsqueeze(input_ids,0)
        input_mask = torch.unsqueeze(input_mask,0)
        segment_ids = torch.unsqueeze(segment_ids,0)
        segment_ids = torch.unsqueeze(input_label,0)

        return input_ids,input_mask,segment_ids,input_label



    def _get_text(self, video_id, caption=None):
        k = 1
        choice_video_ids = [video_id]
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

        for i, video_id in enumerate(choice_video_ids):
            if caption is not None:
                words = self.tokenizer.tokenize(caption)
            else:
                words = self._get_single_text(video_id)

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]
            # print(words)

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment, choice_video_ids

    def _get_single_text(self, video_id):
        rind = random.randint(0, len(self.sentences[video_id]) - 1)
        caption = self.sentences[video_id][rind]
        words = self.tokenizer.tokenize(caption)
        return words

    def _get_rawvideo(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
        max_video_length = [0] * len(choice_video_ids)

        # Pair x L x T x 3 x H x W
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float)

        for i, video_id in enumerate(choice_video_ids):
            # Individual for YoucokII dataset, due to it video format
            video_path = os.path.join(self.features_path, "{}.mp4".format(video_id))
            if os.path.exists(video_path) is False:
                video_path = video_path.replace(".mp4", ".avi")
                if os.path.exists(video_path) is False:
                    video_path = video_path.replace(".avi", "")
                    if os.path.exists(video_path) is False:
                        print('video path = {} is not exists.'.format(video_path))
                        break

            raw_video_data = self.rawVideoExtractor.get_video_data(video_path)
            raw_video_data = raw_video_data['video']
            if len(raw_video_data.shape) > 3:
                # print("video path: {} ok. video id: {}".format(video_path, video_id))
                raw_video_data_clip = raw_video_data
                # L x T x 3 x H x W
                raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)
                if self.max_frames < raw_video_slice.shape[0]:
                    if self.slice_framepos == 0:
                        video_slice = raw_video_slice[:self.max_frames, ...]
                    elif self.slice_framepos == 1:
                        video_slice = raw_video_slice[-self.max_frames:, ...]
                    else:
                        sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                        video_slice = raw_video_slice[sample_indx, ...]
                else:
                    video_slice = raw_video_slice

                video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=self.frame_order)

                slice_len = video_slice.shape[0]
                max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
                if slice_len < 1:
                    pass
                else:
                    video[i][:slice_len, ...] = video_slice
            else:
                # print(raw_video_data)
                _ = self.rawVideoExtractor.get_video_data_check(video_path)
                print("video path: {} error. video id: {}".format(video_path, video_id))

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask

    def __getitem__(self, idx):
        if self.unfold_sentences:
            video_id, caption = self.sentences_dict[idx]
        else:
            video_id, caption = self.csv['video_id'].values[idx], None
        pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, caption)
        video, video_mask = self._get_rawvideo(choice_video_ids)
        
        masked_input_ids,masked_input_mask,masked_segment_ids,masked_input_label = self._get_maskinputid_masklabel(video_id, caption)

        return pairs_text, pairs_mask, pairs_segment, video, video_mask, masked_input_ids,masked_input_mask,masked_segment_ids,masked_input_label,video_id
