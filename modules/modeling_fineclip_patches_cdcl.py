from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F

from modules.until_module import PreTrainedModel, AllGather, CrossEn
from modules.module_cross import CrossModel, CrossConfig, Transformer as TransformerClip

from modules.module_clip import CLIP, convert_weights
from modules.modeling import CLIP4ClipPreTrainedModel, show_log, update_attr, check_attr

# from modules.differential_topk import PerturbedTopK

logger = logging.getLogger(__name__)
allgather = AllGather.apply


class XCLIP(CLIP4ClipPreTrainedModel):
    def __init__(self, cross_config, clip_state_dict, task_config):
        super(XCLIP, self).__init__(cross_config)
        self.task_config = task_config
        self.ignore_video_index = -1

        assert self.task_config.max_words + self.task_config.max_frames <= cross_config.max_position_embeddings

        self._stage_one = True
        self._stage_two = False

        show_log(task_config, "Stage-One:{}, Stage-Two:{}".format(self._stage_one, self._stage_two))

        self.loose_type = False
        if self._stage_one and check_attr('loose_type', self.task_config):
            self.loose_type = True
            show_log(task_config, "Test retrieval by loose type.")

        # CLIP Encoders: From OpenAI: CLIP [https://github.com/openai/CLIP] ===>
        vit = "visual.proj" in clip_state_dict
        assert vit
        if vit:
            vision_width = clip_state_dict["visual.conv1.weight"].shape[0]
            vision_layers = len(
                [k for k in clip_state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            vision_patch_size = clip_state_dict["visual.conv1.weight"].shape[-1]
            grid_size = round((clip_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
            image_resolution = vision_patch_size * grid_size
        else:
            counts: list = [len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"visual.layer{b}"))) for b in
                            [1, 2, 3, 4]]
            vision_layers = tuple(counts)
            vision_width = clip_state_dict["visual.layer1.0.conv1.weight"].shape[0]
            output_width = round((clip_state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
            vision_patch_size = None
            assert output_width ** 2 + 1 == clip_state_dict["visual.attnpool.positional_embedding"].shape[0]
            image_resolution = output_width * 32

        embed_dim = clip_state_dict["text_projection"].shape[1]
        context_length = clip_state_dict["positional_embedding"].shape[0]
        vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
        transformer_width = clip_state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))

        show_log(task_config, "\t embed_dim: {}".format(embed_dim))
        show_log(task_config, "\t image_resolution: {}".format(image_resolution))
        show_log(task_config, "\t vision_layers: {}".format(vision_layers))
        show_log(task_config, "\t vision_width: {}".format(vision_width))
        show_log(task_config, "\t vision_patch_size: {}".format(vision_patch_size))
        show_log(task_config, "\t context_length: {}".format(context_length))
        show_log(task_config, "\t vocab_size: {}".format(vocab_size))
        show_log(task_config, "\t transformer_width: {}".format(transformer_width))
        show_log(task_config, "\t transformer_heads: {}".format(transformer_heads))
        show_log(task_config, "\t transformer_layers: {}".format(transformer_layers))
        
        #########
        self.patches = (image_resolution//vision_patch_size)**2
        ########

        self.linear_patch = '2d'
        if hasattr(task_config, "linear_patch"):
            self.linear_patch = task_config.linear_patch
            show_log(task_config, "\t\t linear_patch: {}".format(self.linear_patch))

        # use .float() to avoid overflow/underflow from fp16 weight. https://github.com/openai/CLIP/issues/40
        cut_top_layer = 0
        show_log(task_config, "\t cut_top_layer: {}".format(cut_top_layer))
        self.clip = CLIP(
            embed_dim,
            image_resolution, vision_layers-cut_top_layer, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers-cut_top_layer,
            linear_patch=self.linear_patch
        ).float()

        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in clip_state_dict:
                del clip_state_dict[key]

        convert_weights(self.clip)
        # <=== End of CLIP Encoders

        self.sim_header = 'meanP' #"seqTransf"
        if hasattr(task_config, "sim_header"):
            self.sim_header = task_config.sim_header
            show_log(task_config, "\t sim_header: {}".format(self.sim_header))
        if self.sim_header == "tightTransf": assert self.loose_type is False

        cross_config.max_position_embeddings = context_length
        if self.loose_type is False:
            # Cross Encoder ===>
            cross_config = update_attr("cross_config", cross_config, "num_hidden_layers", self.task_config, "cross_num_hidden_layers")
            self.cross = CrossModel(cross_config)
            # <=== End of Cross Encoder
            self.similarity_dense = nn.Linear(cross_config.hidden_size, 1)

        if self.sim_header == "seqLSTM" or self.sim_header == "seqTransf":
            self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings, cross_config.hidden_size)
        if self.sim_header == "seqTransf":
            self.transformerClip = TransformerClip(width=transformer_width, layers=self.task_config.cross_num_hidden_layers,
                                                   heads=transformer_heads, )
        if self.sim_header == "seqLSTM":
            self.lstm_visual = nn.LSTM(input_size=cross_config.hidden_size, hidden_size=cross_config.hidden_size,
                                       batch_first=True, bidirectional=False, num_layers=1)

        num_words = task_config.max_words
        num_frames = self.task_config.max_frames

        # recommend set True
        self.use_original_clip_for_frame_features = True    

        # for coarse-grained constrast weights
        self.global_mat_weight = nn.parameter.Parameter(torch.eye(embed_dim), requires_grad=True)

        # for cross-grained constrast weights
        # self.word_logit_weight = nn.parameter.Parameter(torch.eye(num_words), requires_grad=True)
        self.frame_logit_weight = nn.parameter.Parameter(torch.eye(num_frames), requires_grad=True)        

        # for fine-grained constrast weights
        # self.local_mat_weight = nn.parameter.Parameter(torch.eye(embed_dim), requires_grad=True)
        # self.frame_mat_weight = nn.parameter.Parameter(torch.eye(num_frames), requires_grad=True)
        # self.word_mat_weight = nn.parameter.Parameter(torch.eye(num_words), requires_grad=True)
        # self.frame_mat_weight2 = nn.parameter.Parameter(torch.eye(num_frames), requires_grad=True)
        # self.word_mat_weight2 = nn.parameter.Parameter(torch.eye(num_words), requires_grad=True)

        self.loss_fct = CrossEn()
        # self.visual_token_selector = PerturbedTopK(10)

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, video, video_mask=None,multi_augmentation=False):
        if multi_augmentation:
            input_ids = input_ids.squeeze(2)
            token_type_ids = token_type_ids.squeeze(2)
            attention_mask = attention_mask.squeeze(2)
            video_mask = video_mask.view(-1, video_mask.shape[-1])
        else:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        # T x 3 x H x W
        video = torch.as_tensor(video).float()
        b, pair, bs, ts, channel, h, w = video.shape
        video = video.view(b * pair * bs * ts, channel, h, w)
        video_frame = bs * ts

        # [bs, 1, dim], [bs, num_words, dim], [bs, num_frames, dim]

        if multi_augmentation:
            pass
        else:
            (sequence_output, seq_features), visual_output, visual_patches = self.get_sequence_visual_output(input_ids, token_type_ids, attention_mask, 
                                                                    video, video_mask, shaped=True, video_frame=video_frame,multi_augmentation=False)

            if self.training:
                TIB_output,Intra_feature_pearson_constrain_loss = self.get_similarity_logits(sequence_output, seq_features, visual_output, visual_patches, attention_mask, 
                                            video_mask, shaped=True, loose_type=self.loose_type)
                return TIB_output,Intra_feature_pearson_constrain_loss
            else:
                return None

    def get_sequence_output(self, input_ids, token_type_ids, attention_mask, shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        bs_pair = input_ids.size(0)
        sequence_hidden, seq_features = self.clip.encode_text(input_ids, return_hidden=True)
        sequence_hidden, seq_features = sequence_hidden.float(), seq_features.float()
        sequence_hidden = sequence_hidden.view(bs_pair, -1, sequence_hidden.size(-1))

        return sequence_hidden, seq_features

    def get_visual_output(self, video, video_mask, shaped=False, video_frame=-1):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        bs_pair = video_mask.size(0)
        visual_hidden, visual_patches = self.clip.encode_image(video, video_frame=video_frame,return_hidden=True)

        visual_hidden = visual_hidden.float()
        visual_patches = visual_patches.float()

        visual_patches = visual_patches[:, 1:, :]
        # print(visual_patches.shape)
        visual_hidden = visual_hidden.view(bs_pair, -1, visual_hidden.size(-1))
        visual_patches = visual_patches.view(bs_pair, -1 , self.patches, visual_hidden.size(-1))
        # print("visual_patches.shape1",visual_patches.shape) # B F P D
        return visual_hidden, visual_patches

    def get_sequence_visual_output(self, input_ids, token_type_ids, attention_mask, video, video_mask, shaped=False, video_frame=-1,multi_augmentation=False):
        if multi_augmentation:
            if shaped is False:
                input_ids = input_ids.squeeze(2)
                token_type_ids = token_type_ids.squeeze(2)
                attention_mask = attention_mask.squeeze(2)

                video_mask = video_mask.view(-1, video_mask.shape[-1])
                video = torch.as_tensor(video).float()
                b, pair, bs, ts, channel, h, w = video.shape
                video = video.view(b * pair * bs * ts, channel, h, w)
                video_frame = bs * ts
            B,multi,dim = input_ids.shape
            sequence_output_list = []
            seq_features_list = []
            for i in range(multi):
                sequence_output, seq_features = self.get_sequence_output(input_ids[:,i], token_type_ids[:,i], attention_mask[:,i], shaped=True)
                sequence_output_list.append(sequence_output)
                seq_features_list.append(seq_features)
            
            visual_output,visual_patches = self.get_visual_output(video, video_mask, shaped=True, video_frame=video_frame) 

            return (sequence_output_list, seq_features_list), visual_output,visual_patches

        else:
            if shaped is False:
                input_ids = input_ids.view(-1, input_ids.shape[-1])
                token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
                attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
                video_mask = video_mask.view(-1, video_mask.shape[-1])

                video = torch.as_tensor(video).float()
                b, pair, bs, ts, channel, h, w = video.shape
                video = video.view(b * pair * bs * ts, channel, h, w)
                video_frame = bs * ts

            sequence_output, seq_features = self.get_sequence_output(input_ids, token_type_ids, attention_mask, shaped=True) # [bs, 1, dim], [bs, num_words, dim]
            visual_output,visual_patches = self.get_visual_output(video, video_mask, shaped=True, video_frame=video_frame)                  # [bs, num_frames, dim]

            return (sequence_output, seq_features), visual_output,visual_patches

    def _get_cross_output(self, sequence_output, visual_output, attention_mask, video_mask):

        concat_features = torch.cat((sequence_output, visual_output), dim=1)  # concatnate tokens and frames
        concat_mask = torch.cat((attention_mask, video_mask), dim=1)
        text_type_ = torch.zeros_like(attention_mask)
        video_type_ = torch.ones_like(video_mask)
        concat_type = torch.cat((text_type_, video_type_), dim=1)

        cross_layers, pooled_output = self.cross(concat_features, concat_type, concat_mask, output_all_encoded_layers=True)
        cross_output = cross_layers[-1]

        return cross_output, pooled_output, concat_mask

    def _mean_pooling_for_similarity_sequence(self, sequence_output, attention_mask):
        attention_mask_un = attention_mask.to(dtype=torch.float).unsqueeze(-1)
        attention_mask_un[:, 0, :] = 0.
        sequence_output = sequence_output * attention_mask_un
        text_out = torch.sum(sequence_output, dim=1) / torch.sum(attention_mask_un, dim=1, dtype=torch.float)
        return text_out

    def _mean_pooling_for_similarity_visual(self, visual_output, video_mask,):
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum
        return video_out

    def _mean_pooling_for_similarity(self, sequence_output, visual_output, attention_mask, video_mask,):
        text_out = self._mean_pooling_for_similarity_sequence(sequence_output, attention_mask)
        video_out = self._mean_pooling_for_similarity_visual(visual_output, video_mask)

        return text_out, video_out

    def _loose_similarity(self, sequence_output, seq_features, visual_output, visual_patches, attention_mask, video_mask, sim_header="meanP",start_patch_eval=False):
        """
            sequence_output: CLS token of text       # [bs, 1, dim]
            seq_features: all tokens of text         # [bs, num_words, dim]
            visual_output: all frames of video       # [bs, num_frames, dim]
        """
        ##########################Process################################
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()
        visual_patches = visual_patches.contiguous()

        # Sequential type: Transformer Encoder
        visual_output_original = visual_output
        seq_length = visual_output.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=visual_output.device)
        position_ids = position_ids.unsqueeze(0).expand(visual_output.size(0), -1)
        # position_ids : [bs, num_frames, dim]
        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        visual_output = visual_output + frame_position_embeddings


        extended_video_mask = (1.0 - video_mask.unsqueeze(1)) * -1000000.0
        extended_video_mask = extended_video_mask.expand(-1, video_mask.size(1), -1)
        
        visual_output = visual_output.permute(1, 0, 2)  # NLD -> LND
        visual_output = self.transformerClip(visual_output, extended_video_mask)
        visual_output = visual_output.permute(1, 0, 2)  # LND -> NLD
        visual_output = visual_output + visual_output_original

        #patch-level visual feature
        visual_patches = visual_patches / visual_patches.norm(dim=-1, keepdim=True)
        # B,F,P,D = visual_patches.shape
        # visual_patches = visual_patches.reshape(B*F*P,D)

        
        # video-level visual feature 
        visual_output_norm = visual_output / visual_output.norm(dim=-1, keepdim=True)
        video_output = self._mean_pooling_for_similarity_visual(visual_output_norm, video_mask)
        video_output = video_output / video_output.norm(dim=-1, keepdim=True)                    # [bs, dim]

        # frame-level visual features       
        if self.use_original_clip_for_frame_features:
            frame_features = visual_output_original / visual_output_original.norm(dim=-1, keepdim=True)                # [bs, num_frames, dim]
        else:
            frame_features = visual_output / visual_output.norm(dim=-1, keepdim=True)                                  # [bs, num_frames, dim]

        # sentence-level textual feature
        sentence_output = sequence_output.squeeze(1)
        sentence_output  = sentence_output / sentence_output.norm(dim=-1, keepdim=True)          # [bs, dim]
        

        logit_scale = self.clip.logit_scale.exp()

        if self.training:
            video_output = allgather(video_output, self.task_config)
            frame_features = allgather(frame_features, self.task_config)
            sentence_output = allgather(sentence_output, self.task_config)
            # word_features = allgather(word_features, self.task_config)
            visual_output_norm = allgather(visual_output_norm, self.task_config)
            visual_patches = allgather(visual_patches, self.task_config)
            torch.distributed.barrier()

        ##########################Sim Matrix################################
        # video-sentence score 
        sentence_video_logits = logit_scale * torch.matmul(torch.matmul(sentence_output, self.global_mat_weight), video_output.t())

        #  sentence-frame score 
        sentence_frame_weak_logits = logit_scale * torch.sum(torch.matmul(sentence_output, frame_features.permute(0, 2, 1)) \
            * torch.matmul(torch.softmax(torch.matmul(sentence_output, frame_features.permute(0, 2, 1)) / 1e-2, dim=-1), self.frame_logit_weight), dim=-1).t()
        
        # patch-sentence score
        B,F,P,D = visual_patches.shape
        visual_patches_vector = visual_patches.reshape(B*F*P,D)
        BT,D = sentence_output.shape
        sentence_patches_init = torch.matmul(sentence_output, visual_patches_vector.t()).reshape(BT,B,F*P) # BT,BFP  BT,B,F*P
        sentence_patches_sim = sentence_patches_init.div(0.01).softmax(-1)
        sentence_patches_weak_logits = logit_scale * torch.sum(sentence_patches_init*sentence_patches_sim, dim=-1)


        if self.training:
            #strong sentence-frame score 
            sentence2frames_sim = torch.matmul(sentence_output, frame_features.permute(0, 2, 1)) #BT,D BDF -->B,BT,F  # / 1e-2
            sentence2frames_sim = torch.diagonal(sentence2frames_sim).T #FB-->BF
            sentence2frames_sim = sentence2frames_sim.unsqueeze(2)
            sentence2frames_sim = sentence2frames_sim.detach()

            visual_output_adapt = (visual_output_norm * sentence2frames_sim.div(0.1).softmax(1)).sum(1)
            visual_output_adapt = visual_output_adapt / visual_output_adapt.norm(dim=-1, keepdim=True)
            sentence_frame_strong_logits = logit_scale * torch.matmul(sentence_output, visual_output_adapt.t())


            #strong sentence-patch score 
            sentence2patches_sim = sentence_patches_sim.detach() #BT,B,F*P
            sentence2patches_sim = sentence2patches_sim.permute(1,0,2) #B,BT,F*P
            sentence2patches_sim = torch.diagonal(sentence2patches_sim).T #B,F*P
            sentence2patches_sim = sentence2patches_sim.reshape(B,F,P,1)
            #B,F,P,D = visual_patches.shape

            visual_output_adapt_patch = torch.sum((visual_patches * sentence2patches_sim),dim=(1,2))
            visual_output_adapt_patch = visual_output_adapt_patch / visual_output_adapt_patch.norm(dim=-1, keepdim=True)
            sentence_patch_strong_logits = logit_scale * torch.matmul(sentence_output, visual_output_adapt_patch.t())



            text_feat = sentence_output
            cdcr_alpha1 = 0.14
            cdcr_alpha2 = 0.0
            z_a_norm = (text_feat - text_feat.mean(0)) / text_feat.std(0)  # BxD
            # cross-correlation matrix
            B, D = z_a_norm.shape


            # cdcl_video_level 
            video_feat = video_output
            z_b_norm = (video_feat - video_feat.mean(0)) / video_feat.std(0)  # BxD
            c = torch.matmul(z_a_norm.t(),z_b_norm) / B
            #c = torch.einsum('bm,bn->mn', z_a_norm, z_b_norm) / B  # DxD
            # loss
            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
            off_diag = c.flatten()[1:].view(D - 1, D + 1)[:, :-1].pow_(2).sum()
            cdcr_loss_video_level = (on_diag * cdcr_alpha1 + off_diag * cdcr_alpha2)

            ## cdcl_frame_level 
            video_feat_2 = visual_output_adapt
            z_b_norm_2 = (video_feat_2 - video_feat_2.mean(0)) / video_feat_2.std(0)  # BxD
            c = torch.matmul(z_a_norm.t(),z_b_norm_2) / B
            #c = torch.einsum('bm,bn->mn', z_a_norm, z_b_norm_2) / B  # DxD
            # loss
            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
            off_diag = c.flatten()[1:].view(D - 1, D + 1)[:, :-1].pow_(2).sum()
            cdcr_loss_frame_level = (on_diag * cdcr_alpha1 + off_diag * cdcr_alpha2)

            ### cdcl_video_level 
            video_feat_3 = visual_output_adapt_patch
            z_b_norm_3 = (video_feat_3 - video_feat_3.mean(0)) / video_feat_3.std(0)  # BxD
            c = torch.matmul(z_a_norm.t(),z_b_norm_3) / B
            #c = torch.einsum('bm,bn->mn', z_a_norm, z_b_norm_2) / B  # DxD
            # loss
            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
            off_diag = c.flatten()[1:].view(D - 1, D + 1)[:, :-1].pow_(2).sum()
            cdcr_loss_patch_level = (on_diag * cdcr_alpha1 + off_diag * cdcr_alpha2)

            TIB_output = [sentence_video_logits,sentence_frame_weak_logits,sentence_frame_strong_logits,sentence_patches_weak_logits,sentence_patch_strong_logits]
            Intra_feature_pearson_constrain_loss = [cdcr_loss_video_level,cdcr_loss_frame_level,cdcr_loss_patch_level]

            return TIB_output,Intra_feature_pearson_constrain_loss
            #return sentence_video_logits,sentence_frame_weak_logits,sentence_frame_strong_logits,sentence_patches_weak_logits,sentence_patch_strong_logits,cdcr_loss_video_level,cdcr_loss_frame_level,cdcr_loss_patch_level
        else:
            sentence2frames_sim = torch.matmul(sentence_output, frame_features.permute(0, 2, 1)) # B,C,F
            sentence2frames_sim = sentence2frames_sim.div(0.1).softmax(-1)
            sentence2frames_sim = sentence2frames_sim.permute(1,0,2) # C,B,F

            C,D = sentence_output.shape
            B,F,D = visual_output_norm.shape
            C,B,F =  sentence2frames_sim.shape
            wait = torch.zeros((C,B,F,D)).to(visual_output_norm.device)
            for i in range(C):
                tmpsentence2frames_sim = sentence2frames_sim[i].unsqueeze(2) # BF1  BFD
                tmpvisual_output = tmpsentence2frames_sim * visual_output_norm #BFD  
                wait[i] = tmpvisual_output
            tmp = wait
            tmp = tmp.sum(2)# CBD
            tmp = tmp / tmp.norm(dim=-1, keepdim=True) #CBD
            wait2 = torch.zeros((C,B)).to(visual_output_norm.device)
            for i in range(C):
                tmpsentence_output=sentence_output[i].unsqueeze(1)
                tmptmp = tmp[i]
                logit = logit_scale * tmptmp.matmul(tmpsentence_output).reshape(-1)
                wait2[i] = logit
            sentence_frame_strong_logits = wait2

            if start_patch_eval:
                sentence2patches_sim = sentence_patches_sim.detach() #C,B,F*P
                C,D = sentence_output.shape
                B,F,P,D = visual_patches.shape
                visual_patches_reshape = visual_patches.reshape(B,F*P,D)
                B,FP,D = visual_patches_reshape.shape
                C,B,FP =  sentence2patches_sim.shape

                wait = torch.zeros((C,B,FP,D)).to(visual_patches_reshape.device)
                for i in range(C):
                    tmpsentence2patches_sim = sentence2patches_sim[i].unsqueeze(2) # B,FP,1  
                    tmpvisual_output = tmpsentence2patches_sim * visual_patches_reshape #B,FP,D  
                    wait[i] = tmpvisual_output
                tmp = wait
                tmp = tmp.sum(2)# CBD
                tmp = tmp / tmp.norm(dim=-1, keepdim=True) #CBD
                wait2 = torch.zeros((C,B)).to(visual_patches_reshape.device)
                for i in range(C):
                    tmpsentence_output=sentence_output[i].unsqueeze(1)
                    tmptmp = tmp[i]
                    logit = logit_scale * tmptmp.matmul(tmpsentence_output).reshape(-1)
                    wait2[i] = logit
                sentence_patch_strong_logits = wait2
            else:
                sentence_patch_strong_logits = sentence_frame_strong_logits

            TIB_output = [sentence_video_logits,sentence_frame_weak_logits,sentence_frame_strong_logits,sentence_patches_weak_logits,sentence_patch_strong_logits]
            Intra_feature_pearson_constrain_loss = [[],[],[]]
            return TIB_output,Intra_feature_pearson_constrain_loss
            #return sentence_video_logits,sentence_frame_weak_logits,sentence_frame_strong_logits,sentence_patches_weak_logits,sentence_patch_strong_logits,[],[],[]


    def _attenion_over_fine_grained_sim_matrix(self, word_features, frame_features):
        bs_video, num_frames, dim_video = frame_features.shape
        bs_text, num_words, dim_text = word_features.shape
        fine_grained_sim_scores = torch.matmul(torch.matmul(word_features.view(-1, dim_text), self.local_mat_weight), frame_features.view(-1, dim_video).t()).view(bs_text, num_words, bs_video, num_frames)  # [bs_text, num_words, bs_video, num_frames]

        word_level_logit = torch.sum(torch.matmul(torch.softmax(fine_grained_sim_scores/1e-2, dim=1).permute(0,2,3,1), self.word_mat_weight).permute(0,3,1,2) * fine_grained_sim_scores, dim=1)               # [bs_text, bs_video, num_frames]
        frame_level_logit = torch.sum(torch.matmul(torch.softmax(fine_grained_sim_scores/1e-2, dim=-1), self.frame_mat_weight) * fine_grained_sim_scores, dim=-1)                                             # [bs_text, num_words, bs_video]

        sent2frame_logits = torch.sum(torch.matmul(torch.softmax(word_level_logit/1e-2, dim=-1),self.frame_mat_weight2) * word_level_logit, dim=-1)                                # [bs_text, bs_video]
        video2word_logits = torch.sum(torch.matmul(torch.softmax(frame_level_logit/1e-2, dim=1).permute(0,2,1), self.word_mat_weight2).permute(0,2,1) * frame_level_logit, dim=1)  # [bs_text, bs_video]

        return (sent2frame_logits + video2word_logits) / 2

    def _cross_similarity(self, sequence_output, visual_output, attention_mask, video_mask):
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()

        b_text, s_text, h_text = sequence_output.size()
        b_visual, s_visual, h_visual = visual_output.size()

        retrieve_logits_list = []

        step_size = b_text      # set smaller to reduce memory cost
        split_size = [step_size] * (b_text // step_size)
        release_size = b_text - sum(split_size)
        if release_size > 0:
            split_size += [release_size]

        # due to clip text branch retrun the last hidden
        attention_mask = torch.ones(sequence_output.size(0), 1)\
            .to(device=attention_mask.device, dtype=attention_mask.dtype)

        sequence_output_splits = torch.split(sequence_output, split_size, dim=0)
        attention_mask_splits = torch.split(attention_mask, split_size, dim=0)
        for i in range(len(split_size)):
            sequence_output_row = sequence_output_splits[i]
            attention_mask_row = attention_mask_splits[i]
            sequence_output_l = sequence_output_row.unsqueeze(1).repeat(1, b_visual, 1, 1)
            sequence_output_l = sequence_output_l.view(-1, s_text, h_text)
            attention_mask_l = attention_mask_row.unsqueeze(1).repeat(1, b_visual, 1)
            attention_mask_l = attention_mask_l.view(-1, s_text)

            step_truth = sequence_output_row.size(0)
            visual_output_r = visual_output.unsqueeze(0).repeat(step_truth, 1, 1, 1)
            visual_output_r = visual_output_r.view(-1, s_visual, h_visual)
            video_mask_r = video_mask.unsqueeze(0).repeat(step_truth, 1, 1)
            video_mask_r = video_mask_r.view(-1, s_visual)

            cross_output, pooled_output, concat_mask = \
                self._get_cross_output(sequence_output_l, visual_output_r, attention_mask_l, video_mask_r)
            retrieve_logits_row = self.similarity_dense(pooled_output).squeeze(-1).view(step_truth, b_visual)

            retrieve_logits_list.append(retrieve_logits_row)

        retrieve_logits = torch.cat(retrieve_logits_list, dim=0)
        return retrieve_logits

    def get_similarity_logits(self, sequence_output, seq_features, visual_output, visual_patches, attention_mask, video_mask, shaped=False, loose_type=False,start_patch_eval=False):
        if shaped is False:
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        contrastive_direction = ()
        if loose_type:
            assert self.sim_header in ["seqTransf"]
            TIB_output,Intra_feature_pearson_constrain_loss = self._loose_similarity(sequence_output, seq_features, visual_output, visual_patches, attention_mask, video_mask, sim_header=self.sim_header,start_patch_eval=start_patch_eval)
            #sentence_video_logits,sentence_frame_weak_logits,sentence_frame_strong_logits,sentence_patches_weak_logits,sentence_patch_strong_logits,cdcr_loss_video_level,cdcr_loss_frame_level,cdcr_loss_patch_level = self._loose_similarity(sequence_output, seq_features, visual_output, visual_patches, attention_mask, video_mask, sim_header=self.sim_header,start_patch_eval=start_patch_eval)

        return TIB_output,Intra_feature_pearson_constrain_loss

    def get_vis_info(self, sequence_output, seq_features, visual_output, visual_patches, attention_mask, video_mask):
        """
            sequence_output: CLS token of text       # [bs, 1, dim]
            seq_features: all tokens of text         # [bs, num_words, dim]
            visual_output: all frames of video       # [bs, num_frames, dim]
        """

        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])

        ##########################Process################################
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()
        visual_patches = visual_patches.contiguous()

        # Sequential type: Transformer Encoder
        visual_output_original = visual_output
        seq_length = visual_output.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=visual_output.device)
        position_ids = position_ids.unsqueeze(0).expand(visual_output.size(0), -1)
        # position_ids : [bs, num_frames, dim]
        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        visual_output = visual_output + frame_position_embeddings


        extended_video_mask = (1.0 - video_mask.unsqueeze(1)) * -1000000.0
        extended_video_mask = extended_video_mask.expand(-1, video_mask.size(1), -1)
        
        visual_output = visual_output.permute(1, 0, 2)  # NLD -> LND
        visual_output = self.transformerClip(visual_output, extended_video_mask)
        visual_output = visual_output.permute(1, 0, 2)  # LND -> NLD
        visual_output = visual_output + visual_output_original

        #patch-level visual feature
        visual_patches = visual_patches / visual_patches.norm(dim=-1, keepdim=True)
        # B,F,P,D = visual_patches.shape
        # visual_patches = visual_patches.reshape(B*F*P,D)

        
        # video-level visual feature 
        visual_output_norm = visual_output / visual_output.norm(dim=-1, keepdim=True)
        video_output = self._mean_pooling_for_similarity_visual(visual_output_norm, video_mask)
        video_output = video_output / video_output.norm(dim=-1, keepdim=True)                    # [bs, dim]

        # frame-level visual features       
        if self.use_original_clip_for_frame_features:
            frame_features = visual_output_original / visual_output_original.norm(dim=-1, keepdim=True)                # [bs, num_frames, dim]
        else:
            frame_features = visual_output / visual_output.norm(dim=-1, keepdim=True)                                  # [bs, num_frames, dim]


        # sentence-level textual feature
        sentence_output = sequence_output.squeeze(1)
        sentence_output  = sentence_output / sentence_output.norm(dim=-1, keepdim=True)          # [bs, dim]
        

        logit_scale = self.clip.logit_scale.exp()

        ##########################Sim Matrix################################
        # video-sentence score 
        sentence_video_logits = logit_scale * torch.matmul(torch.matmul(sentence_output, self.global_mat_weight), video_output.t())


        #  sentence-frame score 
        
        # patch-sentence score
        B,F,P,D = visual_patches.shape
        visual_patches_vector = visual_patches.reshape(B*F*P,D)
        BFP,D = visual_patches_vector.shape
        BT,D = sentence_output.shape
        B,F,D = frame_features.shape


        sentence_patches_init_new = torch.zeros((BT,B,F,P)).to(visual_patches.device)

        for i in range(BT):
            for j in range(B):
                tmp_patches = visual_patches[j] #F,P,D
                tmp_frame = frame_features[j] #F,D

                tmp_wait = torch.zeros((F,P)).to(visual_patches.device)

                for k in range(F):
                    tmp_wait[k] = torch.matmul(tmp_patches[k],tmp_frame[k].reshape(D,1)).reshape(P)
                sentence_patches_init_new[i,j] = tmp_wait



        sentence_patches_init = torch.matmul(sentence_output, visual_patches_vector.t()).reshape(BT,B,F*P) # BT,BFP  BT,B,F*P
        sentence_patches_sim = sentence_patches_init.div(0.1).softmax(-1)
        sentence_patch_logits = logit_scale * torch.sum(sentence_patches_init*sentence_patches_sim, dim=-1)
        #sentence_patches_sim = sentence_patches_sim.reshape(BT,B,F,P)

        sentence_patches_sim_new = sentence_patches_init_new.div(0.1).softmax(-1)
        sentence_patches_sim_new = sentence_patches_sim_new.reshape(BT,B,F,P)


    
        sentence2frames_sim = torch.matmul(sentence_output, frame_features.permute(0, 2, 1)) # B,C,F

        sentence_frame_logits  = logit_scale * torch.sum(torch.matmul(sentence_output, frame_features.permute(0, 2, 1)) \
            * torch.matmul(torch.softmax(torch.matmul(sentence_output, frame_features.permute(0, 2, 1)) / 1e-2, dim=-1), self.frame_logit_weight), dim=-1).t()

        sentence2frames_sim = sentence2frames_sim.div(0.03).softmax(-1)
        sentence2frames_sim = sentence2frames_sim.permute(1,0,2) # C,B,F


        return sentence_video_logits,sentence_frame_logits,sentence_patch_logits,sentence2frames_sim,sentence_patches_sim_new