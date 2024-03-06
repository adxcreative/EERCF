from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import random
import os
from metrics import compute_metrics, tensor_text_to_video_metrics, tensor_video_to_text_sim
import time
import argparse
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling_fineclip_patches_cdcl import XCLIP
from modules.optimization import BertAdam
from tqdm import tqdm
from modules.until_module import CrossEn

from util import parallel_apply, get_logger
from dataloaders.data_dataloaders import DATALOADER_DICT

torch.distributed.init_process_group(backend="nccl")

global logger

def cal_new(main_sim_matrix,rerank_sim_matrix,args,tem):

    main_sim_matrix = torch.tensor(main_sim_matrix)
    rerank_sim_matrix = torch.tensor(rerank_sim_matrix)


    main_sim_matrix = main_sim_matrix * F.softmax(main_sim_matrix/tem, dim=0)*len(main_sim_matrix) 
    rerank_sim_matrix = rerank_sim_matrix * F.softmax(rerank_sim_matrix/tem, dim=0)*len(rerank_sim_matrix) 


    main_sim_matrix = main_sim_matrix.numpy()
    rerank_sim_matrix = rerank_sim_matrix.numpy()


    ind = np.argsort(main_sim_matrix,axis=1) #选取前topk进行rerank，rerank使用rerank_sim_matrix
    ind = ind[:,int((-1*args.rerantopk)):]


    for i in range(len(main_sim_matrix)):
        main_sim_matrix[i][ind[i]] = rerank_sim_matrix[i][ind[i]]+100

    return main_sim_matrix


def np_softmax(X, theta = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats. 
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the 
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """
    # make X at least 2d
    y = np.atleast_2d(X)
    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)
    # multiply y against the theta parameter, 
    y = y * float(theta)
    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    # exponentiate y
    y = np.exp(y)
    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)
    # finally: divide elementwise
    p = y / ax_sum
    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()
    return p

def get_args(description='FineCLIP on Retrieval Task'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_pretrain", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    

    parser.add_argument('--video_weight', type=float, default=0.5, help='')
    parser.add_argument('--weakframe_weight', type=float, default=0.5, help='')
    parser.add_argument('--strongframe_weight', type=float, default=1.0, help='')
    parser.add_argument('--weakpatch_weight', type=float, default=0.05, help='')
    parser.add_argument('--strongpatch_weight', type=float, default=0.0, help='')
    parser.add_argument('--cdcl_video_weight', type=float, default=1.0, help='')
    parser.add_argument('--cdcl_frame_weight', type=float, default=1.0, help='')
    parser.add_argument('--cdcl_patch_weight', type=float, default=0.0, help='')
    parser.add_argument('--weak_loss_weight', type=float, default=0.8, help='')
    parser.add_argument('--strong_loss_weight', type=float, default=0.2, help='')
    parser.add_argument('--cdcl_loss_weight', type=float, default=0.001, help='')
    parser.add_argument('--rerantopk', type=int, default=50, help='')

    parser.add_argument('--train_csv', type=str, default='data/.train.csv', help='')
    parser.add_argument('--val_csv', type=str, default='data/.val.csv', help='')
    parser.add_argument('--test_csv', type=str, default='data/.test.csv', help='')
    parser.add_argument('--data_path', type=str, default='data/caption.pickle', help='data pickle file path')
    parser.add_argument('--features_path', type=str, default='data/videos_feature.pickle', help='feature path')

    parser.add_argument('--num_thread_reader', type=int, default=1, help='')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=3500, help='batch size eval')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate exp epoch decay')
    parser.add_argument('--n_display', type=int, default=100, help='Information display frequence')
    parser.add_argument('--video_dim', type=int, default=1024, help='video feature dimension')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_words', type=int, default=20, help='')
    parser.add_argument('--max_frames', type=int, default=100, help='')
    parser.add_argument('--feature_framerate', type=int, default=1, help='')
    parser.add_argument('--margin', type=float, default=0.1, help='margin for loss')
    parser.add_argument('--hard_negative_rate', type=float, default=0.5, help='rate of intra negative sample')
    parser.add_argument('--negative_weighting', type=int, default=1, help='Weight the loss for intra negative')
    parser.add_argument('--n_pair', type=int, default=1, help='Num of pair to output from data loader')

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument("--resume_model", default=None, type=str, required=False, help="Resume train model.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--task_type", default="retrieval", type=str, help="Point the task `retrieval` to finetune.")
    parser.add_argument("--datatype", default="msrvtt", type=str, help="Point the dataset to finetune.")

    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
    parser.add_argument("--rank", default=0, type=int, help="distribted training")
    parser.add_argument('--coef_lr', type=float, default=1., help='coefficient for bert branch.')
    parser.add_argument('--use_mil', action='store_true', help="Whether use MIL as Miech et. al. (2020).")
    parser.add_argument('--sampled_use_mil', action='store_true', help="Whether MIL, has a high priority than use_mil.")

    parser.add_argument('--text_num_hidden_layers', type=int, default=12, help="Layer NO. of text.")
    parser.add_argument('--visual_num_hidden_layers', type=int, default=12, help="Layer NO. of visual.")
    parser.add_argument('--cross_num_hidden_layers', type=int, default=4, help="Layer NO. of cross.")

    parser.add_argument('--loose_type', action='store_true', help="Default using tight type for retrieval.")
    parser.add_argument('--expand_msrvtt_sentences', action='store_true', help="")

    parser.add_argument('--train_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")
    parser.add_argument('--eval_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")

    parser.add_argument('--freeze_layer_num', type=int, default=0, help="Layer NO. of CLIP need to freeze.")
    parser.add_argument('--slice_framepos', type=int, default=0, choices=[0, 1, 2],
                        help="0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.")
    parser.add_argument('--linear_patch', type=str, default="2d", choices=["2d", "3d"],
                        help="linear projection of flattened patches.")
    parser.add_argument('--sim_header', type=str, default="meanP",
                        choices=["meanP", "seqLSTM", "seqTransf", "tightTransf"],
                        help="choice a similarity header.")

    parser.add_argument("--pretrained_clip_name", default="ViT-B/32", type=str, help="Choose a CLIP version")
    parser.add_argument("--multi_augmentation", action='store_true', help="multiaug augmentation")
    parser.add_argument('--multi_data_path', type=str, default='data/caption.pickle', help='data pickle file path')


    args = parser.parse_args()

    if args.sim_header == "tightTransf":
        args.loose_type = False

    # Check paramenters
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    return args

def set_seed_logger(args):
    global logger
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(args.local_rank)
    args.world_size = world_size
    rank = torch.distributed.get_rank()
    args.rank = rank

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logger = get_logger(os.path.join(args.output_dir, "log.txt"))

    if args.local_rank == 0:
        logger.info("Effective parameters:")
        for key in sorted(args.__dict__):
            logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    return args

def init_device(args, local_rank):
    global logger

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)

    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    args.n_gpu = n_gpu

    if args.batch_size % args.n_gpu != 0 or args.batch_size_val % args.n_gpu != 0:
        raise ValueError("Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
            args.batch_size, args.n_gpu, args.batch_size_val, args.n_gpu))

    return device, n_gpu

def init_model(args, device, n_gpu, local_rank):

    if args.init_model:
        print(args.init_model)
        model_state_dict = torch.load(args.init_model, map_location='cpu')
    else:
        model_state_dict = None

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    # "cross-base", pytorch_pretrained_bert,init_model_state_dict,args
    model = XCLIP.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

    model.to(device)

    return model

def prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, local_rank, coef_lr=1.):

    if hasattr(model, 'module'):
        model = model.module

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    no_decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    decay_clip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." in n]
    decay_noclip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." not in n]

    no_decay_clip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." in n]
    no_decay_noclip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." not in n]

    weight_decay = 0.2
    optimizer_grouped_parameters = [
        {'params': [p for n, p in decay_clip_param_tp], 'weight_decay': weight_decay, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in decay_noclip_param_tp], 'weight_decay': weight_decay},
        {'params': [p for n, p in no_decay_clip_param_tp], 'weight_decay': 0.0, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in no_decay_noclip_param_tp], 'weight_decay': 0.0}
    ]

    scheduler = None
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_proportion,
                         schedule='warmup_cosine', b1=0.9, b2=0.98, e=1e-6,
                         t_total=num_train_optimization_steps, weight_decay=weight_decay,
                         max_grad_norm=1.0)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=True)

    return optimizer, scheduler, model

def save_model(epoch, args, model, optimizer, tr_loss, type_name=""):
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(
        args.output_dir, "pytorch_model.bin.{}{}".format("" if type_name=="" else type_name+".", epoch))
    optimizer_state_file = os.path.join(
        args.output_dir, "pytorch_opt.bin.{}{}".format("" if type_name=="" else type_name+".", epoch))
    torch.save(model_to_save.state_dict(), output_model_file)
    torch.save({
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': tr_loss,
            }, optimizer_state_file)
    logger.info("Model saved to %s", output_model_file)
    logger.info("Optimizer saved to %s", optimizer_state_file)
    return output_model_file

def load_model(epoch, args, n_gpu, device, model_file=None):
    if model_file is None or len(model_file) == 0:
        model_file = os.path.join(args.output_dir, "pytorch_model.bin.{}".format(epoch))
    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location='cpu')
        if args.local_rank == 0:
            logger.info("Model loaded from %s", model_file)
        # Prepare model, bert cache dir 
        cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
        model = XCLIP.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

        model.to(device)
    else:
        model = None
    return model

def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer, scheduler, global_step, local_rank=0):
    global logger
    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    start_time = time.time()
    total_loss = 0
    # logger.info("start train")
    loss_fct = CrossEn()
    # feat_loss_fun = torch.nn.SmoothL1Loss(reduction='mean')


    for step, batch in enumerate(train_dataloader):
        if n_gpu == 1:
            # multi-gpu does scattering it-self
            batch = tuple(t.to(device=device, non_blocking=True) for t in batch)
        
        multi_input_ids, multi_input_mask, multi_segment_ids, video, video_mask = batch

        input_ids = multi_input_ids[:,0]
        input_mask = multi_input_mask[:,0]
        segment_ids = multi_segment_ids[:,0]
        
        loss = 0.0
        TIB_output,Intra_feature_pearson_constrain_loss =model(input_ids, segment_ids, input_mask, video, video_mask,multi_augmentation=False)
        sentence_video_logits,sentence_frame_weak_logits,sentence_frame_strong_logits,sentence_patches_weak_logits,sentence_patch_strong_logits = TIB_output
        cdcr_loss_video_level,cdcr_loss_frame_level,cdcr_loss_patch_level = Intra_feature_pearson_constrain_loss
        

        sim_maxtrix_weak = sentence_video_logits * args.video_weight + \
                    sentence_frame_weak_logits *  args.weakframe_weight + \
                    sentence_patches_weak_logits * args.weakpatch_weight
        
        sim_maxtrix_strong = sentence_frame_strong_logits*args.strongframe_weight + sentence_patch_strong_logits*args.strongpatch_weight
        

        sim_weak_loss1 = loss_fct(sim_maxtrix_weak)
        sim_weak_loss2 = loss_fct(sim_maxtrix_weak.T)
        sim_weak_loss = (sim_weak_loss1 + sim_weak_loss2) / 2

        sim_strong_loss1 = loss_fct(sim_maxtrix_strong)
        sim_strong_loss2 = loss_fct(sim_maxtrix_strong.T)
        sim_strong_loss = (sim_strong_loss1 + sim_strong_loss2) / 2


        intra_loss = cdcr_loss_video_level * args.cdcl_video_weight \
                    + cdcr_loss_frame_level * args.cdcl_frame_weight \
                    + cdcr_loss_patch_level* args.cdcl_patch_weight
        inter_loss = sim_weak_loss * args.weak_loss_weight + sim_strong_loss * args.strong_loss_weight

        loss = inter_loss + intra_loss * args.cdcl_loss_weight


        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()

        total_loss += float(loss)
        if (step + 1) % args.gradient_accumulation_steps == 0:

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if scheduler is not None:
                scheduler.step()  # Update learning rate schedule

            optimizer.step()
            optimizer.zero_grad()

            # https://github.com/openai/CLIP/issues/46
            if hasattr(model, 'module'):
                torch.clamp_(model.module.clip.logit_scale.data, max=np.log(100))
            else:
                torch.clamp_(model.clip.logit_scale.data, max=np.log(100))

            global_step += 1
            if global_step % log_step == 0 and local_rank == 0:
                
                logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f, WeakLoss: %f, StrongLoss %f, CDCLLoss %f, Time/step: %f", epoch + 1,
                            args.epochs, step + 1,
                            len(train_dataloader), "-".join([str('%.9f'%itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                            float(loss),
                            float(sim_weak_loss),
                            float(sim_strong_loss),
                            float(intra_loss),
                            (time.time() - start_time) / (log_step * args.gradient_accumulation_steps))
                start_time = time.time()

    total_loss = total_loss / len(train_dataloader)
    return total_loss, global_step


def _run_on_single_gpu(args, model, batch_list_t, batch_list_v, batch_sequence_output_list, batch_seq_features_list, batch_visual_output_list,batch_visual_patches_list):
    main_sim_matrix = []
    video_sim_matrix = []
    weakframe_sim_matrix = []
    strongframe_sim_matrix = []
    weakpatch_sim_matrix = []
    strongpatch_sim_matrix = []
    rerank_sim_matrix = []

    print(len(batch_list_t),len(batch_list_v))
    for idx1, b1 in tqdm(enumerate(batch_list_t)):
        input_mask, segment_ids, *_tmp = b1
        sequence_output = batch_sequence_output_list[idx1]
        seq_features = batch_seq_features_list[idx1]
        main_each_row = []
        video_each_row = []
        weakframe_each_row = []
        strongframe_each_row = []
        weakpatch_each_row = []
        strongpatch_each_row = []
        rerank_each_row = []

        for idx2, b2 in enumerate(batch_list_v):
            video_mask, *_tmp = b2
            visual_output = batch_visual_output_list[idx2]
            visual_patches = batch_visual_patches_list[idx2]

            if args.strongpatch_weight>0:
                TIB_output,_ = model.get_similarity_logits(sequence_output, seq_features, visual_output, visual_patches, input_mask, video_mask,
                                                                     loose_type=model.loose_type,start_patch_eval=True)
                sentence_video_logits,sentence_frame_weak_logits,sentence_frame_strong_logits,sentence_patches_weak_logits,sentence_patch_strong_logits = TIB_output
            else:
                TIB_output,_ = model.get_similarity_logits(sequence_output, seq_features, visual_output, visual_patches, input_mask, video_mask,
                                                                     loose_type=model.loose_type,start_patch_eval=False)
                sentence_video_logits,sentence_frame_weak_logits,sentence_frame_strong_logits,sentence_patches_weak_logits,sentence_patch_strong_logits = TIB_output

            sentence_video_logits = sentence_video_logits.cpu().detach().numpy()
            sentence_frame_weak_logits = sentence_frame_weak_logits.cpu().detach().numpy()
            sentence_frame_strong_logits = sentence_frame_strong_logits.cpu().detach().numpy()
            sentence_patches_weak_logits = sentence_patches_weak_logits.cpu().detach().numpy()
            sentence_patch_strong_logits = sentence_patch_strong_logits.cpu().detach().numpy()

            rerank_logits = sentence_video_logits * args.video_weight + sentence_frame_weak_logits * args.weakframe_weight + sentence_patches_weak_logits * args.weakpatch_weight
            add_logits = sentence_frame_strong_logits * args.strongframe_weight + sentence_patch_strong_logits * args.strongpatch_weight
            rerank_logits = rerank_logits * args.weak_loss_weight + add_logits * args.strong_loss_weight


            main_logits = sentence_video_logits
            

            main_each_row.append(main_logits)
            video_each_row.append(sentence_video_logits)
            weakframe_each_row.append(sentence_frame_weak_logits)
            strongframe_each_row.append(sentence_frame_strong_logits)
            weakpatch_each_row.append(sentence_patches_weak_logits)
            strongpatch_each_row.append(sentence_patch_strong_logits)
            rerank_each_row.append(rerank_logits)


        main_each_row = np.concatenate(tuple(main_each_row), axis=-1)
        video_each_row = np.concatenate(tuple(video_each_row), axis=-1)
        weakframe_each_row = np.concatenate(tuple(weakframe_each_row), axis=-1)
        strongframe_each_row = np.concatenate(tuple(strongframe_each_row), axis=-1)
        weakpatch_each_row = np.concatenate(tuple(weakpatch_each_row), axis=-1)
        strongpatch_each_row = np.concatenate(tuple(strongpatch_each_row), axis=-1)
        rerank_each_row = np.concatenate(tuple(rerank_each_row), axis=-1)

        main_sim_matrix.append(main_each_row)
        video_sim_matrix.append(video_each_row)
        weakframe_sim_matrix.append(weakframe_each_row)
        strongframe_sim_matrix.append(strongframe_each_row)
        weakpatch_sim_matrix.append(weakpatch_each_row)
        strongpatch_sim_matrix.append(strongpatch_each_row)
        rerank_sim_matrix.append(rerank_each_row)

    
    main_sim_matrix = np.concatenate(tuple(main_sim_matrix), axis=0)
    video_sim_matrix = np.concatenate(tuple(video_sim_matrix), axis=0)
    weakframe_sim_matrix = np.concatenate(tuple(weakframe_sim_matrix), axis=0)
    strongframe_sim_matrix = np.concatenate(tuple(strongframe_sim_matrix), axis=0)
    weakpatch_sim_matrix = np.concatenate(tuple(weakpatch_sim_matrix), axis=0)
    strongpatch_sim_matrix = np.concatenate(tuple(strongpatch_sim_matrix), axis=0)
    rerank_sim_matrix = np.concatenate(tuple(rerank_sim_matrix), axis=0)

    return main_sim_matrix,video_sim_matrix,weakframe_sim_matrix,strongframe_sim_matrix,weakpatch_sim_matrix,strongpatch_sim_matrix,rerank_sim_matrix


def change_shape(sim_matrix,cut_off_points_clone):
    cut_off_points2len_ = [itm + 1 for itm in cut_off_points_clone]
    max_length = max([e_-s_ for s_, e_ in zip([0]+cut_off_points2len_[:-1], cut_off_points2len_)])
    sim_matrix_new = []
    for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_):
        sim_matrix_new.append(np.concatenate((sim_matrix[s_:e_],
                                            np.full((max_length-e_+s_, sim_matrix.shape[1]), -np.inf)), axis=0))
    sim_matrix_new = np.stack(tuple(sim_matrix_new), axis=0)
    return sim_matrix_new

def eval_epoch(args, model, test_dataloader, device):

    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)

    # #################################################################
    ## below variables are used to multi-sentences retrieval
    # multi_sentence_: important tag for eval
    # cut_off_points: used to tag the label when calculate the metric
    # sentence_num: used to cut the sentence representation
    # video_num: used to cut the video representation
    # #################################################################
    multi_sentence_ = False
    cut_off_points_, sentence_num_, video_num_ = [], -1, -1
    if hasattr(test_dataloader.dataset, 'multi_sentence_per_video') \
            and test_dataloader.dataset.multi_sentence_per_video:
        multi_sentence_ = True
        cut_off_points_ = test_dataloader.dataset.cut_off_points
        sentence_num_ = test_dataloader.dataset.sentence_num
        video_num_ = test_dataloader.dataset.video_num
        cut_off_points_ = [itm - 1 for itm in cut_off_points_]

    if multi_sentence_:
        logger.warning("Eval under the multi-sentence per video clip setting.")
        logger.warning("sentence num: {}, video num: {}".format(sentence_num_, video_num_))

    model.eval()
    with torch.no_grad():
        batch_list_t = []
        batch_list_v = []
        batch_sequence_output_list, batch_visual_output_list = [], []
        batch_seq_features_list = []
        batch_visual_patches_list = []
        total_video_num = 0

        # ----------------------------
        # 1. cache the features
        # ----------------------------
        for bid, batch in enumerate(test_dataloader): # Maybe something went wrong here!!!
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, video, video_mask = batch

            if multi_sentence_:
                # multi-sentences retrieval means: one clip has two or more descriptions.
                b, *_t = video.shape
                sequence_output, seq_features = model.get_sequence_output(input_ids, segment_ids, input_mask)
                batch_sequence_output_list.append(sequence_output)
                batch_seq_features_list.append(seq_features)
                batch_list_t.append((input_mask, segment_ids,))

                s_, e_ = total_video_num, total_video_num + b
                filter_inds = [itm - s_ for itm in cut_off_points_ if itm >= s_ and itm < e_]

                if len(filter_inds) > 0:
                    video, video_mask = video[filter_inds, ...], video_mask[filter_inds, ...]
                    visual_output, visual_patches = model.get_visual_output(video, video_mask)
                    batch_visual_output_list.append(visual_output)
                    batch_visual_patches_list.append(visual_patches)
                    batch_list_v.append((video_mask,))
                total_video_num += b
            else:
                (sequence_output, seq_features), visual_output,visual_patches = model.get_sequence_visual_output(input_ids, segment_ids, input_mask, video, video_mask)

                batch_sequence_output_list.append(sequence_output)
                batch_seq_features_list.append(seq_features)
                batch_list_t.append((input_mask, segment_ids,))
                batch_visual_output_list.append(visual_output)
                batch_list_v.append((video_mask,))
                batch_visual_patches_list.append(visual_patches)

            logger.info("{}/{}\r".format(bid, len(test_dataloader)))

        # ----------------------------------
        # 2. calculate the similarity
        # ----------------------------------
        main_sim_matrix,video_sim_matrix,weakframe_sim_matrix,strongframe_sim_matrix,weakpatch_sim_matrix,strongpatch_sim_matrix,rerank_sim_matrix = _run_on_single_gpu(args, model, batch_list_t, batch_list_v, batch_sequence_output_list, batch_seq_features_list, batch_visual_output_list, batch_visual_patches_list)

    #before   13980 x 1398
    #after  1398 x 10 x 1398

    np.save(os.path.join(args.output_dir,"sim_matrix.npy"),main_sim_matrix)
    print("start save sim_matrix {}.".format(main_sim_matrix.shape))

    dsl_start = False
    cross_dsl = False

    # if dsl_start and not dsl_text:
    #     main_sim_matrix = main_sim_matrix.T
    #     rerank_sim_matrix = rerank_sim_matrix.T

    if cross_dsl:
        tmp_main_sim_matrix_1 = cal_new(main_sim_matrix,rerank_sim_matrix,args,1.0) #1.2
        tmp_main_sim_matrix_2 = cal_new(main_sim_matrix.T,rerank_sim_matrix.T,args,1.0) #1.3
        main_sim_matrix = tmp_main_sim_matrix_1 #tmp_main_sim_matrix_2.T
        
    else:
        if dsl_start:
            main_sim_matrix = torch.tensor(main_sim_matrix)
            rerank_sim_matrix = torch.tensor(rerank_sim_matrix)

            main_sim_matrix = main_sim_matrix * F.softmax(main_sim_matrix/1.0, dim=0)*len(main_sim_matrix) 
            rerank_sim_matrix = rerank_sim_matrix * F.softmax(rerank_sim_matrix/1.0, dim=0)*len(rerank_sim_matrix) 

            main_sim_matrix = main_sim_matrix.numpy()
            rerank_sim_matrix = rerank_sim_matrix.numpy()

        ind = np.argsort(main_sim_matrix,axis=1) #选取前topk进行rerank，rerank使用rerank_sim_matrix
        ind = ind[:,int((-1*args.rerantopk)):]
        ii,jj,kk = 0.8,0.2,0.5#1.0,0.2,0.6  #0.8,0.2,0.5
        for i in range(len(main_sim_matrix)): #前topk + 100用于提高优先级
            main_sim_matrix[i][ind[i]] = rerank_sim_matrix[i][ind[i]]+100
        ind = np.argsort(main_sim_matrix,axis=1)
        ind = ind[:,int((-1*args.rerantopk)):]
        for i in range(len(main_sim_matrix)): #前topk 重排序
             main_sim_matrix[i][ind[i]] = 100.0 + weakframe_sim_matrix[i][ind[i]]*ii + (1-ii)*strongframe_sim_matrix[i][ind[i]] + jj*weakpatch_sim_matrix[i][ind[i]] + kk* video_sim_matrix[i][ind[i]]


    sim_matrix = main_sim_matrix
    if multi_sentence_:
        logger.info("before reshape, sim matrix size: {} x {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
        cut_off_points2len_ = [itm + 1 for itm in cut_off_points_]
        max_length = max([e_-s_ for s_, e_ in zip([0]+cut_off_points2len_[:-1], cut_off_points2len_)])
        sim_matrix_new = []
        for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_):
            sim_matrix_new.append(np.concatenate((sim_matrix[s_:e_],
                                                np.full((max_length-e_+s_, sim_matrix.shape[1]), -np.inf)), axis=0))
        sim_matrix = np.stack(tuple(sim_matrix_new), axis=0)
        logger.info("after reshape, sim matrix size: {} x {} x {}".
                    format(sim_matrix.shape[0], sim_matrix.shape[1], sim_matrix.shape[2]))

        tv_metrics = tensor_text_to_video_metrics(sim_matrix)
        vt_metrics = compute_metrics(tensor_video_to_text_sim(sim_matrix))
    else:
        logger.info("sim matrix size: {}, {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
        tv_metrics = compute_metrics(sim_matrix)
        vt_metrics = compute_metrics(sim_matrix.T)
        logger.info('\t Length-T: {}, Length-V:{}'.format(len(sim_matrix), len(sim_matrix[0])))


    logger.info("Text-to-Video:")
    logger.info('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['MR'], tv_metrics['MeanR']))

    logger.info("Video-to-Text:")
    logger.info('\t>>>  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f} - V2T$Median R: {:.1f} - V2T$Mean R: {:.1f}'.
                format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['MR'], vt_metrics['MeanR']))

    R1 = tv_metrics['R1']+tv_metrics['R5']+tv_metrics['R10']
    return R1

def main():
    global logger
    args = get_args()
    args.output_dir = args.output_dir
    args = set_seed_logger(args)
    device, n_gpu = init_device(args, args.local_rank)

    tokenizer = ClipTokenizer()

    assert  args.task_type == "retrieval"
    model = init_model(args, device, n_gpu, args.local_rank)

    ## ####################################
    # freeze testing
    ## ####################################
    assert args.freeze_layer_num <= 12 and args.freeze_layer_num >= -1
    if hasattr(model, "clip") and args.freeze_layer_num > -1:
        for name, param in model.clip.named_parameters():
            # top layers always need to train
            if name.find("ln_final.") == 0 or name.find("text_projection") == 0 or name.find("logit_scale") == 0 \
                    or name.find("visual.ln_post.") == 0 or name.find("visual.proj") == 0:
                continue    # need to train
            elif name.find("visual.transformer.resblocks.") == 0 or name.find("transformer.resblocks.") == 0:
                layer_num = int(name.split(".resblocks.")[1].split(".")[0])
                if layer_num >= args.freeze_layer_num:
                    continue    # need to train

            if args.linear_patch == "3d" and name.find("conv2."):
                continue
            else:
                # paramenters which < freeze_layer_num will be freezed
                param.requires_grad = False

    ## ####################################
    # dataloader loading
    ## ####################################
    assert args.datatype in DATALOADER_DICT

    assert DATALOADER_DICT[args.datatype]["test"] is not None \
           or DATALOADER_DICT[args.datatype]["val"] is not None

    test_dataloader, test_length = None, 0
    if DATALOADER_DICT[args.datatype]["test"] is not None:
        test_dataloader, test_length = DATALOADER_DICT[args.datatype]["test"](args, tokenizer)

    if DATALOADER_DICT[args.datatype]["val"] is not None:
        val_dataloader, val_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer, subset="val")
    else:
        val_dataloader, val_length = test_dataloader, test_length

    ## report validation results if the ["test"] is None
    if test_dataloader is None:
        test_dataloader, test_length = val_dataloader, val_length
    
    if args.rank == 0:
        logger.info("***** Weight *****")
        logger.info("video_weight = {}".format(args.video_weight))
        logger.info("weakframe_weight = {}".format(args.weakframe_weight))
        logger.info("strongframe_weight = {}".format(args.strongframe_weight))
        logger.info("weakpatch_weight = {}".format(args.weakpatch_weight))
        logger.info("strongpatch_weight = {}".format(args.strongpatch_weight))
        logger.info("cdcl_video_weight = {}".format(args.cdcl_video_weight))
        logger.info("cdcl_frame_weight = {}".format(args.cdcl_frame_weight))
        logger.info("cdcl_patch_weight = {}".format(args.cdcl_patch_weight))
        logger.info("weak_loss_weight = {}".format(args.weak_loss_weight))
        logger.info("strong_loss_weight = {}".format(args.strong_loss_weight))
        logger.info("cdcl_loss_weight = {}".format(args.cdcl_loss_weight))
        logger.info("***** Running test *****")
        logger.info("  Num examples = %d", test_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d", len(test_dataloader))
        logger.info("***** Running val *****")
        logger.info("  Num examples = %d", val_length)


    ## ####################################
    # train and eval
    ## ####################################
    # if args.rank == 0:
    #     eval_epoch(args, model, test_dataloader, device, n_gpu)
    if args.do_train:
        if args.multi_augmentation:
            train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train_multi"](args, tokenizer)
        else:
            train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)
        # train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)
        num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                        / args.gradient_accumulation_steps) * args.epochs

        coef_lr = args.coef_lr
        optimizer, scheduler, model = prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, args.local_rank, coef_lr=coef_lr)

        if args.rank == 0:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", train_length)
            logger.info("  Batch size = %d", args.batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)

        best_score = 0.00001
        best_output_model_file = "None"
        ## ##############################################################
        # resume optimizer state besides loss to continue train
        ## ##############################################################
        logger.info("start resume model ")
        resumed_epoch = 0
        if args.resume_model:
            checkpoint = torch.load(args.resume_model, map_location='cpu')
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            resumed_epoch = checkpoint['epoch']+1
            resumed_loss = checkpoint['loss']
        logger.info("over")
        
        global_step = 0
        for epoch in range(resumed_epoch, args.epochs):
            train_sampler.set_epoch(epoch)
            tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer,
                                               scheduler, global_step, local_rank=args.local_rank)
            if args.rank == 0:
                logger.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.epochs, tr_loss)

                output_model_file = save_model(epoch, args, model, optimizer, tr_loss, type_name="")

                ## Run on val dataset for selecting best model.
                logger.info("Eval on val dataset")
                R1 = eval_epoch(args, model, val_dataloader, device, n_gpu)

                if best_score <= R1:
                    best_score = R1
                    best_output_model_file = output_model_file
                logger.info("The best model is: {}, the R1 is: {:.4f}".format(best_output_model_file, best_score))

        ## Test on the best checkpoint
        if args.rank == 0:
            model = load_model(-1, args, n_gpu, device, model_file=best_output_model_file)
            eval_epoch(args, model, test_dataloader, device)

    elif args.do_eval:
        if args.rank == 0:
            eval_epoch(args, model, test_dataloader, device)

if __name__ == "__main__":
    main()
