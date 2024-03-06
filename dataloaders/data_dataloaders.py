import torch
from torch.utils.data import DataLoader
# from dataloaders.dataloader_msrvtt_retrieval_concept import MSRVTT_DataLoader
from dataloaders.dataloader_msrvtt_retrieval import MSRVTT_TrainDataLoader,MSRVTT_TrainDataLoader_Multi,MSRVTT_TrainDataLoader_Caption,MSRVTT_TrainDataLoader_Multi_Negative
from dataloaders.dataloader_msrvtt_retrieval import MSRVTT_DataLoader
# from dataloaders.dataloader_msrvtt_retrieval_concept import MSRVTT_TrainDataLoader 
from dataloaders.dataloader_msrvtt_retrieval_mlm import MSRVTT_DataLoader_MLM,MSRVTT_TrainDataLoader_MLM,MSRVTT_TrainDataLoader_Gen
from dataloaders.dataloader_msvd_retrieval import MSVD_DataLoader,MSVD_DataLoader_Multi
# from dataloaders.dataloader_lsmdc_retrieval import LSMDC_DataLoader
from dataloaders.dataloader_activitynet_retrieval import ActivityNet_DataLoader
# from dataloaders.dataloader_didemo_retrieval import DiDeMo_DataLoader

#dataloader_msrvtt_train_concept

def dataloader_msrvtt_train_caption(args, tokenizer):
    msrvtt_dataset = MSRVTT_TrainDataLoader_Caption(
        csv_path=args.train_csv,
        json_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        unfold_sentences=args.expand_msrvtt_sentences,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
    )
    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )
    return dataloader, len(msrvtt_dataset)


def dataloader_msrvtt_train_retrieval(args, tokenizer):
    msrvtt_dataset = MSRVTT_TrainDataLoader(
        csv_path=args.train_csv,
        json_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        unfold_sentences=args.expand_msrvtt_sentences,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
    )
    # train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_dataset,num_replicas=args.world_size,rank=args.rank)
    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )
    return dataloader, len(msrvtt_dataset)


def dataloader_msrvtt_train_gen(args, tokenizer):
    msrvtt_dataset = MSRVTT_TrainDataLoader_Gen(
        csv_path=args.train_csv,
        json_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        unfold_sentences=args.expand_msrvtt_sentences,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
        mask_mode=args.mask_mode,
    )

    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )
    return dataloader, len(msrvtt_dataset)

def dataloader_msrvtt_train_mlm(args, tokenizer):
    msrvtt_dataset = MSRVTT_TrainDataLoader_MLM(
        csv_path=args.train_csv,
        json_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        unfold_sentences=args.expand_msrvtt_sentences,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
        # concept_word_vocab_path=args.concept_word_vocab_path
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_dataset)
    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=True,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )
    return dataloader, len(msrvtt_dataset), train_sampler

def dataloader_msrvtt_test_mlm(args, tokenizer,subset="test_mlm"):
    msrvtt_testset = MSRVTT_DataLoader_MLM(
        subset=subset,
        csv_path=args.test_csv,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
    )
    dataloader_msrvtt = DataLoader(
        msrvtt_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(msrvtt_testset)

def dataloader_msrvtt_val_mlm(args, tokenizer,subset="val_mlm"):
    msrvtt_valset = MSRVTT_DataLoader_MLM(
        subset=subset,
        csv_path=args.val_csv,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
    )
    dataloader_msrvtt = DataLoader(
        msrvtt_valset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(msrvtt_valset)

def dataloader_msrvtt_train_concept(args, tokenizer):
    msrvtt_dataset = MSRVTT_TrainDataLoader(
        csv_path=args.train_csv,
        json_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        unfold_sentences=args.expand_msrvtt_sentences,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
        concept_word_vocab_path=args.concept_word_vocab_path
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_dataset)
    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=True,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )
    return dataloader, len(msrvtt_dataset), train_sampler


def dataloader_msrvtt_train(args, tokenizer):
    msrvtt_dataset = MSRVTT_TrainDataLoader(
        csv_path=args.train_csv,
        json_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        unfold_sentences=args.expand_msrvtt_sentences,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
        # concept_word_vocab_path=args.concept_word_vocab_path
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_dataset,num_replicas=args.world_size,rank=args.rank)
    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=True,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )
    return dataloader, len(msrvtt_dataset), train_sampler

def dataloader_msrvtt_train_multi(args, tokenizer):
    print("prepare dataset")
    msrvtt_dataset = MSRVTT_TrainDataLoader_Multi(
        csv_path=args.train_csv,
        json_path=args.multi_data_path, #****
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        unfold_sentences=args.expand_msrvtt_sentences,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
        # concept_word_vocab_path=args.concept_word_vocab_path
    )
    print("prepare dataloader")
    train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_dataset,num_replicas=args.world_size,rank=args.rank)
    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=True,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )
    print("return dataloader")
    return dataloader, len(msrvtt_dataset), train_sampler

def dataloader_msrvtt_train_multi_negative(args, tokenizer):
    print("prepare dataset")
    msrvtt_dataset = MSRVTT_TrainDataLoader_Multi_Negative(
        csv_path=args.train_csv,
        json_path=args.multi_data_path, #****
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        unfold_sentences=args.expand_msrvtt_sentences,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
        # concept_word_vocab_path=args.concept_word_vocab_path
    )
    print("prepare dataloader")
    train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_dataset,num_replicas=args.world_size,rank=args.rank)
    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=True,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )
    print("return dataloader")
    return dataloader, len(msrvtt_dataset), train_sampler
# def dataloader_msrvtt_test(args, tokenizer, subset="test"):
#     msrvtt_testset = MSRVTT_DataLoader(
#         csv_path=args.val_csv,
#         features_path=args.features_path,
#         max_words=args.max_words,
#         feature_framerate=args.feature_framerate,
#         tokenizer=tokenizer,
#         max_frames=args.max_frames,
#         frame_order=args.eval_frame_order,
#         slice_framepos=args.slice_framepos,
#     )
#     dataloader_msrvtt = DataLoader(
#         msrvtt_testset,
#         batch_size=args.batch_size_val,
#         num_workers=args.num_thread_reader,
#         shuffle=False,
#         drop_last=False,
#     )
#     return dataloader_msrvtt, len(msrvtt_testset)

def dataloader_msrvtt_test(args, tokenizer, subset="test"):
    msrvtt_testset = MSRVTT_DataLoader(
        subset=subset,
        csv_path=args.test_csv,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
    )
    dataloader_msrvtt = DataLoader(
        msrvtt_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(msrvtt_testset)

def dataloader_msrvtt_val(args, tokenizer, subset="val"):
    msrvtt_valset = MSRVTT_DataLoader(
        subset=subset,
        csv_path=args.val_csv,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
    )
    dataloader_msrvtt = DataLoader(
        msrvtt_valset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(msrvtt_valset)

# def dataloader_msrvtt_val(args, tokenizer, subset="val"):
#     msrvtt_valset = MSRVTT_TrainDataLoader(
#         csv_path=args.val_csv,
#         json_path=args.data_path,
#         features_path=args.features_path,
#         max_words=args.max_words,
#         feature_framerate=args.feature_framerate,
#         tokenizer=tokenizer,
#         max_frames=args.max_frames,
#         frame_order=args.eval_frame_order,
#         slice_framepos=args.slice_framepos,
#         unfold_sentences=args.expand_msrvtt_sentences,
#     )
#     dataloader_msrvtt = DataLoader(
#         msrvtt_valset,
#         batch_size=args.batch_size_val,
#         num_workers=args.num_thread_reader,
#         shuffle=False,
#         drop_last=False,
#     )
#     return dataloader_msrvtt, len(msrvtt_valset)


# def dataloader_msrvtt_test(args, tokenizer, subset="test"):
#     msrvtt_testset = MSRVTT_TrainDataLoader(
#         csv_path=args.test_csv,
#         json_path=args.data_path,
#         features_path=args.features_path,
#         max_words=args.max_words,
#         feature_framerate=args.feature_framerate,
#         tokenizer=tokenizer,
#         max_frames=args.max_frames,
#         frame_order=args.eval_frame_order,
#         slice_framepos=args.slice_framepos,
#         unfold_sentences=args.expand_msrvtt_sentences,
#     )
#     dataloader_msrvtt = DataLoader(
#         msrvtt_testset,
#         batch_size=args.batch_size_val,
#         num_workers=args.num_thread_reader,
#         shuffle=False,
#         drop_last=False,
#     )
#     return dataloader_msrvtt, len(msrvtt_testset)


def dataloader_msvd_train(args, tokenizer):
    msvd_dataset = MSVD_DataLoader(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(msvd_dataset,num_replicas=args.world_size,rank=args.rank)
    dataloader = DataLoader(
        msvd_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=True,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(msvd_dataset), train_sampler

def dataloader_msvd_train_multi(args, tokenizer):
    msvd_dataset = MSVD_DataLoader_Multi(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(msvd_dataset,num_replicas=args.world_size,rank=args.rank)
    dataloader = DataLoader(
        msvd_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=True,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )
    return dataloader, len(msvd_dataset), train_sampler


def dataloader_msvd_test(args, tokenizer, subset="test"):
    msvd_testset = MSVD_DataLoader(
        subset=subset,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
    )
    dataloader_msrvtt = DataLoader(
        msvd_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(msvd_testset)


# def dataloader_lsmdc_train(args, tokenizer):
#     lsmdc_dataset = LSMDC_DataLoader(
#         subset="train",
#         data_path=args.data_path,
#         features_path=args.features_path,
#         max_words=args.max_words,
#         feature_framerate=args.feature_framerate,
#         tokenizer=tokenizer,
#         max_frames=args.max_frames,
#         frame_order=args.train_frame_order,
#         slice_framepos=args.slice_framepos,
#     )

#     train_sampler = torch.utils.data.distributed.DistributedSampler(lsmdc_dataset)
#     dataloader = DataLoader(
#         lsmdc_dataset,
#         batch_size=args.batch_size // args.n_gpu,
#         num_workers=args.num_thread_reader,
#         pin_memory=True,
#         shuffle=(train_sampler is None),
#         sampler=train_sampler,
#         drop_last=True,
#     )

#     return dataloader, len(lsmdc_dataset), train_sampler

# def dataloader_lsmdc_test(args, tokenizer, subset="test"):
#     lsmdc_testset = LSMDC_DataLoader(
#         subset=subset,
#         data_path=args.data_path,
#         features_path=args.features_path,
#         max_words=args.max_words,
#         feature_framerate=args.feature_framerate,
#         tokenizer=tokenizer,
#         max_frames=args.max_frames,
#         frame_order=args.eval_frame_order,
#         slice_framepos=args.slice_framepos,
#     )
#     dataloader_msrvtt = DataLoader(
#         lsmdc_testset,
#         batch_size=args.batch_size_val,
#         num_workers=args.num_thread_reader,
#         shuffle=False,
#         drop_last=False,
#     )
#     return dataloader_msrvtt, len(lsmdc_testset)


def dataloader_activity_train(args, tokenizer):
    activity_dataset = ActivityNet_DataLoader(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(activity_dataset,num_replicas=args.world_size,rank=args.rank)
    dataloader = DataLoader(
        activity_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=True,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(activity_dataset), train_sampler

def dataloader_activity_test(args, tokenizer, subset="test"):
    activity_testset = ActivityNet_DataLoader(
        subset=subset,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
    )
    dataloader_msrvtt = DataLoader(
        activity_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(activity_testset)


# def dataloader_didemo_train(args, tokenizer):
#     didemo_dataset = DiDeMo_DataLoader(
#         subset="train",
#         data_path=args.data_path,
#         features_path=args.features_path,
#         max_words=args.max_words,
#         feature_framerate=args.feature_framerate,
#         tokenizer=tokenizer,
#         max_frames=args.max_frames,
#         frame_order=args.train_frame_order,
#         slice_framepos=args.slice_framepos,
#     )

#     train_sampler = torch.utils.data.distributed.DistributedSampler(didemo_dataset)
#     dataloader = DataLoader(
#         didemo_dataset,
#         batch_size=args.batch_size // args.n_gpu,
#         num_workers=args.num_thread_reader,
#         pin_memory=True,
#         shuffle=(train_sampler is None),
#         sampler=train_sampler,
#         drop_last=True,
#     )

#     return dataloader, len(didemo_dataset), train_sampler

# def dataloader_didemo_test(args, tokenizer, subset="test"):
#     didemo_testset = DiDeMo_DataLoader(
#         subset=subset,
#         data_path=args.data_path,
#         features_path=args.features_path,
#         max_words=args.max_words,
#         feature_framerate=args.feature_framerate,
#         tokenizer=tokenizer,
#         max_frames=args.max_frames,
#         frame_order=args.eval_frame_order,
#         slice_framepos=args.slice_framepos,
#     )
#     dataloader_didemo = DataLoader(
#         didemo_testset,
#         batch_size=args.batch_size_val,
#         num_workers=args.num_thread_reader,
#         shuffle=False,
#         drop_last=False,
#     )
#     return dataloader_didemo, len(didemo_testset)


DATALOADER_DICT = {}
DATALOADER_DICT["msrvtt"] = {"train_caption":dataloader_msrvtt_train_caption,"train_retrieval":dataloader_msrvtt_train_retrieval,"train_gen":dataloader_msrvtt_train_gen,"train_mlm":dataloader_msrvtt_train_mlm,"val_mlm":dataloader_msrvtt_val_mlm, "test_mlm":dataloader_msrvtt_test_mlm,"train":dataloader_msrvtt_train,"train_multi":dataloader_msrvtt_train_multi,"train_multi_negative":dataloader_msrvtt_train_multi_negative, "val":dataloader_msrvtt_val, "test":dataloader_msrvtt_test}
DATALOADER_DICT["msvd"] = {"train_multi":dataloader_msvd_train_multi,"train":dataloader_msvd_train, "val":dataloader_msvd_test, "test":dataloader_msvd_test}
# DATALOADER_DICT["lsmdc"] = {"train":dataloader_lsmdc_train, "val":dataloader_lsmdc_test, "test":dataloader_lsmdc_test}
DATALOADER_DICT["activity"] = {"train":dataloader_activity_train, "val":dataloader_activity_test, "test":None}
# DATALOADER_DICT["didemo"] = {"train":dataloader_didemo_train, "val":dataloader_didemo_test, "test":dataloader_didemo_test}

