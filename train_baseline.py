import datetime
import os
import logging
import datasets
import torch
import json

from torch import nn
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
import torch.distributed as dist
from transformers import HfArgumentParser
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, EarlyStoppingCallback, set_seed)
from datasets import load_metric

import pandas as pd
import numpy as np

from preprocessed_data.grover import GroverConstructor
from preprocessed_data.mnli import MnliConstructor
from utils.dataset import TokenizedDataset
from utils.configure import Configure
from utils.training_arguments import WrappedTrainingArguments
from utils.trainer import CustomTrainer

from data_gpt2.dataset import gpt2EncodedDataset, load_texts, Corpus
from data_gpt2.download import download
from data_gpt2.utils import distributed
# from utils.dataloader import build_dataloader
import random

def seed_everything(seed):
    torch.manual_seed(seed)       # Current CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    np.random.seed(seed)          # Numpy module
    random.seed(seed)             # Python random module
    torch.backends.cudnn.benchmark = False    # Close optimization
    torch.backends.cudnn.deterministic = True  # Close optimization
    torch.cuda.manual_seed_all(seed)  # All GPU (Optional)

logger = logging.getLogger(__name__)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    f1_metric = load_metric("f1")
    acc_metric = load_metric("accuracy")
    f1_result = f1_metric.compute(predictions=predictions, references=labels)
    acc_result = acc_metric.compute(predictions=predictions, references=labels)
    f1_result.update(acc_result)
    return f1_result

def load_datasets(human_portion, data_dir, real_dataset, fake_dataset):

    download(real_dataset, fake_dataset, data_dir=data_dir)

    real_corpus = Corpus(real_dataset, data_dir=data_dir)

    fake_corpus = Corpus(fake_dataset, data_dir=data_dir)

    real_train, real_valid, real_test = real_corpus.train, real_corpus.valid, real_corpus.test
    fake_train, fake_valid, fake_test = fake_corpus.train, fake_corpus.valid, fake_corpus.test

    portion = 0.1
    limited_train_len = portion * len(real_train)
    limited_valid_len = portion * len(real_valid)
    limited_test_len = portion * len(real_test)

    dic_real_train = []
    for item in real_train:
        dic_item = {'article': item,
                    'label': "human"}
        dic_real_train.append(dic_item)
        if len(dic_real_train)>=limited_train_len * human_portion * 2:
            break
    dic_real_valid = []
    for item in real_valid:
        dic_item = {'article': item,
                    'label': "human"}
        dic_real_valid.append(dic_item)
        if len(dic_real_valid)>=limited_valid_len:
            break
    dic_real_test = []
    for item in real_test:
        dic_item = {'article': item,
                    'label': "human"}
        dic_real_test.append(dic_item)
        if len(dic_real_test)>=limited_test_len:
            break

    dic_fake_train = []
    for item in fake_train:
        dic_item = {'article': item,
                    'label': "machine"}
        dic_fake_train.append(dic_item)
        if len(dic_fake_train)>=limited_train_len * (1 - human_portion) * 2:
            break
    dic_fake_valid = []
    for item in fake_valid:
        dic_item = {'article': item,
                    'label': "machine"}
        dic_fake_valid.append(dic_item)
        if len(dic_fake_valid)>=limited_valid_len:
            break
    dic_fake_test = []
    for item in fake_test:
        dic_item = {'article': item,
                    'label': "machine"}
        dic_fake_test.append(dic_item)
        if len(dic_fake_test)>=limited_test_len:
            break        

    dic_real_train.extend(dic_fake_train)
    dic_real_valid.extend(dic_fake_valid)
    dic_real_test.extend(dic_fake_test)
    random.shuffle(dic_real_train)
    return dic_real_train, dic_real_valid, dic_real_test


def main() -> None:
    # initialize the logger
    logging.basicConfig(level=logging.INFO)

    os.environ['CUDA_VISIBLE_DEVICES'] = "1,2"

    path = os.getcwd()

    # Specify the configuration file we want to use
    cfg_file_path = r"./model/baseline_gpt2.cfg"
    args = Configure.Get(cfg_file_path)
    print(args)
    output_dir = "temp_output"
    args.dataset.use_cache = True

    for seed_item in [2,4,5,10,12]:
        print('============= SEED {}============='.format(seed_item))
        seed_everything(seed_item)
        training_args = WrappedTrainingArguments(
            cfg=cfg_file_path,
            run_name="baseline_gpt2_gpt2_0.1",
            learning_rate=1e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=8,
            weight_decay=0.01,
            input_max_length=512,
            report_to="wandb",
            do_train=True,
            do_eval=True,
            do_predict=False,
            output_dir=output_dir,
            load_best_model_at_end=False,
            evaluation_strategy="epoch",
            # could be specified later in cfg file
            metric_for_best_model=args.evaluate.metric,
            logging_strategy="epoch",
            logging_first_step=True,
            save_strategy="no",
            # save_steps= 1000,
            save_total_limit=1,
            seed=seed_item,
        )
        training_args.num_layers = 5
        training_args.hidden_dims = 300
        training_args.out_dims = 100

        human_portion = 0.5
        temperature, contraloss_weight = 0.2, 0.6
        args.model.temperature, args.model.contraloss_weight, args.dataset.human_portion = temperature, contraloss_weight, human_portion
        print("=====human_portion = {}, temperature={}, contraloss_weight={}=====".format(
            human_portion, temperature, contraloss_weight))

        if "wandb" in training_args.report_to:
            import wandb

            # login to the wandb account
            wandb_api_key = "ed0df8bc480cd3515de03d77799ea0a1f36eb909"
            os.system("wandb login {}".format(wandb_api_key))

            init_args = {}
            if "MLFLOW_EXPERIMENT_ID" in os.environ:
                init_args["group"] = os.environ["MLFLOW_EXPERIMENT_ID"]
            wandb.init(
                project=os.getenv("WANDB_PROJECT", "text_detect2"),
                name=training_args.run_name +
                "_{}_s{}_h{}".format(
                    str(args.dataset.train_portion),seed_item, human_portion),
                entity=os.getenv("WANDB_ENTITY", 'yichen_wang'),
                reinit=True,
                **init_args,
            )
            wandb.config.update(training_args, allow_val_change=True)

        cache_root = os.path.join("preprocessed_data", "cache")
        os.makedirs(cache_root, exist_ok=True)

        # print("=====human_portion = {}, temperature={}, contraloss_weight={}=====".format(args.dataset.human_portion, args.model.temperature, args.model.contraloss_weight))
        if "grover" in args.dataset.loader_path:
            
            grover_raw_dataset: datasets.DatasetDict = datasets.load_dataset(path=args.dataset.loader_path,
                        cache_dir=args.dataset.cache_path)
            pre_train_set, pre_dev_set, pre_test_set = GroverConstructor(args).to_preprocessed(grover_raw_dataset,
                        cache_root=cache_root)

            """
            pre dev set[1]
            {'article': 'Screenshot by Katie Conner/CBS Interactive\n ...Now playing: Watch this: Apple Card FAQ: What you need to know\n', 
            'domain': 'cnet.com', 'title': '5 best ways to check your credit score from your phone or laptop', 'date': 'April 02, 2019', 'authors': 'Katie Conner', 'ind30k': '27333', 'url': 'https://www.cnet.com/how-to/5-best-ways-to-check-your-credit-score-from-your-phone-or-laptop/', 'label': 'human', 'orig_split': 'train_burner', 'split': 'val'}
            """
            train_idx_by_label = pre_train_set.get_train_idx_by_label(
                pre_train_set)
        elif "mnli" in args.dataset.loader_path:
            mnli_raw_dataset: datasets.DatasetDict = datasets.load_dataset(path=args.dataset.loader_path,
                cache_dir=args.dataset.cache_path)
            pre_train_set, pre_dev_set, pre_test_set = MnliConstructor(args).to_preprocessed(mnli_raw_dataset,
                                    cache_root=cache_root)

            train_idx_by_label = pre_train_set.get_train_idx_by_label(
                pre_train_set)
        else: #GPT2 dataset
            pre_train_set, pre_dev_set, pre_test_set = load_datasets(human_portion=args.dataset.human_portion, data_dir=args.dataset.loader_path, real_dataset='webtext', fake_dataset='xl-1542M')

        training_args.train_data_size = len(pre_train_set)

        with open("configure/task2num.json") as f:
            task2num = json.load(f)

        if args.model.task == "baseline":
            from model.base_model import BaseModel
            model = BaseModel(args, task2num)
        elif args.model.task == "bertcl":
            from model.BertCl import BertCl
            model = BertCl(args, training_args,
                        train_idx_by_label, pre_train_set, task2num)
        elif args.model.task == "scl":
            from model.SCL import SCl
            model = SCl(args, training_args, train_idx_by_label,
                        pre_train_set, task2num)
        else:
            raise ValueError("Model Type Not Supported! ")

        dataset_flag = {'train': 1, 'val': 2, 'test': 3}

        train_dataset = TokenizedDataset(
            args, training_args, model.tokenizer, pre_train_set, dataset_flag['train'])
        eval_dataset = TokenizedDataset(
            args, training_args, model.tokenizer, pre_dev_set, dataset_flag['val'])
        test_dataset = TokenizedDataset(
            args, training_args, model.tokenizer, pre_test_set, dataset_flag['test'])

        # early_stopping_callback = EarlyStoppingCallback(
        #     early_stopping_patience=args.seq2seq.patience if args.seq2seq.patience else 30)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=model.tokenizer,
            compute_metrics=compute_metrics,
            # wandb_run_dir=wandb.run.dir if "wandb" in training_args.report_to and training_args.local_rank <= 0 else None,
            callbacks=None #[early_stopping_callback],
        )
        print("Trainer build successfully. ")

        if training_args.do_train:
            train_result = trainer.train()
            # trainer.save_model()  # Saves the tokenizer too for easy upload

            metrics = train_result.metrics
            max_train_samples = len(train_dataset)
            metrics["train_samples"] = min(
                max_train_samples, len(train_dataset))

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

        if training_args.do_eval:
            logger.info("*** Evaluate ***")

            metrics = trainer.evaluate(
                metric_key_prefix="eval"
            )
            max_eval_samples = len(eval_dataset)
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

        if training_args.do_predict:
            logger.info("*** Predict ***")

            predict_results = trainer.predict(
                test_dataset=test_dataset,
                metric_key_prefix="predict"
            )
            metrics = predict_results.metrics
            max_predict_samples = len(test_dataset)
            metrics["predict_samples"] = min(
                max_predict_samples, len(test_dataset))

            trainer.log_metrics("predict", metrics)
            trainer.save_metrics("predict", metrics)

            save_dir = "outputs/bertcl_bs_gptdata_h{}t{}c{}_{}".format(round(args.dataset.human_portion*10), round(
                args.model.temperature*10), round(args.model.contraloss_weight*10), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')[2:])
            os.system("mkdir -p {}".format(save_dir))
            os.system("mv {}/all_results.json {}".format(output_dir, save_dir))
            os.system("rm -rf {}/*".format(output_dir))


if __name__ == "__main__":
    main()
