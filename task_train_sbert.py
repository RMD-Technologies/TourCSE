import luigi
import pandas as pd
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformerTrainingArguments, SentenceTransformerTrainer, losses, util
from sentence_transformers.training_args import BatchSamplers
from peft import LoraConfig, TaskType

import random

random.seed(42)

class TaskTrainSBERT(luigi.Task):

    domain = luigi.ChoiceParameter(choices=['restaurant', "hotel"])
    sbert = luigi.Parameter(default="intfloat/e5-base-v2")
    loss = luigi.ChoiceParameter(choices=['gist','mnrl'])
    is_lora = luigi.BoolParameter(default=False)
    batchs = luigi.IntParameter(default=8)
    epoch = luigi.IntParameter(default=1)
    dataset_train = luigi.Parameter()
    output_dir = luigi.Parameter()

       
    def run(self):

        df_train = pd.read_csv(self.dataset_train, delimiter='\t', header=0)

        
        for df in [df_train]:
            df['text1'] = df['text1'].astype(str)
            df['text2'] = df['text2'].astype(str)

        df_train['label'] = 1

        df_train = df_train.sample(frac=1).reset_index(drop=True)
        
        train_dataset = Dataset.from_dict(
            {
                'text1': df_train['text1'].tolist(),
                'text2': df_train['text2'].tolist(),
                'label': df_train['label'].tolist()
            }
        )    
       
        model = SentenceTransformer(self.sbert)
       
        if self.is_lora:

            target_modules = ["query", "key", "value"]
            if 'mpnet' in self.sbert:
                target_modules = ["q", "k", "v", "o"]

            peft_config = LoraConfig(
                target_modules=target_modules,
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                r=32,
                lora_alpha=32,
                lora_dropout=0.05
            )
        
            model.add_adapter(peft_config)

        loss = None
        if self.loss == 'gist':
            guide = SentenceTransformer(self.sbert)
            loss = losses.GISTEmbedLoss(model=model, guide=guide, temperature=0.05)
        else:
            loss = losses.MultipleNegativesRankingLoss(model=model, scale=20, similarity_fct=util.cos_sim)
          
        training_args = SentenceTransformerTrainingArguments(
                output_dir=self.output().path,
                num_train_epochs=self.epoch,
                learning_rate=5e-5,
                per_device_train_batch_size=self.batchs,
                save_total_limit=1,
                bf16=True,
                save_steps=0.1,
                batch_sampler= BatchSamplers.NO_DUPLICATES,
            )

        trainer = SentenceTransformerTrainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    loss=loss,
        )

        trainer.train()

        model.save_pretrained(self.output().path)

    def output(self):
        return luigi.LocalTarget(f'data/sbert/{self.domain}/{self.output_dir}')

