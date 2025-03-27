import luigi
from glob import glob
from tqdm import tqdm
import nltk
import csv

import gzip
import sys
from datetime import datetime

import random

random.seed(42)

from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    Trainer,
    TrainingArguments,
)

from task_preprocess_pretrain import TaskPreprocessPretrain

class TaskTrainMLM(luigi.Task):

    domain = luigi.ChoiceParameter(choices=["restaurant", "hotel"])
    model_path = luigi.Parameter("google-bert/bert-base-uncased")
    output_dir = luigi.Parameter()

    def requires(self):
        return TaskPreprocessPretrain(self.domain)

    def run(self):

        # save_steps = 1000  # Save model every 1k steps
        per_device_train_batch_size = 16
        num_train_epochs = 1  # Number of epochs
        max_length = 320  # Max length for a text input
        do_whole_word_mask = True  # If set to true, whole words are masked
        mlm_prob = 0.15  # Probability that a word is replaced by a [MASK] token

        # Load the model
        model = AutoModelForMaskedLM.from_pretrained(self.model_path)
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)


        output_dir = self.output().path
        print("Save checkpoints to:", output_dir)


        ##### Load our training datasets

        train_sentences = []
        with self.input()['train'].open('r') as f:
            for line in f:
                train_sentences.append(line.strip())
        
        dev_sentences = []
        with self.input()['dev'].open('r') as f:
            for line in f:
                dev_sentences.append(line.strip()) 
    
        # A dataset wrapper, that tokenizes our data on-the-fly
        class TokenizedSentencesDataset:
            def __init__(self, sentences, tokenizer, max_length, cache_tokenization=False):
                self.tokenizer = tokenizer
                self.sentences = sentences
                self.max_length = max_length
                self.cache_tokenization = cache_tokenization

            def __getitem__(self, item):
                if not self.cache_tokenization:
                    return self.tokenizer(
                        self.sentences[item],
                        add_special_tokens=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_special_tokens_mask=True,
                        #return_attention_mask=True,  # Include attention mask

                    )

                if isinstance(self.sentences[item], str):
                    self.sentences[item] = self.tokenizer(
                        self.sentences[item],
                        add_special_tokens=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_special_tokens_mask=True,
                        #return_attention_mask=True,  # Include attention mask

                    )
                return self.sentences[item]

            def __len__(self):
                return len(self.sentences)


        train_dataset = TokenizedSentencesDataset(train_sentences, tokenizer, max_length)

        dev_dataset = (
            TokenizedSentencesDataset(dev_sentences, tokenizer, max_length, cache_tokenization=True)
            if len(dev_sentences) > 0
            else None
        )


        ##### Training arguments

        if do_whole_word_mask:
            data_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_prob)
        else:
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_prob)

        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=3e-5,
            overwrite_output_dir=True,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            eval_strategy='steps',
            eval_steps=0.1,
            save_steps=0.1,
            save_total_limit=1,
            prediction_loss_only=True,
            bf16=True,
            seed=42
        )

        trainer = Trainer(
            model=model, args=training_args, data_collator=data_collator, train_dataset=train_dataset, eval_dataset=dev_dataset
        )

        print("Save tokenizer to:", output_dir)
        tokenizer.save_pretrained(output_dir)

        trainer.train()

        print("Save model to:", output_dir)
        model.save_pretrained(output_dir)

        print("Training done")

    def output(self):
        return luigi.LocalTarget(f'data/model/mlm/{self.domain}/{self.output_dir}')
    
if __name__ == '__main__':
    tasks = [
        TaskTrainMLM(domain="hotel", output_dir="bert-hotel-mlm"),
    ]
    luigi.build(tasks, local_scheduler=True)