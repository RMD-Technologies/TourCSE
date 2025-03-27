import luigi
from tqdm import tqdm
import nltk
import csv

from task_extract_review import TaskExtractReview
from gensim.utils import simple_preprocess

import random

random.seed(42)

class TaskPreprocessPretrain(luigi.Task):

    domain = luigi.ChoiceParameter(choices=["restaurant", "hotel"])
    dev = luigi.FloatParameter(default=0.01) # 1% as dev

    def requires(self):
        return TaskExtractReview(self.domain)

    def run(self):

        sentences = []

        with self.input().open() as f:
            
            reader = csv.DictReader(f, delimiter='\t')

            total_lines = sum(1 for _ in f) - 1
            f.seek(0) 

            for row in tqdm(reader, total=total_lines, desc=f"Processing {self.domain}"):

                review = row.get('review', '')
                sent_clean = simple_preprocess(review)
                if len(sent_clean) > 3:
                    sentences.append(' '.join(sent_clean))
        
        random.shuffle(sentences)

        dev_size = int(len(sentences) * self.dev)

        # Split the data
        dev_sentences = sentences[:dev_size]
        train_sentences = sentences[dev_size:]

        with self.output()['train'].open('w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(train_sentences)
        
        with self.output()['dev'].open('w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(dev_sentences)

    def output(self):
        return {
            'train': luigi.LocalTarget(f'data/pretrain/{self.domain}_train.tsv'),
            'dev': luigi.LocalTarget(f'data/pretrain/{self.domain}_dev.tsv')
        }
    
if __name__ == '__main__':
    tasks = [
        TaskPreprocessPretrain(domain=domain) for domain in ["hotel", 'restaurant']
    ]
    luigi.build(tasks, local_scheduler=True)