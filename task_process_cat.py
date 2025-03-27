import luigi
from tqdm import tqdm
import json
import csv
from collections import Counter
from collections import defaultdict
import pandas as pd

from cat_aspect_extraction import CAt, SoftmaxAttention
from reach import Reach
from task_extract_topic import TaskExtractTopic
from task_extract_sample import TaskExtractSample
from task_train_w2v import TaskTrainW2V

from utils.preprocess import preprocess
import nltk

class TaskProcessCAt(luigi.Task):

    domain = luigi.ChoiceParameter(choices=["restaurant", "hotel"])
    algo = luigi.ChoiceParameter(choices=["sg", "cbow"])
    K_topic = luigi.IntParameter()
    tresh = luigi.FloatParameter(default=.9)

    def requires(self):
        return {
            "sample": TaskExtractSample(self.domain),
            "topic": TaskExtractTopic(self.domain, self.algo, self.K_topic),
            "w2v": TaskTrainW2V(self.domain, self.algo) 
        }

    def run(self):

        r = Reach.load(self.input()['w2v'].path)
        

        with self.input()['topic'].open('r') as f:
            topics = json.load(f)
        

        cat = CAt(r)
        att = SoftmaxAttention()
        
        for topic in topics:
            cat.add_topic(topic, topics[topic])
            for aspect in topics[topic]:
                cat.add_candidate(aspect)

        df = pd.read_csv(self.input()["sample"].path, delimiter="\t")
        df = df[df["split"] == "train"]
        
        datasets = defaultdict(list)
        for rating, review in tqdm(zip(df["rating"].tolist(), df["review"].tolist()), total=len(df)):
            topics = set()
            for sentence in nltk.sent_tokenize(review):
                tokens = preprocess(sentence).split()
                if not tokens or len(tokens) < 3: continue
                scores = cat.get_scores(tokens, att)
                for t, v in scores:
                    if v < self.tresh: break
                    topics.add(t)
                key_template = "{rating}-{topics}"
                key = key_template.format(rating=rating,topics='.'.join(sorted(topics)))
                datasets[key].append(sentence)
        
        with self.output().open('w') as f:
            json.dump(datasets, f, ensure_ascii=False, indent=4)

    def output(self):
        return luigi.LocalTarget(f'data/cat/{self.domain}/{self.algo}_{self.K_topic}.json')
    
if __name__ == '__main__':
    tasks = [
        TaskProcessCAt(domain='domain', algo='sg', K_topic=250)
    ]
    luigi.build(tasks, local_scheduler=True)