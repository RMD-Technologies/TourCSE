import luigi
import os
import pandas as pd

from task_extract_review import TaskExtractReview
from nltk import sent_tokenize
from gensim.utils import simple_preprocess
from tqdm import tqdm
import fasttext

class TaskExtractSample(luigi.Task):

    domain = luigi.ChoiceParameter(choices=["restaurant", "hotel"])
    sample_train = luigi.IntParameter(default=100_000)
    is_test = luigi.BoolParameter(default=False)

    def requires(self):
        return TaskExtractReview(self.domain)

    def run(self):

        output_path = self.output().path
        output_dir = os.path.dirname(output_path)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        reviews = []
        ratings = []

        if not self.is_test:

            df = pd.read_csv(self.input().path, delimiter='\t', encoding="utf-8")
            df = df.groupby('rating').apply(lambda x: x.sample(n=int((self.sample_train) // 5), random_state=42)).reset_index(drop=True)

            for review, rating in tqdm(zip(reviews, ratings), total=len(df)):
                if len(review) < 5: continue
                review_filter.append(review)
                ratings_filter.append(rating)

        else:
            df = pd.read_csv('data/raw/booking/Hotel_Reviews.csv', sep=",")
            df = df.sample(n=self.sample_train)
            clf_lg = fasttext.load_model('data/fasttext/lid.176.bin')

            def process(text):
                if len(text.split()) < 5: return
                lg = clf_lg.predict(text)[0][0][9:]
                if lg != 'en': return
                return text.lower().strip()

            for negative, positive in tqdm(zip(df['Negative_Review'], df['Positive_Review']), total=len(df)):
                texts = process(negative)
                if texts:
                    reviews.append(texts)
                    ratings.append(1)
                texts = process(positive)
                if texts:
                    reviews.append(texts)
                    ratings.append(5)


        reviews = df['review'].to_list()
        ratings = df['rating'].to_list()

        review_filter = []
        ratings_filter = []

        
        df2 = pd.DataFrame.from_dict({
            'review': reviews,
            'rating': ratings
        })
    
        
        df2.to_csv(output_path, index=False, sep='\t')

    def output(self):
        return luigi.LocalTarget(f"data/sample{'/test/' if self.is_test else '/'}{self.domain}.tsv")
    
if __name__ == '__main__':
    tasks = [
        TaskExtractSample(domain=domain, is_test=True) for domain in ["hotel", 'restaurant']
    ]
    luigi.build(tasks, local_scheduler=True)