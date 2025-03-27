import luigi
import os
from gensim.models import Word2Vec

from task_preprocess_w2v import TaskPreprocessW2V


class TaskTrainW2V(luigi.Task):

    domain = luigi.ChoiceParameter(choices=["restaurant", "hotel"])
    algo = luigi.ChoiceParameter(choices=["sg", "cbow", 'sg_hs', 'cbow_hs'])

    def requires(self):
        return TaskPreprocessW2V(self.domain)

    def run(self):

        output_path = self.output().path
        output_dir = os.path.dirname(output_path)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        w2v = Word2Vec(
            corpus_file=self.input().path,
            vector_size=200,
            sample=0, # Most aspect are very frequent
            ns_exponent=0.75,
            window=10,
            seed=42,
            epochs=1,
            negative=5,
            hs=1 if 'hs' in self.algo else 0,
            sg = 1 if 'sg' in self.algo else 0,
            workers=int(os.cpu_count() * 0.75)
        )
        
        w2v.wv.save_word2vec_format(self.output().path, binary=False)

    def output(self):
        return luigi.LocalTarget(f'data/w2v/{self.domain}_{self.algo}.vec')
    
if __name__ == '__main__':
    tasks = [
        TaskTrainW2V(domain=domain, algo=algo) for domain in ["hotel", 'restaurant'] for algo in ['sg_hs', 'cbow_hs']
    ]
    luigi.build(tasks, local_scheduler=True)