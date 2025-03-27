import luigi
from glob import glob
from tqdm import tqdm
from sentence_transformers import models, SentenceTransformer

class TaskConvertMLMSbert(luigi.Task):

    domain = luigi.ChoiceParameter(choices=["restaurant", "hotel", ''])
    pooling = luigi.Parameter()
    model_path = luigi.Parameter()
    output_dir = luigi.Parameter()

    def run(self):


        transformer = models.Transformer(self.model_path)
        pooling = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode=self.pooling)

        sbert = SentenceTransformer(modules=[transformer, pooling])

        sbert.save_pretrained(self.output().path)
     

    def output(self):
        if self.domain:
            return luigi.LocalTarget(f'data/sbert/{self.domain}/{self.output_dir}')
        return luigi.LocalTarget(f'data/sbert/{self.output_dir}')
    
if __name__ == '__main__':
    tasks = [
        TaskConvertMLMSbert(domain="restaurant", model_path="activebus/BERT-PT_rest", output_dir="bert-pt-mlm-cls", pooling='cls'),
        TaskConvertMLMSbert(domain="restaurant", model_path="activebus/BERT-PT_rest", output_dir="bert-pt-mlm-mean", pooling='mean'),
        TaskConvertMLMSbert(domain="hotel", model_path="data/model/mlm/hotel/bert-hotel-mlm", output_dir="bert-pt-mlm-cls", pooling='cls'),
        TaskConvertMLMSbert(domain="hotel", model_path="data/model/mlm/hotel/bert-hotel-mlm", output_dir="bert-pt-mlm-mean", pooling='mean'),
        TaskConvertMLMSbert(domain='', model_path="google-bert/bert-base-uncased", output_dir="bert-mlm-cls",  pooling='cls'),
        TaskConvertMLMSbert(domain='', model_path="google-bert/bert-base-uncased", output_dir="bert-mlm-mean",  pooling='mean'),
    ]
    luigi.build(tasks, local_scheduler=True)