import luigi
from glob import glob

from task_train_sbert import TaskTrainSBERT

class PipelineTrainEmbeddings(luigi.WrapperTask):

    def requires(self):

        for domain in ['restaurant', 'hotel']:
            for loss in ['mnrl', 'gist']:
                for model in ['all-mpnet-base-v2']:
                        for lora in [True, False]:
                            for dataset in ['senticse', 'simcse', 'cat/sg_10', 'cat/sg_50', 'cat/sg_100', 'cat/sg_150', 'cat/sg_200', 'cat/sg_250']:
                                output = f"{model.split('/')[-1]}{'_lora' if lora else ''}{'_guided' if loss == 'gist' else ''}_{dataset.split('/')[-1]}"
                                yield TaskTrainSBERT(domain=domain, sbert=model, loss=loss, is_lora=lora, batchs=128, dataset_train=f"data/training/{domain}/{dataset}.tsv", output_dir=output, epoch=3)        
                             
if __name__ == '__main__':
    luigi.build([PipelineTrainEmbeddings()], local_scheduler=True)
