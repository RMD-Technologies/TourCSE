import luigi
import glob
import json
from task_evaluate_ir import TaskEvaluateIR

class PipelineEvaluateEmbeddings(luigi.WrapperTask):

    def requires(self):
        
        for domain in ['restaurant', 'hotel']:
            for model in [
                'all-mpnet-base-v2',
                'DILAB-HYU/SentiCSE',
                'sg',
                'cbow',
                'data/sbert/bert-mlm-cls',
                'data/sbert/bert-mlm-mean',
                "intfloat/e5-mistral-7b-instruct",
                "all-MiniLM-L6-v2",
                "all-MiniLM-L12-v2",
                "sentence-t5-xxl",
                "Alibaba-NLP/gte-modernbert-base",
                'intfloat/e5-small-v2',
                'intfloat/e5-base-v2',
                'intfloat/e5-large-v2',
                "gtr-t5-xxl",
                "Alibaba-NLP/gte-base-en-v1.5",
                "Alibaba-NLP/gte-large-en-v1.5",
                "vahidthegreat/StanceAware-SBERT",
            ]:
                yield TaskEvaluateIR(domain=domain, sbert=model, file=model.split('/')[-1])

            for _path in glob.glob(f"data/sbert/{domain}/*"):

                model = _path.split('/')[-1]

                if "lora" in model:
                    try:
                        with open(_path + '/adapter_config.json', 'r') as f:
                            d = json.load(f)
                            print(model, _path) 
                            base_model = d["base_model_name_or_path"]
                            yield TaskEvaluateIR(domain=domain, sbert=base_model, lora_path=_path, file=model)
                    except:
                        pass
                else:
                    yield TaskEvaluateIR(domain=domain, sbert=_path, file=model.split('/')[-1])
      
      
if __name__ == '__main__':
    luigi.build([PipelineEvaluateEmbeddings()], local_scheduler=True)
