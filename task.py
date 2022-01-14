from fairseq.tasks import register_task
from fairseq.tasks.multilingual_translation import MultilingualTranslationTask

from .dataset import MultilingualSampledDataset


@register_task("sample_mnmt")
class SampledMultilingualTask(MultilingualTranslationTask):
    @staticmethod
    def add_args(parser):
        MultilingualTranslationTask.add_args(parser)
        parser.add_argument('--sample-method', default='proportional', choices=['temperature', 'proportional', 'uniform'])
        parser.add_argument('--sample-temperature', default=5, type=int)

    def load_dataset(self, split, epoch=1, **kwargs):
        super(SampledMultilingualTask, self).load_dataset(split, epoch)
        if split != 'train':
            return
        self.datasets[split] = MultilingualSampledDataset(
            self.datasets[split].datasets,
            self.datasets[split].eval_key,
            sample_method=self.args.sample_method,
            temperature=self.args.sample_temperature
        )
