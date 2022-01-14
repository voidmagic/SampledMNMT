import logging
from collections import OrderedDict
from itertools import chain, cycle
import random

from fairseq.data import RoundRobinZipDatasets

logger = logging.getLogger(__name__)


class MultilingualSampledDataset(RoundRobinZipDatasets):
    def __init__(self, datasets, eval_key=None, sample_method='temperature', temperature=5):
        super(MultilingualSampledDataset, self).__init__(datasets, eval_key)
        self.sample_method = sample_method
        self.temperature = temperature

    def map_multilingual_index(self, index):
        for key, dataset in self.datasets.items():
            if index < len(dataset):
                return key, index
            index -= len(dataset)
        raise IndexError(
            "index {} out of range {}".format(index, sum(len(dataset for dataset in self.datasets.values()))))

    def __getitem__(self, index):
        key, index = self.map_multilingual_index(index)
        return key, self.datasets[key][index]

    def collater(self, samples):
        if len(samples) == 0:
            return None
        samples_dict = {
            dataset_key: [sample for sample_key, sample in samples if sample_key == dataset_key]
            for dataset_key in self.datasets.keys()
        }
        batch_dict = {
            key: dataset.collater(samples_dict[key]) if samples_dict[key] else None
            for key, dataset in self.datasets.items()
        }
        return batch_dict if self.eval_key is None else batch_dict[self.eval_key]

    def ordered_indices(self):
        super(MultilingualSampledDataset, self).ordered_indices()
        return self._ordered_indices

    def filter_indices_by_size(self, indices, max_sizes):
        filtered_ignored = OrderedDict([
            (key, dataset.filter_indices_by_size(indices[key], max_sizes if isinstance(max_sizes, tuple) else max_sizes[key]))
            for key, dataset in self.datasets.items()
        ])
        filtered = OrderedDict([(key, value[0]) for key, value in filtered_ignored.items()])
        ignored = OrderedDict([(key, value[1]) for key, value in filtered_ignored.items()])
        return filtered, list(chain.from_iterable(ignored.values()))

    def batch_by_size(self, indices, max_tokens=None, max_sentences=None, required_batch_size_multiple=1):
        batch_sampler = OrderedDict([
            (key, dataset.batch_by_size(indices[key], max_tokens, max_sentences, required_batch_size_multiple))
            for key, dataset in self.datasets.items()
        ])
        batch_sampler = self.map_multilingual_sampler(batch_sampler)
        return batch_sampler

    def map_multilingual_sampler(self, batch_sampler_dict):
        random.seed(0)
        assert self.sample_method in ['uniform', 'temperature', 'proportional']
        if self.sample_method == 'uniform':
            weights = OrderedDict([(key, 1) for key, value in self.datasets.items()])
        elif self.sample_method == 'proportional':
            weights = {key: len(value) for key, value in self.datasets.items()}
        else:
            weights = {key: len(value) ** (1 / self.temperature) for key, value in self.datasets.items()}

        offset = [(key, len(dataset)) for key, dataset in self.datasets.items()]
        offset = [(key, sum([v[1] for v in offset[:i + 1]]) - value) for i, (key, value) in enumerate(offset)]
        offset = OrderedDict(offset)
        for key, dataset in self.datasets.items():
            batch_sampler = batch_sampler_dict[key]
            for row in range(len(batch_sampler)):
                for col in range(len(batch_sampler[row])):
                    batch_sampler[row][col] += offset[key]
            random.shuffle(batch_sampler_dict[key])
        max_key, _ = max(self.datasets.items(), key=lambda item: len(item[1]))
        total_samples = len(self.datasets[max_key]) / weights[max_key]
        estimate_samples = {key: int(total_samples * weight) for key, weight in weights.items()}
        sample_per_batch = {key: len(dataset) / len(batch_sampler_dict[key]) for key, dataset in self.datasets.items()}
        estimate_batch = {key: int(estimate_samples[key] / sample_per_batch[key]) for key in self.datasets.keys()}
        endless_cycle = {key: cycle(batch_sampler) for key, batch_sampler in batch_sampler_dict.items()}
        batch_sampler_weighted = [[next(batch_iter) for _ in range(estimate_batch[key])] for key, batch_iter in endless_cycle.items()]

        logger.info('Sampled sentences: ')
        for key, value in self.datasets.items():
            logger.info('{}: {}-{}'.format(key, len(value), estimate_samples[key]))
        logger.info('Sampled batches: ')
        for key, value in batch_sampler_dict.items():
            logger.info('{}: {}-{}'.format(key, len(value), estimate_batch[key]))
        return list(chain.from_iterable(batch_sampler_weighted))
