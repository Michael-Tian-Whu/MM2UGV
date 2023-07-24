import torch
import numpy as np
from torch.utils.data import Sampler
import copy

__all__ = ["TrainSampler","ContinuousSampler", "SparseSampler", "TestSampler", "BalanceSampler", "SeqSampler"]

class TestSampler_0(Sampler):
    """
        测试集合
    """

    def __init__(self, data_source, seed=2023):
        super(TestSampler, self).__init__(data_source)
        np.random.seed(seed)
        self.data = data_source
        self.length=len(self.data)
        self.idx = list(range(self.length))
        np.random.seed(seed)
        np.random.shuffle(self.idx)#先shuffle
        self.idx = self.idx[self.length*7//10,:]

    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.idx)
    
class TrainSampler_0(Sampler):
    """
        训练集 7/10
    """

    def __init__(self, data_source, seed=2023):
        super(TestSampler, self).__init__(data_source)
        np.random.seed(seed)
        self.data = data_source
        self.length=len(self.data)
        self.idx = list(range(self.length))
        np.random.seed(seed)
        np.random.shuffle(self.idx)#先shuffle
        self.idx = self.idx[:,self.length*7//10]

    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.idx)
    


class ContinuousSampler(Sampler):
    """
    This sampler samples three part of data
    A. random sampled data
    B. frames that follow each one of A
    C. independent sampled data of A and B
    """

    def __init__(self, data_source):
        super(ContinuousSampler, self).__init__(data_source=data_source)
        self.data = data_source
        self.idx = list(range(len(self.data)))
        self.idx2 = self.idx.copy()
        np.random.shuffle(self.idx)
        np.random.shuffle(self.idx2)
        self.res = []
        for i in range(len(self.data)):
            self.res.append(self.idx[i])
            self.res.append(min(self.idx[i] + 1, len(self.data) - 1))
            self.res.append(self.idx2[i])

    def __iter__(self):
        return iter(self.res)

    def __len__(self):
        return len(self.res)


# a = ContinuousSampler()
# for i, v in enumerate(a):
#     print(v)
#     if i == 8:
#         break

class SparseSampler(Sampler):
    """
    a sparse sampler filters data by a step, default set to 10, which is approximately 1 second in time domain
    both starting frame and step length is flexible, one can take advantage of it
    """

    def __init__(self, data_source, start=0, step=10, seed=2022):
        assert start < step
        super(SparseSampler, self).__init__(data_source)
        self.data = data_source
        self.idx = list(range(start, len(self.data), step))
        np.random.seed(seed)
        np.random.shuffle(self.idx)

    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.idx)


class TestSampler(Sampler):
    """
    Sampler used for test, get required number of shuffled data for test,
    if data is not enough for testing, repeat index several times before shuffling
    """

    def __init__(self, data_source, n_test, n_sample, seed=2023):
        super(TestSampler, self).__init__(data_source)
        np.random.seed(seed)
        self.data = data_source
        n = (n_test * n_sample) // len(self.data)
        self.idx = list(range(len(self.data))) * (n + 1)
        np.random.shuffle(self.idx)
        self.idx = self.idx[:n_test * n_sample]

    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.data)


class BalanceSampler(Sampler):
    """
    Another sampler used for test, besides a similar function of TestSampler, this sampler samples data with a balanced
    categories of one specific classification task
    """

    def __init__(self, data_source, n_test, n_per_class, col="landscape", seed=2022):
        super(BalanceSampler, self).__init__(data_source)
        np.random.seed(seed)
        self.data = data_source
        self.idx, self.cache = [], []
        for i in self.data.label[col].unique():
            tmp = self.data.label.loc[self.data.label[col] == i].index.to_list()
            if len(tmp) < n_per_class * n_test:
                tmp = tmp * ((n_per_class * n_test) // len(tmp) + 1)
            tmp = tmp[:n_test * n_per_class]
            np.random.shuffle(tmp)
            self.cache.append(copy.deepcopy(tmp))
        for i in range(n_test):
            for tmp in self.cache:
                self.idx += tmp[i * n_per_class:(i + 1) * n_per_class]

    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.idx)


class SeqSampler(Sampler):
    def __init__(self, data_source, frame_before_end=40, sparse_factor=4, seed=2022, fixed_seed=False):
        super(SeqSampler, self).__init__(data_source)
        self.data = data_source
        self.idx = []
        for i, v in enumerate(self.data.seq):
            if len(v) < frame_before_end + 10:
                continue
            self.idx += list(range(self.data.length[i] + 5, self.data.length[i] + len(v) - frame_before_end - 5))
            # print(self.data.length[i], self.data.length[i] + 5, self.data.length[i] + len(v) - frame_before_end - 5,
            #       self.data.length[i] + len(v))
        if fixed_seed:
            np.random.seed(seed)
        np.random.shuffle(self.idx)
        self.idx = [i for i in self.idx if i % sparse_factor == 0]

    def __len__(self):
        return len(self.idx)

    def __iter__(self):
        return iter(self.idx)
