"""
Pretrained encoder evaluation
"""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.samplers import TestSampler, BalanceSampler

__all__ = ["CosineSimTest", "CosineSimTestBalance"]

class CosineSimTest(object):
    def __init__(self, vision_model, sensor_model, test_data, n_test, batch_size, devices, seed=2022, k=5):
        self.vision = vision_model
        self.sensor = sensor_model
        self.devices = devices
        self.batch_size = batch_size
        self.k = k
        # 一个batch就是一次测试
        np.random.seed(seed)
        seed1, seed2 = np.random.randint(seed), np.random.randint(seed)
        self.query_iter = DataLoader(test_data, sampler=TestSampler(test_data, n_test, batch_size, seed1),
                                     batch_size=batch_size, num_workers=os.cpu_count())
        self.support_iter = DataLoader(test_data, sampler=TestSampler(test_data, n_test, batch_size, seed2),
                                       batch_size=batch_size, num_workers=os.cpu_count())

    def _get_cache(self, dataloader):
        res = []
        for i, (sensor_input, vision_input, landscape_label, terrain_label, a, b) in enumerate(dataloader):
            sensor_feature = self.sensor(sensor_input.float().cuda(self.devices[0])).detach()
            vision_feature = self.vision(vision_input.float().cuda(self.devices[0])).detach()
            res.append((sensor_feature, vision_feature, landscape_label, terrain_label))
        return res

    @staticmethod
    def _validate_with_label(query: torch.Tensor, support: torch.Tensor,
                             query_label: torch.Tensor, support_label: torch.Tensor):
        a = query / torch.norm(query, dim=1, keepdim=True)
        b = support / torch.norm(support, dim=1, keepdim=True)
        c = a @ b.T  # c_ij, similarity of query i and support j
        c = torch.argmax(c, dim=1).tolist()
        n = 0
        for i, v in enumerate(c):
            n += int(query_label[i] == support_label[v])
        return n

    def _get_unbalance_acc(self, q_cache, s_cache):
        # q_cache (sensor_feature, vision_feature, landscape_label, terrain_label)
        s_landscape_acc, s_terrain_acc = [], []
        v_landscape_acc, v_terrain_acc = [], []
        f_landscape_acc, f_terrain_acc = [], []
        assert len(q_cache) == len(s_cache)
        for i in range(len(q_cache)):
            q, s = q_cache[i], s_cache[i]
            s_landscape_acc.append(self._validate_with_label(q[0], s[0], q[2], s[2]) / self.batch_size)
            s_terrain_acc.append(self._validate_with_label(q[0], s[0], q[3], s[3]) / self.batch_size)
            v_landscape_acc.append(self._validate_with_label(q[1], s[1], q[2], s[2]) / self.batch_size)
            v_terrain_acc.append(self._validate_with_label(q[1], s[1], q[3], s[3]) / self.batch_size)
            a = torch.cat((q[0], q[1]), dim=1)
            b = torch.cat((s[0], s[1]), dim=1)
            f_landscape_acc.append(self._validate_with_label(a / torch.norm(a, dim=1, keepdim=True),
                                                             b / torch.norm(b, dim=1, keepdim=True),
                                                             q[2], s[2]) / self.batch_size)
            f_terrain_acc.append(self._validate_with_label(a / torch.norm(a, dim=1, keepdim=True),
                                                           b / torch.norm(b, dim=1, keepdim=True),
                                                           q[3], s[3]) / self.batch_size)

        return s_landscape_acc, s_terrain_acc, v_landscape_acc, v_terrain_acc, f_landscape_acc, f_terrain_acc

    def eval(self):
        support_cache = self._get_cache(self.support_iter)
        query_cache = self._get_cache(self.query_iter)
        s_landscape_acc, s_terrain_acc, v_landscape_acc, v_terrain_acc, f_landscape_acc, f_terrain_acc = \
            self._get_unbalance_acc(query_cache, support_cache)
        return np.mean(s_landscape_acc), np.std(s_landscape_acc), np.mean(s_terrain_acc), np.std(s_terrain_acc), \
               np.mean(v_landscape_acc), np.std(v_landscape_acc), np.mean(v_terrain_acc), np.std(v_terrain_acc), \
               np.mean(f_landscape_acc), np.std(f_landscape_acc), np.mean(f_terrain_acc), np.std(f_terrain_acc),
class CosineSimTest_1(object):
    def __init__(self,  vision_model, sensor_model, test_data, n_test, batch_size, devices, seed=2023, k=5,top=1):
        self.vision = vision_model
        self.sensor = sensor_model
        self.devices = devices
        self.batch_size = batch_size
        self.k = k
        self.top=top
        # 一个batch就是一次测试
        np.random.seed(seed)
        seed1, seed2 = np.random.randint(seed), np.random.randint(seed)
        #2个shuffle的n_test*batch_size大小的数据
        self.query_iter = DataLoader(test_data, sampler=TestSampler(test_data, n_test, batch_size, seed1),
                                     batch_size=batch_size, num_workers=os.cpu_count())
        self.support_iter = DataLoader(test_data, sampler=TestSampler(test_data, n_test, batch_size, seed2),
                                       batch_size=batch_size, num_workers=os.cpu_count())

    def _get_cache(self, dataloader):
        res = []
        for i, (sensor_input, vision_input, landscape_label, terrain_label, a, b) in enumerate(dataloader):
            sensor_feature = self.sensor(sensor_input.float().cuda(self.devices[0])).detach()
            vision_feature = self.vision(vision_input.float().cuda(self.devices[0])).detach()
            res.append((sensor_feature, vision_feature, landscape_label, terrain_label))
        return res

    def _validate_with_label(self,query: torch.Tensor, support: torch.Tensor,
                             query_label: torch.Tensor, support_label: torch.Tensor):
        a = query / torch.norm(query, dim=1, keepdim=True)#[batch,dim]
        b = support / torch.norm(support, dim=1, keepdim=True)
        c = a @ b.T  # c_ij, similarity of query i and support j
        idx = torch.argsort(c)
        idx =idx[:,-self.top-1:-1].tolist()#[self.top,n]
        n = 0
        for i, v in enumerate(idx):
            for j in v:
                if query_label[i] == support_label[j]:
                    n+=1 
                    break
        return n
        
    def _get_unbalance_acc(self, q_cache:list, s_cache:list):
        # q_cache (sensor_feature, vision_feature, landscape_label, terrain_label)
        s_landscape_acc, s_terrain_acc = [], []
        v_landscape_acc, v_terrain_acc = [], []
        f_landscape_acc, f_terrain_acc = [], []
        assert len(q_cache) == len(s_cache)
        for i in range(len(q_cache)):
            q, s = q_cache[i], s_cache[i]
            #单模态
            s_landscape_acc.append(self._validate_with_label(q[0], s[0], q[2], s[2]) / self.batch_size)
            s_terrain_acc.append(self._validate_with_label(q[0], s[0], q[3], s[3]) / self.batch_size)
            v_landscape_acc.append(self._validate_with_label(q[1], s[1], q[2], s[2]) / self.batch_size)
            v_terrain_acc.append(self._validate_with_label(q[1], s[1], q[3], s[3]) / self.batch_size)
            #多模态
            a = torch.cat((q[0], q[1]), dim=1)
            b = torch.cat((s[0], s[1]), dim=1)
            f_landscape_acc.append(self._validate_with_label(a / torch.norm(a, dim=1, keepdim=True),
                                                             b / torch.norm(b, dim=1, keepdim=True),
                                                             q[2], s[2]) / self.batch_size)
            f_terrain_acc.append(self._validate_with_label(a / torch.norm(a, dim=1, keepdim=True),
                                                           b / torch.norm(b, dim=1, keepdim=True),
                                                           q[3], s[3]) / self.batch_size)

        return s_landscape_acc, s_terrain_acc, v_landscape_acc, v_terrain_acc, f_landscape_acc, f_terrain_acc

    def eval(self):
        support_cache = self._get_cache(self.support_iter)
        query_cache = self._get_cache(self.query_iter)
        s_landscape_acc, s_terrain_acc, v_landscape_acc, v_terrain_acc, f_landscape_acc, f_terrain_acc = \
            self._get_unbalance_acc(query_cache, support_cache)
        return np.mean(s_landscape_acc), np.std(s_landscape_acc), np.mean(s_terrain_acc), np.std(s_terrain_acc), \
               np.mean(v_landscape_acc), np.std(v_landscape_acc), np.mean(v_terrain_acc), np.std(v_terrain_acc), \
               np.mean(f_landscape_acc), np.std(f_landscape_acc), np.mean(f_terrain_acc), np.std(f_terrain_acc),


class CosineSimTestBalance(object):
    def __init__(self, vision_model, sensor_model, test_data, n_test, batch_size, devices, seed=2022, k=5,
                 col="landscape"):
        self.vision = vision_model
        self.sensor = sensor_model
        self.devices = devices
        assert batch_size % len(test_data.label[col].unique()) == 0
        per_class = batch_size // len(test_data.label[col].unique())
        self.batch_size = batch_size
        self.k = k
        self.col = col
        # 一个batch就是一次测试
        np.random.seed(seed)
        seed1, seed2 = np.random.randint(seed), np.random.randint(seed)
        self.query_iter = DataLoader(test_data,
                                     sampler=BalanceSampler(test_data, n_test, per_class, col=col, seed=seed1),
                                     batch_size=batch_size, num_workers=os.cpu_count())
        self.support_iter = DataLoader(test_data,
                                       sampler=BalanceSampler(test_data, n_test, per_class, col=col, seed=seed2),
                                       batch_size=batch_size, num_workers=os.cpu_count())

    def _get_cache(self, dataloader):
        res = []
        for i, (sensor_input, vision_input, landscape_label, terrain_label, a, b) in enumerate(dataloader):
            sensor_feature = self.sensor(sensor_input.float().cuda(self.devices[0])).detach()
            vision_feature = self.vision(vision_input.float().cuda(self.devices[0])).detach()
            res.append((sensor_feature, vision_feature, landscape_label, terrain_label))
        return res

    def _validate_with_label(self, query: torch.Tensor, support: torch.Tensor,
                             query_label: torch.Tensor, support_label: torch.Tensor):
        a = query / torch.norm(query, dim=1, keepdim=True)
        b = support / torch.norm(support, dim=1, keepdim=True)
        c = a @ b.T  # c_ij, similarity of query i and support j
        v, idx = torch.topk(c, k=self.k, dim=1)
        n = 0
        for i, row in enumerate(idx):
            hit = 0
            for col in row:
                hit += query_label[i] == support_label[col]
            n += int(hit > self.k // 2)
        return n

    def _get_acc(self, q_cache, s_cache):
        # q_cache (sensor_feature, vision_feature, landscape_label, terrain_label)
        assert len(q_cache) == len(s_cache)
        label = 2 if self.col == "landscape" else 3
        s_acc, v_acc, f_acc = [], [], []
        for i in range(len(q_cache)):
            q, s = q_cache[i], s_cache[i]
            s_acc.append(self._validate_with_label(q[0], s[0], q[label], s[label]) / self.batch_size)
            v_acc.append(self._validate_with_label(q[1], s[1], q[label], s[label]) / self.batch_size)
            a = torch.cat((q[0] / torch.norm(q[0], dim=1, keepdim=True), q[1] / torch.norm(q[1], dim=1, keepdim=True)),
                          dim=1)
            b = torch.cat((s[0] / torch.norm(s[0], dim=1, keepdim=True), s[1] / torch.norm(s[1], dim=1, keepdim=True)),
                          dim=1)
            f_acc.append(self._validate_with_label(a, b, q[label], s[label]) / self.batch_size)
        return s_acc, v_acc, f_acc

    def eval(self):
        support_cache = self._get_cache(self.support_iter)
        query_cache = self._get_cache(self.query_iter)
        s_acc, v_acc, f_acc = self._get_acc(query_cache, support_cache)
        return np.mean(s_acc), np.std(s_acc), np.mean(v_acc), np.std(v_acc), np.mean(f_acc), np.std(f_acc),
