import numpy as np
import collections
from common.layers import EmbeddingDot, SigmoidWithLoss

class UnigramSampler:
    def __init__(self, corpus, power, sample_size):
        self.sample_size = sample_size
        self.cropus = corpus
        counts = collections.Counter()
        for word_id in corpus:
            counts[word_id] += 1
        self.vocab_size = len(counts)
        self.word_p = np.zeros(self.vocab_size)
        for i in range(self.vocab_size):
            self.word_p[i] = counts[i]
        self.word_p = self.word_p ** power
        self.word_p /= np.sum(self.word_p)

    def get_negative_sample(self, target):
        batch_size = target.shape[0]
        negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)
        for i in range(batch_size):
            p = self.word_p.copy()
            target_id = target[i]
            p[target_id] = 0
            p /= np.sum(p)
            negative_sample[i,:] = np.random.choice(self.vocab_size, size=self.sample_size, replace=False, p=p)
        return  negative_sample


class NegativeSamplingLoss:
    def __init__(self, corpus, power, sample_size, W):
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size)
        self.embed_dot_layers = []
        self.loss_layers = []
        for i in range(sample_size + 1):
            self.embed_dot_layers.append(EmbeddingDot(W))
            self.loss_layers.append(SigmoidWithLoss())

        self.params = []
        self.grads = []
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, h, target):
        batch_size = target.shape[0]
        negative_samples = self.sampler.get_negative_sample(target)

        score = self.embed_dot_layers[0].forward(h, target)
        correct_label = np.ones(batch_size, dtype=np.int32)
        loss = self.loss_layers[0].forward(score, correct_label)

        negative_label = np.zeros(batch_size, dtype=np.int32)
        for i in range(self.sample_size):
            negative_target = negative_samples[:, i]
            score = self.embed_dot_layers[1 + i].forward(h, negative_target)
            loss += self.loss_layers[1 + i].forward(score, negative_label)
        return loss 
    
    def backward(self, dout=1):
        dh = 0
        for l0, l1 in zip(self.loss_layers, self.embed_dot_layers):
            dscore = l0.backward(dout)
            dh += l1.backward(dscore)
        return dh