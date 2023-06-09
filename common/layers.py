import numpy as np
from common.functions import softmax, cross_entropy_error

class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(self, x):
        W, = self.params
        out = np.dot(x, W)
        self.x = x
        return out
    
    def backward(self, dout):
        W, = self.params
        dW = np.dot(self.x.T, dout)
        dx = np.dot(dout, W.T)
        self.grads[0][...] = dW
        return dx
    

class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)

        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)
        
        loss = cross_entropy_error(self.y, self.t)
        return loss
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx = dx / batch_size
        return dx
    
class SigmoidWithLoss:
    def __init__(self):
        self.params = []
        self.grads = []
        self.t = None
        self.y = None

    def forward(self, x, t):
        self.t = t
        self.y = 1 / (1 + np.exp(-x))
        loss = cross_entropy_error(np.c_[1 - self.y, self.y], t)
        return loss
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        return dout * (self.y - self.t) / batch_size


class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out
    
    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0
        for i, word_id in enumerate(self.idx):
            dW[word_id] += dout[i]
        return None
    

class EmbeddingDot:
    def __init__(self, W):
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None


    def forward(self, h, idx):
        target_W = self.embed.forward(idx)
        out = np.sum(target_W * h, axis=1)
        self.cache = (target_W, h)
        return out
    
    def backward(self, dout):
        dout = dout.reshape(-1, 1)
        target_W, h = self.cache
        dtarget_w = dout * h
        self.embed.backward(dtarget_w)
        dh = dout * target_W
        return dh
