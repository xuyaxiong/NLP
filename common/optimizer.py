import numpy as np

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m = []
            self.v = []
            for param in params:
                self.m.append(np.zeros_like(param))
                self.v.append(np.zeros_like(param))
                
        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grads[i] ** 2

            self.m_hat = self.m[i] / (1 - self.beta1 ** (1+i))
            self.v_hat = self.v[i] / (1 - self.beta2 ** (1+i))
        
            params[i] -= self.lr * self.m_hat / (np.sqrt(self.v_hat) + 1e-7)


class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = []
            for param in params:
                self.v.append(np.zeros_like(param))
        
        for idx, param in enumerate(params):
            self.v[idx] = self.momentum * self.v[idx] - self.lr * grads[idx]
            param += self.v[idx]

