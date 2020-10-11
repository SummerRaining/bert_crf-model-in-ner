from .backend import optimizers
from .backend import backend as K


__all__ = ['LRMultiplier']


class LRMultiplier(optimizers.Optimizer):

    def __init__(self,
                 optimizer,
                 multipliers,
                 **kwargs):
        """Initialize the optimizer wrapper.

        :param optimizer: The original optimizer.
        :param multipliers: A dict representing the multipliers. key是权重的前缀。
                            The key is the prefix of the weight to be multiplied.
        :param kwargs: Arguments for parent class.
        optimizer=LRMultiplier('adam', {'Dense': 0.5, 'Output': 1.5})
        """
        super(LRMultiplier, self).__init__(**kwargs)
        self.optimizer = optimizers.get(optimizer)          #根据名称得到，optimizer
        self.multipliers = multipliers                      
        if hasattr(self.optimizer, 'learning_rate'):        #keras2.3版本后是learning_rate,2.3之前是lr.
            self.lr_attr = 'learning_rate'
        else:
            self.lr_attr = 'lr'

    @property
    def lr(self):
        return self.optimizer.lr

    @lr.setter
    def lr(self, lr):
        self.optimizer.lr = lr

    @property
    def learning_rate(self):
        return self.optimizer.learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        try:
            self.optimizer.learning_rate = learning_rate
        except ValueError:
            self.optimizer._hyper['learning_rate'] = learning_rate

    def _get_multiplier(self, name):
        multiplier, prefix_len = 1.0, 0         
        for key, val in self.multipliers.items():
            if name.startswith(key):            #如果参数的名称前缀中包含key，就令multiplier作为该param的学习率。
                if len(key) > prefix_len:       
                    prefix_len = len(key)
                    multiplier = val
        return multiplier

    def get_updates(self, loss, params):
        if len(self.updates) > 0:
            return self.updates
        multiplies = {}
        for param in params:
            multiplier = self._get_multiplier(param.name)       #找到对应lr，或者返回1.0。
            if multiplier not in multiplies:                    
                multiplies[multiplier] = [] 
            multiplies[multiplier].append(param)                #该学习率下面的所有参数。

        self.updates, self.weights = [], []
        origin_lr = getattr(self, self.lr_attr)
        for i, (multiplier, params) in enumerate(multiplies.items()):
            lr = origin_lr
            if callable(multiplier):
                lr = lr * multiplier(K.cast(self.optimizer.iterations, K.floatx()))
            elif multiplier != 1.0:
                lr = lr * multiplier
            setattr(self, self.lr_attr, lr)
            with K.name_scope('Group_{}'.format(i)):
                self.updates += self.optimizer.get_updates(loss, params)
            print(self.multipliers, i, self.optimizer.weights)
            for w in self.optimizer.weights:
                if w not in self.weights:
                    self.weights.append(w)
        setattr(self, self.lr_attr, origin_lr)

        return self.updates

    def get_config(self):
        config = {
            'optimizer': optimizers.serialize(self.optimizer),
            'multipliers': self.multipliers
        }
        base_config = super(LRMultiplier, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        optimizer = optimizers.deserialize(config.pop('optimizer'))
        return cls(optimizer, **config)
