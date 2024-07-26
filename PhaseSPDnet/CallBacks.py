import skorch
import numpy as np

class InputShapeSetter(skorch.callbacks.Callback):
    def on_train_begin(self, net, X, y):
        net.set_params(module__dim=X.shape[2],
                       module__classes=len(np.unique(y)))
