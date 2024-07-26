import spdnets
import spdnets.modules as modules
from typing import Optional
import torch.nn as nn
import torch
import spdnets.batchnorm as bn

class SPDNet(nn.Module):
    """
    This class implements the SPDNet model (Based on Kobler)
    """

    def __init__(self,
                 dim=22,
                 classes=1,
                 bnorm: Optional[str] = 'brooks',
                 **kwargs):
        super().__init__(**kwargs)

        self.dim = dim
        self.classes = classes
        self.subspacedims = int(dim/2)
        self.bnorm_ = bnorm
        tsdim = int(self.subspacedims * (self.subspacedims + 1) / 2)

        # Normalization
        if self.bnorm_ == 'brooks':
            self.spdbnorm = bn.SPDBatchNorm((1, self.subspacedims, self.subspacedims), batchdim=0, dtype=torch.double)
        elif self.bnorm_ is not None:
            raise NotImplementedError('requested undefined batch normalization method.')
        elif self.bnorm_ is None:
            print("No Batch Normalization used")

        self.bimap = modules.BiMap((1, self.dim, self.subspacedims), dtype=torch.double)
        self.reeig = modules.ReEig(threshold=1e-4)

        self.logeig = torch.nn.Sequential(
            modules.LogEig(self.subspacedims),
            torch.nn.Flatten(start_dim=1),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(tsdim, self.classes).double(),
        )

    def forward(self, x, return_latent=False, return_prebn=False, return_postbn=False):
        out = ()
        l = self.bimap(x)
        l = self.reeig(l)
        out += (l,) if return_prebn else ()
        l = self.spdbnorm(l) if hasattr(self, 'brooks') else l
        out += (l,) if return_postbn else ()
        l = self.logeig(l)
        out += (l,) if return_latent else ()
        y = self.classifier(l)
        out = y if len(out) == 0 else (y, *out[::-1])
        return out