# Phase-SPDNet

This repository contain the results and the code for the paper Phase-SPDNet [1].
If you use the code please cite the connected paper.

## Abstract 
The integration of Deep Learning (DL) algorithms on brain signal analysis is still in its nascent stages compared to their success in fields like Computer Vision. This is particularly true for BCI, where the brain activity is decoded to control external devices without requiring muscle control.
Electroencephalography (EEG) is a widely adopted choice for designing BCI systems due to its non-invasive and cost-effective nature and excellent temporal resolution. Still, it comes at the expense of limited training data, poor signal-to-noise, and a large variability across and within-subject recordings. 
Finally, setting up a BCI system with many electrodes takes a long time, hindering the widespread adoption of reliable DL architectures in BCIs outside research laboratories. To improve adoption, we need to improve user comfort using, for instance, reliable algorithms that operate with few electrodes. 
Our research aims to develop a DL algorithm that delivers effective results with a limited number of electrodes. Taking advantage of the Augmented Covariance Method and the framework of SPDNet, we propose the Phase-SPDNet architecture and analyze its performance and the interpretability of the results. The evaluation is conducted on 5-fold cross-validation, using only three electrodes positioned above the Motor Cortex. The methodology was tested on nearly 100 subjects from several open-source datasets using the Mother Of All BCI Benchmark (MOABB) framework. 
The results of our Phase-SPDNet demonstrate that the augmented approach combined with the SPDNet significantly outperforms all the current state-of-the-art DL architecture in MI decoding. 
This new architecture is explainable and with a low number of trainable parameters.

## Requirement
To run the following code you need MOABB:
- Nested Cross Validation using Optuna RandomizedGridSearch you can use MOABB 1.1 [2]
- Nested Cross Validation + MDOP use the branch "https://github.com/carraraig/moabb/tree/Takens_NoParallel_1Metric"

All the packages dependencies are listed in the environment.yml file.

## Example of usage (Optuna)
```python

import sys
import os
import moabb
import mne

import resource
from moabb.paradigms import MotorImagery
from moabb.datasets import BNCI2014001
from sklearn.preprocessing import LabelEncoder
from pyriemann.estimation import Covariances
from sklearn.pipeline import Pipeline
from moabb.evaluations import WithinSessionEvaluation
from moabb.pipelines.features import AugmentedDataset
from skorch import NeuralNetClassifier
from skorch.dataset import ValidSplit
from geoopt.optim import RiemannianAdam
import copy
from skorch.callbacks import EarlyStopping, EpochScoring
from skorch.callbacks import WandbLogger
import wandb
from moabb.utils import setup_seed
from PhaseSPDNet.CallBacks import InputShapeSetter
from PhaseSPDNet.Transformer import Transform4D
import torch.nn as nn
import torch
from PhaseSPDNet.models.SPDNet_Ver import SPDNet
from PhaseSPDNet.StandardScaler import StandardScaler_Epoch

wandb.login(key="API_KEY")
wandb_run = wandb.init(project="Project_Name", name="Add_Your_Name", dir=path, reinit=True)

# Set Seed
#setup_seed(seed=42)

# Initialize parameter for the Band Pass filter
fmin = 8
fmax = 32
tmin = 0
tmax = None
fs = 250

# Select the Subject
subjects = [1]

# Deep Learning Parameter
BATCH_SIZE = 64
N_EPOCHS = 300
PATIENCE = 75
LR = 1e-2
VALID_SPLIT = 0.1

# Load the dataset
dataset = BNCI2014001()
dataset.subject_list = [int(param[0])]

# Right Hand vs Left Hand
events = ["right_hand", "left_hand"]

channels_list = [
            "C3", "Cz", "C4"
        ]

paradigm = MotorImagery(events=events, n_classes=len(events), fmin=fmin, fmax=fmax, tmax=tmax, channels=channels_list)

model = SPDNet(bnorm="brooks")

clf = NeuralNetClassifier(
    module=copy.deepcopy(model),
    criterion=nn.CrossEntropyLoss,
    optimizer=RiemannianAdam,
    optimizer__lr=LR,
    batch_size=BATCH_SIZE,
    max_epochs=N_EPOCHS,
    train_split=ValidSplit(VALID_SPLIT, random_state=42),
    device=device,
    callbacks=[EarlyStopping(monitor='valid_loss', patience=PATIENCE),
               EpochScoring(scoring='accuracy', on_train=True, name='train_acc',
                            lower_is_better=False),
               EpochScoring(scoring='accuracy', on_train=False, name='valid_acc',
                            lower_is_better=False),
               InputShapeSetter(),
               WandbLogger(wandb_run, save_model=True) #Comment if you don't want to track
               ],
    verbose=1
)
clf.initialize()

# Pipelines
pipelines = {}
pipelines["Phase-SPDNet(Optuna)"] = Pipeline(steps=[
    ("augmenteddataset", AugmentedDataset()),
    ("StandardScaler", StandardScaler_Epoch()),
    ("Covariances", Covariances("cov")),
    ("Transform4D", Transform4D()),
    ("SPDNet", clf)
])
# ====================================================================================================================
# GridSearch
# ====================================================================================================================
param_grid = {}
param_grid["Phase-SPDNet(Optuna)"] = {
    'augmenteddataset__order': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'augmenteddataset__lag': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
}
# Evaluation For MOABB
# ========================================================================================================
# Select an evaluation Within Session
evaluation = WithinSessionEvaluation(paradigm=paradigm,
                                     datasets=dataset,
                                     overwrite=True,
                                     random_state=42,
                                     hdf5_path=path,
                                     n_jobs=1,
                                     optuna=True)

result = evaluation.process(pipelines, param_grid)

# Close file and save the result
# =================================================================================================================
# Save the final Results
result.to_csv(os.path.join(path, "results.csv"))

```

## References:
[1] Carrara, I., Aristimunha, B., Corsi, M. C., de Camargo, R. Y., Chevallier, S., & Papadopoulo, T. (2024). Geometric neural network based on phase space for BCI decoding. arXiv preprint arXiv:2403.05645.

[2] Aristimunha, B., Carrara, I., Guetschel, P., Sedlar, S., Rodrigues, P., Sosulski, J., Narayanan, D., Bjareholt, E., Quentin, B., Schirrmeister, R. T., Kalunga, E., Darmet, L., Gregoire, C., Abdul Hussain, A., Gatti, R., Goncharenko, V., Thielen, J., Moreau, T., Roy, Y., … Chevallier, S. (2023). Mother of all BCI Benchmarks (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.10034224
