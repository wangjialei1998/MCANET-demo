from torchensemble import VotingClassifier  # voting is a classic ensemble strategy
from torch.utils.data import DataLoader
from torch import nn
import timm
from timm.data import create_loader
from torchensemble.utils.logging import set_logger
from ODIR_DATASET import ODIR_TEST_DATASET,TEST_DATA_PATH,TRAIN_DATA_PATH,ODIR_VALID_DATASET,VALID_DATA_PATH,TEST_DATA_PATH,ODIR_DATASET
import os
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES']='3'
data_ = pd.read_csv('/disk1/wangjialei/research/odir_main/balanced_labels0.csv')
# Load data
dataset_train=ODIR_DATASET(images_names=data_)
dataset_eval= ODIR_VALID_DATASET()
model=timm.create_model("res2net50d",pretrained=True,num_classes=8,drop_rate=0.1)
loader_train = create_loader(
    dataset_train,
    input_size=(3,384,384),
    batch_size=16,
    is_training=True,
    use_prefetcher=False,
    interpolation='bicubic',
    no_aug=False,
    hflip=0.5,
    vflip=0.5,
    color_jitter=0.5,
    mean=(0.4407552, 0.28228086, 0.15446076),
    std=(0.254417, 0.17148255, 0.0995115),
    num_workers=16,
    distributed=False,
    crop_pct=False,
    pin_memory=True
)
dataset_eval= ODIR_VALID_DATASET()

loader_eval = create_loader(
    dataset_eval,
    input_size=(3,384,384),
    batch_size=16,
    is_training=False,
    use_prefetcher=False,
    interpolation='bicubic',
    mean=(0.4407552, 0.28228086, 0.15446076),
    std=(0.254417, 0.17148255, 0.0995115),
    num_workers=16,
    distributed=False,
    crop_pct=False,
    pin_memory=False,
)
# Define the ensemble
ensemble = VotingClassifier(
    estimator=model,               # here is your deep learning model
    n_estimators=10,    
    cuda=True                    # number of base estimators
)
logger = set_logger('ensemble.res2net50d')
# Set the criterion
criterion = nn.BCEWithLogitsLoss()         # training objective
ensemble.set_criterion(criterion)

# Set the optimizer
ensemble.set_optimizer(
    "Adam",                                 # type of parameter optimizer
    lr=1e-1,                       # learning rate of parameter optimizer
    weight_decay=0.0001,              # weight decay of parameter optimizer
)

# Set the learning rate scheduler
ensemble.set_scheduler(
    "CosineAnnealingLR",                    # type of learning rate scheduler
    T_max=50,                           # additional arguments on the scheduler
)

# Train the ensemble
ensemble.fit(
    loader_train,
    epochs=50,  
    test_loader=loader_eval                        # number of training epochs
)
