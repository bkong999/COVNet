# COVNet
This is a PyTorch implementation of the paper "Using Artificial Intelligence to Detect COVID-19 and Community Acquired Pneumonia based on Pulmonary CT: Evaluation of the Diagnostic Accuracy". It supports training, validation and testing for COVNet.

<img src="assets/overview.png" width="600">

## Setup
### Prerequisites
- Anaconda 3.7
- PyTorch 1.4
- SimpleITK
- batchgenerators
- tensorboardX

### Prepare data
Preprocess the data according to the Appendix E1 section in the paper and organize them as the following. A example of train.csv and val.csv are also provided.
```
data
├── caseid1
|   ├── masked_ct.nii
|   └── mask.nii.gz
├── caseid2
|   ├── masked_ct.nii
|   └── mask.nii.gz
├── caseid3
|   ├── masked_ct.nii
|   └── mask.nii.gz
├── caseid4
|   ├── masked_ct.nii
|   └── mask.nii.gz
├── train.csv
└── val.csv
```

## COVNet
<img src="assets/demo.png" width="600">

### Training
Training a COVNet with default arguments. Model checkpoints and tensorboard logs are written out to a unique directory created by default within `experiments/models` and `experiments/logs` respectively after starting training.
```
python main.py
```

### Validation and Testing
You can run validation and testing on the checkpointed best model by:
```
python test.py
```
