# TSGNN-QA

## Installation
* Python 3.x(tested on Python 3.8)
* Pytorch 1.2.0
* torch
* datasets
* numpy
* torchdrug
* dgl
```bash
pip install python 3.6
pip install pytorch
pip install torch
pip install datasets
pip install numpy
pip install torchdrug 
pip install cudatoolkit -c milagraph -c pytorch -c pyg
pip install easydict pyyaml -c conda-forge
pip install dgl
```
## Train and Test

### 1. Preprocess data for TSGNN-QA
Dataset and pre trained model download: Download and decompress data.zip and models.zip from the root directory. Dataset download link:

```bash
https://drive.google.com/drive/folders/1aS2s5sZ0qlDpGZ9rdR7HcHym23N3pUea?usp=sharing
```

### 2. Training for TSGNN-QA

execute the following command for training.
```bash
python ./src/main.py --gpu 1
```

### 3. Testing for TSGNN-QA
Execute the following command to test model performance and save the model
```bash
python ./src/main.py --test True --gpu 1
```

### 4. Other parameter settings in the model
Including other parameters such as batch-size, dataset, multi-step, top-k, n-epochs, etc.
The setting of these parameters is as follows:
```bash
python ./src/main.py --batch-size 1 --d True --multi-step False --topk 10 --n-epochs 30 --gru 1
```
