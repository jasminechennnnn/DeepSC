# Deep Learning Enabled Semantic Communication Systems

<center>Huiqiang Xie, Zhijin Qin, Geoffrey Ye Li, and Biing-Hwang Juang </center>

This is the implementation of Deep Learning Enabled Semantic Communication Systems.

## 0. Environment
```shell
conda create -n deepsc python=3.8
conda activate deepsc
pip install -r requirements.txt
```

## 1. Data Preprocessing
You can skip this step as preprocessed data is already included in the `data/` directory.
```shell
mkdir data
wget http://www.statmt.org/europarl/v7/europarl.tgz
tar zxvf europarl.tgz
python preprocess_text.py
```

## 2. Training
* Please carefully set the mutual information coefficient $\lambda$ using the `--lamb` parameter
* The default work directory is `checkpoints/`

```shell
python train.py 
```

## 3. Evaluation
* Note: If you want to compute sentence similarity, please download the BERT model first.

```shell
python performance.py
```

## Bibtex
```bitex
@article{xie2021deep,
  author={H. {Xie} and Z. {Qin} and G. Y. {Li} and B. -H. {Juang}},
  journal={IEEE Transactions on Signal Processing}, 
  title={Deep Learning Enabled Semantic Communication Systems}, 
  year={2021},
  volume={Early Access}}
```