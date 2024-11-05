## Multi-scale feature reconstruction network for industrial anomaly detection

<p align="center">
  <img src=assets/image/MSFR.png width="80%">
</p>

This is an official PyTorch implementation of the paper [Multi-scale feature reconstruction network for industrial anomaly detection](https://www.sciencedirect.com/science/article/pii/S095070512401284X).
```
@article{iqbal2024multi,
  title={Multi-scale feature reconstruction network for industrial anomaly detection},
  author={Iqbal, Ehtesham and Khan, Samee Ullah and Javed, Sajid and Moyo, Brain and Zweiri, Yahya and Abdulrahman, Yusra},
  journal={Knowledge-Based Systems},
  pages={112650},
  year={2024},
  publisher={Elsevier}
}
```

### Datasets

Compared to existing datasets, AeBAD has the following two characteristics: 1.) The target samples are not aligned and at different sacles. 2.) There is a domain shift between the distribution of normal samples in the test set and the training set, where the domain shifts are mainly caused by the changes in illumination and view.

**Download dataset at [here](https://drive.google.com/file/d/14wkZAFFeudlg0NMFLsiGwS0E593b-lNo/view?usp=share_link) (Google Drive) or [here](https://cloud.189.cn/web/share?code=nYraE3uMRJn2) (access code: g4pr) (Tian Yi Yun Pan).**

* AeBAD-S

<p align="center">
  <img src=assets/image/dataset_s.png width="80%">
</p>

### Get Started

#### Pre-trained models

Download the pre-trained model of MAE (ViT-large) at [here](https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large.pth).

#### Dataset

**MVTec:**

**Create the MVTec dataset directory**. Download the MVTec-AD dataset from [here](https://www.mvtec.com/company/research/datasets/mvtec-ad). The MVTec dataset directory should be as follows. 

```
|-- data
    |-- MVTec-AD
        |-- mvtec_anomaly_detection
            |-- object (bottle, etc.)
                |-- train
                |-- test
                |-- ground_truth
```

**AeBAD:**

Download the AeBAD dataset from the above link. The AeBAD dataset directory should be as follows.

```
|-- AeBAD
    |-- AeBAD_S
        |-- train
            |-- good
                |-- background
        |-- test
                |-- ablation
                    |-- background
        |-- ground_truth
                |-- ablation
                    |-- view
    |-- AeBAD_V
        |-- test
            |-- video1
                |-- anomaly
        |-- train
            |-- good
                |-- video1_train
```

**Note that background, view and illumination in the train set is different from test. The background, view and illumination in test is unseen for the training set.**

#### Virtual Environment

Use the following commands:
```
pip install -r requirements.txt
```

#### Train and Test for MVTec, AeBAD

Train the model and evaluate it for each category or different domains. This will output the results (sample-level AUROC, pixel-level AUROC and PRO) for each category. It will generate the visualization in the directory.

run the following code:

```
sh mvtec_run.sh
```

```
sh AeBAD_S_run.sh
```

TRAIN.MSFR.model_chkpt in MSFR.yaml is the path of above download model. TRAIN.dataset_path (TEST.dataset_path) is the path of data.
Set Test.save_segmentation_images as True or False to save processed image.


## Acknowledgement
We acknowledge the excellent implementation from [MAE](https://github.com/facebookresearch/mae), [ViTDet](https://github.com/facebookresearch/detectron2/tree/main/projects/ViTDet).

## License
The data is released under the CC BY 4.0 license.


