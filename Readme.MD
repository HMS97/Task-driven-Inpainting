# Coarse-to-fine Task-driven Inpainting for Geoscience Images
This repository contains the implementation of the research paper "Coarse-to-fine Task-driven Inpainting for Geoscience Images," accepted in IEEE Transactions on Circuits and Systems for Video Technology (TCSVT) in 2023.




## Dataset and Pretrained Models  for Classification


### 1.Download the [Dataset](https://drive.google.com/file/d/1A8aScQzxEraahqkFRnJVEqBgnKvVFyUW/view?usp=drive_link) into the current folder and unzip


### 2.Download the classification [Pretained Models](https://drive.google.com/drive/folders/1ydRKKrytJugLi7h5vIpkX-ZNdGRjvzqu?usp=sharing) into the save_models folder


# Usage
#### The Requirements

```
pip install -r requirements.txt
```

#### Start the Training

```
python gan_trainer.py
```

#### Start the Testing

```
python gan_test.py

```

#### Start the Evaluation

```
python baseline.py
```


# Citation
If you find the article or code useful for your project, please refer to


```
@ARTICLE{10126081,
  author={Sun, Huiming and Ma, Jin and Guo, Qing and Zou, Qin and Song, Shaoyue and Lin, Yuewei and Yu, Hongkai},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Coarse-to-fine Task-driven Inpainting for Geoscience Images}, 
  year={2023},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TCSVT.2023.3276719}}
```