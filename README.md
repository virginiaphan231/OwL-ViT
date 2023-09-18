# This is OwL-ViT project 
### Dataset 
This is project is fintuned with ***cocominitrain***. This dataset is sampled from orginial COCO dataset with 20% number of images from COCO. 
***cocominitrain*** could be downloaded with this link: https://ln5.sync.com/dl/0324da1d0/rmi7abjx-2dj4ktii-d9jcwgc5-s7fwwrb7/view/default/12056974190004. 

Additionally, we also need to downloaded updated annotation file via this link: https://drive.google.com/file/d/1lezhgY4M_Ag13w0dEzQ7x_zQ_w0ohjin/view

### Data preparation
Once you have everything unzipped, run:
```python index_convert.py```
Note: change number of train images and test images in file ***config.yaml*** to be exact the same of number in the dataset (to avoid running sampling dataset again).

### Install dependencies
```pip install -r requirements.txt```

### Training
```python -W ignore train.py```


#### Config
Please modify values of hyperparameters as needed in file ***config.yalm*** as needed.

