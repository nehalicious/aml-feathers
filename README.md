# aml-feathers

### Project setup
I added the following directories:
1. `baseline`: Contains the notebook to run the pre-trained model from HuggingFace
2. `eda`: Contains any EDA work we might do
3. `model`: Contains the code to train our own model

### Data
The dataset is quite large, so I didn't commit it to github.

1. Download the dataset from [Kaggle](https://www.kaggle.com/competitions/aml-2025-feathers-in-focus/data) and save it in the folder called `data` at the root of the project.
2. The initial dataset contains some nested directories, ie: `test_images/test_images/<...actual images>` and `train_images/train_images/<...actual images>`. This nesting sometimes caused irritating issues around the file not being found. For convenince, unnest it into the following structure.
```shell
| data
| -- test_images
| ------- 1.jpg
| ------- 2.jpg
| ------- and so on
| -- train_images
| ------- 1.jpg
| ------- and so on
| attributes.py
| attributes.npy
| class_names.npy
| test_images_path.csv
| test_images_sample.csv
| train_images.csv
```


### Running the code:
1. Ensure you're running python 3.12 (DO NOT use python 3.13 as some huggingFace and pytorch utilties don't yet support python 3.13)
2. Install the requirements using: `pip install -r requirements.txt`
3. You should now be able to run the notebook  `baseline/baseline.ipnyb` to get results from the pretrained huggingface model

