# Fork of brain train to generate backbone models for [PEFSL project](https://https://github.com/antoine-lavrard/PEFSL)

- Example scripts for training are available in [launch_training.sh](launch_training.sh),[launch_training_resnet12.sh](launch_training_resnet12.sh). 
- One can evaluate the accuracy of the models in few shot tasks with [evaluation.sh](evaluation.sh).
- [to_csv.sh](to_csv.sh) allows to export the results of the evaluation in a csv file.

# setup dataset : 

### working environement :



You can install necessary package using pip : 
```Bash

pip install -r requirements.txt
```


### Setup Datasets
This step is designed to prepare the datasets for later use. In a first place, you must dowload them, and put them into a directory. Once this is done, by running the following command, some meta-data about the dataset (per example number of class, number of samples per class) will be added as a dataset.json in the same folder. 

Main available datasets (make sure the name are corrects : ) (check the [script](create_dataset_file.py) for all available datasets)

- miniimagenetimages
- tieredimagenet
- cifar_fs
- imagenet

linux :
```Bash

python create_dataset_files.py --dataset-path /home/y17bendo/Documents/datasets/
```

## adapt the scripts :
change the dataset-path to correspond to your path to the scripts