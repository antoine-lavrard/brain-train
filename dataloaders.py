### file to generate dataloaders
### for simplicity, we do not offer the choice to load the whole dataset in VRAM anymore

from torchvision import transforms, datasets
from args import args
import torch
import os
import json
import numpy as np
from PIL import Image
from utils import *

### first define dataholder, which will be used as an argument to dataloaders
class DataHolder():
    def __init__(self, data, targets, transforms):
        self.data = data
        if torch.is_tensor(data):
            self.length = data.shape[0]
        else:
            self.length = len(self.data)
        self.targets = targets
        assert(self.length == len(targets))
        self.transforms = transforms
    def __getitem__(self, idx):
        if isinstance(self.data[idx], str):
            elt = transforms.ToTensor()(np.array(Image.open(args.dataset_path + self.data[idx]).convert('RGB')))
        else:
            elt = self.data[idx]
        return self.transforms(elt), self.targets[idx]
    def __len__(self):
        return self.length

def dataLoader(dataholder, shuffle):
    return torch.utils.data.DataLoader(dataholder, batch_size = args.batch_size, shuffle = shuffle, num_workers = min(os.cpu_count(), 8))

def cifar10(dataset):
    pytorchDataset = datasets.CIFAR10(args.dataset_path, train = dataset != "test", download = False)
    data = torch.tensor(pytorchDataset.data).transpose(1,3).transpose(2,3).float() / 256.
    targets = pytorchDataset.targets

    normalization = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))

    trans = torch.nn.Sequential(transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), normalization) if dataset == "train" else normalization

    return {"dataloader": dataLoader(DataHolder(data, targets, trans), shuffle = dataset == "train"), "name":"cifar10_" + dataset, "num_classes":10, "name_classes": pytorchDataset.classes}

def cifar100(dataset):
    pytorchDataset = datasets.CIFAR100(args.dataset_path, train = dataset != "test", download = False)
    data = torch.tensor(pytorchDataset.data).transpose(1,3).transpose(2,3).float() / 256.
    targets = pytorchDataset.targets

    normalization = transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))

    trans = torch.nn.Sequential(transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), normalization) if dataset == "train" else normalization

    return {"dataloader": dataLoader(DataHolder(data, targets, trans), shuffle = dataset == "train"), "name":"cifar100_" + dataset, "num_classes":100, "name_classes": pytorchDataset.classes}

def miniimagenet(datasetName):
    f = open(args.dataset_path + "datasets.json")    
    all_datasets = json.loads(f.read())
    f.close()
    dataset = all_datasets["miniimagenet_" + datasetName]
    data = dataset["data"]
    targets = dataset["targets"]

    normalization = transforms.Normalize([125.3/255, 123.0/255, 113.9/255], [63.0/255, 62.1/255, 66.7/255])

    trans = torch.nn.Sequential(transforms.RandomResizedCrop(84), transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), transforms.RandomHorizontalFlip(), normalization) if datasetName == "train" else torch.nn.Sequential(transforms.Resize(92), transforms.CenterCrop(84), normalization)

    return {"dataloader": dataLoader(DataHolder(data, targets, trans), shuffle = datasetName == "train"), "name":dataset["name"], "num_classes":dataset["num_classes"], "name_classes": dataset["name_classes"]}

def tieredimagenet(datasetName):
    f = open(args.dataset_path + "datasets.json")    
    all_datasets = json.loads(f.read())
    f.close()
    dataset = all_datasets["tieredimagenet_" + datasetName]
    data = dataset["data"]
    targets = dataset["targets"]

    normalization = transforms.Normalize([125.3/255, 123.0/255, 113.9/255], [63.0/255, 62.1/255, 66.7/255])

    trans = torch.nn.Sequential(transforms.RandomResizedCrop(84), transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), transforms.RandomHorizontalFlip(), normalization) if datasetName == "train" else torch.nn.Sequential(transforms.Resize(92), transforms.CenterCrop(84), normalization)

    return {"dataloader": dataLoader(DataHolder(data, targets, trans), shuffle = datasetName == "train"), "name":dataset["name"], "num_classes":dataset["num_classes"], "name_classes": dataset["name_classes"]}

def cifarfs(datasetName):
    f = open(args.dataset_path + "datasets.json")    
    all_datasets = json.loads(f.read())
    f.close()
    dataset = all_datasets["cifarfs_" + datasetName]
    data = dataset["data"]
    targets = dataset["targets"]

    normalization = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    trans = torch.nn.Sequential(transforms.RandomResizedCrop(32), transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), transforms.RandomHorizontalFlip(), normalization) if datasetName == "train" else torch.nn.Sequential(transforms.Resize([int(1.15*32), int(1.15*32)]), transforms.CenterCrop(32), normalization)

    return {"dataloader": dataLoader(DataHolder(data, targets, trans), shuffle = datasetName == "train"), "name":dataset["name"], "num_classes":dataset["num_classes"], "name_classes": dataset["name_classes"]}

def metadataset_imagenet(datasetName):
    f = open(args.dataset_path + "datasets.json")    
    all_datasets = json.loads(f.read())
    f.close()
    dataset = all_datasets["metadataset_imagenet_" + datasetName]
    data = dataset["data"]
    targets = dataset["targets"]
    normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    trans = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), normalization]) if dataset == "train" else transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), normalization])
    return {"dataloader": dataLoader(DataHolder(data, targets, trans), shuffle = datasetName == "train"), "name":dataset["name"], "num_classes":dataset["num_classes"], "name_classes": dataset["name_classes"]}

def imagenet(dataset):
    normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    trans = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalization]) if dataset == "train" else transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalization])

    pytorchDataset = datasets.ImageNet(args.dataset_path + "/imagenet", split = "train" if dataset != "test" else "val", transform = trans)
        
    return {"dataloader": dataLoader(pytorchDataset, shuffle = dataset == "train"), "name":"imagenet_" + dataset, "num_classes":1000, "name_classes": pytorchDataset.classes}

def mnist(dataset):
    pytorchDataset = datasets.MNIST(args.dataset_path, train = dataset != "test", download = False)
    data = pytorchDataset.data.clone().unsqueeze(1).float() / 256.
    targets = pytorchDataset.targets

    normalization = transforms.Normalize((0.1302,), (0.3069,))

    trans = normalization

    return {"dataloader": dataLoader(DataHolder(data, targets, trans), shuffle = dataset == "train"), "name": "mnist_" + dataset, "num_classes": 10, "name_classes": list(range(10))}

def fashionMnist(dataset):
    pytorchDataset = datasets.FashionMNIST(args.dataset_path, train = dataset != "test", download = True)
    data = pytorchDataset.data.clone().unsqueeze(1).float() / 256.
    targets = pytorchDataset.targets

    normalization = transforms.Normalize((0.2849,), (0.3516,))

    trans = torch.nn.Sequential(transforms.RandomCrop(28, padding=4), transforms.RandomHorizontalFlip(), normalization) if dataset == "train" else normalization

    return {"dataloader": dataLoader(DataHolder(data, targets, trans), shuffle = dataset == "train"), "name": "fashion-mnist_" + dataset, "num_classes": 10, "name_classes": pytorchDataset.classes}


def metadataset_dtd(dataset_type):
    f = open(args.dataset_path + "datasets.json")    
    all_datasets = json.loads(f.read())
    f.close()
    dataset = all_datasets["metadataset_dtd_" + dataset_type]
    data = dataset["data"]
    targets = dataset["targets"]
    normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    trans = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), normalization]) if dataset == "train" else transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), normalization])
    return {"dataloader": dataLoader(DataHolder(data, targets, trans), shuffle = dataset_type == "train"), "name":'metadataset_dtd', "num_classes":dataset["num_classes"], "name_classes": dataset["name_classes"]}


def metadataset_cub(dataset_type):
    f = open(args.dataset_path + "datasets.json")    
    all_datasets = json.loads(f.read())
    f.close()
    dataset = all_datasets["metadataset_cub_" + dataset_type]
    data = dataset["data"]
    targets = dataset["targets"]
    normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    trans = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), normalization]) if dataset == "train" else transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),  normalization])
    return {"dataloader": dataLoader(DataHolder(data, targets, trans), shuffle = dataset_type == "train"), "name":'metadataset_cub', "num_classes":dataset["num_classes"], "name_classes": dataset["name_classes"]}


def metadataset_fungi(dataset_type):
    f = open(args.dataset_path + "datasets.json")    
    all_datasets = json.loads(f.read())
    f.close()
    dataset = all_datasets["metadataset_fungi_" + dataset_type]
    data = dataset["data"]
    targets = dataset["targets"]
    normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    trans = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), normalization]) if dataset == "train" else transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), normalization])
    return {"dataloader": dataLoader(DataHolder(data, targets, trans), shuffle = dataset_type == "train"), "name":'metadataset_fungi', "num_classes":dataset["num_classes"], "name_classes": dataset["name_classes"]}

def metadataset_aircraft(dataset_type):
    f = open(args.dataset_path + "datasets.json")    
    all_datasets = json.loads(f.read())
    f.close()
    dataset = all_datasets["metadataset_aircraft_" + dataset_type]
    data = dataset["data"]
    targets = dataset["targets"]
    normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    trans = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), normalization]) if dataset == "train" else transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), normalization])
    return {"dataloader": dataLoader(DataHolder(data, targets, trans), shuffle = dataset_type == "train"), "name":'metadataset_aircraft', "num_classes":dataset["num_classes"], "name_classes": dataset["name_classes"]}

def metadataset_mscoco(dataset_type):
    if dataset_type=='train':
        return 'mscoco is not a train dataset in metadataset (see official split)'
    f = open(args.dataset_path + "datasets.json")    
    all_datasets = json.loads(f.read())
    f.close()
    dataset = all_datasets["metadataset_mscoco_" + dataset_type]
    data = dataset["data"]
    targets = dataset["targets"]
    normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    trans = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), normalization]) if dataset == "train" else transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), normalization])
    return {"dataloader": dataLoader(DataHolder(data, targets, trans), shuffle = dataset_type == "train"), "name":'metadataset_mscoco', "num_classes":dataset["num_classes"], "name_classes": dataset["name_classes"]}

def prepareDataLoader(name):
    if isinstance(name, str):
        name = [name]
    result = []
    dataset_options = {
            "cifar10_train": lambda: cifar10("train"),
            "cifar10_validation": lambda: cifar10("validation"),
            "cifar10_test": lambda: cifar10("test"),
            "mnist_train": lambda: mnist("train"),
            "mnist_validation": lambda: mnist("validation"),
            "mnist_test": lambda: mnist("test"),
            "fashion-mnist_train": lambda: fashionMnist("train"),
            "fashion-mnist_validation": lambda: fashionMnist("validation"),
            "fashion-mnist_test": lambda: fashionMnist("test"),
            "imagenet_train": lambda: imagenet("train"),
            "imagenet_validation": lambda: imagenet("validation"),
            "imagenet_test": lambda: imagenet("test"),
            "miniimagenet_train": lambda: miniimagenet("train"),
            "miniimagenet_validation": lambda: miniimagenet("validation"),
            "miniimagenet_test": lambda: miniimagenet("test"),
            "tieredimagenet_train": lambda: tieredimagenet("train"),
            "tieredimagenet_validation": lambda: tieredimagenet("validation"),
            "tieredimagenet_test": lambda: tieredimagenet("test"),
            "cifarfs_train": lambda: cifarfs("train"),
            "cifarfs_validation": lambda: cifarfs("validation"),
            "cifarfs_test": lambda: cifarfs("test"),
            "metadataset_imagenet_train": lambda: metadataset_imagenet("train"),
            "metadataset_imagenet_validation": lambda: metadataset_imagenet("validation"),
            "metadataset_imagenet_test": lambda: metadataset_imagenet("test"),
            "metadataset_cub_train": lambda: metadataset_cub("train"),
            "metadataset_cub_validation": lambda: metadataset_cub("validation"),
            "metadataset_cub_test": lambda: metadataset_cub("test"),
            "metadataset_dtd_train": lambda: metadataset_dtd("train"),
            "metadataset_dtd_validation": lambda: metadataset_dtd("validation"),
            "metadataset_dtd_test": lambda: metadataset_dtd("test"),
            "metadataset_fungi_train": lambda: metadataset_fungi("train"),
            "metadataset_fungi_validation": lambda: metadataset_fungi("validation"),
            "metadataset_fungi_test": lambda: metadataset_fungi("test"),
            "metadataset_aircraft_train": lambda: metadataset_aircraft("train"),
            "metadataset_aircraft_validation": lambda: metadataset_aircraft("validation"),
            "metadataset_aircraft_test": lambda: metadataset_aircraft("test"),
            "metadataset_mscoco_train": lambda: metadataset_mscoco("train"),
            "metadataset_mscoco_validation": lambda: metadataset_mscoco("validation"),
            "metadataset_mscoco_test": lambda: metadataset_mscoco("test"),
        }
    for elt in name:
        assert elt in dataset_options.keys(), 'The chosen dataset is not existing, please provide a valid option'
        result.append(dataset_options[elt.lower()]())
    return result
    
if args.training_dataset != "":
    try:
        eval(args.training_dataset)
        trainSet = prepareDataLoader(eval(args.training_dataset))
    except NameError:
        trainSet = prepareDataLoader(args.training_dataset)
else:
    trainSet = []

if args.validation_dataset != "":
    try:
        eval(args.validation_dataset)
        validationSet = prepareDataLoader(eval(args.validation_dataset))
    except NameError:
        validationSet = prepareDataLoader(args.validation_dataset)
else:
    validationSet = []

if args.test_dataset != "":
    try:
        eval(args.test_dataset)
        testSet = prepareDataLoader(eval(args.test_dataset))
    except NameError:
        testSet = prepareDataLoader(args.test_dataset)
else:
    testSet = []
    
print(" dataloaders,", end='')
