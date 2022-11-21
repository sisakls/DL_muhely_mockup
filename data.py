import torch
import torchvision
import torchvision.transforms as tfs
import datasets

batch_size = 64

def task_split(data, label_list):
    task_dict = {}
    
    for idx, lbl_set in enumerate(label_list):
        task = data.filter(lambda x: x['label'] in lbl_set)
        task_dict['task_{}'.format(idx)] = task
        
    return datasets.DatasetDict(task_dict)


def transform_func(examples):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transforms = tfs.ToTensor()
    
    examples["image"] = [transforms(img) for img in examples["image"]] #transforms(img).to(device)
    
    return examples


def collate_fun(examples):
    images = []
    labels = []
    for example in examples:
        images.append((example["image"]))
        labels.append(example["label"])
        
    images = torch.stack(images)
    labels = torch.tensor(labels)
    return (images, labels)


def prep_data():
    data = datasets.load_dataset("mnist")
    
    label_list = [{0,1}, {2,3}, {4,5}, {6,7}, {8,9}]
    tasks = task_split(data['train'], label_list)
    subtests = task_split(data['test'], label_list)

    ds_train = tasks.with_transform(transform_func)
    ds_test  = data['test'].with_transform(transform_func)
    ds_subtest = subtests.with_transform(transform_func)

    train_dict = {}
    for key, data in ds_train.items():
        train_dict[key] = torch.utils.data.DataLoader(data, collate_fn=collate_fun, batch_size=batch_size)
    test_dict  = {'all': torch.utils.data.DataLoader(ds_test, collate_fn=collate_fun, batch_size=batch_size)}
    for key, subtest in ds_subtest.items():
        test_dict[key] = torch.utils.data.DataLoader(subtest, collate_fn=collate_fun, batch_size=batch_size)
    
    return train_dict, test_dict