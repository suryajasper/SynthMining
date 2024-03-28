import os
import random
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class MultiClassDataset(Dataset):
    def __init__(self, data_path: str, category_max=999, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.category_max = category_max
        self.classes, self.class_to_idx = self._find_classes()
        self.image_paths = self._load_image_paths()

    def _find_classes(self):
        classes = [d for d in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def _load_image_paths(self):
        accepted_types = ('.png', '.jpg', '.jpeg')
        image_paths = []
        
        for class_name in self.classes:
            class_path = os.path.join(self.data_path, class_name)
            if os.path.isdir(class_path):
                images = [
                    os.path.join(class_path, img) 
                        for img in os.listdir(class_path) 
                            if img.endswith(accepted_types)
                ]
                if self.category_max is not None:
                    random.shuffle(images)
                    images = images[:self.category_max]
                image_paths.append(images)
        
        if len(image_paths) == 0:
            image_paths.append([
                os.path.join(self.data_path, img) 
                    for img in os.listdir(self.data_path) 
                        if img.endswith(accepted_types)
            ])
        
        return image_paths

    def _load_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return sum([min(len(imgs), self.category_max) for imgs in self.image_paths])

    @property
    def num_labels(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        n_traversed = 0
        for label, category in enumerate(self.image_paths):
            if index < n_traversed + len(category):
                image = self._load_image(category[index - n_traversed])
                label_onehot = torch.eye(self.num_labels)[label]
                return image, label_onehot
            else:
                n_traversed += len(category)
        
        raise Exception('Could not get item')

class MultiLabelDataset(Dataset):
    def __init__(self, data_path: str, label_path: str, category_max=999, transform=None):
        self.data_path = data_path
        self.label_path = label_path
        self.transform = transform
        self.category_max = category_max
        self.image_paths, self.image_labels, self.label_names = self._load_images()

    def _load_images(self):
        accepted_image_types = ('.png', '.jpg', '.jpeg')
        image_paths = []
        
        accepted_label_types = ('.csv', '.txt')
        labels = []

        assert self.label_path.endswith(accepted_label_types), \
                f"incorrect label file type, accepts: {', '.join(accepted_label_types)}" 

        df = pd.read_csv(self.label_path)
        label_names = list(df.columns)[1:]

        for label_entry in df.values:
            image_id = label_entry[0]

            assert image_id.endswith(accepted_image_types), \
                    f"image {image_id} incorrect file type, accepts: {', '.join(accepted_image_types)}"

            image_path = os.path.join(self.data_path, image_id)

            assert os.path.exists(image_path), f'could not find image {image_path}'

            image_paths.append(image_path)
            labels.append(label_entry[1:].astype(int)[:self.category_max])

        return image_paths, labels, label_names
    
    def _load_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.image_paths)

    @property
    def num_labels(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path, label = self.image_paths[index], self.image_labels[index]
        image = self._load_image(image_path)
        label = torch.Tensor(label)
        return image, label

if __name__ == '__main__':
    # faces dataset: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
    # animal dataset: https://www.kaggle.com/datasets/alessiocorrado99/animals10
    data_path = 'data/faces/faces'
    label_path = 'data/faces/faces_labels.csv'
    batch_size = 4
    image_size = 256
    print(f'testing dataloader with batch size: {batch_size}, image_size: {image_size}')

    dataset = MultiLabelDataset(data_path=data_path, label_path=label_path, category_max=50,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0) 

    print('Number of Labels: ', len(dataset.label_names))

    print(f'dataloader initialized with {dataset.num_labels} labels and {len(dataloader)} batches')
    for i, (img_batch, label_batch) in enumerate(dataloader):
        print(i, f'image batch shape: {img_batch.shape}, label batch size: {len(label_batch)}')
        break
