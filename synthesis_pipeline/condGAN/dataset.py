import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
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
        accepted_types = ('png', 'jpg', 'jpeg')
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
                return image, label
            else:
                n_traversed += len(category)
        
        raise Exception('Could not get item')

if __name__ == '__main__':
    data_path = 'data/animals'
    batch_size = 4
    image_size = 256
    print(f'testing dataloader with batch size: {batch_size}, image_size: {image_size}')
    
    dataset = MultiClassDataset(data_path=data_path, category_max=50,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0) 
    
    print(f'dataloader initialized with {dataset.num_labels} labels and {len(dataloader)} batches')
    for i, (img_batch, label_batch) in enumerate(dataloader):
        print(i, f'image batch: {img_batch.shape}, label batch: {label_batch.shape}')
        break
        