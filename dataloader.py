from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
from PIL import Image

# load classes
def class_name_list():
    path = "./dataset/classes.txt"
    
    with open(path) as f:
        classes = [x.strip() for x in f.readlines()]
    return classes


class CUB200(Dataset):
    def __init__(self, is_train):
        self.is_train = is_train
        self.base_transform = [transforms.Resize((448, 448), Image.BILINEAR),
                                    transforms.RandomCrop((384, 384)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]

        if self.is_train:
            self.classes_id = []
            self.imgs = []

            path = "./dataset/training_labels.txt"
            with open(path, "r") as f:
                for line in f.readlines():
                    l=line.split(" ")
                    self.imgs.append(Image.open("./dataset/train/" + l[0]).convert('RGB'))
                    c = l[1].split(".")
                    self.classes_id.append(int(c[0])-1)  #-1 for match the class list
                f.close
            
            self.transform = transforms.Compose([transforms.RandomHorizontalFlip(), *self.base_transform])
            
        else:
            self.imgs = []
            path = "./dataset/testing_img_order.txt"
            with open(path) as f:
                self.imgs_dir = [x.strip() for x in f.readlines()]
            
            for i in range(len(self.imgs_dir)):
                self.imgs.append(Image.open("./dataset/test/" + self.imgs_dir[i]).convert('RGB'))

            self.transform = transforms.Compose([transforms.Resize((448, 448), Image.BILINEAR),
                                    transforms.CenterCrop((384, 384)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
) 

    def __getitem__(self, idx):
        if self.is_train:
            return self.transform(self.imgs[idx]), self.classes_id[idx]
        else:
            return self.transform(self.imgs[idx])
    
    def __len__(self):
        return len(self.imgs)

