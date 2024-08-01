from torch.utils.data import IterableDataset, DataLoader, Dataset
from datasets import load_dataset
import json

# 一
class MyDatasetLoader(Dataset):
    def __init__(self, file_path):
        self.data = []
        self.file_path = file_path
        self.load()
        print("数据长度：", len(self.data))

    def load(self):
        with open(self.file_path, "r", encoding='utf-8') as f:
            for line in f.readlines():
                self.data.append(json.loads(line))

    def __getitem__(self, key):
        return self.data[key]
    
    def __len__(self):
        return len(self.data)
    

# 二
class MyIterableDataset(IterableDataset):
    def __init__(self, file_path, batch_size):
        self.data = []
        self.file_path = file_path
        self.start_index = 0
        self.batch_size = batch_size
        self.load()
        print("数据长度：", len(self.data))

    def load(self):
        with open(self.file_path, "r", encoding='utf-8') as f:
            for line in f.readlines():
                self.data.append(json.loads(line))

    def __iter__(self):
        return iter(self.data[self.start_index: self.start_index+self.batch_size])


# 三
class DataGenerator:
    def __init__(self, file_path):
        self.data = []
        self.file_path = file_path
        self.load()
        print("数据长度：", len(self.data))

    def load(self):
        with open(self.file_path, "r", encoding='utf-8') as f:
            for line in f.readlines():
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


batch_size = 4
file_path = "school_math_0.25M.json"

#
mydataset = MyDatasetLoader(file_path)
print(mydataset[0])

print("-"*100)

#
myiterdataset = MyIterableDataset(file_path=file_path, batch_size=batch_size)
print([i for i in myiterdataset])

print("-"*100)

# 
dg = DataGenerator(file_path)
dl = DataLoader(dg, batch_size=batch_size, shuffle=False)
for index, item in enumerate(dl):
    print(item)
    break