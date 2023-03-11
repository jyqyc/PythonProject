import torch
from torch.utils.data import Dataset,DataLoader

data_path=r"C:\Users\Administrator\Desktop\pytorch_data\SMSSpamCollection"

#完成数据集类
class MyDataset(Dataset):
    def __init__(self):
        self.lines=open(data_path,encoding='UTF-8').readlines()

    def __getitem__(self, index):
        #获取索引对应位置的数据
        cur_line=self.lines[index].strip()
        label=cur_line[:4].strip()
        content=cur_line[4:].strip()
        return label,content

    def __len__(self):
        #返回总数量
        return len(self.lines)

my_dataset=MyDataset()
dataloader=DataLoader(dataset=my_dataset,batch_size=2, shuffle=True)
if __name__=='__main__':
    #print(my_dataset[0])
    #print(len(my_dataset))

    for i in dataloader:
        print(i)