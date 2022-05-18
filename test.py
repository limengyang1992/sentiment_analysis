import torch
import torch.nn as nn
import torch.nn.functional as F

import json
import numpy as np
from lstm import BiLSTMWithAttention

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('stoi.json', 'r', encoding="utf-8") as f:
    vocab = json.loads(f.read())

def convert(string,max_len=128):
    result = [1]*max_len
    for i,word in enumerate(string[:max_len]):
        result[i] = int(vocab.get(word, vocab.get("<unk>")))
    return np.array(result)
        

if __name__ == '__main__':
    
    # string = "性价比高，机器做工大方，合金手托板，驱动程序好找，"
    # string = "就是系统难装"
    string = "不错，下次还考虑入住。交通也方便，在餐厅吃的也不错。"
    
    # load model
    model = BiLSTMWithAttention(4640).to(device)
    model.load_state_dict(torch.load("best.pth",map_location=torch.device('cpu')))
    model.eval()
    x = torch.LongTensor(convert(string).reshape(1, -1)).to(device)
    predict = F.softmax(model(x),dim=1) 
    print(predict)
