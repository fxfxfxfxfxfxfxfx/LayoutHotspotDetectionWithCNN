from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import time
import os
from EfficientNet.model import EfficientNet
# from sklearn.metrics import accuracy_score

# some parameters
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
use_gpu = torch.cuda.is_available()
data_dir = 'dataset'
batch_size = 16
lr = 0.01
momentum = 0.9
num_epochs = 15
input_size = 300
class_num = 2 
net_name = 'efficientnet-b0' 


def loaddata(data_dir, batch_size, set_name, shuffle):
	data_transforms = {
		'train': transforms.Compose([
		transforms.Resize(input_size),
		transforms.CenterCrop(input_size),
		transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		]),
		'test': transforms.Compose([
		transforms.Resize(input_size),
		transforms.CenterCrop(input_size),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		]),
}

	image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in [set_name]}
	# num_workers=0 if CPU else =1
	dataset_loaders = {x: torch.utils.data.DataLoader(image_datasets[x],
  		batch_size=batch_size,
  		shuffle=shuffle, num_workers=1) for x in [set_name]}
	data_set_sizes = len(image_datasets[set_name])
	return dataset_loaders, data_set_sizes

def test_model(model, criterion):
	model.eval()
	running_loss = 0.0
	running_corrects = 0
	cont = 0
	outPre = []
	outLabel = []
	dset_loaders, dset_sizes = loaddata(data_dir=data_dir, batch_size=16, set_name='test', shuffle=False)
	for data in dset_loaders['test']:
		inputs, labels = data
		labels = torch.squeeze(labels.type(torch.LongTensor))
		inputs, labels = Variable(inputs), Variable(labels)
		outputs = model(inputs)
		_, preds = torch.max(outputs.data, 1)
		loss = criterion(outputs, labels)
		if cont == 0:
			outPre = outputs.data.cpu()
			outLabel = labels.data.cpu()
		else:
			outPre = torch.cat((outPre, outputs.data.cpu()), 0)
			outLabel = torch.cat((outLabel, labels.data.cpu()), 0)
		running_loss += loss.item() * inputs.size(0)
		running_corrects += torch.sum(preds == labels.data)
		cont += 1
	print('Loss: {:.4f} Acc: {:.4f}'.format(running_loss / dset_sizes,
										running_corrects.double() / dset_sizes))







# train
# pth_map = {
# 	'efficientnet-b0': 'adv-efficientnet-b0-b64d5a18.pth',
# 	'efficientnet-b1': 'efficientnet-b1-f1951068.pth',
# 	'efficientnet-b2': 'efficientnet-b2-8bb594d6.pth',
# 	'efficientnet-b3': 'efficientnet-b3-5fb5a3c3.pth',
# 	'efficientnet-b4': 'efficientnet-b4-6ed6700e.pth',
# 	'efficientnet-b5': 'efficientnet-b5-b6417697.pth',
# 	'efficientnet-b6': 'efficientnet-b6-c76e70fd.pth',
# 	'efficientnet-b7': 'efficientnet-b7-dcc49843.pth',
# }
# 自动下载到本地预训练
# model_ft = EfficientNet.from_pretrained('efficientnet-b0')
# 离线加载预训练，需要事先下载好
model_ft = EfficientNet.from_name(net_name)
state_dict = torch.load('dataset/model/efficientnet-b0.pth')  
model_ft.load_state_dict(state_dict)  
  
# test
print('-' * 10)
print('Test Accuracy:')
criterion = nn.CrossEntropyLoss()
test_model(model_ft, criterion)
