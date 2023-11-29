import torch
import torch.nn as nn
from torch.nn import init
from torch.utils.data import Dataset
import numpy as np

class Temporal_Dataset(Dataset):
	def __init__(self, file_name,starting = 0,skip_rows=0, div =3600):
		self.data = np.loadtxt(fname=file_name, skiprows=skip_rows)[:,[0,1,3]]  # src, dst, pos, time
		self.time = self.data[:,2]  # time
		self.trans_time = (self.time - self.time[0])/div  # 递增的时间，选择时间间隔; 小时
		self.data[:,2] = self.trans_time  # 从新赋值时间
		self.data[:,[0,1]] = self.data[:,[0,1]] - starting  # 从0开始编号

	def __len__(self):
		return self.time.shape[0]

	def __getitem__(self,idx):
		sample = self.data[idx,:]
		return sample


