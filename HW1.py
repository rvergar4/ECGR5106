#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import torch
torch.set_printoptions(edgeitems=2, threshold=50)
import imageio
import os


# In[14]:


#Problem1
#creating dataset of multiple images, N × C × H × W tensor
#preallocate tensor of 39 items
batch_size = 39
batch = torch.zeros(batch_size, 3, 255, 255, dtype=torch.uint8)
batch.shape


# In[15]:


#Loading images and storing in tensor
data_dir = "C:/Users/rosam/OneDrive/Desktop/HW1"
filenames = [name for name in os.listdir(data_dir)
             if os.path.splitext(name)[-1] == '.png']
for i, filename in enumerate(filenames):
    img_arr = imageio.imread(os.path.join(data_dir, filename))
    img_t = torch.from_numpy(img_arr)
    img_t = img_t.permute(2, 0, 1)
    img_t = img_t[:3] # <1>
    batch[i] = img_t


# In[16]:


batch


# In[17]:


#obtain mean
batch = batch.float()
n_channels = batch.shape[1]
for c in range(n_channels):
    mean = batch[:, c].mean(-3)
    batch[:, c] = mean


# In[18]:


batch


# In[ ]:




