import gzip
import numpy as np
import random
import torch 

def train():
    ob_type=np.dtype(np.int32).newbyteorder('>') 
    way='D:\HYPERPC\Desktop\Games\c\Python\Mnist' #Путь до нужных данных
    tr_way=way+'train-images-idx3-ubyte.gz'
    tr_i=gzip.open(tr_way,'r')
     
    magic=np.frombuffer(tr_i.read(4),dtype=ob_type)[0]
    n_images=np.frombuffer(tr_i.read(4),dtype=ob_type)[0]    #60000 
    n_rows=np.frombuffer(tr_i.read(4),dtype=ob_type)[0]      #28
    n_columns=np.frombuffer(tr_i.read(4),dtype=ob_type)[0]   #28
     
    all= tr_i.read(n_images*n_columns*n_rows) 
    images=np.frombuffer(all,dtype=np.uint8).astype(np.float32)
    images=np.reshape(images,(n_images,n_rows,n_columns))  #из 1д в 3д
    images=images/255
     
    images = torch.tensor(images)
    tr_i.close
     
    tr_l_way = way+'train-labels-idx1-ubyte.gz'
    tr_l=gzip.open(tr_l_way,'r')

    l_magic=np.frombuffer(tr_l.read(4), dtype=ob_type)[0]
    n_labels = np.frombuffer(tr_l.read(4), dtype=ob_type)[0]
    l_all= tr_l.read(n_labels)
    labels = np.frombuffer(l_all, dtype = np.uint8)
    labels = torch.tensor(labels, dtype = torch.long)

    tr_l.close()

    permutation = np.random.permutation(len(labels)) 
    images = images[permutation] 
    labels = labels[permutation]
     
    print(images)
    print(len(images))

    print(labels)
    print(len(labels))

 
    images = np.roll(images, random.randint(0,len(images)))

    print(images)
    print(len(images))
    
    print(labels)
    print(len(labels))

train() 