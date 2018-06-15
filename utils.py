
# coding: utf-8

# In[1]:

 # create 5 sizes of image datasets

import numpy as np
import scipy as sp
import tensorflow as tf
import matplotlib.pyplot as plt
#%matplotlib inline

#global variables
datasets = {}
savedata = {}

#This function creates train, test and eval datasets for different scales of the MNIST dataset images.
#Params: scale: integer, specify the image scale of the returned datasets. Default is -1
              # scales: -1 => mixed (all other scales, shuffled)
              #          0 => smallest
              #          1 => small
              #          2 => normal
              #          3 => big
              #          4 => biggest
        #scale_labels: boolean, set True if a label array for image scale should be returned for each dataset. Default is False.
#Returns: dataset, labels, [scale_labels] for train, eval and test dataset for MNIST of the specified scale as numpy arrays.
def get_scaled_mnist(scale=-1, scale_labels=False):
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data = mnist.train.images
  eval_data = mnist.validation.images
  test_data = mnist.test.images 
  
  train_data = train_data.reshape(train_data.shape[0], 28, 28)
  test_data = test_data.reshape(test_data.shape[0], 28, 28)
  eval_data = eval_data.reshape(eval_data.shape[0], 28, 28)
  t_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  tes_labels = np.asarray(mnist.test.labels, dtype=np.int32)
  e_labels = np.asarray(mnist.validation.labels, dtype=np.int32)

  #image sizes sorted ascending by size. all are padded to fit into the biggest image size

  #smallest size
  if scale==0 or scale==-1:
    if (not "t_data_smallest" in datasets.keys() or not "t_data_mixed" in datasets.keys()):
      train_data_smallest = np.pad(np.array(list(map(lambda img : sp.misc.imresize(img, (12, 12)), train_data))), ((0, 0), (16, 16), (16, 16)), 'constant')
      datasets["t_data_smallest"] = train_data_smallest.astype(np.float32)
      datasets["t_labels_smallest"] = t_labels.astype(np.int32)
      if (scale_labels):
        datasets["t_scale_smallest"] = np.full(t_labels.shape, 0)
      test_data_smallest = np.pad(np.array(list(map(lambda img : sp.misc.imresize(img, (12, 12)), test_data))), ((0, 0), (16, 16), (16, 16)), 'constant')
      datasets["tes_data_smallest"] = test_data_smallest.astype(np.float32)
      datasets["tes_labels_smallest"] = tes_labels.astype(np.int32)
      if (scale_labels):
        datasets["tes_scale_smallest"] = np.full(tes_labels.shape, 0)
      eval_data_smallest = np.pad(np.array(list(map(lambda img : sp.misc.imresize(img, (12, 12)), eval_data))), ((0, 0), (16, 16), (16, 16)), 'constant')
      datasets["e_data_smallest"] = eval_data_smallest.astype(np.float32)
      datasets["e_labels_smallest"] = e_labels.astype(np.int32)
      if (scale_labels):
        datasets["e_scale_smallest"] = np.full(e_labels.shape, 0)
    if scale == 0:
      if (scale_labels):
        return datasets["t_data_smallest"], datasets["t_labels_smallest"], datasets["t_scale_smallest"], datasets["tes_data_smallest"], datasets["tes_labels_smallest"], datasets["tes_scale_smallest"], datasets["e_data_smallest"], datasets["e_labels_smallest"], datasets["e_scale_smallest"]
      return datasets["t_data_smallest"], datasets["t_labels_smallest"], datasets["tes_data_smallest"], datasets["tes_labels_smallest"], datasets["e_data_smallest"], datasets["e_labels_smallest"]

  #small size
  if scale==1 or scale==-1:
    if (not "t_data_small" in datasets.keys() or not "t_data_mixed" in datasets.keys()):
      train_data_small = np.pad(np.array(list(map(lambda img : sp.misc.imresize(img, (20, 20)), train_data))), ((0, 0), (12, 12), (12, 12)), 'constant')
      datasets["t_data_small"] = train_data_small.astype(np.float32)
      datasets["t_labels_small"] = t_labels.astype(np.int32)
      if (scale_labels):
        datasets["t_scale_small"] = np.full(t_labels.shape, 1)
      test_data_small = np.pad(np.array(list(map(lambda img : sp.misc.imresize(img, (20, 20)), test_data))), ((0, 0), (12, 12), (12, 12)), 'constant')
      datasets["tes_data_small"] = test_data_small.astype(np.float32)
      datasets["tes_labels_small"] = tes_labels.astype(np.int32)
      if (scale_labels):
        datasets["tes_scale_small"] = np.full(tes_labels.shape, 1)
      eval_data_small = np.pad(np.array(list(map(lambda img : sp.misc.imresize(img, (20, 20)), eval_data))), ((0, 0), (12, 12), (12, 12)), 'constant')
      datasets["e_data_small"] = eval_data_small.astype(np.float32)
      datasets["e_labels_small"] = e_labels.astype(np.int32)
      if (scale_labels):
        datasets["e_scale_small"] = np.full(t_labels.shape, 1)
    if scale == 1:
      if (scale_labels):
        return datasets["t_data_small"], datasets["t_labels_small"], datasets["t_scale_small"], datasets["tes_data_small"], datasets["tes_labels_small"], datasets["tes_scale_small"], datasets["e_data_small"], datasets["e_labels_small"], datasets["e_scale_small"]
      return datasets["t_data_small"], datasets["t_labels_small"], datasets["e_data_small"], datasets["e_labels_small"]

  #normal size
  if scale==2 or scale==-1:
    if (not "t_data_normal" in datasets.keys() or not "t_data_mixed" in datasets.keys()):
      train_data_norm = np.pad(train_data, ((0, 0), (8, 8), (8, 8)), 'constant')
      datasets["t_data_normal"] = train_data_norm.astype(np.float32)
      datasets["t_labels_normal"] = t_labels.astype(np.int32)
      if (scale_labels):
        datasets["t_scale_normal"] = np.full(t_labels.shape, 2)
      test_data_norm = np.pad(test_data, ((0, 0), (8, 8), (8, 8)), 'constant')
      datasets["tes_data_normal"] = test_data_norm.astype(np.float32)
      datasets["tes_labels_normal"] = tes_labels.astype(np.int32)
      if (scale_labels):
        datasets["tes_scale_normal"] = np.full(tes_labels.shape, 2)
      eval_data_norm = np.pad(eval_data, ((0, 0), (8, 8), (8, 8)), 'constant')
      datasets["e_data_normal"] = eval_data_norm.astype(np.float32)
      datasets["e_labels_normal"] = e_labels.astype(np.int32)
      if (scale_labels):
        datasets["e_scale_normal"] = np.full(t_labels.shape, 2)
    if scale == 2:
      if scale_labels:
        return datasets["t_data_normal"], datasets["t_labels_normal"], datasets["t_scale_normal"], datasets["tes_data_normal"], datasets["tes_labels_normal"], datasets["tes_scale_normal"], datasets["e_data_normal"], datasets["e_labels_normal"], datasets["e_scale_normal"]
      return datasets["t_data_normal"], datasets["t_labels_normal"], datasets["tes_data_normal"], datasets["tes_labels_normal"], datasets["e_data_normal"], datasets["e_labels_normal"]

  #big size
  if scale==3 or scale==-1:
    if (not "t_data_big" in datasets.keys() or not "t_data_mixed" in datasets.keys()):
      train_data_big = np.pad(np.array(list(map(lambda img : sp.misc.imresize(img, (36, 36)), train_data))), ((0, 0), (4, 4), (4, 4)), 'constant')
      datasets["t_data_big"] = train_data_big.astype(np.float32)
      datasets["t_labels_big"] = t_labels.astype(np.int32)
      if (scale_labels):
        datasets["t_scale_big"] = np.full(t_labels.shape, 3)
      test_data_big = np.pad(np.array(list(map(lambda img : sp.misc.imresize(img, (36, 36)), test_data))), ((0, 0), (4, 4), (4, 4)), 'constant')
      datasets["tes_data_big"] = test_data_big.astype(np.float32)
      datasets["tes_labels_big"] = tes_labels.astype(np.int32)
      if (scale_labels):
        datasets["tes_scale_big"] = np.full(tes_labels.shape, 3)
      eval_data_big = np.pad(np.array(list(map(lambda img : sp.misc.imresize(img, (36, 36)), eval_data))), ((0, 0), (4, 4), (4, 4)), 'constant')
      datasets["e_data_big"] = eval_data_big.astype(np.float32)
      datasets["e_labels_big"] = e_labels.astype(np.int32)
      if (scale_labels):
        datasets["e_scale_big"] = np.full(t_labels.shape, 3)
    if scale == 3:
      if scale_labels:
        return datasets["t_data_big"], datasets["t_labels_big"], datasets["t_scale_big"], datasets["tes_data_big"], datasets["tes_labels_big"], datasets["tes_scale_big"], datasets["e_data_big"], datasets["e_labels_big"], datasets["e_scale_big"]
      return datasets["t_data_big"], datasets["t_labels_big"], datasets["tes_data_big"], datasets["tes_labels_big"], datasets["e_data_big"], datasets["e_labels_big"]

  #biggest size
  if scale==4 or scale==-1:
    if (not "t_data_biggest" in datasets.keys() or not "t_data_mixed" in datasets.keys()):
      train_data_biggest = np.array(list(map(lambda img : sp.misc.imresize(img, (44, 44)), train_data)))
      datasets["t_data_biggest"] = train_data_biggest.astype(np.float32)
      datasets["t_labels_biggest"] = t_labels.astype(np.int32)
      if (scale_labels):
        datasets["t_scale_biggest"] = np.full(t_labels.shape, 4)
      test_data_biggest = np.array(list(map(lambda img : sp.misc.imresize(img, (44, 44)), test_data)))
      datasets["tes_data_biggest"] = test_data_biggest.astype(np.float32)
      datasets["tes_labels_biggest"] = tes_labels.astype(np.int32)
      if (scale_labels):
        datasets["tes_scale_biggest"] = np.full(tes_labels.shape, 4)
      eval_data_biggest = np.array(list(map(lambda img : sp.misc.imresize(img, (44, 44)), eval_data)))
      datasets["e_data_biggest"] = eval_data_biggest.astype(np.float32)
      datasets["e_labels_biggest"] = e_labels.astype(np.int32)
      if (scale_labels):
        datasets["e_scale_biggest"] = np.full(t_labels.shape, 4)
    if scale == 4:
      if scale_labels:
        return datasets["t_data_biggest"], datasets["t_labels_biggest"], datasets["t_scale_biggest"], datasets["tes_data_biggest"], datasets["tes_labels_biggest"], datasets["tes_scale_biggest"], datasets["e_data_biggest"], datasets["e_labels_biggest"], datasets["e_scale_biggest"]
      return datasets["t_data_biggest"], datasets["t_labels_biggest"], datasets["tes_data_biggest"], datasets["tes_labels_biggest"], datasets["e_data_biggest"], datasets["e_labels_biggest"]

  #mixed
  if scale==-1:
    if (not "t_data_mixed" in datasets.keys()):
      rng_state = np.random.get_state()
      train_data_mixed = np.random.permutation(np.concatenate((train_data_smallest, train_data_small, train_data_norm, train_data_big, train_data_biggest)))[:train_data.shape[0], :, :]
      np.random.set_state(rng_state)
      train_labels_mixed = np.random.permutation(np.concatenate((t_labels, t_labels, t_labels, t_labels, t_labels)))[:t_labels.shape[0]]
      datasets["t_data_mixed"] = train_data_mixed.astype(np.float32)
      datasets["t_labels_mixed"] = train_labels_mixed.astype(np.int32)
      if scale_labels:
        np.random.set_state(rng_state)
        datasets["t_scale_mixed"] = np.random.permutation(np.concatenate((datasets["t_scale_smallest"], datasets["t_scale_small"], datasets["t_scale_normal"], datasets["t_scale_big"], datasets["t_scale_biggest"])))[:datasets["t_scale_smallest"].shape[0]]

      rng_state = np.random.get_state()
      test_data_mixed = np.random.permutation(np.concatenate((test_data_smallest, test_data_small, test_data_norm, test_data_big, test_data_biggest)))[:test_data.shape[0], :, :]
      np.random.set_state(rng_state)
      test_labels_mixed = np.random.permutation(np.concatenate((tes_labels, tes_labels, tes_labels, tes_labels, tes_labels)))[:tes_labels.shape[0]]
      datasets["tes_data_mixed"] = test_data_mixed.astype(np.float32)
      datasets["tes_labels_mixed"] = test_labels_mixed.astype(np.int32)
      if scale_labels:
        np.random.set_state(rng_state)
        datasets["tes_scale_mixed"] = np.random.permutation(np.concatenate((datasets["tes_scale_smallest"], datasets["tes_scale_small"], datasets["tes_scale_normal"], datasets["tes_scale_big"], datasets["tes_scale_biggest"])))[:datasets["tes_scale_smallest"].shape[0]]

      rng_state = np.random.get_state()
      eval_data_mixed = np.random.permutation(np.concatenate((eval_data_smallest, eval_data_small, eval_data_norm, eval_data_big, eval_data_biggest)))[:eval_data.shape[0], :, :]
      np.random.set_state(rng_state)
      eval_labels_mixed = np.random.permutation(np.concatenate((e_labels, e_labels, e_labels, e_labels, e_labels)))[:e_labels.shape[0]]
      datasets["e_data_mixed"] = eval_data_mixed.astype(np.float32)
      datasets["e_labels_mixed"] = eval_labels_mixed.astype(np.int32)
      if scale_labels:
        np.random.set_state(rng_state)
        datasets["e_scale_mixed"] = np.random.permutation(np.concatenate((datasets["e_scale_smallest"], datasets["e_scale_small"], datasets["e_scale_normal"], datasets["e_scale_big"], datasets["e_scale_biggest"])))[:datasets["e_scale_smallest"].shape[0]]
    if scale == -1:
      if scale_labels:
        return datasets["t_data_mixed"], datasets["t_labels_mixed"], datasets["t_scale_mixed"], datasets["tes_data_mixed"], datasets["tes_labels_mixed"], datasets["tes_scale_mixed"], datasets["e_data_mixed"], datasets["e_labels_mixed"], datasets["e_scale_mixed"]
      return datasets["t_data_mixed"], datasets["t_labels_mixed"], datasets["tes_data_mixed"], datasets["tes_labels_mixed"], datasets["e_data_mixed"], datasets["e_labels_mixed"]

def  getActivations (sess,t_data,layer,stimuli):
    activations = sess.run(layer,feed_dict={t_data:stimuli})
    #plotNNFilter(activations)
    return(activations[0, :])

import tensorflow as tf
import numpy as np
import scipy as sp

class Data:
    
    scaled_data = []
    
    scalename = ['smallest','small','normal','bigger','biggest']
    scales = np.arange(5)
    
    sizes = [12,20,28,36,44]    #for resizing the original images
    pads = [16,12,8,4,0]        #for padding the scaled images with zero
    
    def __init__(self,fashion=False):
        ''' create datasets as ScData classes for different scales and saves the objects in scaled_data
        '''
        self.scaled_data = []
        #load data
        if fashion:
            (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.fashion_mnist.load_data()
        else:
            (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data()
        print(self.x_train.shape)
        print(self.x_test.shape)
        
        for i in self.scales:
            self.scaled_data.append(ScData(self,self.sizes[i],self.pads[i]))
            
    def get_mixed(self,scale_list,scale_labels=False):
        ''' in: scale_list: list of scales e.g. [0,2,4]
            out: xtrain,ytrain,xval,yval,xtest,ytest = Data.get_mixed(scale_labels=False)
                 xtrain,ytrain,scaletrain,xval,yval,scaleval,xtest,ytest,scaletest = Data.get_mixed(scale_labels=True)
        '''
        output = []
        dataset = []
        
        for i in scale_list:
            sets = self.scaled_data[0].get_sets()
            if scale_labels:
                sets.append(np.ones(sets[1].shape)*i)
                sets.append(np.ones(sets[3].shape)*i)
                sets.append(np.ones(sets[5].shape)*i)
                new_order = [0,1,6,2,3,7,4,5,8]
                sets=[sets[f] for f in new_order]
            output.append(sets)
            
        output = np.array(output)
        
        state = np.random.get_state()
        for i in range(output.shape[1]):
            
            dset = np.concatenate(output[:,i],axis=0)
            np.random.set_state(state)
            np.random.shuffle(dset)
            size = int(dset.shape[0]/len(scale_list))
            dataset.append(dset[:size])
        
        return dataset
        
class ScData:
    def __init__(self,Data,size,pad):
        ''' create mnist dataset with specific scale
        '''
        self.x_train = np.pad(np.array(list(map(lambda img : sp.misc.imresize(img, (size, size)), Data.x_train))), ((0, 0), (pad, pad), (pad, pad)), 'constant')
        self.y_train = Data.y_train
        self.x_test = np.pad(np.array(list(map(lambda img : sp.misc.imresize(img, (size, size)), Data.x_test))), ((0, 0), (pad, pad), (pad, pad)), 'constant')
        self.y_test = Data.y_test
        
    def get_sets(self):
        ''' returns the train,validation and test sets for this object
            out = x_train,y_train,x_val,y_val,x_test,y_test
        '''
        # shuffling can be included here
        x_t = self.x_train[:50000,:,:]
        y_t = self.y_train[:50000]
        x_val = self.x_train[50000:,:,:]
        y_val = self.y_train[50000:]
        
        return x_t,y_t,x_val,y_val,self.x_test,self.y_test