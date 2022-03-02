#!/usr/bin/env python
# coding: utf-8

# In[1]:


# imports

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


# ## Breast-w Dataset

# In[223]:


class Breast_W_Dataset:
    def __init__(self, dataset_path, dataset_name, output_coloumn_name='Class', 
                 train_size=0.1, normalization_method='zero_mean_unit_var', balance=False):
        """
        self.x_train
        self.x_test
        self.y_train
        self.y_test
        """
        print("Started reading dataset ", dataset_name, "...")
        
        self.dataset_path = dataset_path
        self.train_size = train_size
        self.output_coloumn_name = output_coloumn_name
        self.normalization_method = normalization_method
        self.sep = ','
        self.coloumn_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
        self.num_of_features = len(self.coloumn_names)-1
        self.class_names = ['benign', 'malignant']   # 2 for benign, 4 for malignant
        
        # read data using pandas
        self.dataframe = pd.read_csv(self.dataset_path, sep=self.sep, names=self.coloumn_names)
        # replace all '?' with NAN and then remove all NAN rows
        self.dataframe = self.dataframe.replace('?', np.nan).dropna(axis = 0, how = 'any')
        # rename class-names to +1 & -1 to be used in LibSVM & proposed SimpleMLK method
        self.dataframe.replace({self.output_coloumn_name : {2: +1, 4: -1}}, inplace=True)
                
        self.dataframe = shuffle(self.dataframe)
        
        if balance==True:
            data_y = self.dataframe[self.output_coloumn_name].to_numpy().reshape(-1, 1)
            data_x = self.dataframe.drop([self.output_coloumn_name], axis=1).to_numpy(dtype='float64')
            
            # since all class labesl are either +1 or -1
            class_1_data_x = data_x[np.where(data_y==1)[0]]
            class_1_data_y = data_y[data_y==1]
            class_minus_1_data_x = data_x[np.where(data_y==-1)[0]]
            class_minus_1_data_y = data_y[data_y==-1]
            
            # to create a balanced dataset, get random datas from each class equaly
            self.x_train_size = int(len(self.dataframe) * self.train_size)
            class_1_train_size = (int(self.x_train_size/2))/len(class_1_data_x)
            class_minus_1_train_size = (self.x_train_size - int(self.x_train_size/2))/len(class_minus_1_data_x)
            
            class_1_x_train, class_1_x_test, class_1_y_train, class_1_y_test =                         train_test_split(class_1_data_x, class_1_data_y, 
                                            train_size=class_1_train_size, random_state=42)
            class_minus_1_x_train, class_minus_1_x_test, class_minus_1_y_train, class_minus_1_y_test =                         train_test_split(class_minus_1_data_x, class_minus_1_data_y, 
                                            train_size=class_minus_1_train_size, random_state=42)
            
            self.x_train = np.concatenate((class_1_x_train, class_minus_1_x_train), axis=0)
            self.x_test = np.concatenate((class_1_x_test, class_minus_1_x_test), axis=0)
            self.y_train = np.concatenate((class_1_y_train, class_minus_1_y_train), axis=0)
            self.y_test = np.concatenate((class_1_y_test, class_minus_1_y_test), axis=0)
            
            self.y_train = self.y_train.reshape(-1, 1)
            self.y_test = self.y_test.reshape(-1, 1)
            
        elif balance==False:
            # split data test & train
            self.x_train_size = int(len(self.dataframe) * self.train_size)
            self.x_test_size = len(self.dataframe) - self.x_train_size
            self.x_train = self.dataframe.head(self.x_train_size)
            self.x_test = self.dataframe.tail(self.x_test_size)
            
            # remove Class from x_ttrain & x_test and create y_train & y_test
            self.y_train = self.x_train[self.output_coloumn_name]
            self.y_test = self.x_test[self.output_coloumn_name]

            # convert to numpy array
            self.y_train = self.y_train.to_numpy().reshape(-1, 1)
            self.y_test = self.y_test.to_numpy().reshape(-1, 1)
            self.x_train = self.x_train.drop([self.output_coloumn_name], axis=1).to_numpy(dtype='float64')
            self.x_test = self.x_test.drop([self.output_coloumn_name], axis=1).to_numpy(dtype='float64')
        
        # remove self.dataframe to avoid storing data too much
#         del self.dataframe
        
        # Normalize data
        self.normalize(self.normalization_method)

        print("Finished reading dataset ", dataset_name, "...")
        
        
    """
    Normalizing data improves the convergence of learning model and causes that smaller features also be able to affect the model parameters.
    """
    def normalize(self, normalization_method):
        if normalization_method == 'none':
            print("No normalization.")
            return
        
        if normalization_method == 'zero_mean_unit_var':
            print("zero-mean & unit_variance normalization.")
            self.x_train_without_x0 = self.zero_mean_unit_variance(self.x_train)
            self.x_test_without_x0 = self.zero_mean_unit_variance(self.x_test)
            
            
        if normalization_method == 'scale_0_1':
            print("scaling to [0, 1] normalization.")
            self.x_train_without_x0 = self.scaling_between_0_1(self.x_train)
            self.x_test_without_x0 = self.scaling_between_0_1(self.x_test)
     
    
    def scaling_between_0_1(self, numpy_array):
        '''
        Scaling
        '''
        normed_numpy_array = (numpy_array - numpy_array.min(axis=0)) / (numpy_array.max(axis=0) - numpy_array.min(axis=0))
        return normed_numpy_array


    def zero_mean_unit_variance(self, numpy_array):
        '''
        Standardization
        '''
        normed_numpy_array = (numpy_array - numpy_array.mean(axis=0)) / numpy_array.std(axis=0)
        return normed_numpy_array


# In[224]:


# No normalization
# breast_w_dataset = Breast_W_Dataset('./Datasets/breast-cancer-wisconsin.data', "Breast-W", 
#                                     'Class', normalization_method='None')

# print("data shape: ", breast_w_dataset.dataframe.to_numpy().shape)
# print("data-train shape: ", breast_w_dataset.x_train.shape)
# print("data-test shape: ", breast_w_dataset.x_test.shape)
# print("output classes: ", np.unique(breast_w_dataset.y_test))
# breast_w_dataset.dataframe.head()


# ## Diabetic Retinopathy Debrecen Data Set(Messidor)
# 
# ### * ?**is_class_label_a_feature** -> shows if label is considered as a feature in the dataset or not?
# 
# ### output labels are:
#     * b'1'
#     * b'0'

# In[23]:


# imports

from scipy.io import arff


# In[47]:


class Messidor_Dataset:
    def __init__(self, dataset_path, dataset_name, 
                 train_size=0.1, normalization_method='zero_mean_unit_var', 
                 is_class_label_a_feature=False, balance=False):
        """
        self.x_train
        self.x_test
        self.y_train
        self.y_test
        """
        print("Started reading dataset ", dataset_name, "...")
        
        self.dataset_path = dataset_path
        self.train_size = train_size
        self.output_coloumn_name = 'Class'
        self.normalization_method = normalization_method
        self.sep = ','
        
        # read data using scipy.io.arff
        data = arff.loadarff(self.dataset_path)
        self.dataframe = pd.DataFrame(data[0])

        # remove "'" from class names
#         self.dataframe.iloc[: , -1] = int(self.dataframe.iloc[: , -1])
        self.dataframe['Class'] = self.dataframe['Class'].astype(int)
        # replace name of class b'1' to 1 and b'0' to 0
        self.dataframe.replace({'Class' : { 1 : +1, 0 : -1}}, inplace=True)
        
        self.dataframe = shuffle(self.dataframe)
                
        # split data test & train
        self.x_train_size = int(len(self.dataframe) * self.train_size)
        self.x_test_size = len(self.dataframe) - self.x_train_size
        self.x_train = self.dataframe.head(self.x_train_size)
        self.x_test = self.dataframe.tail(self.x_test_size)
        
        # remove Class from x_train & x_test and create y_train & y_test
        self.y_train = self.x_train[self.output_coloumn_name]
        self.y_test = self.x_test[self.output_coloumn_name]

        # convert to numpy array
        self.y_train = self.y_train.to_numpy().reshape(-1, 1)
        self.y_test = self.y_test.to_numpy().reshape(-1, 1)
        if is_class_label_a_feature == False:
            self.x_train = self.x_train.drop([self.output_coloumn_name], axis=1).to_numpy()
            self.x_test = self.x_test.drop([self.output_coloumn_name], axis=1).to_numpy()
        elif is_class_label_a_feature == True:
            self.x_train = self.x_train.to_numpy()
            self.x_test = self.x_test.to_numpy()
        
        # remove self.dataframe to avoid storing data too much
#         del self.dataframe

        # Normalize data
        self.normalize(self.normalization_method)
        
        print("Finished reading dataset ", dataset_name, "...")

        
    """
    Normalizing data improves the convergence of learning model and causes that smaller features also be able to affect the model parameters.
    """
    def normalize(self, normalization_method):
        if normalization_method == 'none':
            print("No normalization.")
            return
        
        if normalization_method == 'zero_mean_unit_var':
            print("zero-mean & unit_variance normalization.")
            self.x_train_without_x0 = self.zero_mean_unit_variance(self.x_train)
            self.x_test_without_x0 = self.zero_mean_unit_variance(self.x_test)
            
            
        if normalization_method == 'scale_0_1':
            print("scaling to [0, 1] normalization.")
            self.x_train_without_x0 = self.scaling_between_0_1(self.x_train)
            self.x_test_without_x0 = self.scaling_between_0_1(self.x_test)
     
    
    def scaling_between_0_1(self, numpy_array):
        '''
        Scaling
        '''
        normed_numpy_array = (numpy_array - numpy_array.min(axis=0)) / (numpy_array.max(axis=0) - numpy_array.min(axis=0))
        return normed_numpy_array


    def zero_mean_unit_variance(self, numpy_array):
        '''
        Standardization
        '''
        normed_numpy_array = (numpy_array - numpy_array.mean(axis=0)) / numpy_array.std(axis=0)
        return normed_numpy_array


# In[60]:


# messidor = Messidor_Dataset('./Datasets/messidor_features.arff', "Messidor", 
#                             normalization_method='None', is_class_label_a_feature=False)

# print("data shape: ", messidor.dataframe.to_numpy().shape)
# print("data-train shape: ", messidor.x_train.shape)
# print("data-test shape: ", messidor.x_test.shape)
# print("output classes: ", np.unique(messidor.y_test))
# messidor.dataframe.head()


# ## Car Dataset

# In[241]:


class Car_Dataset:
    def __init__(self, dataset_path, dataset_name, output_coloumn_name='Class', 
                 train_size=0.1, balance=False):
        """
        self.x_train
        self.x_test
        self.y_train
        self.y_test
        """
        print("Started reading dataset ", dataset_name, "...")
        
        self.dataset_path = dataset_path
        self.train_size = train_size
        self.output_coloumn_name = output_coloumn_name
        self.sep = ','
        self.coloumn_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'Class']
        self.num_of_features = len(self.coloumn_names)-1
        self.class_names = ['benign', 'malignant']   # 2 for benign, 4 for malignant
        
        # read data using pandas
        self.dataframe = pd.read_csv(self.dataset_path, sep=self.sep, names=self.coloumn_names)
        # there is no missing attribute in this dataset
        
        self.dataframe = shuffle(self.dataframe)
        
        # replace all attribute values to start from 1 and increment by 1
        self.dataframe = self.dataframe.replace({'buying'   : { 'vhigh' : 4, 'high' : 3, 'med' : 2, 'low' : 1 }})
        self.dataframe = self.dataframe.replace({'maint'    : { 'vhigh' : 4, 'high' : 3, 'med' : 2, 'low' : 1 }})
        self.dataframe = self.dataframe.replace({'doors'    : { '2' : 1, '3' : 2, '4' : 3 , '5more' : 4 }})
        self.dataframe = self.dataframe.replace({'persons'  : { '2' : 1, '4' : 2, 'more' : 3 }})
        self.dataframe = self.dataframe.replace({'lug_boot' : { 'small' : 1, 'med' : 2, 'big' : 3 }})
        self.dataframe = self.dataframe.replace({'safety'   : { 'low' : 1, 'med' : 2, 'high' : 3 }})
        self.dataframe = self.dataframe.replace({'Class'    : { 'unacc' : 1, 'acc' : 2, 'good' : 3, 'vgood' : 4 }})
        
        self.data_ = self.dataframe.to_numpy()       
        np.random.shuffle(self.data_)

        # seperate class labels from data
        data_y = self.data_[:, -1]          # for last column
        # unacc vs other classes
        data_y[data_y==2] = -1
        data_y[data_y==3] = -1
        data_y[data_y==4] = -1
        # data_y[data_y==1] = +1
        data_x = self.data_[:, :-1]     # for all but last column
        
        # since all class labesl are either +1 or -1
        class_1_data_x = data_x[np.where(data_y==1)[0]]
        class_1_data_y = data_y[data_y==1]
        class_minus_1_data_x = data_x[np.where(data_y==-1)[0]]
        class_minus_1_data_y = data_y[data_y==-1]
        
        # to create a balanced dataset, get random datas from each class equaly
        self.x_train_size = int(len(self.dataframe) * self.train_size)
        class_minus_1_train_size = (int(self.x_train_size/2))/len(class_minus_1_data_x)
        class_1_train_size = (self.x_train_size - int(self.x_train_size/2))/len(class_1_data_x)
        
        class_1_x_train, class_1_x_test, class_1_y_train, class_1_y_test =                         train_test_split(class_1_data_x, class_1_data_y, 
                                         train_size=class_1_train_size, random_state=42)
        class_minus_1_x_train, class_minus_1_x_test, class_minus_1_y_train, class_minus_1_y_test =                         train_test_split(class_minus_1_data_x, class_minus_1_data_y, 
                                         train_size=class_minus_1_train_size, random_state=42)
        
        self.x_train = np.concatenate((class_1_x_train, class_minus_1_x_train), axis=0)
        self.x_test = np.concatenate((class_1_x_test, class_minus_1_x_test), axis=0)
        self.y_train = np.concatenate((class_1_y_train, class_minus_1_y_train), axis=0)
        self.y_test = np.concatenate((class_1_y_test, class_minus_1_y_test), axis=0)        

        self.y_train = self.y_train.reshape(-1, 1)
        self.y_test = self.y_test.reshape(-1, 1)
        
        # split data test & train
#         self.x_train_size = int(len(self.dataframe) * self.train_size)
#         self.x_test_size = len(self.dataframe) - self.x_train_size
#         self.x_train = self.dataframe.head(self.x_train_size)
#         self.x_test = self.dataframe.tail(self.x_test_size)
        
#         # remove Class from x_ttrain & x_test and create y_train & y_test
#         self.y_train = self.x_train[self.output_coloumn_name]
#         self.y_test = self.x_test[self.output_coloumn_name]

#         # convert to numpy array
#         self.y_train = self.y_train.to_numpy().reshape(-1, 1)
#         self.y_test = self.y_test.to_numpy().reshape(-1, 1)
#         self.x_train = self.x_train.drop([self.output_coloumn_name], axis=1).to_numpy()
#         self.x_test = self.x_test.drop([self.output_coloumn_name], axis=1).to_numpy()
        
        # remove self.dataframe to avoid storing data too much
#         del self.dataframe

        print("Finished reading dataset ", dataset_name, "...")


# In[242]:


# No normalization
# car_dataset = Car_Dataset('./Datasets/car.data', "Car", 'Class')

# print("data shape: ", car_dataset.dataframe.to_numpy().shape)
# print("data-train shape: ", car_dataset.x_train.shape)
# print("data-test shape: ", car_dataset.x_test.shape)
# print("output classes: ", np.unique(car_dataset.y_test))
# car_dataset.dataframe.head()


# ## Spambase Dataset

# In[228]:


class Spambase_Dataset:
    def __init__(self, dataset_path, dataset_name, 
                 train_size=0.1, normalization_method='zero_mean_unit_var', balance=False):
        """
        self.x_train
        self.x_test
        self.y_train
        self.y_test
        """
        print("Started reading dataset ", dataset_name, "...")
        
        self.dataset_path = dataset_path
        self.train_size = train_size
        self.normalization_method = normalization_method

        self.data_ = []
        with open(self.dataset_path) as data_file_pointer:
            for line in data_file_pointer:
                tmp = line.split(",")
                instance = []
                for f in tmp:
                    instance.append(float(f))
                self.data_.append(instance)
                
        self.data_ = np.array(self.data_)
        np.random.shuffle(self.data_)
        
        # seperate class labels from data
        data_y = self.data_[:, -1]          # for last column
        data_y[data_y==0] = -1
        data_y[data_y==1] = +1
        data_x = self.data_[:, :-1]     # for all but last column

        if balance==True:
            data_y = data_y.reshape(-1, 1)

            # since all class labesl are either +1 or -1
            class_1_data_x = data_x[np.where(data_y==1)[0]]
            class_1_data_y = data_y[data_y==1]
            class_minus_1_data_x = data_x[np.where(data_y==-1)[0]]
            class_minus_1_data_y = data_y[data_y==-1]
            
            # to create a balanced dataset, get random datas from each class equaly
            self.x_train_size = int(len(data_x) * self.train_size)
            class_1_train_size = (int(self.x_train_size/2))/len(class_1_data_x)
            class_minus_1_train_size = (self.x_train_size - int(self.x_train_size/2))/len(class_minus_1_data_x)
            
            class_1_x_train, class_1_x_test, class_1_y_train, class_1_y_test =                         train_test_split(class_1_data_x, class_1_data_y, 
                                            train_size=class_1_train_size, random_state=42)
            class_minus_1_x_train, class_minus_1_x_test, class_minus_1_y_train, class_minus_1_y_test =                         train_test_split(class_minus_1_data_x, class_minus_1_data_y, 
                                            train_size=class_minus_1_train_size, random_state=42)
            
            self.x_train = np.concatenate((class_1_x_train, class_minus_1_x_train), axis=0)
            self.x_test = np.concatenate((class_1_x_test, class_minus_1_x_test), axis=0)
            self.y_train = np.concatenate((class_1_y_train, class_minus_1_y_train), axis=0)
            self.y_test = np.concatenate((class_1_y_test, class_minus_1_y_test), axis=0)
            
            self.y_train = self.y_train.reshape(-1, 1)
            self.y_test = self.y_test.reshape(-1, 1)
        
        elif balance==False:
            # split data test & train
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.xdata_, self.y, test_size=1-self.train_size, random_state=42)
            
            self.y_train = self.y_train.reshape(-1, 1)
            self.y_test = self.y_test.reshape(-1, 1)
        
        # remove self.dataframe to avoid storing data too much
#         del self.data_
        
        # Normalize data
        self.normalize(self.normalization_method)

        print("Finished reading dataset ", dataset_name, "...")
        
        
    """
    Normalizing data improves the convergence of learning model and causes that smaller features also be able to affect the model parameters.
    """
    def normalize(self, normalization_method):
        if normalization_method == 'none':
            print("No normalization.")
            return
        
        if normalization_method == 'zero_mean_unit_var':
            print("zero-mean & unit_variance normalization.")
            self.x_train_without_x0 = self.zero_mean_unit_variance(self.x_train)
            self.x_test_without_x0 = self.zero_mean_unit_variance(self.x_test)
            
            
        if normalization_method == 'scale_0_1':
            print("scaling to [0, 1] normalization.")
            self.x_train_without_x0 = self.scaling_between_0_1(self.x_train)
            self.x_test_without_x0 = self.scaling_between_0_1(self.x_test)
     
    
    def scaling_between_0_1(self, numpy_array):
        '''
        Scaling
        '''
        normed_numpy_array = (numpy_array - numpy_array.min(axis=0)) / (numpy_array.max(axis=0) - numpy_array.min(axis=0))
        return normed_numpy_array


    def zero_mean_unit_variance(self, numpy_array):
        '''
        Standardization
        '''
        normed_numpy_array = (numpy_array - numpy_array.mean(axis=0)) / numpy_array.std(axis=0)
        return normed_numpy_array


# In[229]:


# spambase_dataset = Spambase_Dataset('./Datasets/spambase.data', "Spambase", 
#                             normalization_method='None')

# print("data shape: ", spambase_dataset.data_.shape)
# print("data-train shape: ", spambase_dataset.x_train.shape)
# print("data-test shape: ", spambase_dataset.x_test.shape)
# print("output classes: ", np.unique(spambase_dataset.y_test))
# dataframe = pd.DataFrame(spambase_dataset.data_, columns =[i for i in range(58)])
# dataframe.head()


# ## Coil2000 Dataset

# In[232]:


class Coil2000_Dataset:
    def __init__(self, dataset_path, dataset_name, 
                 train_size=0.1, normalization_method='zero_mean_unit_var', balance=False):
        """
        self.x_train
        self.x_test
        self.y_train
        self.y_test
        """
        print("Started reading dataset ", dataset_name, "...")
        
        self.dataset_path = dataset_path
        self.train_size = train_size
        self.normalization_method = normalization_method

        self.data_ = []
        line_number = 0        # skip first 90 lines
        with open(self.dataset_path) as data_file_pointer:
            for line in data_file_pointer:
                if line_number < 90:
                    pass
                else:
                    tmp = line.split(",")
                    instance = []
                    for f in tmp:
                        instance.append(float(f))
                    self.data_.append(instance)
                line_number += 1
        
        self.data_ = np.array(self.data_)
        np.random.shuffle(self.data_)
        
        if balance==True:
            # seperate class labels from data
            data_y = self.data_[:, -1]          # for last column
            data_y[data_y==0] = -1
            data_y[data_y==1] = +1
            data_x = self.data_[:, :-1]     # for all but last column
            
            data_y = data_y.reshape(-1, 1)

            # since all class labesl are either +1 or -1
            class_1_data_x = data_x[np.where(data_y==1)[0]]
            class_1_data_y = data_y[data_y==1]
            class_minus_1_data_x = data_x[np.where(data_y==-1)[0]]
            class_minus_1_data_y = data_y[data_y==-1]
            
            # to create a balanced dataset, get random datas from each class equaly
            self.x_train_size = int(len(data_x) * self.train_size)
            class_1_train_size = (int(self.x_train_size/2))/len(class_1_data_x)
            class_minus_1_train_size = (self.x_train_size - int(self.x_train_size/2))/len(class_minus_1_data_x)
            
            class_1_x_train, class_1_x_test, class_1_y_train, class_1_y_test =                         train_test_split(class_1_data_x, class_1_data_y, 
                                            train_size=class_1_train_size, random_state=42)
            class_minus_1_x_train, class_minus_1_x_test, class_minus_1_y_train, class_minus_1_y_test =                         train_test_split(class_minus_1_data_x, class_minus_1_data_y, 
                                            train_size=class_minus_1_train_size, random_state=42)
            
            self.x_train = np.concatenate((class_1_x_train, class_minus_1_x_train), axis=0)
            self.x_test = np.concatenate((class_1_x_test, class_minus_1_x_test), axis=0)
            self.y_train = np.concatenate((class_1_y_train, class_minus_1_y_train), axis=0)
            self.y_test = np.concatenate((class_1_y_test, class_minus_1_y_test), axis=0)
            
            self.y_train = self.y_train.reshape(-1, 1)
            self.y_test = self.y_test.reshape(-1, 1)
        
        elif balance==False:
            # seperate class labels from data
            self.y = self.data_[:, -1]          # for last column
            self.y[self.y==0] = -1
            self.y[self.y==1] = +1
            self.xdata_ = self.data_[:, :-1]     # for all but last column
            
            # split data test & train
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.xdata_, self.y, test_size=1-self.train_size, random_state=42)
            
            self.y_train = self.y_train.reshape(-1, 1)
            self.y_test = self.y_test.reshape(-1, 1)
        
        # remove self.dataframe to avoid storing data too much
#         del self.data_
        
        # Normalize data
        self.normalize(self.normalization_method)

        print("Finished reading dataset ", dataset_name, "...")
        
        
    """
    Normalizing data improves the convergence of learning model and causes that smaller features also be able to affect the model parameters.
    """
    def normalize(self, normalization_method):
        if normalization_method == 'none':
            print("No normalization.")
            return
        
        if normalization_method == 'zero_mean_unit_var':
            print("zero-mean & unit_variance normalization.")
            self.x_train_without_x0 = self.zero_mean_unit_variance(self.x_train)
            self.x_test_without_x0 = self.zero_mean_unit_variance(self.x_test)
            
            
        if normalization_method == 'scale_0_1':
            print("scaling to [0, 1] normalization.")
            self.x_train_without_x0 = self.scaling_between_0_1(self.x_train)
            self.x_test_without_x0 = self.scaling_between_0_1(self.x_test)
     
    
    def scaling_between_0_1(self, numpy_array):
        '''
        Scaling
        '''
        normed_numpy_array = (numpy_array - numpy_array.min(axis=0)) / (numpy_array.max(axis=0) - numpy_array.min(axis=0))
        return normed_numpy_array


    def zero_mean_unit_variance(self, numpy_array):
        '''
        Standardization
        '''
        normed_numpy_array = (numpy_array - numpy_array.mean(axis=0)) / numpy_array.std(axis=0)
        return normed_numpy_array


# In[233]:


# coil2000_dataset = Coil2000_Dataset('./Datasets/coil2000.dat', "Coil2000", 
#                             normalization_method='None')

# print("data shape: ", coil2000_dataset.data_.shape)
# print("data-train shape: ", coil2000_dataset.x_train.shape)
# print("data-test shape: ", coil2000_dataset.x_test.shape)
# print("output classes: ", np.unique(coil2000_dataset.y_test))
# dataframe = pd.DataFrame(coil2000_dataset.data_, columns =[i for i in range(86)])
# dataframe.head()


# ## Bank Marketing Dataset
# ### * ?**is_class_label_a_feature** -> shows if label is considered as a feature in the dataset or not?
# 

# In[234]:


class Bank_Marketing_Dataset:
    def __init__(self, dataset_path, dataset_name, output_coloumn_name='y', 
                 train_size=0.1, normalization_method='zero_mean_unit_var', balance=False):
        """
        self.x_train
        self.x_test
        self.y_train
        self.y_test
        """
        print("Started reading dataset ", dataset_name, "...")
        
        self.dataset_path = dataset_path
        self.train_size = train_size
        self.output_coloumn_name = output_coloumn_name
        self.normalization_method = normalization_method
        self.sep = ';'
        self.coloumn_names = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y"]
        self.num_of_features = len(self.coloumn_names)-1
        self.class_names = ['no', 'yes']
        
        # read data using pandas
        self.dataframe = pd.read_csv(self.dataset_path, sep=self.sep)

        
#         job_dict = {'management': 0, 'technician': 1, 'entrepreneur': 2, 'blue-collar': 3, 'unknown': 4, 'retired': 5, 'admin.': 6, 'services': 7, 'self-employed': 8, 'unemployed': 9, 'housemaid': 10, 'student': 11}
#         edu_dict = {'tertiary': 0, 'secondary': 1, 'unknown': 3, 'primary': 2}
#         mar_dict = {'married': 0, 'single': 1, 'divorced': 2}
#         default_dict = {'no': 0, 'yes': 1}
#         hous_dict = {'yes': 0, 'no': 1}
#         loan_dict = default_dict
#         contact_dict = {'unknown': 0, 'cellular': 1, 'telephone': 2}
#         month_dict = {'may': 0, 'jun': 1, 'jul': 2, 'aug': 3, 'oct': 4, 'nov': 5, 'dec': 6, 'jan': 7, 'feb': 8, 'mar': 9, 'apr': 10, 'sep': 11}
#         pout_dict = {'unknown': 0, 'failure': 1, 'other': 3, 'success': 2}

#         self.dataframe.replace({'job' : job_dict}, inplace=True)
#         self.dataframe.replace({'education' : edu_dict}, inplace=True)
#         self.dataframe.replace({'marital' : mar_dict}, inplace=True)
#         self.dataframe.replace({'default' : default_dict}, inplace=True)
#         self.dataframe.replace({'housing' : hous_dict}, inplace=True)
#         self.dataframe.replace({'loan' : loan_dict}, inplace=True)
#         self.dataframe.replace({'contact' : contact_dict}, inplace=True)
#         self.dataframe.replace({'month' : month_dict}, inplace=True)
#         self.dataframe.replace({'poutcome' : pout_dict}, inplace=True)

        # convert feature names to numbers
        self.nominal_coloumns= ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
        for col_nomi in self.nominal_coloumns:
            self.dataframe[col_nomi] = self.dataframe[col_nomi].astype('category').cat.codes

        # rename class-names to +1 & -1 to be used in LibSVM & proposed SimpleMLK method
        self.dataframe.replace({self.output_coloumn_name : {'yes': +1, 'no': -1}}, inplace=True)
                   
        
        self.dataframe = shuffle(self.dataframe)
        
        if balance==True:
            data_y = self.dataframe[self.output_coloumn_name].to_numpy().reshape(-1, 1)
            data_x = self.dataframe.drop([self.output_coloumn_name], axis=1).to_numpy(dtype='float64')
            
            # since all class labesl are either +1 or -1
            class_1_data_x = data_x[np.where(data_y==1)[0]]
            class_1_data_y = data_y[data_y==1]
            class_minus_1_data_x = data_x[np.where(data_y==-1)[0]]
            class_minus_1_data_y = data_y[data_y==-1]
            
            # to create a balanced dataset, get random datas from each class equaly
            self.x_train_size = int(len(self.dataframe) * self.train_size)
            class_1_train_size = (int(self.x_train_size/2))/len(class_1_data_x)
            class_minus_1_train_size = (self.x_train_size - int(self.x_train_size/2))/len(class_minus_1_data_x)
            
            class_1_x_train, class_1_x_test, class_1_y_train, class_1_y_test =                         train_test_split(class_1_data_x, class_1_data_y, 
                                            train_size=class_1_train_size, random_state=42)
            class_minus_1_x_train, class_minus_1_x_test, class_minus_1_y_train, class_minus_1_y_test =                         train_test_split(class_minus_1_data_x, class_minus_1_data_y, 
                                            train_size=class_minus_1_train_size, random_state=42)
            
            self.x_train = np.concatenate((class_1_x_train, class_minus_1_x_train), axis=0)
            self.x_test = np.concatenate((class_1_x_test, class_minus_1_x_test), axis=0)
            self.y_train = np.concatenate((class_1_y_train, class_minus_1_y_train), axis=0)
            self.y_test = np.concatenate((class_1_y_test, class_minus_1_y_test), axis=0)
            
            self.y_train = self.y_train.reshape(-1, 1)
            self.y_test = self.y_test.reshape(-1, 1)
            
        elif balance==False:
#           # split data test & train
            self.x_train_size = int(len(self.dataframe) * self.train_size)
            self.x_test_size = len(self.dataframe) - self.x_train_size
            self.x_train = self.dataframe.head(self.x_train_size)
            self.x_test = self.dataframe.tail(self.x_test_size)
            
            # remove Class from x_ttrain & x_test and create y_train & y_test
            self.y_train = self.x_train[self.output_coloumn_name]
            self.y_test = self.x_test[self.output_coloumn_name]
            
            # convert to numpy array
            self.y_train = self.y_train.to_numpy().reshape(-1, 1)
            self.y_test = self.y_test.to_numpy().reshape(-1, 1)
            self.x_train = self.x_train.drop([self.output_coloumn_name], axis=1).to_numpy(dtype='float64')
            self.x_test = self.x_test.drop([self.output_coloumn_name], axis=1).to_numpy(dtype='float64')
            
        # remove self.dataframe to avoid storing data too much
#         del self.dataframe
        
        # Normalize data
        self.normalize(self.normalization_method)
        

        print("Finished reading dataset ", dataset_name, "...")

        
    """
    Normalizing data improves the convergence of learning model and causes that smaller features also be able to affect the model parameters.
    """
    def normalize(self, normalization_method):
        if normalization_method == 'none':
            print("No normalization.")
            return
        
        if normalization_method == 'zero_mean_unit_var':
            print("zero-mean & unit_variance normalization.")
            self.x_train_without_x0 = self.zero_mean_unit_variance(self.x_train)
            self.x_test_without_x0 = self.zero_mean_unit_variance(self.x_test)
            
            
        if normalization_method == 'scale_0_1':
            print("scaling to [0, 1] normalization.")
            self.x_train_without_x0 = self.scaling_between_0_1(self.x_train)
            self.x_test_without_x0 = self.scaling_between_0_1(self.x_test)
     
    
    def scaling_between_0_1(self, numpy_array):
        '''
        Scaling
        '''
        normed_numpy_array = (numpy_array - numpy_array.min(axis=0)) / (numpy_array.max(axis=0) - numpy_array.min(axis=0))
        return normed_numpy_array

    def zero_mean_unit_variance(self, numpy_array):
        '''
        Standardization
        '''
        normed_numpy_array = (numpy_array - numpy_array.mean(axis=0)) / numpy_array.std(axis=0)
        return normed_numpy_array


# In[235]:


# No normalization
# bank_dataset = Bank_Marketing_Dataset('./Datasets/bank-full.csv', "Bank Marketing", 'y', normalization_method="None")

# print("data shape: ", bank_dataset.dataframe.to_numpy().shape)
# print("data-train shape: ", bank_dataset.x_train.shape)
# print("data-test shape: ", bank_dataset.x_test.shape)
# print("output classes: ", np.unique(bank_dataset.y_test))
# bank_dataset.dataframe.head()


# ## Skin Segmentation Dataset
# ### * ?**is_class_label_a_feature** -> shows if label is considered as a feature in the dataset or not?
# 

# In[236]:


class Skin_NonSkin_Dataset:
    def __init__(self, dataset_path, dataset_name, train_size=0.1, normalization_method='zero_mean_unit_var', balance=False):
        """
        self.x_train
        self.x_test
        self.y_train
        self.y_test
        """
        print("Started reading dataset ", dataset_name, "...")
        
        self.dataset_path = dataset_path
        self.train_size = train_size
        self.normalization_method = normalization_method

        self.data_ = []
        with open(self.dataset_path) as data_file_pointer:
            for line in data_file_pointer:
                tmp = line.split()
                instance = []
                for f in tmp:
                    instance.append(float(f))
                self.data_.append(instance)
        
        self.data_ = np.array(self.data_)
        np.random.shuffle(self.data_)
        
        if balance==True:
            # seperate class labels from data
            data_y = self.data_[:, -1]          # for last column
            data_y[data_y==2] = -1
            data_y[data_y==1] = +1
            data_x = self.data_[:, :-1]     # for all but last column
            
            data_y = data_y.reshape(-1, 1)

            # since all class labesl are either +1 or -1
            class_1_data_x = data_x[np.where(data_y==1)[0]]
            class_1_data_y = data_y[data_y==1]
            class_minus_1_data_x = data_x[np.where(data_y==-1)[0]]
            class_minus_1_data_y = data_y[data_y==-1]
            
            # to create a balanced dataset, get random datas from each class equaly
            self.x_train_size = int(len(data_x) * self.train_size)
            class_1_train_size = (int(self.x_train_size/2))/len(class_1_data_x)
            class_minus_1_train_size = (self.x_train_size - int(self.x_train_size/2))/len(class_minus_1_data_x)
            
            class_1_x_train, class_1_x_test, class_1_y_train, class_1_y_test =                         train_test_split(class_1_data_x, class_1_data_y, 
                                            train_size=class_1_train_size, random_state=42)
            class_minus_1_x_train, class_minus_1_x_test, class_minus_1_y_train, class_minus_1_y_test =                         train_test_split(class_minus_1_data_x, class_minus_1_data_y, 
                                            train_size=class_minus_1_train_size, random_state=42)
            
            self.x_train = np.concatenate((class_1_x_train, class_minus_1_x_train), axis=0)
            self.x_test = np.concatenate((class_1_x_test, class_minus_1_x_test), axis=0)
            self.y_train = np.concatenate((class_1_y_train, class_minus_1_y_train), axis=0)
            self.y_test = np.concatenate((class_1_y_test, class_minus_1_y_test), axis=0)
            
            self.y_train = self.y_train.reshape(-1, 1)
            self.y_test = self.y_test.reshape(-1, 1)

        elif balance==False:    
            # seperate class labels from data
            self.y = self.data_[:, -1]          # for last column
            self.y[self.y==2] = -1
            self.y[self.y==1] = +1
            self.xdata_ = self.data_[:, :-1]     # for all but last column
            
            # split data test & train
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.xdata_, self.y, test_size=1-self.train_size, random_state=42)
            
            self.y_train = self.y_train.reshape(-1, 1)
            self.y_test = self.y_test.reshape(-1, 1)
        
        # remove self.dataframe to avoid storing data too much
#         del self.data_
        
        # Normalize data
        self.normalize(self.normalization_method)

        print("Finished reading dataset ", dataset_name, "...")
        
        
    """
    Normalizing data improves the convergence of learning model and causes that smaller features also be able to affect the model parameters.
    """
    def normalize(self, normalization_method):
        if normalization_method == 'none':
            print("No normalization.")
            return
        
        if normalization_method == 'zero_mean_unit_var':
            print("zero-mean & unit_variance normalization.")
            self.x_train_without_x0 = self.zero_mean_unit_variance(self.x_train)
            self.x_test_without_x0 = self.zero_mean_unit_variance(self.x_test)
            
            
        if normalization_method == 'scale_0_1':
            print("scaling to [0, 1] normalization.")
            self.x_train_without_x0 = self.scaling_between_0_1(self.x_train)
            self.x_test_without_x0 = self.scaling_between_0_1(self.x_test)
     
    
    def scaling_between_0_1(self, numpy_array):
        '''
        Scaling
        '''
        normed_numpy_array = (numpy_array - numpy_array.min(axis=0)) / (numpy_array.max(axis=0) - numpy_array.min(axis=0))
        return normed_numpy_array


    def zero_mean_unit_variance(self, numpy_array):
        '''
        Standardization
        '''
        normed_numpy_array = (numpy_array - numpy_array.mean(axis=0)) / numpy_array.std(axis=0)
        return normed_numpy_array


# In[237]:


# skin_dataset = Skin_NonSkin_Dataset('./Datasets/Skin_NonSkin.txt', "Skin Segmentation", normalization_method="None")

# print("data shape: ", skin_dataset.data_.shape)
# print("data-train shape: ", skin_dataset.x_train.shape)
# print("data-test shape: ", skin_dataset.x_test.shape)
# print("output classes: ", np.unique(skin_dataset.y_test))
# dataframe = pd.DataFrame(skin_dataset.data_, columns =[i for i in range(4)])
# dataframe.head()


# ## Covertype Dataset

# In[238]:


class Covertype_Dataset:
    def __init__(self, dataset_path, dataset_name, output_coloumn_name='44', 
                 train_size=0.1, normalization_method='zero_mean_unit_var', balance=False):
        """
        self.x_train
        self.x_test
        self.y_train
        self.y_test
        """
        print("Started reading dataset ", dataset_name, "...")
        
        self.dataset_path = dataset_path
        self.train_size = train_size
        self.normalization_method = normalization_method

        self.data_ = []
        with open(self.dataset_path) as data_file_pointer:
            for line in data_file_pointer:
                tmp = line.split(",")
                instance = []
                for f in tmp:
                    instance.append(float(f))
                self.data_.append(instance)
        
        self.data_ = np.array(self.data_)       
        np.random.shuffle(self.data_)

        # seperate class labels from data
        data_y = self.data_[:, -1]          # for last column
        # in the article, it is said to only consider Aspen vs other classes
        # due to dataset description, Aspen class is the class label '5'.
        data_y[data_y==1] = -1
        data_y[data_y==2] = -1
        data_y[data_y==3] = -1
        data_y[data_y==4] = -1
        data_y[data_y==6] = -1
        data_y[data_y==7] = -1
        data_y[data_y==5] = +1
        data_x = self.data_[:, :-1]     # for all but last column
        
        data_y = data_y.reshape(-1, 1)

        # since all class labesl are either +1 or -1
        class_1_data_x = data_x[np.where(data_y==1)[0]]
        class_1_data_y = data_y[data_y==1]
        class_minus_1_data_x = data_x[np.where(data_y==-1)[0]]
        class_minus_1_data_y = data_y[data_y==-1]
        
        # to create a balanced dataset, get random datas from each class equaly
        self.x_train_size = int(len(data_x) * self.train_size)
        class_1_train_size = (int(self.x_train_size/2))/len(class_1_data_x)
        class_minus_1_train_size = (self.x_train_size - int(self.x_train_size/2))/len(class_minus_1_data_x)
        
        class_1_x_train, class_1_x_test, class_1_y_train, class_1_y_test =                         train_test_split(class_1_data_x, class_1_data_y, 
                                         train_size=class_1_train_size, random_state=42)
        class_minus_1_x_train, class_minus_1_x_test, class_minus_1_y_train, class_minus_1_y_test =                         train_test_split(class_minus_1_data_x, class_minus_1_data_y, 
                                         train_size=class_minus_1_train_size, random_state=42)
        
        self.x_train = np.concatenate((class_1_x_train, class_minus_1_x_train), axis=0)
        self.x_test = np.concatenate((class_1_x_test, class_minus_1_x_test), axis=0)
        self.y_train = np.concatenate((class_1_y_train, class_minus_1_y_train), axis=0)
        self.y_test = np.concatenate((class_1_y_test, class_minus_1_y_test), axis=0)
        
        self.y_train = self.y_train.reshape(-1, 1)
        self.y_test = self.y_test.reshape(-1, 1)
        
#         # split data test & train
#         self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.xdata_, self.y, test_size=1-self.train_size, random_state=42)
        
#         self.y_train = self.y_train.reshape(-1, 1)
#         self.y_test = self.y_test.reshape(-1, 1)
        
        # remove self.dataframe to avoid storing data too much
#         del self.data_
        
        # Normalize data
        self.normalize(self.normalization_method)

        print("Finished reading dataset ", dataset_name, "...")
        
        
    """
    Normalizing data improves the convergence of learning model and causes that smaller features also be able to affect the model parameters.
    """
    def normalize(self, normalization_method):
        if normalization_method == 'none':
            print("No normalization.")
            return
        
        if normalization_method == 'zero_mean_unit_var':
            print("zero-mean & unit_variance normalization.")
            self.x_train_without_x0 = self.zero_mean_unit_variance(self.x_train)
            self.x_test_without_x0 = self.zero_mean_unit_variance(self.x_test)
            
            
        if normalization_method == 'scale_0_1':
            print("scaling to [0, 1] normalization.")
            self.x_train_without_x0 = self.scaling_between_0_1(self.x_train)
            self.x_test_without_x0 = self.scaling_between_0_1(self.x_test)
     
    
    def scaling_between_0_1(self, numpy_array):
        '''
        Scaling
        '''
        normed_numpy_array = (numpy_array - numpy_array.min(axis=0)) / (numpy_array.max(axis=0) - numpy_array.min(axis=0))
        return normed_numpy_array


    def zero_mean_unit_variance(self, numpy_array):
        '''
        Standardization
        '''
        normed_numpy_array = (numpy_array - numpy_array.mean(axis=0)) / numpy_array.std(axis=0)
        return normed_numpy_array


# In[239]:


# covertype_dataset = Covertype_Dataset('./Datasets/covtype.data', "Covertype", 
#                                       train_size=0.02, normalization_method="None")

# print("data shape: ", covertype_dataset.data_.shape)
# print("data-train shape: ", covertype_dataset.x_train.shape)
# print("data-test shape: ", covertype_dataset.x_test.shape)
# print("output classes: ", np.unique(covertype_dataset.y_test))
# dataframe = pd.DataFrame(covertype_dataset.data_, columns =[i for i in range(55)])
# dataframe.head()


# In[ ]:




