import timm
import numpy as np
import torch
import matplotlib.pyplot as plt
import transformers
import pandas as pd
import seaborn as sns
from transformers import pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from PIL import Image



def init_data():
    data = pd.read_table('./list/weather_test_list.txt', names=["path", "class"], delimiter='\s')
    label = data['class']
    num_class = len(data['class'].unique())
    names = ['fire','fogsmog', 'lightning', 'rain', 'sandstorm','snow']
    data['name'] = [names[i] for i in data['class']]
    print(f'data : {data}')

    image_list = list()
    for path in data['path']:
        image = Image.open(path)
        image_list.append(image)
    print(f'number of image_list : {len(image_list)}')

    return data, label, image_list, num_class
