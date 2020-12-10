from torch.autograd import Variable
import torch.nn as nn
import torch
import wave
import numpy as np
import os
import librosa
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn.pipeline import Pipeline
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.preprocessing import OneHotEncoder

label = {0: "Animals", 1: "Natural soundscapes & water sounds",2: "Human non-speech sounds",3:"Interior/domestic sounds" ,4:"Exterior/urban noises"}
labels_complet = {0: 'dog',
 1: 'rooster',
 2: 'pig',
 3: 'cow',
 4: 'frog',
 5: 'cat',
 6: 'hen',
 7: 'insects',
 8: 'sheep',
 9: 'crow',
 10: 'rain',
 11: 'sea_waves',
 12: 'crackling_fire',
 13: 'crickets',
 14: 'chirping_birds',
 15: 'water_drops',
 16: 'wind',
 17: 'pouring_water',
 18: 'toilet_flush',
 19: 'thunderstorm',
 20: 'crying_baby',
 21: 'sneezing',
 22: 'clapping',
 23: 'breathing',
 24: 'coughing',
 25: 'footsteps',
 26: 'laughing',
 27: 'brushing_teeth',
 28: 'snoring',
 29: 'drinking_sipping',
 30: 'door_wood_knock',
 31: 'mouse_click',
 32: 'keyboard_typing',
 33: 'door_wood_creaks',
 34: 'can_opening',
 35: 'washing_machine',
 36: 'vacuum_cleaner',
 37: 'clock_alarm',
 38: 'clock_tick',
 39: 'glass_breaking',
 40: 'helicopter',
 41: 'chainsaw',
 42: 'siren',
 43: 'car_horn',
 44: 'engine',
 45: 'train',
 46: 'church_bells',
 47: 'airplane',
 48: 'fireworks',
 49: 'hand_saw'}

list_difficult_feature = ["water_drops","wind","airplane","washing_machine","helicopter"]
list_simple_feature = ["door_wood_knock","pouring_water","toilet_flush","sea_waves","thunderstorm"]
LEN_WAVEFORM = 22050 * 20
local_config = {
            'batch_size': 64, 
            'load_size': 22050*20,
            'phase': 'extract'
            }

"""
    This script defines the structure of SouneNet
    @reference: https://github.com/EsamGhaleb/soundNet_pytorch
"""

class SoundNet(nn.Module):
    def __init__(self):
        """
            The constructor of SoundNet
        """
        super().__init__()
    
        # Conv-1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(64, 1), stride=(2, 1), padding=(32, 0))
        self.batchnorm1 = nn.BatchNorm2d(16, eps=1e-5, momentum=0.1)
        self.relu1 = nn.ReLU(True)
        self.maxpool1 = nn.MaxPool2d((8, 1), stride=(8, 1)) 

        # Conv-2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(32, 1), stride=(2, 1), padding=(16, 0))
        self.batchnorm2 = nn.BatchNorm2d(32, eps=1e-5, momentum=0.1)
        self.relu2 = nn.ReLU(True)
        self.maxpool2 = nn.MaxPool2d((8, 1), stride=(8, 1)) 

        # Conv-3
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(16, 1), stride=(2, 1), padding=(8, 0))
        self.batchnorm3 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1)
        self.relu3 = nn.ReLU(True)  

        # Conv-4
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(8, 1), stride=(2, 1), padding=(4, 0))
        self.batchnorm4 = nn.BatchNorm2d(128, eps=1e-5, momentum=0.1)
        self.relu4 = nn.ReLU(True)  

        # Conv-5
        self.conv5 = nn.Conv2d(128, 256, kernel_size=(4, 1), stride=(2, 1), padding=(2, 0))
        self.batchnorm5 = nn.BatchNorm2d(256, eps=1e-5, momentum=0.1)
        self.relu5 = nn.ReLU(True)
        self.maxpool5 = nn.MaxPool2d((4, 1), stride=(4, 1)) 

        # Conv-6
        self.conv6 = nn.Conv2d(256, 512, kernel_size=(4, 1), stride=(2, 1), padding=(2, 0))
        self.batchnorm6 = nn.BatchNorm2d(512, eps=1e-5, momentum=0.1)
        self.relu6 = nn.ReLU(True)  

        # Conv-7
        self.conv7 = nn.Conv2d(512, 1024, kernel_size=(4, 1), stride=(2, 1), padding=(2, 0))
        self.batchnorm7 = nn.BatchNorm2d(1024, eps=1e-5, momentum=0.1)
        self.relu7 = nn.ReLU(True)  

        # Conv-8
        self.conv8_objs = nn.Conv2d(1024, 1000, kernel_size=(8, 1), stride=(2, 1))
        self.conv8_scns = nn.Conv2d(1024, 401, kernel_size=(8, 1), stride=(2, 1))

    def forward(self, waveform):
        """
            The forward process of SoundNet
            Arg:    waveform     (torch.autograd.Variable)   - Raw 20s waveform.
            Ret:    The list of each layer's output
        """    
        out1 = self.maxpool1(self.relu1(self.batchnorm1(self.conv1(waveform))))
        out2 = self.maxpool2(self.relu2(self.batchnorm2(self.conv2(out1))))
        out3 = self.relu3(self.batchnorm3(self.conv3(out2)))
        out4 = self.relu4(self.batchnorm4(self.conv4(out3)))
        out5 = self.maxpool5(self.relu5(self.batchnorm5(self.conv5(out4))))
        out6 = self.relu6(self.batchnorm6(self.conv6(out5)))
        out7 = self.relu7(self.batchnorm7(self.conv7(out6)))
        snds = self.conv8_objs(out7)    
        scns = self.conv8_scns(out7)    
        return [out1, out2, out3, out4, out5, out6, out7, [snds, scns]]

def classe(nom_fichier):
    res= nom_fichier.split("-")[-1]
    label=int(res.split(".")[0])
    fold=int(nom_fichier.split("-")[2][-1])
    return label,fold

def load_from_txt(txt_name, config=local_config):
    with open(txt_name, 'r') as handle:
        txt_list = handle.read().splitlines()

    audios = []
    audio_paths = []
    for idx, audio_path in enumerate(txt_list):
        if idx % 100 is 0:
            print('Processing: {}'.format(idx))
        sound_sample, _ = load_audio(audio_path)
        audios.append(preprocess(sound_sample, config))
        audio_paths.append(audio_path)
    return audios, audio_paths

def preprocess(raw_audio, config=local_config):
    # Select first channel (mono)
    if len(raw_audio.shape) > 1:
        raw_audio = raw_audio[0]

    # Make range [-256, 256]
    raw_audio *= 256.0

    # Make minimum length available
    length = config['load_size']
    if length > raw_audio.shape[0]:
        raw_audio = np.tile(raw_audio, int(length/raw_audio.shape[0] + 1))

    # Make equal training length
    if config['phase'] != 'extract':
        raw_audio = raw_audio[:length]

    # Check conditions
    assert len(raw_audio.shape) == 1, "It seems this audio contains two channels, we only need the first channel"
    assert np.max(raw_audio) <= 256, "It seems this audio contains signal that exceeds 256"
    assert np.min(raw_audio) >= -256, "It seems this audio contains signal that exceeds -256"

    # Shape to 1 x DIM x 1 x 1
    raw_audio = np.reshape(raw_audio, [1, 1, -1, 1])

    return raw_audio.copy()

def load_audio(audio_path, sr=None):
    # By default, librosa will resample the signal to 22050Hz(sr=None). And range in (-1., 1.)
    sound_sample, sr = librosa.load(audio_path, sr=sr, mono=False)

    return sound_sample, sr

def load_data():
  audio_txt = 'data.txt'
  sound_samples, audio_paths = load_from_txt(audio_txt, config=local_config)
  return sound_samples, audio_paths

def extract_complete(sound_samples, audio_paths,model):
  X_1=[]
  y_1=[]
  X_2=[]
  y_2=[]
  X_3=[]
  y_3=[]
  X_4=[]
  y_4=[]
  X_5=[]
  y_5=[]
  X_out_1_1=[]
  X_out_1_2=[]
  X_out_1_3=[]
  X_out_2_1=[]
  X_out_2_2=[]
  X_out_2_3=[]
  X_out_3_1=[]
  X_out_3_2=[]
  X_out_3_3=[]
  
  X_out_4_1=[]
  X_out_4_2=[]
  X_out_4_3=[]
  X_out_6_1=[]
  X_out_6_2=[]
  X_out_6_3=[]
  X_out_7_1=[]
  X_out_7_2=[]
  X_out_7_3=[]

  # Extract Feature
  model.eval()
  i=0
  for idx, sound_sample in enumerate(sound_samples):
    label,fold=classe(audio_paths[idx])    
    new_sample = torch.from_numpy(sound_sample)
    output = model.forward(new_sample)
    output_1 = output[0]
    output_2 = output[1]
    output_3 = output[2]
    output_4 = output[3]
    output_good = output[4]
    output_6 = output[5]
    output_7 = output[6]
    
    output_1 = output_1.detach().numpy()
    output_1 = output_1.mean(axis=2).reshape(-1)
    
    output_2 = output_2.detach().numpy()
    output_2 = output_2.mean(axis=2).reshape(-1)
    
    output_3 = output_3.detach().numpy()
    output_3 = output_3.mean(axis=2).reshape(-1)

    output_4 = output_4.detach().numpy()
    output_4 = output_4.mean(axis=2).reshape(-1)
    #output_4 = output_4.reshape(-1)

    output_good = output_good.detach().numpy()
    output_good = output_good.mean(axis=2).reshape(-1)
    #output_good = output_good.reshape(-1)

    output_6 = output_6.detach().numpy()
    output_6 = output_6.mean(axis=2).reshape(-1)
    #output_6 = output_6.reshape(-1)

    output_7 = output_7.detach().numpy()
    output_7 = output_7.mean(axis=2).reshape(-1)
    #output_7 = output_7.reshape(-1)
    if(fold==1):
      X_1.append(output_good)
      X_out_1_1.append(output_1)
      X_out_2_1.append(output_2)
      X_out_3_1.append(output_3)
      X_out_4_1.append(output_4)
      X_out_6_1.append(output_6)
      X_out_7_1.append(output_7)
      y_1.append(label)
    if(fold==2):
      X_2.append(output_good)
      X_out_1_2.append(output_1)
      X_out_2_2.append(output_2)
      X_out_3_2.append(output_3)
      X_out_4_2.append(output_4)
      X_out_6_2.append(output_6)
      X_out_7_2.append(output_7)
      y_2.append(label)
    if(fold==3):
      X_3.append(output_good)
      X_out_1_3.append(output_1)
      X_out_2_3.append(output_2)
      X_out_3_3.append(output_3)
      X_out_4_3.append(output_4)
      X_out_6_3.append(output_6)
      X_out_7_3.append(output_7)
      y_3.append(label)
    if(fold==4):
      X_4.append(output_good)
      y_4.append(label)
    if(fold==5):
      X_5.append(output_good)
      y_5.append(label)
    if i%100==0:
      print(i)
    i+=1
  X=X_1+X_2+X_3+X_4+X_5
  X_out_1 = X_out_1_1 + X_out_1_2 + X_out_1_3
  X_out_2 = X_out_2_1 + X_out_2_2 + X_out_2_3
  X_out_3 = X_out_3_1 + X_out_3_2 + X_out_3_3
  X_out_4 = X_out_4_1 + X_out_4_2 + X_out_4_3
  X_out_6 = X_out_6_1 + X_out_6_2 + X_out_6_3
  X_out_7 = X_out_7_1 + X_out_7_2 + X_out_7_3
  y=y_1+y_2+y_3+y_4+y_5
  return X,X_out_1,X_out_2,X_out_3,X_out_4,X_out_6,X_out_7,y

def creation_data():
    fichier = open("data.txt", "a")
    flag=True
    for filenames in os.walk('ESC-50-master/audio/'):
        for file_list in filenames:
               for file_name in file_list:
                    if file_name.endswith((".wav")):
                      if(flag):
                        fichier.write("ESC-50-master/audio/"+file_name)
                        flag=False
                      else:
                        fichier.write("\nESC-50-master/audio/"+file_name)
    fichier.close()
    
def custom_cv_5folds(X):
  i = 0
  while i < 5:
    train = np.concatenate((np.arange(0,i*400,dtype=int),np.arange((i+1)*400,2000,dtype=int)))
    test = np.arange(i*400,(i+1)*400,dtype=int)
    yield train, test
    i += 1
    
def plot_confusion_matrix(cm, classes, normalize=False, title='confusion matrix', cmap=plt.cm.Blues):
    # This function prints and plots the confusion matrix. Normalization can be applied by setting `normalize=True`
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("normalized confusion matrix")
    else:
        print('confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('true label')
    plt.xlabel('predicted label')
    
def five_major_class(X,y,labels_complet):
    X_animals = []
    y_animals = []
    y_labels_animals = []
    X_natural = []
    y_natural = []
    y_labels_natural = []
    X_human = []
    y_human = []
    y_labels_human = []
    X_interior = []
    y_interior = []
    y_labels_interior = []
    X_exterior = []
    y_exterior = []
    y_labels_exterior = []
    for i in range(len(y)):
      categorie = y[i]//10
      if(categorie==0):
        X_animals.append(X[i])
        y_animals.append(y[i])
        y_labels_animals.append(labels_complet[y[i]])
      if(categorie==1):
        X_natural.append(X[i])
        y_natural.append(y[i])
        y_labels_natural.append(labels_complet[y[i]])
      if(categorie==2):
        X_human.append(X[i])
        y_human.append(y[i])
        y_labels_human.append(labels_complet[y[i]])
      if(categorie==3):
        X_interior.append(X[i])
        y_interior.append(y[i])
        y_labels_interior.append(labels_complet[y[i]])
      if(categorie==4):
        X_exterior.append(X[i])
        y_exterior.append(y[i])
        y_labels_exterior.append(labels_complet[y[i]])
    return X_animals,y_animals,y_labels_animals,X_natural,y_natural,y_labels_natural,X_human,y_human,y_labels_human,X_interior,y_interior,y_labels_interior,X_exterior,y_exterior,y_labels_exterior

def easy_difficult(X,y,labels_complet):
    X_difficult = []
    y_difficult = []
    y_labels_difficult = []
    X_simple = []
    y_simple = []
    y_labels_simple = []
    for i in range(len(y)):
      if labels_complet[y[i]] in list_difficult_feature:
        X_difficult.append(X[i])
        y_difficult.append(y[i])
        y_labels_difficult.append(labels_complet[y[i]])
      if labels_complet[y[i]] in list_simple_feature:
        X_simple.append(X[i])
        y_simple.append(y[i])
        y_labels_simple.append(labels_complet[y[i]])
    return X_difficult,y_difficult,y_labels_difficult,X_simple,y_simple,y_labels_simple


def neural_matrix(X,y,activation_perc=95):
  enc = OneHotEncoder()
  y_enc = enc.fit_transform(y.reshape(-1,1))
  threshold = np.percentile(X, activation_perc, axis=0)
  stats = (X > threshold).T @ y_enc
  return stats, enc

def plot_activation(X, neurons):
    fig,axes = plt.subplots(1,8,figsize=(20,5),sharey=True)
    axes[0].set_ylabel("Number of activation")
    for i,ax in enumerate(axes.ravel()):
        neuron_idx = neurons[i]
        neuron = X[:,neuron_idx]
        indices_sorted = np.argsort(neuron)
        best_samples = indices_sorted[int(activation_percentage*X.shape[0]):]
        labels_samples = [labels_complet[k] for k in y[best_samples]]
        unique, counts = np.unique(labels_samples, return_counts=True)
        best_labels = counts > 5
        ax.set_title("neuron {}".format(neuron_idx))
        ax.xaxis.set_tick_params(rotation=90)
        ax.bar(unique[best_labels], counts[best_labels])
    return fig, axes