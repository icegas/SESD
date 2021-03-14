import torch
import numpy as np
import random
import pdb
import os
import threading
import time
import math
import glob
from scipy import signal
from scipy.io import wavfile
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import librosa


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def loadWAV(filename, max_frames, evalmode=True, num_eval=10, sr=160.0):

    # Maximum audio length
    max_audio = int(max_frames * sr)

    # Read wav file and convert to torch tensor
    audio, _  = librosa.load(filename, sr=16000) 

    audiosize = audio.shape[0]

    if audiosize <= max_audio:
        shortage    = max_audio - audiosize + 1 
        audio       = np.pad(audio, (0, shortage), 'wrap')
        audiosize   = audio.shape[0]

    if evalmode:
        startframe = np.linspace(0,audiosize-max_audio,num=num_eval)
    else:
        startframe = np.int64(random.random()*(audiosize-max_audio))
    
    feats = []
    if evalmode and max_frames == 0:
        feats.append(audio)
    else:
        feats.append(audio[startframe:startframe+max_audio])

    feat = np.stack(feats,axis=0).astype(np.float)

    return feat;

class TrainLoader(Dataset):
    def __init__(self, df, audio_path, max_frames, classes, hop_len, sample_rate):

        self.max_frames = max_frames
        self.filenames = []
        self.labels = []
        self.classes = classes

        self.map_cls = {}
        for i in range(len(classes)):
            self.map_cls[classes[i]] = i

        for idx, r in df.iterrows():

            self.filenames.append(audio_path+r.filename)

            if r.category in self.classes:
                self.labels.append(self.map_cls[r.category])
            else:
                self.labels.append(self.map_cls["background"])
        
        self.labels = np.asarray(self.labels)
        self.label_size =int( (max_frames * sample_rate / 100) / hop_len ) + 1
        self.hop_len = hop_len

    def __getitem__(self, train_data):

        label = np.zeros((self.label_size, len(self.classes)-1))
        audios = np.array([])
        windows = train_data['windows']
        batch_labels = train_data["classes"]

        start_part = 0
        
        for i in range(len(windows)):
            
            idx = np.random.choice(
                np.where(self.labels == batch_labels[i])[0]
            )

            audio = loadWAV(self.filenames[idx], windows[i], evalmode=False).ravel()
            audios = np.append(audios, audio)

            part_size = int(audio.shape[0] / self.hop_len) + 1
            if self.classes[batch_labels[i]] != "background":
                label[start_part:start_part+part_size, batch_labels[i]] = 1

            start_part += part_size 
            
        return torch.FloatTensor(audios), torch.FloatTensor(label)

    def __len__(self):
        return len(self.filenames)

class TrainSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, batch_size, max_frames, sample_rate, gen_config):

        self.data_source = data_source
        self.batch_size = batch_size;
        self.max_frames = max_frames 
        #self.sr = sample_rate / 100
        self.class_num = len(self.data_source.classes)

        self.max_size = gen_config['max_frame_size']
        self.min_size = gen_config['min_frame_size']
        self.samples_in_epoch = gen_config['gen_samples']
        
    def __iter__(self):

        out = []

        for _ in range(self.samples_in_epoch):

            train_data = {}
            windows = []

            for i in range(self.class_num-1):
                windows.append(int(np.random.uniform(self.min_size, self.max_size)) )
            windows.append(self.max_frames - np.sum(windows))
            train_data['windows'] = windows
            train_data['classes'] =  np.random.choice(range(self.class_num), replace=False, 
                                                size=self.class_num)

            out.append(train_data)


        return  iter([out[i] for i in range(len(out)) ])
    
    def __len__(self):
        return len(self.data_source)
    

def get_data_loader(config, df, generator):
    
    train_dataset = TrainLoader(df, config["audio_path"],
                                config["max_frames"], config["classes"],
                                config["melfb"]['hop_len'], config['sample_rate'])

    train_sampler = TrainSampler(train_dataset, config["batch_size"], 
                                config["max_frames"], config["sample_rate"], generator)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        num_workers=config["threads"],
        sampler=train_sampler,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )
    
    return train_loader

class TestLoader(Dataset):
    def __init__(self, eval_frames, max_frames, sample_rate, **kwargs):
        self.max_frames = eval_frames;
        self.window = int(max_frames * sample_rate / 100)
        self.filenames = []

    def __getitem__(self, fn):
        audio = loadWAV(fn, self.max_frames, evalmode=True).ravel()
        out_data = []

        for i in range(0, len(audio), self.window):
            out_data.append(audio[i:i+self.window])
        
        out_data[-1] = audio[-self.window:]

        return torch.FloatTensor(out_data), fn.split('/')[-1]

    def __len__(self):
        return len(self.test_list)

class TestSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, df, audio_path):
        self.data_source = data_source
        self.df = df
        self.audio_path = audio_path
    
    def __iter__(self):

        filenames = []
        for (fn, data) in self.df.groupby('filename'):
            filenames.append(self.audio_path + fn)

        return  iter([filenames[i] for i in range(len(filenames)) ])

    def __len__(self):
        return len(self.data_source)


def get_test_data_loader(config, df):
    
    test_dataset = TestLoader(config["eval_frames"],
                                config["max_frames"], config['sample_rate'])

    test_sampler = TestSampler(test_dataset, df, config["audio_path"])

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=config["threads"],
        sampler=test_sampler,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )
    
    return test_loader

