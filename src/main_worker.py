from DatasetLoader import get_data_loader, get_test_data_loader
import numpy as np
from CRNN_model import CRNN
import time
import sys
import pandas as pd

import torch
from sklearn.model_selection import train_test_split

import yaml

import sed_eval
import dcase_util
from tqdm.notebook import tqdm

def get_train_val_data_loaders(config):
    df = pd.read_csv(config["meta_path"])
    X_train, X_val, _, _ = train_test_split(np.arange(df.shape[0]),  
                                       df.category, stratify=df.category, test_size=0.25)

    return get_data_loader(config, df.iloc[X_train], config["train_generator"]), \
           get_data_loader(config, df.iloc[X_val],   config["val_generator"])

def evaluate_model(data_loader, model, cls_names, thr, hop_len, s_r, iou_thr, criterion):
    ref_event_list, est_event_list = [], []
    file_counter = 0; counter = 0; loss = 0;

    for data, label in data_loader:

        with torch.no_grad():
            out = model(data.cuda())
            loss += criterion(out, label.cuda())

        map_event(label.detach().cpu().numpy(), ref_event_list, cls_names, thr, hop_len, s_r, file_counter)
        map_event(out.detach().cpu().numpy(), est_event_list, cls_names, thr, hop_len, s_r, file_counter)
        
        file_counter += label.shape[0]
        counter += 1
        torch.cuda.empty_cache()
    
    return print_metrics(ref_event_list, est_event_list, iou_thr), loss / counter

def print_metrics(ref_event_list, est_event_list, iou_thr):

    cols = ['f_measure', 'precision', 'recall']
    metrics = {}
    for c in cols:
        metrics[c] = 0

    if len(est_event_list) != 0:

        reference_event_list = dcase_util.containers.MetaDataContainer(ref_event_list)
        estimated_event_list = dcase_util.containers.MetaDataContainer(est_event_list)

        event_based_metrics = sed_eval.sound_event.EventBasedMetrics(
        event_label_list=reference_event_list.unique_event_labels,
        t_collar=0.250
        )

        for filename in reference_event_list.unique_files:
            reference_event_list_for_current_file = reference_event_list.filter(
                filename=filename
            )

            estimated_event_list_for_current_file = estimated_event_list.filter(
                filename=filename
            )


            event_based_metrics.evaluate(
                reference_event_list=reference_event_list_for_current_file,
                estimated_event_list=estimated_event_list_for_current_file
            )

        metrics = event_based_metrics.overall_f_measure()

    print("#################################################")
    print()
    print("F1: {}, Precsion: {}, Recall: {}".format(
        metrics['f_measure'], metrics['precision'], metrics['recall']
    ))
    print()
    print("#################################################")
    

    return metrics

def map_event(data, event_list, cls_names, thr, hop_len, s_r, file_counter=None, fn=None):

    for batch_data in data:

        for i in range(batch_data.shape[-1]):
            d = batch_data[:, i]; 
            d[np.where(d>thr)] = 1; d[np.where(d<=thr)] = 0;
        
            if len(np.where(d > thr)[0]) == 0:
                continue

            idx = np.diff(d)
            split_index = np.where( (idx == 1) | (idx==-1) )[0]
            split_index += 1
            split_values = np.split(d, split_index)

            for v, j in zip(split_values, range(len(split_values)) ):      

                if v[0] != 0:
                    event = {}
                    event['event_label'] = cls_names[i]

                    if j == 0:
                        event['event_onset'] = 0.0
                    else:
                        event['event_onset'] = split_index[j-1] * hop_len / s_r

                    event['event_offset'] = event['event_onset'] + len(v) * hop_len / s_r

                    if fn == None:
                        event['file'] = str(file_counter)
                    else:
                        event['file'] = fn
                    event_list.append(event)

        if file_counter != None:
            file_counter += 1


def step(data_loader, model, opt, criterion, batch_size):

    stepsize = batch_size;

    counter = 0;
    index   = 0;
    loss    = 0;

    tstart = time.time()

    for data, label in data_loader:

        model.zero_grad()
        out = model(data.cuda())
        nloss = criterion(out, label.cuda())
        nloss.backward();
        opt.step();

        loss    += nloss.detach().cpu();
        counter += 1;
        index   += stepsize;

        telapsed = time.time() - tstart
        tstart = time.time()

        sys.stdout.write("\rProcessing (%d) "%(index));
        sys.stdout.write("Loss %f - %.2f Hz "%(loss/counter, stepsize/telapsed));
        sys.stdout.flush();

        torch.cuda.empty_cache()

    sys.stdout.write("\n");
        
    return loss/counter

def train_network():

    with open("../config/experiment.yaml", 'r') as stream:
        config = yaml.safe_load(stream)

    out_data = {'train_loss' : [], 'val_loss' : [], 'F1' : []}

    np.random.seed(config["SEED"])

    train_data_loader, val_data_loader = get_train_val_data_loaders(config)

    model = CRNN(num_classes=len(config['classes'])-1, melfb=config["melfb"])
    model = model.cuda()

    EPOCHS = config['epochs']
    BATCH_SIZE = config['batch_size']
    #criterion = torch.nn.BCEWithLogitsLoss()#torch.nn.CrossEntropyLoss()
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    for it in range(EPOCHS):


        print(time.strftime("%Y-%m-%d %H:%M:%S"), it, "Training epoch %d  "%(it) );

        train_loss = step(train_data_loader, model, optimizer, criterion, BATCH_SIZE)
        metrics, val_loss = evaluate_model(val_data_loader, model, config["classes"], config['threshold'], 
                            config['melfb']['hop_len'], config["sample_rate"], config["iou_thr"], criterion)

        print(time.strftime("%Y-%m-%d %H:%M:%S"), "TRAIN LOSS %f"%(train_loss), "VAL LOSS %f"%(val_loss));

        out_data['train_loss'].append(train_loss)
        out_data['val_loss'].append(val_loss)
        out_data['F1'].append(metrics['f_measure'])

        torch.save(model.state_dict(), config['savepath'] + "%04d.pt"%it )
    
    return out_data

def get_ref_test_event_list(df, cls_desed):
    ret = []

    for (fn, data) in df.groupby('filename'):

        for _, r in data.iterrows():
            event = {}

            if r.event_label in cls_desed:
                for k, v in dict(r).items():
                    event[k] = v
            
                ret.append(event)

    return ret


def evaluate_test(model_path):
    with open("../config/test_evaluation.yaml", 'r') as stream:
        config = yaml.safe_load(stream)
    
    df = pd.read_csv(config["meta_path"], sep='\t')
    
    test_data_loader = get_test_data_loader(config, df )
    model = CRNN(num_classes=len(config['classes'])-1, melfb=config["melfb"])
    model.load_state_dict(torch.load(model_path))
    model = model.cuda()
    model.eval()

    ref_event_list = get_ref_test_event_list(df, config['classes_desed'])
    est_event_list = []

    total = len(df.groupby('filename'))

    for data, fn in tqdm(test_data_loader, total=total):
        with torch.no_grad():
            out = model(data.squeeze(0).cuda())
        map_event(out.detach().cpu().numpy(), est_event_list, config["classes_desed"], 
                    config["threshold"], config["melfb"]["hop_len"], config["sample_rate"], fn=fn[0])
    
    print_metrics(ref_event_list, est_event_list, config['iou_thr'])



#if __name__=="__main__":
#    evaluate_test("../models/model0010.pt")