#!/usr/bin/env python
import os, re, torch, shutil
import torch.nn as nn
from cnns.resnet import resnet101
#from models.inception import inception_v3
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import urllib, json
from urllib import request
#from shapeworld import Dataset

def trace():
    print('Tracing!')
    dataset = Dataset.create(dtype='agreement', name='existential')
    generated = dataset.generate(n=128, mode='train', include_model=True)
    print('\n'.join(dataset.to_surface(value_type='language', word_ids=generated['caption'][:5])))

def load_resnet101(data_dir):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    # downloads pretrained resnet-101
    model = resnet101(pretrained=True)
    traindir = os.path.join(data_dir + 'grounded/')
    
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset)

    classes = [
            d for d in os.listdir(
                train_dataset.root) if os.path.isdir(
                os.path.join(
                    train_dataset.root,
                    d))]
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    rev_class = {class_to_idx[key]: key for key in class_to_idx.keys()}

    
    for i, (input, target) in enumerate(train_loader):
        x = model.forward(input)
        feat_dict = {}; feat_dict[rev_class[i]] = x.data.numpy().tolist()
        f = open(data_dir + 'feats/' + rev_class[i] + '.json', 'w+')
        f.write(json.dumps(feat_dict)); f.close()
        #print(rev_class[i])

def build():
    vocab = ['brown', 'robotics', 'navigation', 'great', 'is', 'for', 'and', 'linguistics', 'cognition']
    seq2seq = Seq2SeqEncoder()
    rnn = RNNEncoder(vocab=vocab, text_field_embedder=None, num_highway_layers=0, phrase_layer=seq2seq)
    sent = 'linguistics is great for robotics'

def clean_sw():
    dirpath = os.getcwd() + '/ShapeWorld/examples/agreement/'
    ctypes = [name for name in os.listdir(dirpath) if '.DS' not in name]
    img_dict = {}
    for ctype in ctypes:
        if os.path.isdir(dirpath + ctype) is False:
            os.mkdir(dirpath + ctype)
        imgs = [fname for fname in os.listdir(dirpath + ctype) if fname.endswith('png')]
        print(imgs)
        newdirpath = os.getcwd() + '/shapeworld/figs/'
        if os.path.isdir(newdirpath + ctype) is False:
            os.mkdir(newdirpath + ctype)

        indices = {}
        f = open(dirpath + ctype + '/caption.txt', 'r')
        lines = f.readlines(); chunks = [lines[i:i+6] for i in range(0, len(lines), 6)]
        
        f = open(dirpath + ctype + '/agreement.txt', 'r')
        lines = f.readlines()

        for i in range(len(chunks)):
            captions, labels, index = chunks[i][:-1], [int(float(item)) for item in lines[i].strip().split(';')], len(indices)
            indices[index] = {'1': [], '0': []}
            for j in range(len(labels)):
                indices[index][str(labels[j])].append(captions[j].strip())
            
        for img in imgs:
            index = int(img.split('.')[0].split('-')[-1])
            if os.path.isdir(newdirpath + ctype + '/' + str(len(img_dict))) is False:
                os.mkdir(newdirpath + ctype + '/' + str(len(img_dict)))
            if os.path.isfile(newdirpath + ctype + '/' + str(len(img_dict))) is False:
                shutil.copy(dirpath + ctype + '/' + img, newdirpath + ctype + '/' + str(len(img_dict)))
            img_dict[len(img_dict)] = {'image_feat':[], 'orig_index': index, 'ctype': ctype, 'captions': indices[index]}

    f = open(os.getcwd() + '/shapeworld/data.json', 'w+')
    f.write(json.dumps(img_dict))

def create_sw():
    f = open(os.getcwd() + '/shapeworld/data.json', 'r')
    for line in f: data = json.loads(line)

    remove = ['logical-existential', 'logical-full']

    train, val, test = [], [], []

    ctypes = [item for item in os.listdir(os.getcwd() + '/shapeworld/figs/') if '.DS' not in item]
    temp = {}
    for key in data:
        ctype, img, captions = data[key]['ctype'], key, data[key]['captions']
        if ctype not in temp.keys(): temp[ctype] = []
        temp[ctype].append([img, captions])

    for ctype in temp:
        items = temp[ctype]
        val.extend(items[:5]); test.extend(items[5:10]); train.extend(items[10:])

    def fin_sw(datatype, data):
        f = open(os.getcwd() + '/shapeworld/' + datatype + '.tsv', 'w+')
        for item in data:
            img, captions = item[0], item[1]
            for label in captions:
                for caption in captions[label]:
                    s = caption + '\t' + str(label) + '\t' + str(img) + '\n'
                    f.write(s)

    fin_sw('test', test); fin_sw('dev', dev); fin_sw('train', train)

def clean_abs():
    return

if __name__ == '__main__':
    clean_abs()
