import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import PIL
import time
import matplotlib.pyplot as plt
import json
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from torchvision.transforms.transforms import Compose, Normalize, Resize, ToTensor, RandomHorizontalFlip, RandomCrop
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
import torch.nn.functional as F
import datetime
import seaborn as sns
from tqdm.auto import tqdm
from m1CNNRNN import EncoderDecodertrain152
from dataset1 import GetDataset, CapsCollate
from vocab import Vocabulary
import pickle
from timeit import default_timer as timer
from torch.nn.parallel import DataParallel

start_time = timer()

#device_ids = list(range(torch.cuda.device_count()))
#device_ids = list(range(13,16))

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

tqdm.pandas()

vocabulary_file_path = '/data1/casual/luyitan/a325/vocabulary_file.json'
vocab = Vocabulary(1)
with open(vocabulary_file_path, 'r') as f:
    load_dict = json.load(f)
vocab.stoi = load_dict
vocab.itos = {v: k for k, v in vocab.stoi.items()}

transform = Compose([
    Resize((256, 256), PIL.Image.LANCZOS),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = GetDataset(transform, '/data1/casual/luyitan/a325/data5w/label/train_label.csv', 'train', vocab)
val_dataset = GetDataset(transform, '/data1/casual/luyitan/a325/data5w/label/val_label.csv', 'val', vocab)

pad_idx = vocab.stoi["<pad>"]

batch_size = 16

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=10,
    collate_fn=CapsCollate(pad_idx=pad_idx, batch_first=True)
)

val_dataloader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=10,
    collate_fn=CapsCollate(pad_idx=pad_idx, batch_first=True)
)

embed_size = 512
attention_dim = 1024
encoder_dim = 2048
decoder_dim = 2048
learning_rate = 1e-5
vocab_size = len(vocab)

model = EncoderDecodertrain152(
    embed_size=embed_size,
    vocab_size=vocab_size,
    attention_dim=attention_dim,
    encoder_dim=encoder_dim,
    decoder_dim=decoder_dim
).to(device)

#model = DataParallel(model, device_ids=device_ids).to(device_ids[0])  

# class_weights = torch.ones(len(vocab), device = device)

# special_tokens = ["<sos>", "<eos>"]
# special_tokens2 = ["(", ")"]

# special_weight = 3.0
# special_weight2 = 2.0

# for token in special_tokens:
#     idx = vocab.stoi[token]
#     class_weights[idx] = special_weight

# for token in special_tokens2:
#     idx = vocab.stoi[token]
#     class_weights[idx] = special_weight2

# criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=vocab.stoi["<pad>"])

criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<pad>"])
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def save_model(model, num_epochs, acc):
    model_state = {
        'num_epochs': num_epochs,
        'embed_size': embed_size,
        'vocab_size': len(vocab),
        'attention_dim': attention_dim,
        'encoder_dim': encoder_dim,
        'decoder_dim': decoder_dim,
        'state_dict': model.state_dict(),
        'val_acc': acc
    }

    torch.save(model_state,
               '/data1/casual/luyitan/a325/model_files5w_m1/' + 'attention_model_state_' + str(
                   num_epochs) + '.pth')

num_epochs = 20
print_every = 50
print_every_val = 20

epoch_count = []

train_loss_values = []
val_loss_values = []

train_accuracy_values = []
val_accuracy_values = []

def accuracy(outputs, targets):
    pred = outputs.argmax(dim=-1)
    pred = pred.cpu().numpy()
    targets = targets.cpu().numpy()
    # print(pred)
    # print(pred.shape)
    # print(targets)
    # print(target.shape)
    pred_list = np.concatenate([item for item in pred])
    # print(pred_list)
    # print(pred_list.shape)
    target_list = np.concatenate([item for item in targets])
    # print(target_list)
    # print(target_list.shape)
    correct_cnt = torch.eq(torch.from_numpy(pred_list), torch.from_numpy(target_list)).view(-1).float().sum().item()
    # print(correct_cnt)
    # print(len(pred_list))
    return correct_cnt / len(pred_list)

for epoch in range(1, num_epochs + 1):
    epoch_count.append(epoch)
    train_loss, train_acc = 0, 0
    for idx, (image, captions) in enumerate(iter(train_dataloader)):
        #image, captions = image.to(device_ids[0]), captions.to(device_ids[0])
        image, captions = image.to(device), captions.to(device)

        # Zero the gradients.
        optimizer.zero_grad()

        # Feed forward
        outputs, attentions = model(image, captions)
        # print(outputs)
        # print(outputs.shape)

        # Calculate the batch loss.
        targets = captions[:, 1:]
        # print(targets.shape)
        loss = criterion(outputs.view(-1, vocab_size), targets.reshape(-1))
        train_loss += loss.item()

        # Backward pass.
        loss.backward()

        # Update the parameters in the optimizer.
        optimizer.step()
        
        # Calculate the batch accuracy
        acc = accuracy(outputs, targets)
        train_acc += acc

        if (idx + 1) % print_every == 0:
            print('Epoch: [{}][{}/{}]\tLoss {}\tAcc {}'.format(epoch, idx, len(train_dataloader), loss.item(),
                                                                acc))

            model.train()
    
    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)
    
    torch.cuda.empty_cache()

    # one epoch validation
    count = 0
    val_loss, val_acc = 0, 0
    model.eval()
    with torch.no_grad():
        for idx_val, (image, captions) in enumerate(iter(val_dataloader)):
            # Move to device, if available
            img, target = image.to(device), captions.to(device)

            # Forward pass
            outputs_val, unused = model(img, target)
            target_val = target[:, 1:]

            # Calculate batch loss
            loss_val = criterion(outputs_val.view(-1, vocab_size), target_val.reshape(-1))
            val_loss += loss_val.item()
            
            # Calculate batch acc
            acc_val = accuracy(outputs_val, target_val)
            val_acc += acc_val
            
            if (idx_val + 1) % print_every_val == 0:
                print('Epoch: [{}][{}/{}]\tLoss {}\tAcc {}'.format(epoch, idx_val, len(val_dataloader), loss_val.item(),
                                                                   acc_val))
        val_loss /= len(val_dataloader)
        val_acc /= len(val_dataloader)
        
    model.train()
    
    train_loss_values.append(train_loss)
    train_accuracy_values.append(train_acc)
    
    val_loss_values.append(val_loss)
    val_accuracy_values.append(val_acc)
    
    # save the latest model
    #model = model.module
    save_model(model, epoch, val_acc)
    torch.cuda.empty_cache()
    
    if epoch % 5 == 0:
        with open('/data1/casual/luyitan/a325/result/train_data5w_m1.pkl', 'wb') as f:
            data = {
                'epoch_count': epoch_count,
                'train_loss_values': train_loss_values,
                'val_loss_values': val_loss_values,
                'train_accuracy_values': train_accuracy_values,
                'val_accuracy_values': val_accuracy_values
            }
            pickle.dump(data, f)

end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")
