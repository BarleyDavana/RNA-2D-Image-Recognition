import os
import numpy as np
from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.transforms import Compose, Normalize, Resize, ToTensor, RandomHorizontalFlip, RandomCrop
from m1CNNRNN import EncoderDecodertrain152
from dataset1 import GetDataset, CapsCollate
from vocab import Vocabulary
from utils import get_best_model
import matplotlib.pyplot as plt
from PIL import Image
import PIL
import cv2
import json

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

tqdm.pandas()

dataset = 'data1'

# generate caption
def get_caps_from(model, features_tensors, vocab):
    # generate the caption
    model.eval()
    with torch.no_grad():
        features = model.encoder(features_tensors.to(device))
        caps, alphas = model.decoder.generate_caption(features, stoi=vocab.stoi,
                                                      itos=vocab.itos)
        caption = ' '.join(caps)
    #print(alphas)
    return caps, alphas

def show_images_and_save(img, result, target, index):
    img = img.numpy().transpose((1, 2, 0))
    temp_image = img
    fig = plt.figure(figsize=(14, 14))
    fontsize = 25
    result = result[:-1]

    # Show the predicted label
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Predicted: ' + ''.join(result), fontsize=fontsize)
    img = ax.imshow(temp_image)
    ax.imshow(np.ones_like(temp_image), alpha=0, extent=img.get_extent())

    # Save the figure
    plt.savefig(f"image_{index}.png")
    plt.close(fig)


# Show attention
def plot_attention(img, result, attention_plot, target):
    # untransform
    img[0] = img[0] * 0.229
    img[1] = img[1] * 0.224
    img[2] = img[2] * 0.225
    img[0] += 0.485
    img[1] += 0.456
    img[2] += 0.406

    fontsize = 30

    img = img.numpy().transpose((1, 2, 0))
    temp_image = img
    fig = plt.figure(figsize=(15, 15))

    result = result[:-1]

    len_result = len(result)
    all_att = np.zeros(shape=(14, 14))

    for l in range(len_result):
        temp_att = attention_plot[l].reshape(14, 14)
        ax = fig.add_subplot(5,10, l + 1)
        ax.set_title(result[l], fontsize=fontsize)
        img = ax.imshow(temp_image, alpha=1)
        plt.axis('off')
        all_att += temp_att

        # normalize the attention map
        mask = cv2.resize(temp_att, (256, 256))
        normed_mask = mask / mask.max()
        normed_mask = (normed_mask * 255).astype('uint8')
        ax.imshow(normed_mask, alpha=0.2, interpolation='nearest', cmap="jet", extent = img.get_extent())

    plt.tight_layout()
    plt.show()
    plt.savefig('attentionm1.png')

if __name__ == '__main__':
    transform = Compose([
        # RandomHorizontalFlip(),
        Resize((256, 256), PIL.Image.BICUBIC),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    vocab = Vocabulary(1)
    with open('/data1/casual/luyitan/a325/vocabulary_file.json', 'r') as f:
        load_dict = json.load(f)
    vocab.stoi = load_dict
    vocab.itos = {v: k for k, v in vocab.stoi.items()}

    test_dataset = GetDataset(transform, dataset + '/label/test4_label.csv', 'test', vocab)

    pad_idx = vocab.stoi["<pad>"]

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        # shuffle=True,
        num_workers=10,
        collate_fn=CapsCollate(pad_idx=pad_idx, batch_first=True)
    )

    MODEL_PATH = '/data1/casual/luyitan/a325/turn1/model_files5w_m1/attention_model_state_20.pth'
    model_state = torch.load(MODEL_PATH)
    model = EncoderDecodertrain152(
        embed_size=model_state['embed_size'],
        vocab_size=model_state['vocab_size'],
        attention_dim=model_state['attention_dim'],
        encoder_dim=model_state['encoder_dim'],
        decoder_dim=model_state['decoder_dim']
    ).to(device)
    model.load_state_dict(model_state['state_dict'])
    model = model.to(device)

    # show any 1
    dataiter = iter(test_dataloader)
    flag = True
    index = 0
    while True:
        images, _ = next(dataiter)
        target = vocab.tensor_to_captions(_)
        for idx, item in enumerate(target):
            img = images[idx].detach().clone()
            img1 = images[idx].detach().clone()
            caps, alphas = get_caps_from(model, img.unsqueeze(0), vocab)
            #show_images_and_save(img1, caps, target[idx], index)
            result = caps[:-1]
            results = ''.join(result)
            print(results)
            plot_attention(img1, caps, alphas, target[index])
            index += 1
            flag = False
            break
