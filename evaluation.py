import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.transforms import Compose, Normalize, Resize, ToTensor, RandomHorizontalFlip, RandomCrop
from tqdm.auto import tqdm
import time
from m1CNNRNN import EncoderDecodertrain152
from dataset1 import GetDataset, CapsCollate
from vocab import Vocabulary
import matplotlib.pyplot as plt
import Levenshtein
import json
from utils import get_best_model
from PIL import Image
import PIL

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

tqdm.pandas()

if __name__ == '__main__':
    #dataset_name = 'data1'
    dataset_name = 'data5w'

    transform = Compose([
        #RandomHorizontalFlip(),
        Resize((256, 256), PIL.Image.LANCZOS),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    vocab = Vocabulary(1)
    with open('/data1/casual/luyitan/a325/vocabulary_file.json', 'r') as f:
        load_dict = json.load(f)
    vocab.stoi = load_dict
    vocab.itos = {v: k for k, v in vocab.stoi.items()}

    test_dataset = GetDataset(transform, dataset_name + '/label/test_label.csv', 'test', vocab)
    #test_dataset = GetDataset(transform, dataset_name + '/label/testdb.csv', 'test', vocab)

    pad_idx = vocab.stoi["<pad>"]

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=10,
        collate_fn=CapsCollate(pad_idx=pad_idx, batch_first=True),
    )

    MODEL_PATH = get_best_model('model_files5w_m1')
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

    f = open('result/test5w_m1.txt', 'w')

    print(len(test_dataset))
    
    total_correct = 0
    total_samples = 0
    ALDs = []

    model.eval()
    with torch.no_grad():
        for i, (image, captions) in enumerate(test_dataloader):
            img, target = image.to(device), captions.to(device)
            features = model.encoder(img)
            caps = model.decoder.generate_caption_batch(features, stoi=vocab.stoi, itos=vocab.itos)
            gen_captions = vocab.tensor_to_captions(caps)
            targets = vocab.tensor_to_captions(target)
            
            # correct_ratios = []

            # for gen_caption, target_caption in zip(gen_captions, targets):
            #     size = min(len(gen_caption), len(target_caption))
            #     same_char = sum(1 for i in range(size) if gen_caption[i] == target_caption[i])
            #     correct_ratio = same_char / len(target_caption)
            #     correct_ratios.append(correct_ratio)
            LDs = []
            for gen_caption, target_caption in zip(gen_captions, targets):
                LD = Levenshtein.distance(gen_caption, target_caption)
                LDs.append(LD)
                ALDs.append(LD)
            
            # for correct_ratio, gen_caption, target_caption in zip(correct_ratios, gen_captions, targets):
            #     f.write('{}\t{}\t{:.2f}\n'.format(gen_caption, target_caption, correct_ratio))
            #     print("Correct ratio for sample {}: {:.2f}".format(i, correct_ratio))
            for LD, gen_caption, target_caption in zip(LDs, gen_captions, targets):
                f.write('{}\t{}\t{:.2f}\n'.format(gen_caption, target_caption, LD))
                print("Levenshtein distance for sample {}: {:.2f}".format(i, LD))

            same_list = [x for _, x in enumerate(gen_captions) if x == targets[_]]
            
            total_correct += len(same_list)
            total_samples += len(target)
        
            print('Process: [{}/{}]\tAcc {}'.format(i, len(test_dataloader), len(same_list) / len(target)))
            
            average_distance = sum(LDs) / len(LDs)
            print('Process: [{}/{}]\tLD {}'.format(i, len(test_dataloader), average_distance))

    f.close()

accuracy = total_correct / total_samples
average_distance = sum(ALDs) / len(ALDs)
print("Total accuracy: {:.2f}".format(accuracy))
print("Total ALD: {:.2f}".format(average_distance))

print(total_correct)
print(total_samples)

import pandas as pd

original_label_file = '/data1/casual/luyitan/a325/data5w/label/test_label.csv'
#original_label_file = '/data1/casual/luyitan/a325/data1/label/testdb.csv'

original_df = pd.read_csv(original_label_file)

result_file = '/data1/casual/luyitan/a325/result/test5w_m1.txt'
# result_file = '/data1/casual/luyitan/a325/result/test5wdb_m1.txt'

result_df = pd.read_csv(result_file, delimiter='\t', header=None)
result_df.columns = ['Predicted', 'Target', 'Distance']

merged_df = pd.concat([original_df[['urs_id']], result_df[['Predicted', 'Target', 'Distance']]], axis=1)

merged_df.to_csv(result_file, index=False)
