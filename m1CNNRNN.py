import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from tqdm.auto import tqdm

tqdm.pandas()
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.attention_dim = attention_dim
        self.W = nn.Linear(decoder_dim, attention_dim) # transform the decoder's hidden state to the attention dimension
        self.U = nn.Linear(encoder_dim, attention_dim) # transform the encoder's outputs to the attention dimension
        self.A = nn.Linear(attention_dim, 1) # map the combined states to a scalar for computing attention weights

    def forward(self, features, hidden_state):
        u_hs = self.U(features)  # (batch_size,196,attention_dim)
        w_ah = self.W(hidden_state)  # (batch_size,attention_dim)
        combined_states = torch.tanh(u_hs + w_ah.unsqueeze(1))  # (batch_size,196,attemtion_dim)
        attention_scores = self.A(combined_states)  # (batch_size,196,1)
        attention_scores = attention_scores.squeeze(2)  # (batch_size,196)
        alpha = F.softmax(attention_scores, dim=1)  # (batch_size,196)
        attention_weights = features * alpha.unsqueeze(2)  # (batch_size,196,features_dim)
        attention_weights = attention_weights.sum(dim=1)  # (batch_size,features_dim)
        return alpha, attention_weights

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, drop_prob=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.attention_dim = attention_dim
        self.decoder_dim = decoder_dim
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.lstm_cell = nn.LSTMCell(embed_size + encoder_dim, decoder_dim, bias=True)
        self.fcn = nn.Linear(decoder_dim, vocab_size)
        self.drop = nn.Dropout(drop_prob)

    def forward(self, features, captions):
        embeds = self.embedding(captions)
        h, c = self.init_hidden_state(features)  # (batch_size, decoder_dim)
        seq_length = len(captions[0]) - 1  # Exclude the last one
        batch_size = captions.size(0)
        num_features = features.size(1)

        preds = torch.zeros(batch_size, seq_length, self.vocab_size).to(device)
        alphas = torch.zeros(batch_size, seq_length, num_features).to(device)

        for s in range(seq_length):
            alpha, context = self.attention(features, h)
            lstm_input = torch.cat((embeds[:, s], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))

            output = self.fcn(self.drop(h))

            preds[:, s] = output
            alphas[:, s] = alpha

        return preds, alphas

    def generate_caption(self, features, max_len=51, itos=None, stoi=None):
        # Inference part
        # Given the image features generate the captions

        batch_size = features.size(0)
        h, c = self.init_hidden_state(features)  # (batch_size, decoder_dim)

        alphas = []

        # starting input
        word = torch.tensor(stoi['<sos>']).view(1, -1).to(device)
        embeds = self.embedding(word)

        captions = []

        for i in range(max_len):
            alpha, context = self.attention(features, h)
            alphas.append(alpha.cpu().detach().numpy())
            # print('embeds',embeds.shape)
            # print('embeds[:,0]',embeds[:,0].shape)
            # print('context',context.shape)
            lstm_input = torch.cat((embeds[:, 0], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            output = self.fcn(self.drop(h))
            # print('output',output.shape)
            output = output.view(batch_size, -1)

            # select the word with most val
            predicted_word_idx = output.argmax(dim=1)

            # save the generated word
            captions.append(predicted_word_idx.item())
            # print('predicted_word_idx',predicted_word_idx.shape)

            # end if <EOS detected>
            if itos[predicted_word_idx.item()] == "<eos>":
                break

            # send generated word as the next caption
            embeds = self.embedding(predicted_word_idx.unsqueeze(0))

        # covert the vocab idx to words and return sentence
        return [itos[idx] for idx in captions], alphas

    def generate_caption_batch(self, features, max_len=51, itos=None, stoi=None):
        # Inference part
        # Given the image features generate the captions

        batch_size = features.size(0)
        h, c = self.init_hidden_state(features)  # (batch_size, decoder_dim)

        alphas = []

        word = torch.full((batch_size, 1), stoi['<sos>']).to(device)
        embeds = self.embedding(word)

        captions = torch.zeros((batch_size, max_len), dtype=torch.long).to(device)
        captions[:, 0] = word.squeeze()

        for i in range(max_len):
            alpha, context = self.attention(features, h)
            lstm_input = torch.cat((embeds[:, 0], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            output = self.fcn(self.drop(h))
            output = output.view(batch_size, -1)
            predicted_word_idx = output.argmax(dim=1)
            captions[:, i] = predicted_word_idx
            embeds = self.embedding(predicted_word_idx).unsqueeze(1)

        return captions

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

class EncoderCNNtrain152(nn.Module):
    def __init__(self, encoded_image_size=14):
        super(EncoderCNNtrain152, self).__init__()
        resnet = torchvision.models.resnet152(pretrained = True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.enc_image_size = encoded_image_size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

    def forward(self, images):
        features = self.resnet(images)  # (batch_size,2048,14,14)
        features = self.adaptive_pool(features)
        features = features.permute(0, 2, 3, 1)  # (batch_size,14,14,2048)
        features = features.view(features.size(0), -1, features.size(-1))  # (batch_size,196,2048)
        return features

class EncoderDecodertrain152(nn.Module):
    def __init__(self, embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, drop_prob=0.3):
        super().__init__()
        self.encoder = EncoderCNNtrain152()
        self.decoder = DecoderRNN(
            embed_size=embed_size,
            vocab_size=vocab_size,
            attention_dim=attention_dim,
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim
        )

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs
