import torch
from torch import nn
import numpy as np
import torchvision

device = torch.device("cuda:9" if torch.cuda.is_available() else "cpu")

class ScaledDotProductAttention(nn.Module):
    def __init__(self, QKVdim):
        super(ScaledDotProductAttention, self).__init__()
        self.QKVdim = QKVdim

    def forward(self, Q, K, V, attn_mask):
        """
        :param Q: [batch_size, n_heads, -1(len_q), QKVdim]
        :param K, V: [batch_size, n_heads, -1(len_k=len_v), QKVdim]
        :param attn_mask: [batch_size, n_heads, len_q, len_k]
        """
        # scores: [batch_size, n_heads, len_q, len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.QKVdim)
        scores.to(device).masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)  # [batch_size, n_heads, len_q, len_k]
        context = torch.matmul(attn, V).to(device)  # [batch_size, n_heads, len_q, QKVdim]
        return context, attn

class Multi_Head_Attention(nn.Module):
    def __init__(self, Q_dim, K_dim, QKVdim, n_heads=8, dropout=0.1):
        super(Multi_Head_Attention, self).__init__()
        self.W_Q = nn.Linear(Q_dim, QKVdim * n_heads).to(device)
        self.W_K = nn.Linear(K_dim, QKVdim * n_heads).to(device)
        self.W_V = nn.Linear(K_dim, QKVdim * n_heads).to(device)
        self.n_heads = n_heads
        self.QKVdim = QKVdim
        self.embed_dim = Q_dim
        self.dropout = nn.Dropout(p=dropout)
        self.W_O = nn.Linear(self.n_heads * self.QKVdim, self.embed_dim).to(device)

    def forward(self, Q, K, V, attn_mask):
        """
        Self-encoder attention:
                Q = K = V: [batch_size, num_pixels=196, encoder_dim=2048]
                attn_mask: [batch_size, len_q=196, len_k=196]
        Self-decoder attention:
                Q = K = V: [batch_size, max_len=52, embed_dim=512]
                attn_mask: [batch_size, len_q=52, len_k=52]
        encoder-decoder attention:
                Q: [batch_size, 52, 512] from decoder
                K, V: [batch_size, 196, 2048] from encoder
                attn_mask: [batch_size, len_q=52, len_k=196]
        return _, attn: [batch_size, n_heads, len_q, len_k]
        """
        residual, batch_size = Q, Q.size(0)
        # q_s: [batch_size, n_heads=8, len_q, QKVdim] k_s/v_s: [batch_size, n_heads=8, len_k, QKVdim]
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.QKVdim).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.QKVdim).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.QKVdim).transpose(1, 2)
        # attn_mask: [batch_size, self.n_heads, len_q, len_k]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        # attn: [batch_size, n_heads, len_q, len_k]
        # context: [batch_size, n_heads, len_q, QKVdim]
        context, attn = ScaledDotProductAttention(self.QKVdim)(q_s, k_s, v_s, attn_mask)
        # context: [batch_size, n_heads, len_q, QKVdim] -> [batch_size, len_q, n_heads * QKVdim]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.QKVdim).to(device)
        # output: [batch_size, len_q, embed_dim]
        output = self.W_O(context)
        output = self.dropout(output)
        return nn.LayerNorm(self.embed_dim).to(device)(output + residual), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, embed_dim, d_ff, dropout):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=d_ff, kernel_size=1).to(device)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=embed_dim, kernel_size=1).to(device)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_dim = embed_dim

    def forward(self, inputs):
        """
        encoder: inputs: [batch_size, len_q=196, embed_dim=2048]
        decoder: inputs: [batch_size, max_len=52, embed_dim=512]
        """
        residual = inputs
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        output = self.dropout(output)
        return nn.LayerNorm(self.embed_dim).to(device)(output + residual)

class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, dropout, attention_method, n_heads):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = Multi_Head_Attention(Q_dim=embed_dim, K_dim=embed_dim, QKVdim=64, n_heads=n_heads, dropout=dropout)
        if attention_method == "ByPixel":
            self.dec_enc_attn = Multi_Head_Attention(Q_dim=embed_dim, K_dim=2048, QKVdim=64, n_heads=n_heads, dropout=dropout)
            self.pos_ffn = PoswiseFeedForwardNet(embed_dim=embed_dim, d_ff=2048, dropout=dropout)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        """
        :param dec_inputs: [batch_size, max_len=52, embed_dim=512]
        :param enc_outputs: [batch_size, num_pixels=196, 2048]
        :param dec_self_attn_mask: [batch_size, 52, 52]
        :param dec_enc_attn_mask: [batch_size, 52, 196]
        """
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn

class Decoder(nn.Module):
    def __init__(self, n_layers, vocab_size, embed_dim, dropout, attention_method, n_heads):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.tgt_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding.from_pretrained(self.get_position_embedding_table(embed_dim), freeze=True)
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList([DecoderLayer(embed_dim, dropout, attention_method, n_heads) for _ in range(n_layers)])
        self.projection = nn.Linear(embed_dim, vocab_size, bias=False).to(device)
        self.attention_method = attention_method

    def get_position_embedding_table(self, embed_dim):
        def cal_angle(position, hid_idx):
            return position / np.power(10000, 2 * (hid_idx // 2) / embed_dim)
        def get_posi_angle_vec(position):
            return [cal_angle(position, hid_idx) for hid_idx in range(embed_dim)]

        embedding_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(51)])
        embedding_table[:, 0::2] = np.sin(embedding_table[:, 0::2])  # dim 2i
        embedding_table[:, 1::2] = np.cos(embedding_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(embedding_table).to(device)

    def get_attn_pad_mask(self, seq_q, seq_k):
        # generate an attention mask for masking padding tokens
        batch_size, len_q = seq_q.size()
        batch_size, len_k = seq_k.size()
        # In wordmap, <pad>:0
        # pad_attn_mask: [batch_size, 1, len_k], one is masking
        pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
        return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]

    def get_attn_subsequent_mask(self, seq):
        # generate an attention mask for masking future tokens
        attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
        subsequent_mask = np.triu(np.ones(attn_shape), k=1)
        subsequent_mask = torch.from_numpy(subsequent_mask).byte().to(device)
        return subsequent_mask

    def forward(self, encoder_out, encoded_captions):
        """
        :param encoder_out: [batch_size, num_pixels=196, 2048]
        :param encoded_captions: [batch_size, 52]
        :param caption_lengths: [batch_size, 1]
        """
        batch_size = encoder_out.size(0)
        decode_lengths = len(encoded_captions[0]) - 1  # Exclude the last one
        # dec_outputs: [batch_size, max_len=52, embed_dim=512]
        # dec_self_attn_pad_mask: [batch_size, len_q=52, len_k=52], 1 if id=0(<pad>)
        # dec_self_attn_subsequent_mask: [batch_size, 52, 52], Upper triangle of an array with 1.
        # dec_self_attn_mask for self-decoder attention, the position whose val > 0 will be masked.
        # dec_enc_attn_mask for encoder-decoder attention.
        # e.g. 9488, 23, 53, 74, 0, 0  |  dec_self_attn_mask:
        # 0 1 1 1 2 2
        # 0 0 1 1 2 2
        # 0 0 0 1 2 2
        # 0 0 0 0 2 2
        # 0 0 0 0 1 2
        # 0 0 0 0 1 1
        dec_outputs = self.tgt_emb(encoded_captions) + self.pos_emb(torch.LongTensor([list(range(len(encoded_captions[0])))]*batch_size).to(device))
        dec_outputs = self.dropout(dec_outputs)
        dec_self_attn_pad_mask = self.get_attn_pad_mask(encoded_captions, encoded_captions)
        dec_self_attn_subsequent_mask = self.get_attn_subsequent_mask(encoded_captions)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
        if self.attention_method == "ByPixel":
            dec_enc_attn_mask = (torch.tensor(np.zeros((batch_size, len(encoded_captions[0]), 196))).to(device) == torch.tensor(np.ones((batch_size, len(encoded_captions[0]), 196))).to(device))

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # attn: [batch_size, n_heads, len_q, len_k]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, encoder_out, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        predictions = self.projection(dec_outputs)
        return predictions, encoded_captions, decode_lengths, dec_self_attns, dec_enc_attns

class EncoderLayer(nn.Module):
    def __init__(self, dropout, attention_method, n_heads):
        super(EncoderLayer, self).__init__()
        if attention_method == "ByPixel":
            self.enc_self_attn = Multi_Head_Attention(Q_dim=2048, K_dim=2048, QKVdim=64, n_heads=n_heads, dropout=dropout)
            self.pos_ffn = PoswiseFeedForwardNet(embed_dim=2048, d_ff=4096, dropout=dropout)

    def forward(self, enc_inputs, enc_self_attn_mask):
        """
        :param enc_inputs: [batch_size, num_pixels=196, 2048]
        :param enc_outputs: [batch_size, len_q=196, d_model=2048]
        :return: attn: [batch_size, n_heads=8, 196, 196]
        """
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn

class Encoder(nn.Module):
    def __init__(self, n_layers, dropout, attention_method, n_heads):
        super(Encoder, self).__init__()
        if attention_method == "ByPixel":
            self.pos_emb = nn.Embedding.from_pretrained(self.get_position_embedding_table(), freeze=True)
        self.layers = nn.ModuleList([EncoderLayer(dropout, attention_method, n_heads) for _ in range(n_layers)])
        self.attention_method = attention_method

    def get_position_embedding_table(self):
    #     def cal_angle(position, hid_idx):
    #         x = position % 14
    #         y = position // 14
    #         x_enc = x / np.power(10000, hid_idx / 1024)
    #         y_enc = y / np.power(10000, hid_idx / 1024)
    #         return np.sin(x_enc), np.sin(y_enc)
    #     def get_posi_angle_vec(position):
    #         return [cal_angle(position, hid_idx)[0] for hid_idx in range(1024)] + [cal_angle(position, hid_idx)[1] for hid_idx in range(1024)]

        def cal_angle(position, hid_idx):
            x = position % 14
            y = position // 14
            x_enc = x / np.power(10000, hid_idx / 1024)
            y_enc = y / np.power(10000, hid_idx / 1024)
            if hid_idx % 2 ==0:
                return np.sin(x_enc), np.cos(y_enc)
            else:
                return np.cos(x_enc), np.sin(y_enc)
        def get_posi_angle_vec(position):
            return [cal_angle(position, hid_idx)[0] for hid_idx in range(1024)] + [cal_angle(position, hid_idx)[1] for hid_idx in range(1024)]
        
        embedding_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(196)])
        return torch.FloatTensor(embedding_table).to(device)

    def forward(self, encoder_out):
        """
        :param encoder_out: [batch_size, num_pixels=196, dmodel=2048]
        """
        batch_size = encoder_out.size(0)
        positions = encoder_out.size(1)
        if self.attention_method == "ByPixel":
            encoder_out = encoder_out + self.pos_emb(torch.LongTensor([list(range(positions))]*batch_size).to(device))
        # enc_self_attn_mask: [batch_size, 196, 196]
        enc_self_attn_mask = (torch.tensor(np.zeros((batch_size, positions, positions))).to(device)
                              == torch.tensor(np.ones((batch_size, positions, positions))).to(device))
        enc_self_attns = []
        for layer in self.layers:
            encoder_out, enc_self_attn = layer(encoder_out, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return encoder_out, enc_self_attns

class Transformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, encoder_layers, decoder_layers, dropout=0.1, attention_method="ByPixel", n_heads=8):
        super(Transformer, self).__init__()
        self.encoder = Encoder(encoder_layers, dropout, attention_method, n_heads)
        self.decoder = Decoder(decoder_layers, vocab_size, embed_dim, dropout, attention_method, n_heads)
        self.embedding = self.decoder.tgt_emb
        self.attention_method = attention_method

    def load_pretrained_embeddings(self, embeddings):
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def forward(self, enc_inputs, encoded_captions):
        """
        preprocess: enc_inputs: [batch_size, 14, 14, 2048]/[batch_size, 196, 2048] -> [batch_size, 196, 2048]
        encoded_captions: [batch_size, 52]
        caption_lengths: [batch_size, 1], not used
        The encoder or decoder is composed of a stack of n_layers=6 identical layers.
        One layer in encoder: Multi-head Attention(self-encoder attention) with Norm & Residual
                            + Feed Forward with Norm & Residual
        One layer in decoder: Masked Multi-head Attention(self-decoder attention) with Norm & Residual
                            + Multi-head Attention(encoder-decoder attention) with Norm & Residual
                            + Feed Forward with Norm & Residual
        """
        batch_size = enc_inputs.size(0)
        encoder_dim = enc_inputs.size(-1)
        if self.attention_method == "ByPixel":
            enc_inputs = enc_inputs.view(batch_size, -1, encoder_dim)
        encoder_out, enc_self_attns = self.encoder(enc_inputs)
        # encoder_out: [batch_size, 196, 2048]
        predictions, encoded_captions, decode_lengths, dec_self_attns, dec_enc_attns = self.decoder(encoder_out, encoded_captions)
        alphas = {"enc_self_attns": enc_self_attns, "dec_self_attns": dec_self_attns, "dec_enc_attns": dec_enc_attns}
        return predictions, alphas
    
    def generate_caption_batch(self, enc_inputs, max_len = 51, itos=None, stoi=None):
        """
        Generate captions batch for the given image features (enc_inputs).
        """
        # Initialize lists to store attention weights
        batch_size = enc_inputs.size(0)
        
        #dec_self_attns_list = []
        #dec_enc_attns_list = []
        encoder_dim = enc_inputs.size(-1)
        
        if self.attention_method == "ByPixel":
            enc_inputs = enc_inputs.view(batch_size, -1, encoder_dim)

        encoder_out, _ = self.encoder(enc_inputs)

        #start_token = torch.tensor(stoi['<sos>']).unsqueeze(0).unsqueeze(0).to(device)
        
        start_tokens = torch.full((batch_size, 1), stoi['<sos>']).to(device)
        captions = torch.zeros((batch_size, max_len), dtype=torch.long).to(device)

        # print(captions)
        # print(captions.shape)
        
        input_tokens = start_tokens
        
        #input_tokens = captions
        
        # print(input_tokens)
        # print(input_tokens.shape)
        
        for i in range(max_len):
            predictions, _, _, dec_self_attns, dec_enc_attns = self.decoder(encoder_out, input_tokens)
            # print(predictions)
            # print(predictions.shape)
            
            predictions = predictions[:, i, :].squeeze(1)  # [s, 1, vocab_size] -> [s, vocab_size]
            # print(predictions)
            # print(predictions.shape)
            
            predictions = torch.argmax(predictions, dim=1)
            # print(predictions)
            # print(predictions.shape)
            
            captions[:, i] = predictions
            # print(captions)
            
            predicted_next_token = predictions.unsqueeze(1)
            # print(predicted_next_token)
            input_tokens = torch.cat([input_tokens, predicted_next_token], dim=1)
            # print(input_tokens)
        
        return captions

    def generate_caption(self, enc_inputs, max_len = 51, itos=None, stoi=None):
        """
        Generate captions batch for the given image features (enc_inputs).
        """
        batch_size = enc_inputs.size(0)
        
        #dec_self_attns_list = []
        #dec_enc_attns_list = []
        encoder_dim = enc_inputs.size(-1)
        
        if self.attention_method == "ByPixel":
            enc_inputs = enc_inputs.view(batch_size, -1, encoder_dim)

        encoder_out, _ = self.encoder(enc_inputs)

        # Initialize lists to store attention weights        
        #dec_self_attns_list = []
        #dec_enc_attns_list = []
        #start_token = torch.tensor(stoi['<sos>']).unsqueeze(0).unsqueeze(0).to(device)
        start_tokens = torch.full((1, 1), stoi['<sos>']).to(device)

        #start_tokens = torch.tensor(stoi['<sos>']).view(1, -1).to(device)
        captions = []
        alphas = []
        #captions = torch.zeros((batch_size, max_len), dtype=torch.long).to(device)

        # print(captions)
        # print(captions.shape)
        
        input_tokens = start_tokens

        for i in range(max_len):
            predictions, _, _, dec_self_attns, dec_enc_attns = self.decoder(encoder_out, input_tokens)
            # print(predictions)
            #print(predictions.shape)

            alpha = dec_enc_attns[-1]
            alpha = alpha[:, 0, i, :].view(batch_size, 1, 14, 14).squeeze(1)
            #print(alpha.shape)
            alphas.append(alpha.cpu().detach().numpy())
            #print(alphas)

            predictions = predictions[:, i, :].view(batch_size, -1).squeeze(1)  # [s, 1, vocab_size] -> [s, vocab_size]
            # print(predictions)
            #print(predictions.shape)
            
            predictions = torch.argmax(predictions, dim=1)
            # print(predictions)
            # print(predictions.shape)
            
            captions.append(predictions.item())

            #print(captions)
            
            predicted_next_token = predictions.unsqueeze(1)
            # print(predicted_next_token)
            input_tokens = torch.cat([input_tokens, predicted_next_token], dim=1)
            # print(input_tokens)
            if itos[predicted_next_token.item()] == "<eos>":
                break
        #print(captions)
        captions = [itos[idx] for idx in captions]
        return captions, alphas

class CNN_Encoder(nn.Module):
    def __init__(self, encoded_image_size=14, attention_method="ByPixel"):
        super(CNN_Encoder, self).__init__()
        self.enc_image_size = encoded_image_size
        self.attention_method = attention_method
        resnet = torchvision.models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

    def forward(self, images):
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # [batch_size, 2048/512, 8, 8] -> [batch_size, 2048/512, 14, 14]
        out = out.permute(0, 2, 3, 1)
        return out

class EncoderDecoderTransformer(nn.Module):
    def __init__(self, embed_dim, vocab_size, encoder_layers, decoder_layers, dropout=0.1, attention_method="ByPixel", n_heads=8):
        super().__init__()
        self.encoder = CNN_Encoder()
        self.decoder = Transformer(
            embed_dim=embed_dim,
            vocab_size=vocab_size,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers
        )

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs
