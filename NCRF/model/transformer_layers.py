import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_bert import *


class MyBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config, data):
        
        self.gpu = data.HP_gpu
        super(MyBertEmbeddings, self).__init__()
        #self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        #self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
#         print (cuda.is_available())
#         print (config.gpu)
#         exit()

        if self.gpu:
        
            self.position_embeddings = self.position_embeddings.cuda()
            self.LayerNorm = self.LayerNorm.cuda()
            self.dropout = self.dropout.cuda()


    def forward(self, word_embeddings, token_type_ids=None, position_ids=None):
        #seq_length = input_ids.size(1)
        seq_length = word_embeddings.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=word_embeddings.device)
            
           # print ("word_embeddings")
           # print (word_embeddings, word_embeddings.shape, "size", word_embeddings.shape[0], word_embeddings.shape[1])
           # print ()
           # print ("position_ids", position_ids, position_ids.shape)
           # print ()
            #print ("position_ids.unsqueeze(0)", position_ids.unsqueeze(0), position_ids.unsqueeze(0).shape)
            #print ()
           # print ("position_ids.unsqueeze(0).expand(input_shape)", position_ids.unsqueeze(0).expand((8,32)), 
           #        position_ids.unsqueeze(0).expand((8,32)).shape)
            
            position_ids = position_ids.unsqueeze(0).expand((word_embeddings.shape[0], word_embeddings.shape[1]))
            
           # print ("Llega aqui")
            #print ("position_ids expanded", position_ids)
            #exit()
            #position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
       # if token_type_ids is None:
       #     token_type_ids = torch.zeros_like(input_ids)

        if self.gpu:
            position_ids = position_ids.cuda()
       # print ("Position ids", position_ids, position_ids.shape)
        position_embeddings = self.position_embeddings(position_ids)
       # print ("Position embeddings", position_embeddings, position_embeddings.shape)
       
       
        embeddings = word_embeddings + position_embeddings #+ token_type_embeddings
        #print (embeddings, embeddings.shape)
        #print ()
       # input("NEXT")
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class MyBertEncoder(nn.Module):
    def __init__(self, config, data):
        super(MyBertEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
    #    print ("ENTRA forward MyBERTEncoder")
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if head_mask is None:
              #  print ("hidden_states", hidden_states, hidden_states.shape)
              #  print ("attention_mask", attention_mask)
              #  input("NEXT")
                layer_outputs = layer_module(hidden_states, attention_mask, None)
            else:
                layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i])
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)

        return outputs  # last-layer hidden state, (all hidden states), (all attentions)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):

    def __init__(self, ninp, nhead, nhid, nlayers, dropout=0.5):
    #def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        #self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        #self.decoder = nn.Linear(ninp, 800)

#        self.init_weights()

#     def _generate_square_subsequent_mask(self, sz):
#         mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
#         mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#         return mask

#     def init_weights(self):
#         initrange = 0.1
#         #self.encoder.weight.data.uniform_(-initrange, initrange)
#         #self.decoder.bias.data.zero_()
#         self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, mask=None):
       # if self.src_mask is None or self.src_mask.size(0) != len(src):
       #     device = src.device
          #  mask = self._generate_square_subsequent_mask(len(src)).to(device)
          #  self.src_mask = mask
        
        #src = src * math.sqrt(self.ninp) #This does not apply, because we are allowed here to look at next words
      #  src = self.pos_encoder(src)
        
        
       # print ("src after pos encoder", src, src.shape)
       # print ()
        
    
        
        output = self.transformer_encoder(src)
        
       # exit()
        #print (output.shape)
        #exit()
        #output = self.decoder(output)
        
        #print (output)
        #exit()
        
        return output