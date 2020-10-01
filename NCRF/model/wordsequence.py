# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2017-10-17 16:47:32
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-02-01 15:59:26
from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .wordrep import WordRep
from .transformer_layers import TransformerModel, MyBertEncoder, MyBertEmbeddings

import transformers
#from torchnlp.modules import transformer as torchnlp_transformer

class Encoder(nn.Module):
    """
    A Transformer Encoder module. 
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """
    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=100, input_dropout=0.0, layer_dropout=0.0, 
                 attention_dropout=0.0, relu_dropout=0.0, use_mask=False):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
            use_mask: Set to True to turn on future value masking
        """
        
        super(Encoder, self).__init__()
        
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)
        
        params =(hidden_size, 
                 total_key_depth or hidden_size,
                 total_value_depth or hidden_size,
                 filter_size, 
                 num_heads, 
                 _gen_bias_mask(max_length) if use_mask else None,
                 layer_dropout, 
                 attention_dropout, 
                 relu_dropout)
        
        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.enc = nn.Sequential(*[EncoderLayer(*params) for l in range(num_layers)])
        
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)
        
    
    def forward(self, inputs):
        #Add input dropout
        x = self.input_dropout(inputs)
        
        # Project to hidden size
        x = self.embedding_proj(x)
        
        # Add timing signal
        x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
        
        y = self.enc(x)
        
        y = self.layer_norm(y)
        return y

class WordSequence(nn.Module):
    def __init__(self, data):
        super(WordSequence, self).__init__()
        print("build word sequence feature extractor: %s..." % (data.word_feature_extractor))
        self.tasks = data.HP_tasks
        self.data = data
        self.gpu = data.HP_gpu
        self.use_char = data.use_char
        self.droplstm = nn.Dropout(data.HP_dropout)
        self.bilstm_flag = data.HP_bilstm
        self.lstm_layer = data.HP_lstm_layer
        self.wordrep = WordRep(data)
        self.input_size = data.word_emb_dim
        self.status = data.status
        self.index_of_main_tasks = data.index_of_main_tasks
        self.feature_num = data.feature_num
        if self.use_char:
            self.input_size += data.HP_char_hidden_dim
            if data.char_feature_extractor == "ALL":
                self.input_size += data.HP_char_hidden_dim
        for idx in range(self.feature_num):
            self.input_size += data.feature_emb_dims[idx]
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        if self.bilstm_flag:
            lstm_hidden = data.HP_hidden_dim // 2
        else:
            lstm_hidden = data.HP_hidden_dim

        self.word_feature_extractor = data.word_feature_extractor
        if self.word_feature_extractor == "GRU":
            self.lstm = nn.GRU(self.input_size, lstm_hidden, num_layers=self.lstm_layer, batch_first=True,
                               bidirectional=self.bilstm_flag)
        elif self.word_feature_extractor == "LSTM":
            self.lstm = nn.LSTM(self.input_size, lstm_hidden, num_layers=self.lstm_layer, batch_first=True,
                                bidirectional=self.bilstm_flag)
        elif self.word_feature_extractor == "CNN":
            # cnn_hidden = data.HP_hidden_dim
            self.word2cnn = nn.Linear(self.input_size, data.HP_hidden_dim)
            self.cnn_layer = data.HP_cnn_layer
            print("CNN layer: ", self.cnn_layer)
            self.cnn_list = nn.ModuleList()
            self.cnn_drop_list = nn.ModuleList()
            self.cnn_batchnorm_list = nn.ModuleList()
            kernel = 3
            pad_size = int((kernel - 1) / 2)
            for idx in range(self.cnn_layer):
                self.cnn_list.append(
                    nn.Conv1d(data.HP_hidden_dim, data.HP_hidden_dim, kernel_size=kernel, padding=pad_size))
                self.cnn_drop_list.append(nn.Dropout(data.HP_dropout))
                self.cnn_batchnorm_list.append(nn.BatchNorm1d(data.HP_hidden_dim))
        
        elif self.word_feature_extractor.upper() == "TRANSFORMER":

            bert_config = transformers.modeling_bert.BertConfig()
            bert_config.hidden_size = self.input_size
            
            bert_config.num_attention_heads = data.HP_transformer_num_attention_heads
            bert_config.num_hidden_layers = data.HP_transformer_num_hidden_layers
            bert_config.intermediate_size = data.HP_transformer_intermediate_size
            bert_config.hidden_dropout_prob = data.HP_transformer_hidden_dropout_prob
            bert_config.max_position_embeddings = data.MAX_SENTENCE_LENGTH
            #bert_config
            print ("bert_config", bert_config)
    
            self.transformer_embeddings = MyBertEmbeddings(bert_config, data)

            self.transformer_encoder = MyBertEncoder(bert_config, data)
            
            data.HP_hidden_dim = bert_config.hidden_size #self.input_size
        
            #torchnlp version
#             self.transformer_encoder = torchnlp_transformer.Encoder(
#                                     self.input_size,
#                                     nhid,
#                                     nlayers,
#                                     nhead,
#                                     attention_key_channels=0,
#                                     attention_value_channels=0,
#                                     filter_size,
#                                     max_length,
#                                     input_dropout,
#                                     0.2,
#                                     attention_dropout,
#                                     relu_dropout,
#                                     use_mask=False
#                                 )
        
        
        
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tagList = nn.ModuleList([nn.Linear(data.HP_hidden_dim,
                                                       data.label_alphabet_sizes[idtask])
                                             for idtask in range(data.HP_tasks)])

        if self.gpu:
            self.droplstm = self.droplstm.cuda()

            self.hidden2tagList.cuda()
            if self.word_feature_extractor == "CNN":
                self.word2cnn = self.word2cnn.cuda()
                for idx in range(self.cnn_layer):
                    self.cnn_list[idx] = self.cnn_list[idx].cuda()
                    self.cnn_drop_list[idx] = self.cnn_drop_list[idx].cuda()
                    self.cnn_batchnorm_list[idx] = self.cnn_batchnorm_list[idx].cuda()
            elif self.word_feature_extractor == "LSTM":
                self.lstm = self.lstm.cuda()
            elif self.word_feature_extractor.upper() == "TRANSFORMER":
                self.transformer_encoder = self.transformer_encoder.cuda()
                #self.t_decoder = self.t_decoder.cuda()
                

    def forward(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, 
                mask, #NEW testing
                inference):
        """
            input:
                word_inputs: (batch_size, sent_len)
                feature_inputs: [(batch_size, sent_len), ...] list of variables
                word_seq_lengths: list of batch_size, (batch_size,1)
                char_inputs: (batch_size*sent_len, word_length)
                char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
                char_seq_recover: variable which records the char order information, used to recover char order
            output:
                Variable(batch_size, sent_len, hidden_dim)
        """

        word_represent = self.wordrep(
            word_inputs,
            feature_inputs,
            word_seq_lengths,
            char_inputs,
            char_seq_lengths,
            char_seq_recover)

        if self.word_feature_extractor == "CNN":
            word_in = F.tanh(
                self.word2cnn(word_represent)).transpose(
                2, 1).contiguous()
            for idx in range(self.cnn_layer):
                if idx == 0:
                    cnn_feature = F.relu(self.cnn_list[idx](word_in))
                else:
                    cnn_feature = F.relu(self.cnn_list[idx](cnn_feature))
                cnn_feature = self.cnn_drop_list[idx](cnn_feature)
                cnn_feature = self.cnn_batchnorm_list[idx](cnn_feature)
            feature_out = cnn_feature.transpose(2, 1).contiguous()
 
        elif self.word_feature_extractor == "LSTM":

            packed_words = pack_padded_sequence(
                word_represent, word_seq_lengths.cpu().numpy(), True)
            hidden = None

            lstm_out, (hidden_x, cell_x) = self.lstm(packed_words, hidden)
            lstm_out, _ = pad_packed_sequence(lstm_out)
            feature_out = self.droplstm(lstm_out.transpose(1, 0))
            
        else:       
            word_represent = self.transformer_embeddings(word_represent)
            feature_out = self.transformer_encoder(word_represent)[0]
            

        if self.status == "train":
            
            outputs = [self.hidden2tagList[idtask](feature_out) for idtask in range(
                self.tasks) if not inference or idtask < self.main_tasks]
            
        else:

            self.main_tasks = self.index_of_main_tasks
            outputs = [self.hidden2tagList[idtask](feature_out) for idtask in range(
                self.tasks) if not inference or idtask in self.main_tasks]

        return outputs

    def sentence_representation(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
                                char_seq_recover):
        """
            input:
                word_inputs: (batch_size, sent_len)
                feature_inputs: [(batch_size, ), ...] list of variables
                word_seq_lengths: list of batch_size, (batch_size,1)
                char_inputs: (batch_size*sent_len, word_length)
                char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
                char_seq_recover: variable which records the char order information, used to recover char order
            output:
                Variable(batch_size, sent_len, hidden_dim)
        """

        word_represent = self.wordrep(word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
                                      char_seq_recover)
        ## word_embs (batch_size, seq_len, embed_size)
        batch_size = word_inputs.size(0)
        if self.word_feature_extractor == "CNN":
            word_in = torch.tanh(self.word2cnn(word_represent)).transpose(2, 1).contiguous()
            for idx in range(self.cnn_layer):
                if idx == 0:
                    cnn_feature = F.relu(self.cnn_list[idx](word_in))
                else:
                    cnn_feature = F.relu(self.cnn_list[idx](cnn_feature))
                cnn_feature = self.cnn_drop_list[idx](cnn_feature)
                if batch_size > 1:
                    cnn_feature = self.cnn_batchnorm_list[idx](cnn_feature)
            feature_out = F.max_pool1d(cnn_feature, cnn_feature.size(2)).view(batch_size, -1)
        else:
            packed_words = pack_padded_sequence(word_represent, word_seq_lengths.cpu().numpy(), True)
            hidden = None
            lstm_out, hidden = self.lstm(packed_words, hidden)
            ## lstm_out (seq_len, seq_len, hidden_size)
            ## feature_out (batch_size, hidden_size)
            feature_out = hidden[0].transpose(1, 0).contiguous().view(batch_size, -1)

        feature_list = [feature_out]
        for idx in range(self.feature_num):
            feature_list.append(self.feature_embeddings[idx](feature_inputs[idx]))
        final_feature = torch.cat(feature_list, 1)
        outputs = self.hidden2tag(self.droplstm(final_feature))
        ## outputs: (batch_size, label_alphabet_size)
        return outputs
