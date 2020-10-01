# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BERT finetuning runner."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from curses.ascii import isalnum
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import *
from sklearn.metrics import accuracy_score
from tqdm import tqdm, trange
from shutil import copyfile

import distutils
import sys
import os
import csv
import logging
import argparse
import random
import tempfile
import subprocess
import string
import numpy as np
import torch
import time
import unicodedata

#sys.path.append(os.path.join(os.path.dirname(__file__), "tree2labels"))

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

BERT_MODEL = "bert_model"
OPENIA_GPT_MODEL = "openai_gpt_model"
GPT2_MODEL = "gpt2_model"
TRANSFORXL_MODEL = "transforxl_model"
XLNET_MODEL = "xlnet_model"
XLM_MODEL = "xlm_modeL"
DISTILBERT_MODEL = "distilbert_model"
ROBERT_MODEL = "robert_model"

MODELS = {BERT_MODEL: (BertModel, BertTokenizer, 'bert-base-uncased'),
          DISTILBERT_MODEL: (DistilBertModel, DistilBertTokenizer, 'distilbert-base-cased'),
         }


class MTLBertForTokenClassification(BertPreTrainedModel):

    def __init__(self, config, finetune, use_bilstms=False):
        
        super(MTLBertForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.num_tasks = len(self.num_labels)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob) 
        self.use_bilstms = use_bilstms
        self.lstm_size = 400
        self.lstm_layers = 2
        self.bidirectional_lstm = True
        
        if self.use_bilstms:
            self.lstm = nn.LSTM(config.hidden_size, self.lstm_size, num_layers=self.lstm_layers, batch_first=True,
                                bidirectional=self.bidirectional_lstm)
            
            self.hidden2tagList = nn.ModuleList([nn.Linear(self.lstm_size * (2 if self.bidirectional_lstm else 1),
                                                       self.num_labels[idtask])
                                                       for idtask in range(self.num_tasks)])
        else:
            
            self.hidden2tagList = nn.ModuleList([nn.Linear(config.hidden_size,
                                                       self.num_labels[idtask])
                                                       for idtask in range(self.num_tasks)])

        self.finetune = finetune
        self.init_weights()
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):
        
        hidden_outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            head_mask=head_mask)

        sequence_output = hidden_outputs[0]
        
        if not self.finetune:
            sequence_output = sequence_output.detach()        

        if self.use_bilstms:
            self.lstm.flatten_parameters()
            sequence_output, hidden = self.lstm(sequence_output, None)
        
        sequence_output = self.dropout(sequence_output)
        outputs = [(classifier(sequence_output),) for classifier in self.hidden2tagList]
        losses = []   
        
        for idtask, out in enumerate(outputs):
            
            logits = out[0]
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels[idtask])[active_loss]
                    active_labels = labels[:, idtask, :].reshape(-1)[active_loss]
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels[idtask]), labels.view(-1))
                
                losses.append(loss)  
                 
                outputs = (sum(losses),) + hidden_outputs

        return outputs


class MTLDistilBertForTokenClassification(DistilBertPreTrainedModel):

    def __init__(self, config, finetune, use_bilstms=False):
        super(MTLDistilBertForTokenClassification, self).__init__(config)
        
        self.num_labels = config.num_labels
        self.num_tasks = len(self.num_labels)
    
        self.distilbert = DistilBertModel(config)
        self.dropout = nn.Dropout(config.dropout)

        self.use_bilstms = use_bilstms
        self.lstm_size = 400
        self.lstm_layers = 2
        self.bidirectional_lstm = True
        
        if self.use_bilstms:
            self.lstm = nn.LSTM(config.hidden_size, self.lstm_size, num_layers=self.lstm_layers, batch_first=True,
                                bidirectional=self.bidirectional_lstm)
            
            self.hidden2tagList = nn.ModuleList([nn.Linear(self.lstm_size * (2 if self.bidirectional_lstm else 1),
                                                       self.num_labels[idtask])
                                                       for idtask in range(self.num_tasks)])
        else:
            
            self.hidden2tagList = nn.ModuleList([nn.Linear(config.hidden_size,
                                                       self.num_labels[idtask])
                                                       for idtask in range(self.num_tasks)])

        self.finetune = finetune
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):

        hidden_outputs = self.distilbert(input_ids,
                            attention_mask=attention_mask,
                            head_mask=head_mask)

        sequence_output = hidden_outputs[0]
        
        if not self.finetune:
            sequence_output = sequence_output.detach()        

        if self.use_bilstms:
            self.lstm.flatten_parameters()
            sequence_output, hidden = self.lstm(sequence_output, None)
        
        sequence_output = self.dropout(sequence_output)
        outputs = [(classifier(sequence_output),) for classifier in self.hidden2tagList]
        losses = []   
        
        for idtask, out in enumerate(outputs):
            
            logits = out[0]
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels[idtask])[active_loss]
                    active_labels = labels[:, idtask, :].reshape(-1)[active_loss]
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels[idtask]), labels.view(-1))
                
                losses.append(loss)  
                 
                outputs = (sum(losses),) + hidden_outputs

        return outputs


        
class InputSLExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a,
                 text_a_list,
                 text_a_postags, labels=None, num_tasks=1):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the sentence
            label: (Optional) list. The labels for each token. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = None
        self.text_a_list = text_a_list 
        self.text_a_postags = text_a_postags
        self.labels = labels
        self.num_tasks = num_tasks


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, position_ids, segment_ids, labels_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.position_ids = position_ids
        self.segment_ids = segment_ids
        self.labels_ids = labels_ids


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class SLProcessor(DataProcessor):
    """Processor for PTB formatted as sequence labeling seq_lu file"""
    
    def __init__(self, args):
        self.transformer_pretrained_model = args.transformer_pretrained_model
        self.label_split_char = args.label_split_char

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "dev")

    def get_labels(self, data_dir):
        """See base class."""
        
        train_samples = self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
        dev_samples = self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")
        
        train_labels = [sample.labels for sample in train_samples] 
        dev_labels = [sample.labels for sample in dev_samples]
        labels = []
        
        if self.label_split_char is None:
            tasks_range = 1
        else:
            tasks_range = len(train_labels[0])
        
        for idtask in range(tasks_range): 
            labels.append([])
            labels[idtask].append("[MASK_LABEL]")
            labels[idtask].append("-EOS-")
            labels[idtask].append("-BOS-")
        train_labels.extend(dev_labels)

        for label in train_labels:
            for id_label_component, sent_label_component in enumerate(label):
                for word_label in sent_label_component:
                    if word_label not in labels[id_label_component]:
                        labels[id_label_component].append(word_label)     

        return labels



    def _preprocess_disco(self, word):
    

        if word == "-LRB-": 
            word = "("
        elif word == "-RRB-": 
            word = ")"
        elif "`" in word:
            word = word.replace("`", "'")
        elif "’’" in word:
            word = word.replace("’’", "'")
        

        #Needed at least to be able to do a 'correct word - 1st-subword-piece' alignment between
        #the original input and the tokenized output obtained by bert-base-german-dbmdz-uncased
        if self.transformer_pretrained_model == "bert-base-german-dbmdz-uncased":
            if '"' in word:
                word = word.replace('"',"'")
            if "Ä" in word:
                word = word.replace("Ä", "A") 
            if "Ë" in word:
                word = word.replace("Ë", "E") 
            if "Ï" in word:
                word = word.replace("Ï", "I")
            if "Ö" in word:
                word = word.replace("Ö", "O")
            if "Ü" in word:
                word = word.replace("Ü", "U")
            if "ä" in word:
                word = word.replace("ä", "a")
            if "ë" in word:
                word = word.replace("ë", "e")
            if "ï" in word:
                word = word.replace("ï", "i")
            if "ö" in word:
                word = word.replace("ö", "o")
            if "ü" in word:
                word = word.replace("ü", "u")

            if "Â" in word:
                word = word.replace("Â","A")
            if "Ê" in word:
                word = word.replace("Ê","E")
            if "Î" in word:
                word = word.replace("Î","I")
            if "Ô" in word:
                word = word.replace("Ô","O") 
            if "Û" in word:
                word = word.replace("Û","U")

            if "â" in word:
                word = word.replace("â", "a")
            if "ê" in word:
                word = word.replace("ê", "e")
            if "î" in word:
                word = word.replace("î", "i")
            if "ô" in word:
                word = word.replace("ô", "o")
            if "û" in word:
                word = word.replace("û", "u")

            if "À" in word:
                word = word.replace("À", "a")
            if "È" in word:
                word = word.replace("È", "e")
            if "Ì" in word:
                word = word.replace("Ì", "i")
            if "Ò" in word:
                word = word.replace("Ò", "o")
            if "Ù" in word:
                word = word.replace("Ù", "u")

            if "à" in word:
                word = word.replace("à", "a")
            if "è" in word:
                word = word.replace("è", "e")
            if "ì" in word:
                word = word.replace("ì", "i")
            if "ò" in word:
                word = word.replace("ò", "o")
            if "ù" in word:
                word = word.replace("ù", "u")

            if "Á" in word:
                word = word.replace("Á", "A")
            if "É" in word:
                word = word.replace("É", "E")
            if "Í" in word:
                word = word.replace("Í", "I")
            if "Ó" in word:
                word = word.replace("Ó", "O")
            if "Ú" in word:
                word = word.replace("Ú", "U")

            if "á" in word:
                word = word.replace("á", "a")
            if "é" in word:
                word = word.replace("é", "e")
            if "í" in word:
                word = word.replace("í", "i")
            if "ó" in word:
                word = word.replace("ó", "o")
            if "ú" in word:
                word = word.replace("ú", "u")

            if "Ç" in word:
                word = word.replace("Ç","C")
            if "ç" in word:
                word = word.replace("ç","c")

        if word == "":
            raise ValueError("Generating an empty word")     
        return word

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        sentences_texts = []
        sentences_postags = []
        sentences_labels = []
        sentences_tokens = []
        sentence, sentence_postags, sentence_labels = [], [], []
        tokens = []
        
        for l in lines:
            if l != []:
                
                if l[0] in ["-EOS-", "-BOS-"]:
                    tokens.append(l[0])
                    sentence_postags.append(l[-2]) 
                else:     
                    tokens.append(l[0])
                    sentence.append(self._preprocess_disco(l[0]))
                    
                    if self.label_split_char != None:
                        values = l[-1].strip().split(self.label_split_char)
                    else:
                        values = [l[-1].strip()]

                    for idtask, value in enumerate(values):
                        try:
                            sentence_labels[idtask].append(value)                        
                        except IndexError:
                            sentence_labels.append([value])                        
                    sentence_postags.append(l[-2]) 
            else:
                
                sentences_texts.append(" ".join(sentence))
                sentences_labels.append(sentence_labels)
                sentences_postags.append(sentence_postags)
                sentences_tokens.append(tokens)
                sentence, sentence_postags, sentence_labels = [], [] , [[] for idtask in sentence_labels]
                tokens = []

        for guid, (sent, labels) in enumerate(zip(sentences_texts, sentences_labels)):
 
            examples.append(
                InputSLExample(guid=guid, text_a=sent,
                               text_a_list=sentences_tokens[guid],
                               text_a_postags=sentences_postags[guid],
                               labels=labels))
          
        return examples


def _valid_wordpiece_indexes(sent, wp_sent): 
      
    valid_idxs = []
    chars_to_process = ""
    idx = 0
      
    wp_idx = 0
    case = -1

    # print ("sent", list(enumerate(sent)))
    # print ("wp_sent", list(enumerate(wp_sent)))
    for idword, word in enumerate(sent):

        # print ("Last case", case)
        # print ("sent", list(enumerate(sent)))
        # print ("wp_sent", list(enumerate(wp_sent)))
        # print ("idword, word", idword, word)
        # print ("valid_idxs", valid_idxs)
        # print ()
        chars_to_process = word
        
        '''
        (0) The word fully matches the word piece when no index has been assigned yet, easy case.
        '''
        if word == wp_sent[wp_idx]: 
            case = 0
            valid_idxs.append(wp_idx)
            wp_idx += 1
            
        else:
            
            while chars_to_process != "":

                #print (word, type(word), len(word), wp_sent[wp_idx], type(wp_sent[wp_idx]), len(wp_sent[wp_idx]))
                if word.startswith(wp_sent[wp_idx]) and chars_to_process == word:
                    '''
                    (1) The wordpiece wp_sent[wp_idx] is the prefix of the original word, i.e. first word piece, 
                     we assign its index to the word
                    '''
                    case = 1
                    chars_to_process = chars_to_process[len(wp_sent[wp_idx]):]
                    valid_idxs.append(wp_idx)
                    wp_idx += 1
                    continue
                
                elif not wp_sent[wp_idx].startswith("##") and chars_to_process.startswith(wp_sent[wp_idx]): 
                    '''
                    (2) To control errors in BERT tokenizer at word level. For example a token
                    that is split into to actual tokens and not two or more wordpieces
                    '''
                    case = 2
                    chars_to_process = chars_to_process[len(wp_sent[wp_idx]):]
                    wp_idx += 1 
       
                    continue
            
                elif wp_sent[wp_idx].startswith("##"): 
                    
                    '''
                    (3) It is a wordpiece of the form ##[piece]. If this happens,
                    we skip word pieces until a new word is read because in this scenario
                    the original word (word) in the sentence has been assigned a wp_index already, according to (1)
                    '''
                    case = 3
                    while wp_sent[wp_idx].startswith("##"):
                        
                        chars_to_process = chars_to_process[len(wp_sent[wp_idx][2:]):]
                        wp_idx += 1
                    continue
                
                elif wp_sent[wp_idx] == "[UNK]":
                    '''  
                    (4) The word could not be tokenized and the BERT tokenizer  generated an [UNK]
                    This can be a problematic case: sometime an original token is split on two, and then each of those
                    generate two consecutive [UNK] symbols. This complicates a lot the alignment between words and word pieces
                    '''
                    
                    case = 4
                    
                    '''
                    We found an [UNK] when the current word still has not been assigned a wp_idx,
                    we consider that [UNK] index must be aligned with word
                    '''
                    if chars_to_process == word:
                        chars_to_process = ""
                        valid_idxs.append(wp_idx)
                        wp_idx += 1

                    else:
                        '''
                        We found an UNK, but the current word has been already assigned an wp_idx. However, 
                        there still missing chars to process from that word (for example if it was generated according to (1))
                        but we know this [UNK] should be a ##wordpiece. To correct this problem, we skip word pieces
                        until a word pieces matches the next word to assign an index to, to get back the alignment to valid
                        scenario.
                        '''
                        chars_to_process = ""
                        while idword + 1 < len(sent) and not sent[idword + 1].startswith(wp_sent[wp_idx]):
                            wp_idx += 1
                    
                    continue    

                elif not word.startswith(wp_sent[wp_idx]) and chars_to_process == word:
                    '''
                    Some kind of unpredictable tokenization mismatching between the input samples and BERT
                    caused a mismatch in the alignment. We try to move forward to get the alignment back to
                    a valid position, iff the word has still not received any index
                    '''    
                    case = 5 
                    wp_idx += 1      
                elif chars_to_process != word:
                    '''
                    otherwise we just move to the next word
                    ''' 
                    case = 6
                    break
                else:
                    raise RuntimeError("Potential infinite loop caused by the sentence" + 
                                       "Sentence: {}\n".format(list(enumerate(sent))) + 
                                       "Word piece sentence: {}\n".format(list(enumerate(wp_sent))) + 
                                       "Selected indexes: {}\n".format(list(enumerate(valid_idxs)))
                                       
                                       )
                    
    return valid_idxs      



def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, args):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = [{l:i for i, l in enumerate(component_label)} for j, component_label in enumerate(label_list)]
    #label_map_reverse = [{i:l for i, l in enumerate(component_label)} for j, component_label in enumerate(label_list)]
    num_tasks = len(label_map)

    features = []
    for (ex_index, example) in enumerate(examples):
        #To make it work the _valid_wordpiece_indexes() function with bert-uncased too and not just bert-cased 
        ori_tokens_a = example.text_a.split(" ") if not args.do_lower_case else example.text_a.lower().split(" ")
        #ori_tokens_a = example.text_a.split(" ")
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

#         print ("ex_index", ex_index)
#         print ("example", example)
#         print ("ori_tokens_a", ori_tokens_a, len(ori_tokens_a))
#         print ("tokens_a", tokens_a, len(tokens_a))
#         input("NEXT")
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        ori_tokens_a = ["[CLS]"] + ori_tokens_a + ["[SEP]"]
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        
        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        position_ids = list(range(max_seq_length))
        valid_indexes = _valid_wordpiece_indexes(ori_tokens_a, tokens)
        
        input_mask = [1 if idtoken in valid_indexes else 0 
                      for idtoken, _ in enumerate(tokens)]

        labels_ids = [[] for i in range(num_tasks)]
        i = 0
        for idtoken, token in enumerate(tokens):
            
            for idtask in range(num_tasks):
                if idtoken in valid_indexes:
                    
                    if token == "[CLS]":
                        labels_ids[idtask].append(label_map[idtask]["-BOS-"])
                    elif token == "[SEP]":
                        labels_ids[idtask].append(label_map[idtask]["-EOS-"])
                    else:
                        try:
                            label_mapped = label_map[idtask][example.labels[idtask][i]]
                            labels_ids[idtask].append(label_mapped)
                        except KeyError:
                            labels_ids[idtask].append(0)
                        
                        if  idtask == num_tasks - 1:
                            i += 1
                else:        
                    try:        
                        labels_ids[idtask].append(label_map[idtask][example.labels[idtask][min(i, len(example.labels[idtask]) - 1)]])
                    except KeyError:
                        labels_ids[idtask].append(0)
            
        padding = [0] * (max_seq_length - len(input_ids))
        
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        
        labels_ids = [lids + padding for lids in labels_ids]

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        
        for l in labels_ids:
            assert len(l) == max_seq_length
        
        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              position_ids=position_ids,
                              segment_ids=segment_ids,
                              labels_ids=labels_ids))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def accuracy(out, labels, mask):

    output = out * mask
    gold = labels * mask
    mask = list()
    o_flat = list(output.flatten())
    g_flat = list(gold.flatten())

    o_filtered, g_filtered = [], []
    
    for o, g in zip(o_flat, g_flat):
        if g != 0:
            g_filtered.append(g)
            o_filtered.append(o)

    assert(len(o_filtered), len(g_filtered))
    return accuracy_score(o_filtered, g_filtered)
    

def evaluate(model, device, logger, processor, tokenizer, label_list, args):    

    start_raw_time = time.time()
    if args.do_test:
        eval_examples = processor.get_test_examples(args.data_dir)
    else:
        eval_examples = processor.get_dev_examples(args.data_dir)
    
    eval_features = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer, args)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_position_ids = torch.tensor([f.position_ids for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.labels_ids for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_position_ids, all_segment_ids, all_label_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    
    label_map_reverse = {i:l for i, l in enumerate(label_list)}
    
    examples_texts = [example.text_a_list for example in eval_examples]
    examples_postags = [example.text_a_postags for example in eval_examples]
    # examples_preds = []
    examples_preds = [[] for i in range(len(label_list))]   
    model.eval()

    eval_loss, eval_accuracy = [0] * len(label_list), [0] * len(label_list)
    # eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    
    for input_ids, input_mask, position_ids, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        position_ids = position_ids.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
    
        with torch.no_grad():
            outputs = model(input_ids=input_ids,
                          #  position_ids=position_ids,
                            token_type_ids=segment_ids,
                            attention_mask=input_mask)  # , input_mask)
            
            for idtask, task_output in enumerate(outputs):
                logits_task = task_output[0]
                logits_task = logits_task.detach().cpu().numpy()
                task_label_ids = label_ids[:, idtask, :].to('cpu').numpy()
                masks = input_mask.cpu().numpy()
                outputs = np.argmax(logits_task, axis=2)
                
                for prediction, mask in zip(outputs, masks):
                    examples_preds[idtask].append([label_map_reverse[idtask][element] for element, m in zip(prediction, mask)
                                           if m != 0])

                for idx_out, (o, l) in enumerate(zip(outputs, task_label_ids)):
                    eval_accuracy[idtask] += accuracy(o, l, masks[idx_out])
    
            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1
    
    # Join the preds into one single label
    new_examples_preds = []
    for idsample in range(len(examples_texts)):
        for idtask , component in enumerate(examples_preds):

            if idtask == 0:
                new_examples_preds.append(component[idsample])
            else:
                new_examples_preds[-1] = [c + "{}" + n for c, n in zip(new_examples_preds[-1], component[idsample])]
    
    output_file_name = args.output_dir + ".tsv"
    with open(output_file_name, "w") as temp_out:
        print ("Saving the output at", output_file_name)
        content = []
        for tokens, postags, preds in zip(examples_texts, examples_postags, new_examples_preds):
         #   assert(len(tokens), len(preds))
            content.append("\n".join(["\t".join(element) for element in zip(tokens, postags, preds)]))
        temp_out.write("\n\n".join(content))
        temp_out.write("\n\n")
     
    raw_time = time.time() - start_raw_time   
    eval_accuracy = [e / nb_eval_examples for e in eval_accuracy]
    
    print ("Eval accuracy per task", eval_accuracy)
    eval_accuracy = sum(eval_accuracy) / len(eval_accuracy)
    
    result = {'eval_loss': eval_loss,
              'eval_accuracy': eval_accuracy}
    score = eval_accuracy
    out = eval_accuracy

    print ("Average eval accuracy", eval_accuracy)

    if args.parsing_paradigm == "constituency":
         
        tmp_trees_file = tempfile.NamedTemporaryFile(delete=False)
        command = ["python", "decode.py ",
        "--input", output_file_name,
        "--output", tmp_trees_file.name,
        "--disc" if args.disco_encoder is not None else "",
        "--split_char {}",
        "--os",
        "--disco_encoder " + args.disco_encoder if args.disco_encoder is not None else "",
        "" if not args.add_leaf_unary_column else "--add_leaf_unary_column",
        "--path_reduced_tagset " + args.path_reduced_tagset if args.path_reduced_tagset is not None else ""]

        p = subprocess.Popen(" ".join(command), stdout=subprocess.PIPE, shell=True)
        out_decoding, err = p.communicate()        
        out_decoding = out_decoding.decode("utf-8")
        raw_decode_time = float(out_decoding.split("\n")[0].split(":")[1])
        
        output_trees = output_file_name.replace(".tsv",".trees")
        with open(output_trees, "w") as f:
            with open(tmp_trees_file.name) as f_temp_trees:
                f.write(f_temp_trees.read())

        detailed_score = ""
        if not args.disco_encoder:

            command = [args.evalb, output_trees, args.path_gold_parenthesized]                
            if args.evalb_param is not None:
                    command.extend(["-p", args.evalb_param])
            p = subprocess.Popen(" ".join(command), stdout=subprocess.PIPE, shell=True)
            
            out, err = p.communicate()
            out = out.decode("utf-8")
                
            detailed_score += out
            score_all = float([l for l in out.split("\n")
                                           if l.startswith("Bracketing FMeasure")][0].split("=")[1])
            score_disco = -1
        
        else:

            command = ["discodop", "eval",
                       args.path_gold_parenthesized,
                       output_trees,
                       args.evalb_param,
                       "--fmt", "discbracket",
                       "--disconly"]
             
            p = subprocess.Popen(" ".join(command), stdout=subprocess.PIPE, shell=True)
            out, err = p.communicate()
            out = out.decode("utf-8")         
             
            detailed_score += """
                            ********************************
                            *******  Disco scores    *******
                            ********************************
                            """
            detailed_score += out + "\n"
             
            score_disco = float([l for l in out.split("\n")
                             if l.startswith("labeled f-measure:")][0].rsplit(" ", 1)[1])
            
            command = ["discodop", "eval",
                       args.path_gold_parenthesized,
                       output_trees,
                       args.evalb_param,
                       "--fmt", "discbracket"]

            p = subprocess.Popen(" ".join(command), stdout=subprocess.PIPE, shell=True)
            out, err = p.communicate()
            out = out.decode("utf-8")
            
            detailed_score += """
                            ********************************
                            *******  Overall score   *******
                            ********************************
                            """
            detailed_score += out            
            
            score_all = float([l for l in out.split("\n")
                             if l.startswith("labeled f-measure:")][0].rsplit(" ", 1)[1])
        
        score = (score_all, score_disco)        
        os.remove(tmp_trees_file.name)

        
    results = {"eval_loss":eval_loss,
               "eval_accuracy": eval_accuracy,
               "score": score,
               "detailed_score": detailed_score,
               "output_file_name": output_file_name,
               "raw_time": raw_time,
               "raw_decode_time": raw_decode_time}


    return results


def main():
    parser = argparse.ArgumentParser()

    # # Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    
    parser.add_argument("--transformer_model",
                        help="Specify the architecture of the transformer model: bert_model|distilbert_model")
    
    parser.add_argument("--transformer_pretrained_model",
                        help="Specify the pretrained model to be finetuned: bert-base-german-dbmdz-cased|distilbert-base-german-cased (for German) and bert-base-cased|bert-large-cased|distilbert-base-cased (for English). Check the full list of pre-trained models at: https://github.com/huggingface/transformers",
                        required=True)
    
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train. In this work: sl_tsv (sequence labeling in .tsv format)")
    
    parser.add_argument("--model_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output path where the model will be stored.")
    
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                      #  required=True,
                        help="The output path where the model predictions will be stored")

    # # Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test",
                        action="store_true",
                        help="Whether to run eval on the test set")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    
    parser.add_argument("--not_finetune", dest="not_finetune", default=False, action="store_true",
                        help="[DEPRECATED] Determine where to finetune BERT (flag to True) or just the output layer (flag set to False)")
    
    parser.add_argument("--use_bilstms",
                        default=False,
                        action="store_true",
                        help="Further contextualized BERT outputs with BILSTMs")
    
    parser.add_argument("--cache_dir", default=PYTORCH_PRETRAINED_BERT_CACHE)

    parser.add_argument("--parsing_paradigm",
                        type=str,
                        help="constituency")
    ### Constituent parsing parameters
    parser.add_argument("--evalb",
                        type=str,
                        help="Path to the script for EVALB")
    parser.add_argument("--evalb_param",
                        type=str,
                        help="Path to the script for EVALB param file",
                        default=None)
    parser.add_argument("--path_gold_parenthesized",
                        type=str,
                        help="Path to the constituency parenthesized gold file used by the EVALB script. This should be the dev set during the --do_eval phase or the test set in the --do_test phase")
    parser.add_argument("--label_split_char",
                        type=str,
                        help="Character used in labels to split their components",
                        default=None)
    
    parser.add_argument("--disco_encoder",
                        type=str,
                        help="Strategy to predict the inorder position of the tokens")
    
    parser.add_argument("--path_reduced_tagset",
                        help="Path to the file containing the mapping from the original to the reduced tagset. Used together with the DiscPrecedentProp() strategy")
    
    parser.add_argument("--add_leaf_unary_column", action="store_true", default=False,
                        help="[DEPRECATED] To add a dummy, empty leaf unary chain to every label of the sequence. Used only when the data_dir was encoded without including leaft unary chains in the labels")
    
    parser.add_argument("--log",
    help="The path to a log file where to write the evaluation results and other logging info")
    
    args = parser.parse_args()

    processors = {"sl_tsv": SLProcessor(args)}

    if args.log is not None:
        f_log = open(args.log, "w")
        f_log.write("\t".join(["Epoch", "Score"]) + "\n")
        f_log_last_output = open(args.log + ".last_output", "w")

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval and not args.do_test:
        raise ValueError("At least one of `do_train` or `do_eval` or `do_test` must be True.")

#     if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
#         raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
#     os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    processor = processors[task_name]  # ()
    
    num_labels = [len(task_labels) for task_labels in processor.get_labels(args.data_dir)]
    label_list = processor.get_labels(args.data_dir)
    label_reverse_map = [{i:label for i, label in enumerate(labels)} 
                         for labels in label_list]
    num_tasks = len(label_list)

    if args.transformer_model in MODELS:
        model_class, tokenizer_class, pretrained_model = MODELS[args.transformer_model]
    else:
        raise KeyError("The transformer model ({}) does not exist".format(args.transformer_model))
    
    tokenizer = tokenizer_class.from_pretrained(args.transformer_pretrained_model, do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir)

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:

        train_examples = processor.get_train_examples(args.data_dir)



        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    if args.transformer_model == BERT_MODEL:
    
        model = MTLBertForTokenClassification.from_pretrained(args.transformer_pretrained_model,
                                                       cache_dir=args.cache_dir,
                                                      # cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank),
                                                       num_labels=num_labels,
                                                       finetune=not args.not_finetune,
                                                       use_bilstms=args.use_bilstms)    
    elif args.transformer_model == DISTILBERT_MODEL:

        model = MTLDistilBertForTokenClassification.from_pretrained(args.transformer_pretrained_model,
                                                          cache_dir=args.cache_dir,
                                                          # cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank),
                                                           num_labels=num_labels,
                                                           finetune=not args.not_finetune,
                                                           use_bilstms=args.use_bilstms)

    else: raise NotImplementedError("The selected transformer is not available for parsing as token classification")
    
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, args)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_position_ids = torch.tensor([f.position_ids for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)            
        all_label_ids = torch.tensor([f.labels_ids for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_position_ids, all_segment_ids, all_label_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        best_dev_evalb = -sys.maxsize - 1
        best_dev_evalb_disco = -sys.maxsize - 1
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, position_ids, segment_ids, label_ids = batch
                
                outputs = model(input_ids=input_ids,
                            #   position_ids=position_ids,
                               token_type_ids=segment_ids,
                               attention_mask=input_mask,
                               labels=label_ids)
                
                loss = outputs[0]
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss = loss.mean()
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_optimization_steps, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
            
            # if args.parsing_paradigm.lower() == "constituency":
            #    # path_evaluation = "EVALB"
            #     path_gold = args.path_gold_parenthesized
            #     path_evaluation_params = args.evalb_param
            
            dev_results = evaluate(model, device, logger, processor, tokenizer, label_list,
                                                   args)
            dev_loss = dev_results["eval_loss"]
            dev_acc = dev_results["eval_accuracy"]
            dev_eval_score = dev_results["score"]
            path_output_file = dev_results["output_file_name"]
            
            print ("Current best score on the dev set: ", best_dev_evalb)
            
            f_log.write("\t".join([str(epoch), "\t".join(map(str, dev_eval_score))]) + "\n")
            copyfile(path_output_file, f_log_last_output.name)
            f_log.flush()
            if best_dev_evalb < dev_eval_score[0]:
                print ("New best score (continuous, discontinuous) on the dev set: ", dev_eval_score)
                best_dev_evalb = dev_eval_score[0]
                best_dev_evalb_disco = dev_eval_score[1]
                
                # Save a trained model
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(args.model_dir)
                
                if args.do_train:
                    print ("Saving the best new model...")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    
            model.train()  # If not, following error: cudnn RNN backward can only be called in training mode

    # Load a trained model that you have fine-tuned
    output_model_file = os.path.join(args.model_dir)
    
    if torch.cuda.is_available():
        model_state_dict = torch.load(output_model_file)
    else:
        model_state_dict = torch.load(output_model_file, map_location=torch.device('cpu'))
        
    if args.transformer_model == BERT_MODEL:
        model = MTLBertForTokenClassification.from_pretrained(args.transformer_pretrained_model,
                                                              state_dict=model_state_dict,
                                                            cache_dir=args.cache_dir,
                                                      # cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank),
                                                       num_labels=num_labels,
                                                       finetune=not args.not_finetune,
                                                       use_bilstms=args.use_bilstms
                                                       )    
    elif args.transformer_model == DISTILBERT_MODEL:
        model = MTLDistilBertForTokenClassification.from_pretrained(args.transformer_pretrained_model,
                                                            state_dict=model_state_dict,
                                                            cache_dir=args.cache_dir,
                                                          # cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank),
                                                           num_labels=num_labels,
                                                           finetune=not args.not_finetune,
                                                           use_bilstms=args.use_bilstms
                                                           )

    model.to(device)

    if (args.do_eval or args.do_test) and (args.local_rank == -1 or torch.distributed.get_rank() == 0):

        if args.parsing_paradigm.lower() == "constituency":
            path_evaluation = args.evalb
            path_evaluation_params = args.evalb_param
            path_gold = args.path_gold_parenthesized
        else:
            raise NotImplementedError("Unknown parsing paradigm")
        
        results = evaluate(model, device, logger, processor, tokenizer, label_list,
                                               args)
        
        loss = results["eval_loss"]
        acc = results["eval_accuracy"]
        eval_score = results["score"]
        detailed_score = results["detailed_score"]
        path_output_file = results["output_file_name"]
        raw_time = results["raw_time"]
        raw_decode_time = results["raw_decode_time"]
        total_time = raw_time + raw_decode_time
        print (detailed_score)
        
        if args.do_test:
            eval_examples = processor.get_test_examples(args.data_dir)
        else:
            eval_examples = processor.get_dev_examples(args.data_dir)
        
    
        pytorch_total_params = sum(p.numel() for p in model.parameters()) #if p.requires_grad)
        print ("### Output summary")
        print ("Number_parameters:", pytorch_total_params)
        print ("Number of samples:", len(eval_examples))
        print ("Decode time: ", raw_time)
        print ("Tree inference time:", raw_decode_time)
        print ("Total_time:", raw_time + raw_decode_time)
        print ("Sents/s", len(eval_examples) / total_time)
        print ("Sents/s (without tree inference):", len(eval_examples) / raw_time)


if __name__ == "__main__":
    main()
