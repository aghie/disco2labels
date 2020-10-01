import tempfile
import subprocess
import warnings
import os
import codecs
import tempfile
import copy
import sys
import time
from tree import SeqTree, RelativeLevelTreeEncoder, Token
from logging import warning


def rebuild_input_sentence(lines, include_label=False):
     
    if len(lines[0].split("\t")) > 3:
    
        sentence = [ (l.split("\t")[0],
                      l.split("\t")[1] + "##" + "|".join(feat for feat in l.split("\t")[2:-1]
                                                     if feat != "-") + "##")         
                                for l in lines]                    
    else:
        sentence = [tuple(l.split("\t")[0:2]) for l in lines]
    return sentence


"""
Returns a label as a tuple of 3 elements: (level,label,leaf_unary_chain)
"""
def split_label(label, split_char):
    if label in ["-BOS-", "-EOS-", "NONE"]:
        return (label, "-EMPTY-", "-EMPTY-")
    if len(label.split(split_char)) == 2:
        return (label.split(split_char)[0], label.split(split_char)[1], "-EMPTY-")
    
    return tuple(label.split(split_char))

"""
Transforms a list of lists into a single list
"""
def flat_list(l):
    flat_l = []
    for sublist in l:
        for item in sublist:
            flat_l.append(item)
    return flat_l

"""
Determines if a stringx can be converted into a string
"""
def is_int(x):
    try:
        int(x)
        return True
    except ValueError:
        return False

"""
Auxiliar function to compute the accuracy in an homogeneous way respect
to the enriched approach and the .seq_lu format
"""
def get_enriched_labels_for_retagger(preds, unary_preds):
 
    # TODO: Update for SPRML corpus
    warning("The model will not work well if + is not an unique joint char for collapsed branches (update for SPRML)")
    new_preds = []
    for zpreds, zunaries in zip(preds, unary_preds):
        aux = []
        for zpred, zunary in zip(zpreds, zunaries):
            if "+" in zunary and zpred not in ["-EOS-", "NONE", "-BOS-"]:
                
                if zpred == "ROOT":
                    new_zpred = "+".join(zunary.split("+")[:-1])
                else:
                    new_zpred = zpred + "_" + "+".join(zunary.split("+")[:-1])
            else:
                new_zpred = zpred
            aux.append(new_zpred)
        new_preds.append(aux)
    return new_preds


def label_contains_leaf_unary(label, split_char="@"):
    return len(label.split(split_char)) > 2 and label.split(split_char)[2] != "NONE"



def remove_duplicate_end_disco(preds, disco_encoder, split_char):
          
    none_elements = [idx for idx, p in enumerate(preds) if p.startswith("NONE")]
    code = len(none_elements)
    
    if code == 0 and len(preds) > 1:
        preds[-1] = split_char.join(["NONE","NONE","NONE",str(disco_encoder.DEFAULT)])
    elif code >= 2:
        for idx in none_elements[:-1]:
            preds[idx] = "1ROOT" + split_char + split_char.join(preds[idx].split(split_char)[1:])
        
    if len(preds) == 1 and preds[0].startswith("NONE"):
        preds[-1] = split_char.join(["1ROOT","S","NONE","0"])
    
    for ipred, pred in enumerate(preds):
        level, label, unary, offset = pred.split(split_char)
        
        if "-EOS-" in pred or "-BOS-" in pred or "[MASK_LABEL]" in pred:
    
            level = "1" if level in ["-EOS-", "-BOS-", "[MASK_LABEL]"] else level
            label = "S" if label in ["-EOS-", "-BOS-", "[MASK_LABEL]"] else label
            unary = "NONE" if unary in ["-EOS-", "-BOS-", "[MASK_LABEL]"] else unary
            offset = str(disco_encoder.DEFAULT) if offset in ["-EOS-", "-BOS-", "[MASK_LABEL]"] else offset
            # preds[ipred] = "@".join(["1","S","NONE","0"])
            preds[ipred] = split_char.join([level, label, unary, offset])
            continue

        if level == str(0) and len(preds) == 1:
            preds[ipred] = split_char.join([str(1), label, unary, offset])
            continue

        if level != "NONE" and label == "NONE" and unary == "NONE":
            preds[ipred] = split_char.join([level, "S", unary, offset])
            continue
        
        if ipred == len(preds) and label == "NONE" and unary == "NONE":
            preds[ipred] = split_char.join([level, "S", unary, offset])
            continue
        
    return preds



def remove_duplicate_end_conti(preds, split_char):
             
    none_elements = [idx for idx, p in enumerate(preds) if p.startswith("NONE")]
    code = len(none_elements)
    
    if code == 0 and len(preds) > 1:
        preds[-1] = split_char.join(["NONE","NONE","NONE"])
    elif code >= 2:
        for idx in none_elements[:-1]:
            preds[idx] = "1ROOT" + split_char + split_char.join(preds[idx].split(split_char)[1:])
            
    if len(preds) == 1 and preds[0].startswith("NONE"):
        preds[-1] = split_char.join(["1ROOT","S","NONE"])
    
    for ipred, pred in enumerate(preds):
        level, label, unary = pred.split(split_char)
        
        if "-EOS-" in pred or "-BOS-" in pred or "[MASK_LABEL]" in pred:
            preds[ipred] = split_char.join(["1", "S", "NONE"])
            continue

        if level == str(0) and len(preds) == 1:
            preds[ipred] = split_char.join([str(1), label, unary])
            continue

        if level != "NONE" and label == "NONE" and unary == "NONE":
            preds[ipred] = split_char.join([level, "S", unary])
            continue
        
        if ipred == len(preds) and label == "NONE" and unary == "NONE":
            preds[ipred] = split_char.join([level, "S", unary])
            continue
            
    return preds




"""
Transforms a list of sentences (a list of Token instances) into the corresponding parenthesized trees
@param sentences: A list of lists of Token instances
@return A list of parenthesized trees
"""
def sequence_to_parenthesis(sentences, join_char="~", split_char="@", os=True):
      

    #parenthesized_trees = []  
    relative_encoder = RelativeLevelTreeEncoder(join_char=join_char, split_char=split_char)
    
    f_max_in_common = SeqTree.maxincommon_to_tree
    f_uncollapse = relative_encoder.uncollapse
    f_empty = relative_encoder.remove_empty
    
    
    def _sequence_to_parenthesis(sentence):
         
        for token in sentence:
             
            if label_contains_leaf_unary(token.label, split_char):
                token.postag = token.label.split(split_char)[2] + join_char + token.postag

        tree = f_max_in_common(sentence, relative_encoder)
        tree = f_empty(tree) #relative_encoder.remove_empty(tree)
        tree = f_uncollapse(tree)
        tree = tree.pformat(margin=100000000)
         
        if tree.startswith("( ("):  # Ad-hoc workaround for sentences of length 1 in German SPRML
            tree = tree[2:-1]
         
        return tree
    
    if os:
        sentences = [sentence[1:-1] for sentence in sentences]
        
    return  [_sequence_to_parenthesis(sentence) for sentence in sentences]    
    




def get_treebank_postags(path_treebank):
    
    with open(path_treebank) as f:
        trees = f.readlines()
    
    postags = set([])
    for tree in trees:
        tree = SeqTree.fromstring(tree, remove_empty_top_bracketing=True)
        postags = postags.union(set([s.label() for s in tree.subtrees(lambda t: t.height() == 2)]))
    # postags = set([l.split("\t")[1] for l in lines
    #           if l != "\n"])
    
    return postags



def postprocess(sentences, disco_encoder, os=True, split_char="@"):
    
    new_sentences = []
    for sentence in sentences:
    
        if os:
            sentence = sentence[1:-1]
    
        if disco_encoder:
            labels = remove_duplicate_end_disco([token.label for token in sentence], disco_encoder, split_char)
        else:
            labels = remove_duplicate_end_conti([token.label for token in sentence], split_char)
         
        for idtoken, token in enumerate(sentence):
            token.label = labels[idtoken]               
        new_sentences.append(sentence)

    return new_sentences




def sequence_to_discbracket(sentences, disc_offset_strategy,
                            postprocess=True, split_char="@"):
    
    
    disc_sents = []
    for sentence in sentences:

        if postprocess:
            poffsets = disc_offset_strategy.postprocess([token.label.split(split_char)[-1] for token in  sentence], 
                                                        sentence)
        else:
            poffsets = [token.label.split(split_char)[-1] for token in  sentence]
            
            
        disc_sents.append(disc_offset_strategy.decode(sentence,
                                                     poffsets))
    return disc_sents

    

