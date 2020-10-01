'''
It receives as input the splits of a constituent treebank (each of them in one file, 
and each sample of the file represented in a one-line format) and transforms it into
a sequence of labels, one per word, in a TSV format.
'''

from argparse import ArgumentParser
from tree import *
from discontinuities_encodings import *
from nltk.corpus import treebank

import time
import nltk
import argparse
import os
import codecs
import utils
import random
import filecmp
import subprocess
import tempfile

BOS = "-BOS-"
EOS = "-EOS-"
EMPTY = "-EMPTY-"


#LANGUAGES_WITH_EXTRA_ROOT_PARENTHESIS = ["french_spmrl", "swedish_spmrl"]

"""
Intended for morphologically rich corpora (e.g. SPRML)
"""
def get_feats_dict(sequences, feats_dict):
    idx = len(feats_dict)+1
    for sequence in sequences:
        for token in sequence:
            #The first element seems to be a summary of the coarse-grained tag plus the lemma
            if "##" in token.postag:
                for feat in token.postag.split("##")[1].split("|"):
                    feat_split = feat.split("=")
                    if not feat_split[0] in feats_dict:
                        feats_dict[feat_split[0]] = idx
                        idx+=1
                        
                

"""
Intended for morphologically rich corpora (e.g. SPRML)
"""
def feats_from_tag(tag, tag_split_symbol, feats_dict):
    
    feats = ["-"]*(len(feats_dict)+1)
    feats[0] = tag.split("##")[0]
    if "##" in tag:
        for feat in tag.split("##")[1].split(tag_split_symbol):     
            feat_split = feat.split("=")
            feats[feats_dict[feat_split[0]]] = feat
    return feats


"""
Transforms a constituent treebank (parenthesized trees, one line format) into a sequence labeling format
"""
def transform_split(path, binarized,dummy_tokens, root_label,abs_top,
                    abs_neg_gap, join_char,split_char, discontinous,
                    disco_encoder):
           
    with codecs.open(path,encoding="utf-8") as f:
        trees = f.readlines()
    
    sequences = []
    for tree in trees:
  
        tree = SeqTree.fromstring(tree, remove_empty_top_bracketing=True)  
        tree.set_encoding(RelativeLevelTreeEncoder(join_char=join_char, split_char=split_char))
        tree.set_discontinous(discontinous, disco_encoder)

        gold_tokens = tree.to_maxincommon_sequence(is_binary=binarized,
                                       root_label=root_label,
                                       encode_unary_leaf=True,
                                       abs_top=abs_top,
                                       abs_neg_gap=abs_neg_gap)
        
        if dummy_tokens:
            gold_tokens.insert(0, Token(BOS, BOS, split_char.join([BOS]*len(gold_tokens[0].label.split(split_char)) ) ) )
            gold_tokens.append( Token(EOS, EOS, split_char.join([EOS]*len(gold_tokens[0].label.split(split_char)) ) ) )
            
        sequences.append(gold_tokens)
         
    
    return sequences


def _set_tag_for_leaf_unary_chain(leaf_unary_chain, join_char="+"):

    if join_char in leaf_unary_chain:
        return join_char.join(leaf_unary_chain.split(join_char)[:-1]) #[:-1] not to take the PoS tag
    else:
        return EMPTY    



def write_linearized_trees(path_dest, sequences, feats_dict):
    
    with codecs.open(path_dest,"w",encoding="utf-8") as f:
        for sentence in sequences:
             
            for token in sentence:

                if (feats_dict == {}):
                    f.write("\t".join([token.word,token.postag,token.label])+"\n")
                else:
                    feats = "\t".join(feats_from_tag(token.postag, "|", feats_dict))
                    f.write("\t".join([token.word,feats,token.label])+"\n")
            f.write("\n")    



if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument("--train", dest="train", help="Path to the parenthesized training file",default=None, required=True)
    parser.add_argument("--dev", dest="dev", help="Path to the parenthesized development file",default=None, required=True)
    parser.add_argument("--test", dest="test", help="Path to the parenthesized test file",default=None, required=True)
    parser.add_argument("--output", dest="output", 
                        help="Path to the output directory to store the dataset", default=None, required=True)
    parser.add_argument("--output_no_unaries", dest="output_no_unaries", 
                        help="Path to the output directory to store the dataset without encoding leaf unary chains", default=None)
    parser.add_argument("--os", action="store_true",help="Activate this option to add both a dummy beggining- (-BOS-) and an end-of-sentence (-EOS-) token to every sentence")
    parser.add_argument("--root_label", action="store_true", help="Activate this option to add a simplified root label to the nodes that are directly mapped to the root of the constituent tree",
                        default=False)
    parser.add_argument("--abs_top", dest="abs_top", type=int,
                        help="To indicate which levels should be stored in absolute scale. This feature also depends on the value assigned to --abs_neg_gap",
                        default=None)
    parser.add_argument("--abs_neg_gap", type=int, 
                        dest="abs_neg_gap",
                        default=None,
                        help="To indicate that only relative levels that encode a gap < -abs_neg_gap should be codified in an absolute scale")
    parser.add_argument("--join_char",type=str,dest="join_char",
                        default="~",
                        help="Symbol used to to collapse branches")
    parser.add_argument("--split_char",type=str,dest="split_char",
                        default={},
                        help="Symbol used to split the different components of the output label")
    parser.add_argument("--split_tags", action="store_true",
                        help="To create various features from the input PoS-tag (used only in SPMRL datasets with morphologycally rich PoS-tags)")
    parser.add_argument("--split_tag_symbol", type=str, default="|")
    parser.add_argument("--disc", action="store_true", default=False, help="To specify that it is a discontinuous treebanks and that discontinuities will be encoded")
    parser.add_argument("--disco_encoder", default=None, help="To specify the strategy to encode the discontinuities in the treebank. Options = abs-idx|rel-idx|lehmer|lehmer-inverse|pos-pointer|pos-pointer-reduced")
    parser.add_argument("--use_samedisc_offset", action="store_true", default=False, 
                        help="After identifying a new discontinuous span, all next words belonging to the same discontinuity are tagged with a NEXT label. Valid for the relative index, Lehmer, and inverse Lehmer strategies")
    parser.add_argument("--path_reduced_tagset",
                        help="Path to the file containing the mapping from the original to the reduced tagset. Used together with the POSPointerDiscoLocation() strategy to encode discontinuities")
    parser.add_argument("--add_root_parenthesis", default=False, action="store_true",
                        help="Add root parentheses of the form '( ' and ' )' if this option is activated. Use together with --check_decode. Required to homogenize formats in a couple of SPMRL treebanks")
    parser.add_argument("--check_decode", default=False, action="store_true",
                        help="To decode back the files and check that whether they are equivalent to the input ones")
#    parser.add_argument("--avoid_postprocessing", default=False, action="store_true",
#                        help="To specify whether not to post-process the encoded files. If so, you must make sure the generated offsets allow to build a valid discontinuous tree. This option might be interesting to check whether we are encoding/decoding gold trees correctly.")

    parser.add_argument("--binarized", action="store_true", help="[DEPRECATED] Activate this options if you first want to binarize the constituent trees", default=False)
    args = parser.parse_args()
    
    transform = transform_split
    if args.disc:
        if args.disco_encoder.lower() == "lehmer":
            disco_encoder = LehmerDiscoLocation(args.use_samedisc_offset)
        elif args.disco_encoder.lower() == "lehmer-inverse":
            disco_encoder = LehmerInverseDiscoLocation(args.use_samedisc_offset)
        elif args.disco_encoder.lower() == "abs-idx":
            disco_encoder = AbsIndexDiscoLocation()
        elif args.disco_encoder.lower() == "rel-idx":
            disco_encoder = RelIndexDiscoLocation(args.use_samedisc_offset)
        elif args.disco_encoder.lower() == "pos-pointer":
            disco_encoder = POSPointerDiscoLocation()
        elif args.disco_encoder.lower() == "pos-pointer-reduced":
            disco_encoder = POSPointerDiscoLocation()
            disco_encoder.build_reduced_tagset(args.path_reduced_tagset)
            
        else:
            raise NotImplementedError("The strategy %s to compute the offset for disctrees is not implemented", args.disco_encoder)
         
    else:
        disco_encoder=None
        
    train_sequences= transform(args.train, args.binarized, args.os, args.root_label, 
                                                         args.abs_top, args.abs_neg_gap,
                                                         args.join_char, args.split_char,
                                                         args.disc, disco_encoder) 
    
    dev_sequences = transform(args.dev, args.binarized, args.os, args.root_label, 
                                                     args.abs_top, args.abs_neg_gap,
                                                     args.join_char, args.split_char,
                                                     args.disc, disco_encoder) 
    
    feats_dict = {}
    if args.split_tags:
        get_feats_dict(train_sequences, feats_dict)
        get_feats_dict(dev_sequences, feats_dict)

              
    test_sequences = transform(args.test, args.binarized, args.os, args.root_label, 
                               args.abs_top, args.abs_neg_gap,
                               args.join_char, args.split_char,
                               args.disc, disco_encoder) 

    print ("Encoding the training set... ")
    write_linearized_trees("/".join([args.output, "train.tsv"]), train_sequences,
                           feats_dict)
    
    print ("Encoding the dev set...")
    write_linearized_trees("/".join([args.output, "dev.tsv"]), dev_sequences, 
                           feats_dict)
    
    print ("Encoding the test set...")
    write_linearized_trees("/".join([args.output, "test.tsv"]), test_sequences, 
                           feats_dict)
    print ()
    
    if args.check_decode:
    
        for input_file, split in zip([args.train, args.dev, args.test],["train", "dev", "test"]):
            #Checking that the encoding is complete (we decode back and verify that the original files are the same)   
            f_tmp = tempfile.NamedTemporaryFile(delete=False)
            path_input_to_decode =  "/".join([args.output, split+".tsv"])               
                    
            command = ["python", "decode.py", "--input",path_input_to_decode,
                       "--output", f_tmp.name, 
                       "--disc" if args.disc else "", 
                       "--disco_encoder "+ args.disco_encoder if args.disco_encoder is not None else "",
                       "--split_char", args.split_char,
                       "--os" if args.os else "",
                       "--use_samedisc_offset" if args.use_samedisc_offset else "",
                       "--path_reduced_tagset "+args.path_reduced_tagset if args.path_reduced_tagset is not None else "",
                       "--avoid_postprocessing", #if args.avoid_postprocessing else "",
                       "--add_root_parenthesis" if args.add_root_parenthesis else ""] #Needed for some SPMRL datasets, not relevant for discontinuous
            
            p = subprocess.Popen(" ".join(command),stdout=subprocess.PIPE, shell=True)
            out, err = p.communicate(command)
            out = out.decode("utf-8")
            
            if not filecmp.cmp(input_file, f_tmp.name):
                print ("File {} encoded in {} {}".format(input_file,"/".join([args.output,split+".tsv"]),"[FAILURE]"))
            else:
                print ("File {} encoded in {} {}".format(input_file,"/".join([args.output,split+".tsv"]),"[OK]"))

            os.remove(f_tmp.name)

    #To create a version of the data where the unaries are not included as part of the labels to predict
    if args.output_no_unaries is not None:
    
        print ("\nGenerating a SL version of the treebank without encoding leaf unary branches...")
        for split in ["train","dev","test"]:
            with codecs.open("/".join([args.output_no_unaries, split+".tsv"]),"w") as f_out:

                with codecs.open("/".join([args.output,split+".tsv"])) as f:
                    sentences = f.read().split("\n\n")
                    for idsent, sentence in enumerate(sentences):
                        for l in sentence.split("\n"):
                            if l != "":
                                lsplit = l.split("\t")
                                label = lsplit[-1].split(args.split_char)
                                label = args.split_char.join([component for i,component in enumerate(label)
                                                                if i !=2])
                                f_out.write("\t".join([lsplit[0],lsplit[1],label])+"\n")
                        if idsent < len(sentences)-1: 
                            f_out.write("\n")
                                
                   