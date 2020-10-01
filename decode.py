from argparse import ArgumentParser
from tree import *
from utils import sequence_to_parenthesis, sequence_to_discbracket, postprocess
from discontinuities_encodings import *
import os
import time


def prepare_sents_and_labels(instances):
    
    sents = []
    for instance in instances:
        sents.append([Token(e[0],e[1],e[-1]) for e in [element.split("\t") for element in instance]])   
    return sents              


def write_parenthesized_trees(args, parenthesized_trees):

    with open(args.output,"w") as f_out:
        for tree in parenthesized_trees:
            if args.add_root_parenthesis:
                f_out.write("( "+tree+")\n")
            else:
                f_out.write(tree+"\n")


def write_linearized_trees(args, sents):
    
    path_sequential_output = args.output+".tsv"
    with open(path_sequential_output, "w") as f_out:
        for sent in sents :
            for token in sent:           
                f_out.write("\t".join([token.word, token.postag, token.label])+"\n")
            f_out.write("\n")
                
                


if __name__ == '__main__':
    
        
    parser = ArgumentParser()
    parser.add_argument("--input", dest="input", help="Path to the input .tsv (the sequence labeling file).",default=None, required=True)
    parser.add_argument("--output", dest="output", help="Path to the output file to save the parenthesized trees.",default=None, required=True)

    parser.add_argument("--join_char",type=str,dest="join_char",
                        default="~",
                        help="Symbol used to to collapse branches.")
    parser.add_argument("--split_char",type=str,dest="split_char",
                        default="{}",
                        help="Symbol used to split the different components of the output label.")
    parser.add_argument("--split_tags", action="store_true",
                        help="To create various features from the input PoS-tag (used only in SPMRL datasets with morphologycally rich PoS-tags).")
    parser.add_argument("--split_tag_symbol", type=str, default="|")
    parser.add_argument("--disc", action="store_true", default=False, help="To specify that it is a discontinuous treebanks and that discontinuities will be encoded.")
    parser.add_argument("--disco_encoder", default=None, help="To specify the strategy to encode the discontinuities in the treebank.")
    parser.add_argument("--os", action="store_true")
    parser.add_argument("--add_leaf_unary_column",  action="store_true")
    parser.add_argument("--use_samedisc_offset", action="store_true", default=False, 
                        help="After identifying a new discontinuous span, all next words belonging to the same discontinuity are tagged with a NEXT label. Valid for the relative index, Lehmer, and inverse Lehmer strategies.")
    parser.add_argument("--path_reduced_tagset",
                        help="Path to the file containing the mapping from the original to the reduced tagset. Used together with the POSBasedLocation() strategy.")
    parser.add_argument("--avoid_postprocessing", default=False, action="store_true",
                        help="To specify whether not to post-process the offsets. If so, you must make sure the generated offsets allow to build a valid discontinuous tree. This option might be interesting to check whether we are encoding/decoding gold trees correctly.")
    parser.add_argument("--add_root_parenthesis", default=False, action="store_true",
                        help="Add root parentheses of the form '( ' and ' )' if this option is activated. Required to homogenize formats in a couple of SPMRL treebanks.")
    
    args = parser.parse_args()
    
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
    
    
    start_decode_time = time.time()
    
    with open(args.input) as f_in:
        
        content = f_in.read()        
        instances = [instance.split("\n") 
                     for instance in content.split("\n\n")[:-1]]
        
        sents = prepare_sents_and_labels(instances)

        sents = postprocess(sents, disco_encoder=disco_encoder, os=args.os,
                            split_char=args.split_char)

        if args.disc:  
            sents = sequence_to_discbracket(sents, disco_encoder,
                                               postprocess= not args.avoid_postprocessing,
                                               split_char=args.split_char)

        parenthesized_trees = sequence_to_parenthesis(sents, 
                                                      split_char=args.split_char,
                                                      os= not args.os)
        
        decode_time = time.time()-start_decode_time
        print ("raw decode time: ",decode_time)
        write_parenthesized_trees(args, parenthesized_trees)
        #write_linearized_trees(args, sents)      
        
        