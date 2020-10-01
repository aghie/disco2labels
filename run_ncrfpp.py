from tree import SeqTree, RelativeLevelTreeEncoder
from argparse import ArgumentParser
from nltk.tree import Tree
from utils import sequence_to_parenthesis, get_enriched_labels_for_retagger, flat_list, rebuild_input_sentence
from sklearn.metrics import accuracy_score
import codecs
import os
import tempfile
import copy
import time
import sys
import subprocess
import uuid

if __name__ == '__main__':
     
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--test", dest="test", help="Path to the input test file as sequences", default=None)
    arg_parser.add_argument("--gold", dest="gold", help="Path to the original linearized trees, without preprocessing")
    arg_parser.add_argument("--model", dest="model", help="Path to the model")
    arg_parser.add_argument("--gpu", dest="gpu", default="False")
    arg_parser.add_argument("--output", dest="output", default="/tmp/trees.txt", required=True)
    arg_parser.add_argument("--evalb", dest="evalb", help="Path to the script EVALB")
    arg_parser.add_argument("--evalb_param", dest="evalb_param", help="Path to the EVALB param file", default=None)
    arg_parser.add_argument("--ncrfpp", dest="ncrfpp", help="Path to the NCRFpp repository")
    arg_parser.add_argument("--disc", action="store_true", default=False)
    arg_parser.add_argument("--disco_encoder",
                            help="Strategy to encode discontinuities [abs-idx|rel-idx|lehmer|lehmer-inorder|pos-pointer|pos-pointer-reduced]",
                            default=None)
    arg_parser.add_argument("--os", action="store_true")
    arg_parser.add_argument("--add_leaf_unary_column", action="store_true", default=False)
    arg_parser.add_argument("--path_reduced_tagset")
    arg_parser.add_argument("--label_split_char", default="{}")
    
    args = arg_parser.parse_args()
    
    # If not, it gives problem with Chinese chracters
 #   reload(sys)
 #   sys.setdefaultencoding('UTF8')
    
    gold_trees = codecs.open(args.gold).readlines()
    
    # Check if we need to add an a pair of ROOT brackets (needed for SPRML)?
    add_root_brackets = False
    if gold_trees[0].startswith("( ("):
        add_root_brackets = True
    
    gold_trees = codecs.open(args.gold).readlines()
    path_raw_dir = args.test
    path_name = args.model
    path_output = args.output+".tsv"
    path_ncrfpp_log = "/tmp/" + path_name.split("/")[-1] + ".tagger.log"
    path_dset = path_name + ".dset"
    path_model = path_name + ".model"
    
    # Reading stuff for evaluation
    sentences = []
    gold_labels = []
    for s in codecs.open(path_raw_dir).read().split("\n\n"):
        sentence = []
        for element in s.split("\n"):
            if element == "": break
            word, postag, label = element.strip().split("\t")[0], "\t".join(element.strip().split("\t")[1:-1]), element.strip().split("\t")[-1]
            sentence.append((word, postag))
            gold_labels.append(label)
        if sentence != []: sentences.append(sentence)

    conf_str = """
    ### Decode ###
    status=decode
    """
    conf_str += "raw_dir=" + path_raw_dir + "\n"
    conf_str += "decode_dir=" + path_output + "\n"
    conf_str += "dset_dir=" + path_dset + "\n"
    conf_str += "load_model_dir=" + path_model + "\n"
    conf_str += "gpu=" + args.gpu + "\n"

    decode_fid = str(uuid.uuid4())
    file_conf_ncrfpp = codecs.open("/tmp/" + decode_fid, "w")
    file_conf_ncrfpp.write(conf_str)

    time_init = time.time()

    os.system("python " + args.ncrfpp + "/main.py --config " + file_conf_ncrfpp.name + " > " + path_ncrfpp_log)
    
    log_lines = codecs.open(path_ncrfpp_log).readlines()
    raw_time = float([l for l in log_lines
                    if l.startswith("raw: time:")][0].split(",")[0].replace("raw: time:", "").replace("s", ""))

    number_parameters = int([l for l in log_lines
                    if l.startswith("Number_parameters:")][0].split(":")[1])
    
    file_output_trees = open(args.output+".trees", "w") 
    command = ["python", "decode.py ",
               "--input", path_output,
               "--output", file_output_trees.name,
               "--disc" if args.disco_encoder is not None else "",
               "--split_char", args.label_split_char,
               "--os" if args.os else "",
               "--disco_encoder " + args.disco_encoder if args.disco_encoder is not None else "",
               "" if not args.add_leaf_unary_column else "--add_leaf_unary_column",
               "--path_reduced_tagset " + args.path_reduced_tagset if args.path_reduced_tagset is not None else ""]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
    p = subprocess.Popen(" ".join(command), stdout=subprocess.PIPE, shell=True)
    out_decoding, err = p.communicate()
    out_decoding = out_decoding.decode("utf-8")
    raw_decode_time = float(out_decoding.split("\n")[0].split(":")[1])
    
    total_without_tree_inference_time = raw_time 
    total_time = raw_time + raw_decode_time 
    
    command = ["discodop", "eval",
               args.gold,
               file_output_trees.name,
               args.evalb_param,
               "--fmt", "discbracket"]
    
    p = subprocess.Popen(" ".join(command), stdout=subprocess.PIPE, shell=True)
    out, err = p.communicate()

    out = out.decode("utf-8")

    print ("*****************************")
    print ("******* Overall scores ******")
    print ("*****************************")
    print (out)
    
    command = ["discodop", "eval",
               args.gold,
               file_output_trees.name,
               args.evalb_param,
               "--fmt", "discbracket",
               "--disconly"]

    p = subprocess.Popen(" ".join(command), stdout=subprocess.PIPE, shell=True)
    out, err = p.communicate()
    out = out.decode("utf-8")
    
    print ("*****************************")
    print ("* Only discontinuous scores *")
    print ("*****************************")
    print (out)
    
    print ("Number_parameters:",number_parameters)
    print ("Total time:", round(total_time, 2)) 
    print ("Total time tree inference", raw_decode_time)
    print ("Sents/s:   ", round(len(gold_trees) / (total_time), 2))
    print ("Sents/s (without tree inference): ", round(len(gold_trees) / (total_without_tree_inference_time), 2)) 

    os.remove(file_conf_ncrfpp.name)
    os.remove(path_ncrfpp_log)
