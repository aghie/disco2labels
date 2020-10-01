from argparse import ArgumentParser
import os 
import nltk
from nltk.tree import Tree



def change_postags(tree, d_tags={}):

    new_children = []
    for idchild, child in enumerate(tree):
        
        if type(child) == type(u'') or type(child) == type(""):
        
           idx_word = int(child.split("=")[0])
           word =  child.split("=")[1]
           tree.set_label(d_tags[idx_word])
        else:                

           change_postags(child, d_tags)
     
    return tree
    
            
            
def read_pred_tags(path_pred_tags):
    
    sentences_tags = []
    with open(path_pred_tags) as f:
        
        sentences = f.read().split("\n\n")
        
        for sentence in sentences:
            d = {}
            if sentence != '':
                for idx, l in enumerate(sentence.split("\n")[1:-1]):
                    ls = l.split("\t")
                    token, postag = ls[0], ls[1]
                    d[idx] = postag
    
                sentences_tags.append(d)

    
    return sentences_tags
    
if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument("--input_disbracket", help="Gold file in disbracket format ",
                        default=None, required=True)
    parser.add_argument("--input_pred_tags", help="Input file in tsv format with the predicted postags",
                        default=None)
    parser.add_argument("--out_disbracket", help="Gold file in disbracket format ",
                        default=None)

        
    args = parser.parse_args()
    
    sentences_tags = read_pred_tags(args.input_pred_tags)
    
    with open(args.input_disbracket) as f:
        
        with open(args.out_disbracket, "w") as f_out:
            
        
            trees = f.readlines()
            for idtree,tree in enumerate(trees):
               # print (tree)
                tree = Tree.fromstring(tree, remove_empty_top_bracketing=True)
                tags_proof = [(idx, s.label()) for idx,s in enumerate(tree.subtrees(lambda t: t.height() == 2))]
                
                tree = change_postags(tree,sentences_tags[idtree])
                tags_proof2 = [(idx, s.label()) for idx,s in enumerate(tree.subtrees(lambda t: t.height() == 2))]

                f_out.write(tree.pformat(1000000)+"\n")


        