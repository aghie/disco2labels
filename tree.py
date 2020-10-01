from nltk.tree import Tree
import copy
import itertools
import warnings
from numpy import insert
from collections import Counter
from numpy.core.defchararray import index
from setuptools.unicode_utils import try_encode



class Token(object):
    
    def __init__(self, word, postag, label):
        self.word = word
        self.postag = postag
        self.label = label



"""
Class to manage the transformation of a constituent tree into a sequence of labels
and vice versa. It extends the Tree class from the NLTK framework to address constituent Parsing as a 
sequential labeling problem.
"""
class SeqTree(Tree):
    
    EMPTY_LABEL = "EMPTY-LABEL"
    
    def __init__(self,label,children, discontinous=False):
         
        self.encoding = None
        self.discontinous = discontinous
        
        # if self.discontinous:
        #     self.disc_strategy = DiscOffsetAbsStrategy()
        
        super(SeqTree, self).__init__(label,children) 
    
    #TODO: At the moment only the RelativeLevelTreeEncoder is supported
    def set_encoding(self, encoding):
        self.encoding = encoding
        
    def set_discontinous(self, discontinous, disc_strategy):
        self.discontinous = discontinous
        
        if self.discontinous:
            self.disc_strategy = disc_strategy


    """
    Transforms a constituent tree with N leaves into a sequence of N labels.
    @param is_binary: True if binary trees are being encoded and want to use an optimized
    encoding [Not tested at the moment]
    @param root_label: Set to true to include a special label to words directly attached to the root
    @param encode_unary_leaf: Set to true to encode leaf unary chains as a part of the label
    """
    def to_maxincommon_sequence(self,is_binary=False, root_label=False, encode_unary_leaf=False,
                                abs_top=None, abs_neg_gap=None, join_char="~"):
        
        
        tags = [s.label() for s in self.subtrees(lambda t: t.height() == 2)]
        self.collapse_unary(collapsePOS=True, collapseRoot=True, joinChar=join_char) 
        
        if self.encoding is None: raise ValueError("encoding attribute is None")
        leaves_paths = []
        self.path_to_leaves([self.label()],leaves_paths)
        leaves = self.leaves()
        
        unary_sequence =  [s.label() for s in self.subtrees(lambda t: t.height() == 2)] #.split("+")
        
        
        labels = self.encoding.to_maxincommon_sequence(leaves, leaves_paths, unary_sequence, binarized=is_binary, 
                                                         root_label= root_label,
                                                         encode_unary_leaf=encode_unary_leaf,
                                                         abs_top=abs_top,
                                                         abs_neg_gap=abs_neg_gap)

        #We sort the words and labels according to the indexes
        #We then add the offset as part of the encoding
        if not self.discontinous:
            tokens = [Token(w,t,l) for w,t,l in zip(leaves,tags,labels)]
            
        else:
                
            unsorted_elements = list(zip(leaves, tags, labels))
            f = lambda x: int(x[0].split("=")[0])
            f_removed_index = lambda x: x.split("=",1)[1]
            sorted_elements = sorted(unsorted_elements, key=f)
            leaves = []
            tags = []
            labels = []

            tokens = []
            for word, tag, label in sorted_elements:
                leaves.append(f_removed_index(word))
                tags.append(tag)
                labels.append(label)
                     
            inorder_seq = list(map(f,unsorted_elements))

            offsets = self.disc_strategy.encode(inorder_seq,**{"postags":tags})
            #Concatenating the offset to the labels
            assert(len(labels)==len(offsets))
            labels = [self.encoding.split_char.join([l,str(o)]) for l,o in  zip(labels, offsets)]
        
            tokens = [Token(w,t,l) for w,t,l in zip(leaves,tags,labels)]
        
        return tokens#leaves, tags, labels
                        
            
        
            
            
                
        
    """
    Transforms a predicted sequence into a constituent tree
    @params sequence: A list of the predictions 
    @params sentence: A list of (word,postag) representing the sentence (the postags must also encode the leaf unary chains)
    @precondition: The postag of the tuple (word,postag) must have been already preprocessed to encode leaf unary chains, 
    concatenated by the '+' symbol (e.g. UNARY[0]+UNARY[1]+postag)
    """
    @classmethod
    def maxincommon_to_tree(cls, sentence, encoding):
        if encoding is None: raise ValueError("encoding parameter is None")
        return encoding.maxincommon_to_tree(sentence)


    """
    Gets the path from the root to each leaf node
    Returns: A list of lists with the sequence of non-terminals to reach each 
    terminal node
    """
    def path_to_leaves(self, current_path, paths):

        for i, child in enumerate(self):
            
            pathi = []
            if isinstance(child,Tree):
                common_path = copy.deepcopy(current_path)
                common_path.append(child.label()+"*"+str(i))
                child.path_to_leaves(common_path, paths)
            else:
                for element in current_path:
                    pathi.append(element)
                pathi.append(child)
                paths.append(pathi)
    
        return paths
    
    
    
"""
Encoder/Decoder class to transform a constituent tree into a sequence of labels by representing
how many levels in the tree there are in common between the word_i and word_(i+1) (in a relative scale) 
and the label (constituent) at that lowest ancestor.
"""
class RelativeLevelTreeEncoder(object):
        
    ROOT_LABEL = "ROOT"
    NONE_LABEL = "NONE"


    SPLIT_LABEL_SURNAME_SYMBOL = "*"


    def __init__(self, join_char="~",split_char="@"):
        self.join_char = join_char
        self.split_char = split_char


    #TODO: The binarized option has not been tested/evaluated
    """
    Transforms a tree into a sequence encoding the "deepest-in-common" phrase between words t and t+1
    @param leaves: A list of words representing each leaf node
    @param leaves_paths: A list of lists that encodes the path in the tree to reach each leaf node
    @param unary_sequence: A list of the unary sequences (if any) for every leaf node
    @param binarized: If True, when predicting an "ascending" level we map the tag to -1, as it is possible to determine in which
    level the word t needs to be located
    @param root_label: Set to true to include a special label ROOT to the words that are directly attached to the root of the sentence
    @param encode_unary_leaf: Set to true to encode leaf unary chains as a part of the label
    """  
    def to_maxincommon_sequence(self, leaves, leaves_paths, unary_sequence, 
                                binarized, root_label, encode_unary_leaf=False,
                                abs_top=None, abs_neg_gap=None):
        
        sequence = []
        previous_ni = 0
        ni=0
        relative_ni = 0 
        previous_relative_ni=0
        
        for j,leaf in enumerate(leaves):
            
            #It is the last real word of the sentence
            if j == len(leaves)-1: 
                #NEWJOINT   
                if encode_unary_leaf and self.join_char in unary_sequence[j]:
                    encoded_unary_leaf = self.split_char+self.join_char.join(unary_sequence[j].split(self.join_char)[:-1]) #The PoS tags is not encoded
                else:
                    encoded_unary_leaf = self.split_char+"NONE"          

                #TODO: This is a computation trick that seemed to work better in the dev set
                #Sentences of length on are annotated with ROOT_UNARYCHAIN instead NONE_UNARYCHAIN   
                if (root_label and len(leaves)==1):
                    sequence.append("1"+self.ROOT_LABEL+self.split_char+self.NONE_LABEL+encoded_unary_leaf)
                else:
                    sequence.append((self.NONE_LABEL+self.split_char+self.NONE_LABEL+encoded_unary_leaf))    
                              
                break
            
            explore_up_to = min( len(leaves_paths[j]), len(leaves_paths[j+1]) )+1   
            ni = 0

            for i in range(explore_up_to):   
                     
                if leaves_paths[j][i] == leaves_paths[j+1][i]:
                    ni+=1
                else:              
                    relative_ni = ni - previous_ni              
                    if binarized:
                        relative_ni = relative_ni if relative_ni >=0 else -1
                        
                    #NEWJOINT
                    if encode_unary_leaf and self.join_char in unary_sequence[j]:
                        encoded_unary_leaf = self.split_char+self.join_char.join(unary_sequence[j].split(self.join_char)[:-1]) #The PoS tags is not encoded
                    else:
                        encoded_unary_leaf = self.split_char+"NONE"

                    
                    #The root_label is activated and it is a top two level
                    if (root_label and ni==1) or (abs_top is not None and 
                                                  abs_neg_gap is not None and 
                                                  root_label and ni < (abs_top+1) 
                                                  and relative_ni <= abs_neg_gap): #and ni==1:

           
                        sequence.append(str(ni)+self.ROOT_LABEL+self.split_char+leaves_paths[j][ni-1].split(self.SPLIT_LABEL_SURNAME_SYMBOL)[0]+encoded_unary_leaf)                        
                    else:
                        sequence.append(self._tag(relative_ni, leaves_paths[j][ni-1])+encoded_unary_leaf)
                    
                    previous_ni = ni
                    previous_relative_ni = relative_ni
                    break

        return sequence   
    
    
    
    def remove_empty(self,tree):
      
        # Removing empty label from root
        if tree.label() == SeqTree.EMPTY_LABEL:
            # If a node has more than two children
            # it means that the constituent should have been filled.
            if len(tree) > 1:
            
                tree.set_label("S")     
            else:  
                while (tree.label() == SeqTree.EMPTY_LABEL) and len(tree) == 1:
                    tree = tree[0]

        return self._remove_empty(tree)
       # return tree
    
    
    def _remove_empty(self, tree):
    
        new_children = []
        
#         while (tree.label() == SeqTree.EMPTY_LABEL) and len(tree) == 1:
#                 tree = tree[0]
        
        for child in tree:
            
            if type(child) == type(u'') or type(child) == type(""):
                
                new_children.append(child)
            else:                
                if child.label() == SeqTree.EMPTY_LABEL and len(child) != 0:

#                     new_children.extend([self.remove_empty(grand_child) 
#                                                    for grand_child in child])
                     for grand_child in child:
                        new_children.append(self.remove_empty(grand_child))
                        
                else:
      #              print ("Entra else ", tree, len(child), child.label() )
      #              print ()
                    new_children.append(self.remove_empty(child))

        
        return  Tree(tree.label(),new_children)
        #tree = Tree(tree.label(),new_children)
        #return tree



    """
    Uncollapses the INTERMEDIATE unary chains and also removes empty nodes that might be created when
    transforming a predicted sequence into a tree.
    """
    def uncollapse(self, tree):
        
        if self.join_char in tree.label():
            aux = SeqTree(tree.label().split(self.join_char)[0], [])
            aux.append(SeqTree(self.join_char.join(tree.label().split(self.join_char)[1:]), tree))
            tree = aux
        
        return self._uncollapse(tree)
        
        
    def _uncollapse(self, tree):

        uncollapsed = []
        for  child in tree:

            if type(child) == type(u'') or type(child) == type(""):
                uncollapsed.append(child)
            else:

                label = child.label()
                #NEWJOINT
                if self.join_char in label:
                    
                    label_split = label.split(self.join_char) 
                    swap = Tree(label_split[0],[])

                    last_swap_level = swap
                    for unary in label_split[1:]:
                        last_swap_level.append(Tree(unary,[]))
                        last_swap_level = last_swap_level[-1]
                    last_swap_level.extend(child)
                    uncollapsed.append(self._uncollapse(swap))
                #We are uncolapsing the child node
                else:                    
                    uncollapsed.append(self._uncollapse(child))
        
        tree = Tree(tree.label(),uncollapsed)
        return tree

    
    """
    Gets a list of the PoS tags from the tree
    @return A list containing the PoS tags
    """
    def get_postag_trees(self,tree):
        
        postags = []
        
        for child in tree:
            
            if len(child) == 1 and type(child[-1]) == type(""):
                postags.append(child)
            else:
                postags.extend(self.get_postag_trees(child))
        
        return postags


    #TODO: The unary chain is not needed here.
    """
    Transforms a prediction of the form LEVEL_LABEL_[UNARY_CHAIN] into a tuple
    of the form (level,label):
    level is an integer or None (if the label is NONE or NONE_leafunarychain).
    label is the constituent at that level
    @return (level, label)
    """
    def preprocess_tags(self,pred):
        
        try:         
            pred_split = pred.split(self.split_char)
            level, label = pred_split[0],pred_split[1]  
            try:
                return (int(level), label)
            except ValueError:
                
                #It is a NONE label with a leaf unary chain
                if level == self.NONE_LABEL: #or level == self.ROOT:
                    return (None,pred_split[1])
                return (level,label)
            
        except IndexError:
            return (None, pred)


    """
    Transforms a predicted sequence into a constituent tree
    @params sequence: A list of the predictions 
    @params sentence: A list of (word,postag) representing the sentence (the postags must also encode the leaf unary chains)
    @precondition: The postag of the tuple (word,postag) must have been already preprocessed to encode leaf unary chains, 
    concatenated by the '+' symbol (e.g. UNARY[0]+UNARY[1]+postag)
    """
    def maxincommon_to_tree(self, sentence):
        
        tree = SeqTree(SeqTree.EMPTY_LABEL,[])
        current_level = tree
        previous_at = None
        first = True

        #sequence = list(map(self.preprocess_tags,[t.label for t in sentence]))
        sequence = [self.preprocess_tags(t.label) for t in sentence] #list(map(self.preprocess_tags,[t.label for t in sentence]))
        sequence = self._to_absolute_encoding(sequence)  
        printing = False
        
        for j,(level,label) in enumerate(sequence):
            
            token = sentence[j]
            
            if level is None:
                prev_level, _ = sequence[j-1]
                previous_at = tree
                while prev_level > 1:
                    previous_at = previous_at[-1]
                    prev_level-=1
          
                if self.NONE_LABEL == label: #or self.ROOT_LABEL:
                    previous_at.append( Tree( token.postag,[token.word]) )
                #It is a leaf unary chain
                else:
                    if label[0].isdigit() and self.ROOT_LABEL in label:
                        previous_at.append(Tree(self.join_char+token.postag,[token.word]))
                    else:
                        previous_at.append(Tree(label+self.join_char+token.postag,[ token.word]))   

                return tree
                #continue
                   
            i=0
            for i in range(level-1):
                if len(current_level) == 0 or i >= sequence[j-1][0]-1: 
                    child_tree = Tree(SeqTree.EMPTY_LABEL,[])                      
                    current_level.append(child_tree)   
                    current_level = child_tree

                else:
                    current_level = current_level[-1]
                     
            if current_level.label() == SeqTree.EMPTY_LABEL and label != self.NONE_LABEL:    
                current_level.set_label(label)
                        
            if first:
                previous_at = current_level
                previous_at.append(Tree( token.postag,[token.word]))
                first=False
            else:         
                #If we are at the same or deeper level than in the previous step
                if i >= sequence[j-1][0]-1: 
                    current_level.append(Tree(token.postag,[token.word]))
                else:
                    previous_at.append(Tree( token.postag,[token.word]))        
                previous_at = current_level
                
            current_level = tree
        return tree

        
        

#     """
#     Transforms a predicted sequence into a constituent tree
#     @params sequence: A list of the predictions 
#     @params sentence: A list of (word,postag) representing the sentence (the postags must also encode the leaf unary chains)
#     @precondition: The postag of the tuple (word,postag) must have been already preprocessed to encode leaf unary chains, 
#     concatenated by the '+' symbol (e.g. UNARY[0]+UNARY[1]+postag)
#     """
#     def maxincommon_to_tree(self, sequence, sentence):
#         
# #         print ("sequence", sequence)
# #         print ("sentence", sentence)
#         tree = SeqTree(SeqTree.EMPTY_LABEL,[])
#         current_level = tree
#         previous_at = None
#         first = True
# 
#         sequence = list(map(self.preprocess_tags,sequence))
#         sequence = self._to_absolute_encoding(sequence)  
#         printing = False
#         
#         for j,(level,label) in enumerate(sequence):
#             if level is None:
#                 prev_level, _ = sequence[j-1]
#                 previous_at = tree
#                 while prev_level > 1:
#                     previous_at = previous_at[-1]
#                     prev_level-=1
#           
#                 if self.NONE_LABEL == label: #or self.ROOT_LABEL:
#                     previous_at.append( Tree( sentence[j][1],[ sentence[j][0]]) )
#                 #It is a leaf unary chain
#                 else:
#                     if label[0].isdigit() and self.ROOT_LABEL in label:
#                         previous_at.append(Tree(self.join_char+sentence[j][1],[ sentence[j][0]]))
#                     else:
#                         previous_at.append(Tree(label+self.join_char+sentence[j][1],[ sentence[j][0]]))   
# 
#                 return tree
#                 continue
#                    
#             i=0
#             for i in range(level-1):
#                 if len(current_level) == 0 or i >= sequence[j-1][0]-1: 
#                     child_tree = Tree(SeqTree.EMPTY_LABEL,[])                      
#                     current_level.append(child_tree)   
#                     current_level = child_tree
# 
#                 else:
#                     current_level = current_level[-1]
#                      
#             if current_level.label() == SeqTree.EMPTY_LABEL and label != self.NONE_LABEL:    
#                 current_level.set_label(label)
#                         
#             if first:
#                 previous_at = current_level
#                 previous_at.append(Tree( sentence[j][1],[ sentence[j][0]]))
#                 first=False
#             else:         
#                 #If we are at the same or deeper level than in the previous step
#                 if i >= sequence[j-1][0]-1: 
#                     current_level.append(Tree( sentence[j][1],[sentence[j][0]]))
#                 else:
#                     previous_at.append(Tree( sentence[j][1],[ sentence[j][0]]))        
#                 previous_at = current_level
#                 
#             current_level = tree
#         return tree



#D: New to_absolute_encoding function to consider both "top" and "down" encodings
    """
    Transforms an encoding of a tree in a relative scale into an
    encoding of the tree in an absolute scale.
    """
    def _to_absolute_encoding(self, relative_sequence):
        
    
        absolute_sequence = [0]*len(relative_sequence)
        current_level = 0
        for j,(level,phrase) in enumerate(relative_sequence):
        
            if level is None:
                absolute_sequence[j] = (level,phrase)
            elif type(level) == type("") and self.ROOT_LABEL in level:
                
                try:
                    aux_level = int(level.replace(self.ROOT_LABEL,""))
                    absolute_sequence[j] = (aux_level, phrase)
                except ValueError:
                    aux_level = 1
                    absolute_sequence[j] = (aux_level,phrase)
            #elif level == self.ROOT_LABEL:
                #absolute_sequence[j] = (1, phrase)
                current_level=aux_level
            else:                
                current_level+= level
                absolute_sequence[j] = (current_level,phrase)
        return absolute_sequence

    
    def _tag(self,level,tag):
        #NEWJOINT
        return str(level)+self.split_char+tag.rsplit("*",1)[0]
      #  return str(level)+"_"+tag.rsplit("-",1)[0]
    
    
class SyntacticDistanceEncoder(object):
    
    
    def is_leaf(self, node):
        return type(node) == type(u'') or type(node) == type("")
            
    
    def rightmost_leaf(self, tree):
        
        if self.is_leaf(tree):
            return tree
        else:
            return self.rightmost_leaf(tree[-1])
    
    def leftmost_leaf(self, tree):
        
        if self.is_leaf(tree):
            return tree
        else:
            return self.leftmost_leaf(tree[0])
        
    
    def _words_to_indices(self,tree, index=0, indexes=[]):
        
        if self.is_leaf(tree):
            indexes.append(index)
            return index+1
        else:
            for child in tree:

                index = self._words_to_indices(child, index,indexes)
            return index
        
    


