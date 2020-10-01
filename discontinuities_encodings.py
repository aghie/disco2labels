from tree import Token
import warnings
import copy


class IndexDiscoLocation(object):


    def encode(self, inorder_seq, **kwargs):
        raise NotImplementedError

    def decode(self, sentence, offsets):
        raise NotImplementedError

    def add_index_to_token(self, token, idx):
        token.word = str(idx) + "=" + token.word
        token.label = "@".join(token.label.split("@")[0:3])
        return token


class AbsIndexDiscoLocation(IndexDiscoLocation):
    
    
    INVARIANT="-INV-"
    DEFAULT=INVARIANT

    def encode(self, inorder_seq,**kwargs):
        """
        @param inorder_seq: A list of the indexes of the leave nodes of the tree 
        encoding their position in the correspoding continuous arrangement.
        @return A list of offsets. Each offset encodes the absolute index of the position
        of a word in the continuous arrangement of the tree. 
        The words that remain in the same location get assigned the special label -INV-.
        """
        
        offsets = [self.INVARIANT]*len(inorder_seq)
        
        for idx, element in enumerate(inorder_seq):
            if idx != element:
                offsets[element] = idx 
                
        return offsets


    def postprocess(self, offsets, sentence):
        """
        @param sentence: The input sentence. A list of Token instances.
        @param offsets: A list of word offsets for 'sentence', encoded 
        by an AbsIndexDiscoLocation instance
        @return A valid list of word offsets to build a well-formed tree
        @precondition: Only a word must have been labeled as the last word in 'labels',
        i.e. only a word must have its predicted as NONE.
        """
        
        last_idx = len(offsets) - 1
        indexes_to_fill = []
        remaining_offsets = []
                
        #We fill words that remain in the same place, i.e. those assigned an -INV- label
        p_offsets = [offset if offset==self.INVARIANT else None for offset in offsets] 
        
        #We make sure the word labeled with a NONE syntactic level stays in the last place   
        none_level_idx = [idtoken for idtoken,token in enumerate(sentence)
                            if token.label.startswith("NONE")]
        assert(len(none_level_idx) < 2)
        
        if len(none_level_idx) == 1:
        
            p_offsets[none_level_idx[0]] = last_idx
            if none_level_idx[0] < last_idx:
                p_offsets[last_idx] = None
            
        #Calculate indexes that have still not been assigned a word
        indexes_to_fill = [idx for idx, o in enumerate(p_offsets)
                            if o is None]

        remaining_offsets =  [idx for idx in range(len(offsets))
                                if idx not in p_offsets and p_offsets[idx] != self.INVARIANT]
    
        #We fill those empty positions and postprocess them if necessary
        for idx in indexes_to_fill:
                
            if offsets[idx] == self.INVARIANT:
                
                if idx in remaining_offsets:
                    p_offsets[idx] = self.INVARIANT
                    remaining_offsets.remove(idx)
                else:
                    p_offsets[idx] = min(remaining_offsets, key = lambda x: abs(x-idx) )
                    remaining_offsets.remove(p_offsets[idx])                    
            
            else:
                offset_idx = int(offsets[idx])
                if offset_idx in remaining_offsets:
                    p_offsets[idx] = offset_idx
                    remaining_offsets.remove(offset_idx)
                else:
                    p_offsets[idx] = min(remaining_offsets, key = lambda x: abs(x-offset_idx) )
                    remaining_offsets.remove(p_offsets[idx])                       
                    
        return p_offsets



    def decode(self, sentence, offsets):
        """
        @param sentence: The input sentence. A list of Token instances.
        @param offsets: A list of word offsets for 'sentence', encoded by an 
        AbsIndexDiscoLocation instance, that can form a valid tree.
        @return A list of Token Instances sorted according to the position of 
        the tokens in the continuous arrangement of the tree.
        """
        
        idxs = [None]*len(sentence)
        assert(len(sentence)==len(offsets))

        for idxtoken,(token, offset) in enumerate(zip(sentence, offsets)):
            if offset == self.INVARIANT:
                idxs[idxtoken] = idxtoken
            else:
                idxs[int(offset)] = idxtoken
            
        
        return [self.add_index_to_token(sentence[idx], idx) for  idx in idxs ]    

    


class RelIndexDiscoLocation(IndexDiscoLocation):
    
    #[Optional] used to indicate that the index of a word is the same as the one
    #generated for the previous token
    SAME_AS_PREVIOUS_OFFSET= "SAME"
    INVARIANT = "-INV-"
    DEFAULT=INVARIANT
    
    def __init__(self, use_samedisc_offset=False):
        self.use_samedisc_offset = use_samedisc_offset


    def encode(self, inorder_seq,**kwargs):
        """
        @param inorder_seq: A list of the indexes of the leave nodes of the tree 
        encoding their position in the correspoding continuous arrangement.
        @return A list of offsets. Each offset encodes the relative index of the 
        position of a word in the continuous arrangement of the tree. 
        The words that remain in the same location get assigned the special label -INV-.
        """
         
        offsets = [self.INVARIANT]*len(inorder_seq)
        prev = None
     
        for idx, element in enumerate(inorder_seq):
            if idx != element:
                 
                if idx - element == prev and self.use_samedisc_offset:
                    offsets[element] = self.SAME_AS_PREVIOUS_OFFSET
                else:
                    offsets[element] = idx - element
 
            prev = idx-element
 
        return offsets


    

    def postprocess(self,offsets, sentence):
        """
        @param sentence: The input sentence. A list of Token instances.
        @param offsets: A list of word offsets for 'sentence', encoded by an RelIndexDiscoLocation instance
        @return A valid list of word offsets to build a well-formed tree
        @precondition: Only a word must have been labeled as the last word in 'labels',
        i.e. only a word must have its predicted as NONE.
        """
        
        last_idx = len(offsets) - 1
        indexes_to_fill = []
        remaining_offsets = []

        offsets = self._change_same_label_offsets(offsets)

        #We fill words that remain in the same place
        p_offsets = [offset if offset==self.INVARIANT else None for offset in offsets]
        #We make sure that the word labeled with NONE as level stays in the last place   
        none_level_idx = [idtoken for idtoken, token in enumerate(sentence)
                          if token.label.startswith("NONE")]
        assert(len(none_level_idx) < 2)
        
        if len(none_level_idx) == 1:
        
            p_offsets[none_level_idx[0]] = (last_idx - none_level_idx[0])
            if none_level_idx[0] < last_idx:
                p_offsets[last_idx] = None
                
        #Calculate indexes that have still not been assigned a word
        indexes_to_fill = [idx for idx, o in enumerate(p_offsets)
                           if o is None]

        remaining_offsets =  list(range(len(offsets)))
        for io, o in enumerate(p_offsets):
            
            diff = 0 if o == self.INVARIANT else o  
            if o is not None and io+diff in remaining_offsets:
                remaining_offsets.remove(io+diff)
    
        for idx in indexes_to_fill:
            offset_idx = int(offsets[idx]) if offsets[idx] != self.INVARIANT else 0
            if idx + offset_idx in remaining_offsets:
                p_offsets[idx] = offsets[idx]
                remaining_offsets.remove(idx + offset_idx)
            else: #This case represents an incongruency in the prediction
                new_abs_idx = min(remaining_offsets, key = lambda x: abs(x-idx+offset_idx))
                p_offsets[idx] = new_abs_idx - idx
                remaining_offsets.remove(new_abs_idx)         

        return p_offsets




    def _change_same_label_offsets(self,offsets):
        """
        @param offsets: It changes the SAME_AS_PREVIOUS_OFFSET dummy label by its real value
        @return The same list of offsets, with SAME_AS_PREVIOUS_OFFSET values changed to its real offset
        """
        
        new_offsets = []
        prev_offset = None
        for o in offsets:
        #for io, o in enumerate(offsets):
            if o != self.SAME_AS_PREVIOUS_OFFSET:
                new_offsets.append(o)
            else:
                if prev_offset is None:
                    warnings.warn("First label means 'same label' as previous element (but there is no previous element")
                    new_offsets.append(self.INVARIANT)
                else:
                    new_offsets.append(prev_offset)

            if o != self.SAME_AS_PREVIOUS_OFFSET:
                prev_offset = o

        return new_offsets




    def decode(self, sentence, offsets):
        """
        @param sentence: The input sentence. A list of Token instances.
        @param offsets: A list of word offsets for 'sentence', encoded by an 
        AbsIndexDiscoLocation instance, that can form a valid tree.
        @return A list of Token Instances sorted according to the position of 
        the tokens in the continuous arrangement of the tree.
        """
         
        #sorted_sentence = [None]*len(sentence)
        idxs = [None]*len(sentence)
        assert(len(sentence)==len(offsets))
         
        prev_offset = None
        for idxtoken,(token, offset) in enumerate(zip(sentence, offsets)):   
            if offset == self.SAME_AS_PREVIOUS_OFFSET:
                idxs[idxtoken+prev_offset] = idxtoken 
            else:
        
                if offset == self.INVARIANT:
                    idxs[idxtoken] = idxtoken
                else:
                    offset = int(offset)
                    idxs[idxtoken+offset] = idxtoken
             
            
                prev_offset = offset

        assert (set(idxs) == set(range(len(sentence))))
        return [self.add_index_to_token(sentence[idx], idx) for  idx in idxs ]    


    
class LehmerDiscoLocation(IndexDiscoLocation):
    
    #[Optional] used to indicate that the index of a word is the same as the one
    #generated for the previous token
    SAME_AS_PREVIOUS_OFFSET = -1
    DEFAULT=0
    
    def __init__(self, use_samedisc_offset=False):
        self.use_samedisc_offset = use_samedisc_offset


    def encode(self, inorder_seq,**kwargs):
        """
        @param inorder_seq: A list of the indexes of the leave nodes of the tree 
        encoding their position in the correspoding continuous arrangement.
        @return A list of offsets. Each offset encodes the position of a word in 
        the continuous arrangement of the tree using the Lehmer code.
        """
         
        offsets = []    
        idxs_not_visited = sorted(inorder_seq)
        prev_offset = None

        for e in inorder_seq:
        #for ie, e in enumerate(inorder_seq):
            
            offset = idxs_not_visited.index(e)
            
            if self.use_samedisc_offset  and offset != 0 and offset == prev_offset:
                offsets.append(self.SAME_AS_PREVIOUS_OFFSET)
            else:
                offsets.append(offset)
            
            prev_offset = offset    
            del idxs_not_visited[offset]
        
        return offsets
    


    def _change_same_label_offsets(self,offsets):
        """
        @param offsets: It changes the SAME_AS_PREVIOUS_OFFSET dummy label by its real value
        @return The same list of offsets, with SAME_AS_PREVIOUS_OFFSET values changed to its real offset
        """
        
        new_offsets = []
        prev_offset = None
        for o in offsets:
        #for io, o in enumerate(offsets):
             
            if o != self.SAME_AS_PREVIOUS_OFFSET:
                new_offsets.append(o)
            else:
                if prev_offset is None:
                    warnings.warn("First label means 'same label' as previous element (but there is no previous element")
                    new_offsets.append(0)
                else:
                    new_offsets.append(prev_offset)
 
            if o != self.SAME_AS_PREVIOUS_OFFSET:
                prev_offset = o

        return new_offsets


    def postprocess(self, offsets, sentence):
        """
        @param sentence: The input sentence. A list of Token instances.
        @param offsets: A list of word offsets for 'sentence', encoded 
        by an RelIndexDiscoLocation instance
        @return A valid list of word offsets to build a well-formed tree
        @precondition: Only a word must have been labeled as the last word 
        in 'labels', i.e. only a word must have its predicted as NONE.
        """
        
        def update_offsets(offsets, idxs, pivot):
            
            new_offsets = []
            new_idxs = []
            copy_idxs = idxs.copy()

            while offsets != []:
                offset = offsets.pop(0)
                if offset >= len(copy_idxs):
                    offset = len(copy_idxs)-1
                new_idxs.append(copy_idxs[offset])
                del copy_idxs[offset]
                
            new_idxs.remove(pivot)
            new_idxs.append(pivot)
            new_offsets = self.encode(new_idxs)
            return new_offsets


        offsets = list(map(int, offsets))
        offsets = self._change_same_label_offsets(offsets)
        
        p_offsets = []
        idxs = list(range(0, len(offsets)))
        
        
        #Getting index of the word labeled with NONE, i.e. the last word
        last_word_idx = [idtoken for idtoken,token in enumerate(sentence)
                         if token.label.startswith("NONE")]
        
        if len(sentence) > 1:
            last_word_idx = last_word_idx[0]

        
        while offsets != []:
            
            offset = offsets[0]         

            if offsets == []:
                p_offsets.append(0)
            else:

                if offset >= len(offsets):           
                    p_offset = len(offsets)-1
                else:
                    p_offset = offset        

                if sentence[idxs[p_offset]].label.startswith("NONE") and len(offsets) > 1:
                    offsets = update_offsets(offsets, idxs, last_word_idx)  
                    continue
                else:
                    del idxs[p_offset] 
                    offsets.pop(0)
                    p_offsets.append(p_offset)
                
        return p_offsets


    
    def decode(self, sentence, offsets):
        """
        @param sentence: The input sentence. A list of Token instances.
        @param offsets: A list of word offsets for 'sentence', encoded by 
        an LehmerDiscoLocation instance, that can form a valid tree.
        @return A list of Token Instances sorted according to the position 
        of the tokens in the continuous arrangement of the tree.
        """
        
        offsets = list(map(int, offsets))
        offsets = self._change_same_label_offsets(offsets)
        idxs = []
        assert(len(sentence)==len(offsets))
        idxs_not_visited = list(range(len(sentence)))

        for idx, offset in zip(range(len(sentence)),offsets):
            idxs.append(idxs_not_visited[offset])
            idxs_not_visited.remove(idxs_not_visited[offset])
            
        return [self.add_index_to_token(sentence[idx], idx) for  idx in idxs ]       




class LehmerInverseDiscoLocation(IndexDiscoLocation):

    #Optionally used to indicate that the index of a word is the same as the one
    #generated for the previous token
    SAME_AS_PREVIOUS_OFFSET = -1
    DEFAULT=0
    
    def __init__(self, use_samedisc_offset=False):
        self.use_samedisc_offset = use_samedisc_offset

    def encode(self, inorder_seq,**kwargs):
        """
        @param inorder_seq: A list of the indexes of the leave nodes of the tree 
        encoding their position in the correspoding continuous arrangement.
        @return A list of offsets. Each offset encodes the position of a word in 
        the continuous arrangement of the tree using the Lehmer code of the inverse permutation
        """
        
        offsets = []
        idxs_not_visited = inorder_seq.copy()
        prev_offset = None
        for element in sorted(inorder_seq):# enumerate(inorder_seq):
            
            
            offset = idxs_not_visited.index(element)
            
            if self.use_samedisc_offset and offset != 0 and offset == prev_offset:
                offsets.append(self.SAME_AS_PREVIOUS_OFFSET)
            else:
                offsets.append(idxs_not_visited.index(element))
            
            prev_offset = offset
            idxs_not_visited.remove(element)

        return offsets

    def postprocess(self, offsets, sentence):
        """
        @param sentence: The input sentence. A list of Token instances.
        @param offsets: A list of word offsets for 'sentence', encoded 
        by an RelIndexDiscoLocation instance
        @return A valid list of word offsets to build a well-formed tree
        @precondition: Only a word must have been labeled as the last word in 'labels',
        i.e. only a word must have its predicted as NONE.
        """
        
        offsets = list(map(int, offsets))    
        offsets = self._change_same_label_offsets(offsets)
        p_offsets = []
        blanks = len(offsets) - 1
        last_assigned = False
        
        i=0      
        while offsets != []:
            
            offset = offsets.pop(0) 
            if not sentence[i].label.startswith("NONE"):
                if offset < blanks:
                    p_offsets.append(offset)
                else:         
                    if last_assigned:
                        p_offsets.append(max(blanks,0))
                    else:        
                        p_offsets.append(max(blanks-1,0)) #The last blank is reserved for the word predicted as the last word. The label starts by NONE
            else:    
                p_offsets.append(max(blanks,0))     
                last_assigned = True     
                    
            i+=1
            blanks-=1

        return p_offsets

    

    def decode(self, sentence, offsets):
        """
        @param sentence: The input sentence. A list of Token instances.
        @param offsets: A list of word offsets for 'sentence', encoded by an 
        LehmerInverseDiscoLocation instance, that can form a valid tree.
        @return A list of Token Instances sorted according to the position of 
        the tokens in the continuous arrangement of the tree.
        """
        
        idxs = [None]*len(sentence)
        assert(len(sentence)==len(offsets))
        idxs_not_visited = list(range(len(sentence)))
        offsets = list(map(int,offsets))
        
        offsets = self._change_same_label_offsets(offsets)
        
        #We fill the offset indexes that have not been filled yet
        for idx, offset in zip(range(len(sentence)), offsets):
            
            idx_at = idxs_not_visited[offset]
            idxs[idx_at] = idx
            del idxs_not_visited[offset]
        
        return [self.add_index_to_token(sentence[idx], idx) for  idx in idxs ]    
        
        
    def _change_same_label_offsets(self,offsets):
        """
        @param offsets: It changes the SAME_AS_PREVIOUS_OFFSET dummy label by 
        its real value
        @return The same list of offsets, with SAME_AS_PREVIOUS_OFFSET values 
        changed to its real offset
        """
        
        new = []
        prev_offset = None
        for o in offsets:
        #for io, o in enumerate(offsets):
            
            if o != self.SAME_AS_PREVIOUS_OFFSET:
                new.append(o)
            else:
                if prev_offset is None:
                    warnings.warn("First label means 'same label' as previous element (but there is no previous element")
                    new.append(0)
                else:
                    new.append(prev_offset)

            if o != self.SAME_AS_PREVIOUS_OFFSET:
                prev_offset = o

        return new
        
        


class POSPointerDiscoLocation(IndexDiscoLocation):
    
    LAST_INDEX = "LAST_INDEX"
    DEFAULT=0
    
    def __init__(self):
        self.reduce_tagset = False
    
    def encode(self, inorder_seq,**kwargs):
        """
        @param inorder_seq: A list of the indexes of the leave nodes of the tree 
        encoding their position in the correspoding continuous arrangement.
        @return A list of offsets. Each offset encodes the position of a word in 
        the continuous arrangement of the tree using a pointer encoding based on PoS tags
        """
        
        offsets = []
        
        postags = [self._tag2reduced(tag, self.reduce_tagset) for tag in  kwargs["postags"]]
        
        input_seq = list(range(0, len(inorder_seq)))
        values_idxs = {value:idx for idx, value in enumerate(inorder_seq)}

        for value in input_seq: 
            idx = values_idxs[value]     
            prev_value = value-1 if value > 0 else -1
            prev_idx = values_idxs[prev_value] if prev_value != -1 else -1
        
            prev_inorder_value =  self._get_prev_inorder_value(inorder_seq, value, idx) if idx > 0 else -1

            if idx > prev_idx:
                
                if self._belongs_to_right_constituent(inorder_seq, idx, prev_idx, prev_value):
                    displacement_label = self._get_displacement_label(postags, value, prev_inorder_value)
                    offsets.append(displacement_label)
                else:
                    offsets.append(str(0))
                    
            else:
                displacement_label = self._get_displacement_label(postags, value, prev_inorder_value)
                offsets.append(displacement_label)
    
        return offsets


    def postprocess(self, offsets,  sentence):
        """
        @param sentence: The input sentence. A list of Token instances
        @param offsets: A list of word offsets for 'sentence', according to an 
        POSPointerDiscoLocation instance
        @return A valid list of word offsets to build a well-formed tree
        @precondition: Only a word must have been labeled as the last word in 'labels',
         i.e. only a word must have its predicted as NONE.
        """
        
        poffsets = []
        
        last_word_idx = [idtoken for idtoken, token in enumerate(sentence) 
                         if token.label.startswith("NONE")]
        assert(len(last_word_idx)<=1)
        
        if len(last_word_idx) == 1:
            last_word_idx = last_word_idx[0]
        else:
            last_word_idx = None
        
        for idx, (token, offset) in enumerate(zip(sentence, offsets)):
            #This is the last word problem
            if token.label.startswith("NONE"): 
                poffsets.append(self.LAST_INDEX)
                continue

            if offset == "0":
                poffsets.append(offset)
                
            #We have generated a label, we need to check that the postag exists
            #to assign that or, alternatively, other (valid) offset 
            else:
                
                if self._label2idx(sentence, idx, offsets[idx]) == last_word_idx:           
                    poffsets.append("0")
                                  
                else:
                    offset_idx, offset_postag = offset.split("~")
                    offset_idx = abs(int(offset_idx))

                    
                    valid_idxs = [(aux_idx, aux_token) for aux_idx, aux_token in enumerate(sentence[:idx]) 
                                  if self._tag2reduced(aux_token.postag, self.reduce_tagset) == offset_postag]
                          
                    if len(valid_idxs) >= offset_idx: #This is the successful case
                        poffsets.append(offset)
                    
                    elif len(valid_idxs) == 0:
                        poffsets.append("0")
                    else:
                        poffsets.append("-"+str(len(valid_idxs))+"~"+offset_postag)
            
                
        return poffsets
      
       

    def decode(self, sentence, offsets):
        """
        @param sentence: A list of Token instances
        @offsets offsets: A list of offsets generated according to an LehmerInOrderDiscoLocation
        @return (sorted_sentence, idxs) The sentence sorted according to its inorder traversal and
        the list of indexes in inorder traversal too.
        """
        
        idxs = []
        pointer = -1
        last_idx, last_element = None, None
        
        
        if self.reduce_tagset:
            
            sequence_reduced_tagset = [Token(token.word, self._tag2reduced(token.postag, self.reduce_tagset), token.label) 
                                   for token in sentence]
        else:
            sequence_reduced_tagset = sentence

        for iprocessed, (token, offset) in enumerate(zip(sentence, offsets)):

            if offset == "0":
                pointer = pointer+1
                idxs.insert(pointer, iprocessed)
            #We need to insert and update the pointer
            elif offset == self.LAST_INDEX:
                last_idx, last_element = iprocessed, token
            else:
                pointer = self._get_pointer(sequence_reduced_tagset, idxs, offset, iprocessed-1)
                if pointer != last_idx:
                    pointer = idxs.index(pointer)+1
                idxs.insert(pointer, iprocessed)

        if last_idx is not None and last_element is not None:
            idxs.append(last_idx)



        return [self.add_index_to_token(sentence[idx], idx) for  idx in idxs ]    


    

    def build_reduced_tagset(self, path_reduced_tagset):
        
        self.tagset2reduced = {}
        self.reduced2tagset = {}
        self.reduce_tagset = True
        
        assigned_reduced_tags = set([])
        
        with open(path_reduced_tagset) as f:
            lines = f.readlines()
            
            for l in lines:
                if l.startswith("#"): continue
                else:
                    
                    ls = l.strip("\n").split("=")
                    tags_to_reduce, reduced_tag = [t.strip(" " )
                                                   for t in ls[0].split(",") ], ls[1].strip(" ")               
                    if reduced_tag in assigned_reduced_tags:
                        pass
                    else:
                        assigned_reduced_tags.add(reduced_tag)
                    
                    for tag in tags_to_reduce:
                        self.tagset2reduced[tag] =  reduced_tag
                    self.reduced2tagset[reduced_tag] = tags_to_reduce
                

    def _belongs_to_right_constituent(self, inorder_seq, idx, prev_idx, prev_value):
        return any([prev_value > e for e in inorder_seq[prev_idx:idx]])
    
    
    def _tag2reduced(self,tag,reduce=False):        
        return tag if not reduce or tag not in self.tagset2reduced else self.tagset2reduced[tag] 
    
    def _get_displacement_label(self, postags, value, prev_value):
     
        d_tags={}
     
        if value > prev_value:       
            for itag in range(value-1, prev_value-1, -1):    
                tag = postags[itag]
                try:
                    d_tags[tag]+=1
                except KeyError:
                    d_tags[tag]=1
                     
            return str(str(-d_tags[tag])+"~"+postags[prev_value])
         
        else:
            
            print (postags, value, prev_value)
            raise NotImplementedError
            
    def _get_prev_inorder_value(self, inorder_seq, value, idx):
        
        for e in inorder_seq[idx::-1]:
            if e < value:
                return e
            
    def _get_pointer(self, sequence, indexes, pointer, curidx):
        
        offset_idx, offset_postag = pointer.split("~")
        offset_idx = abs(int(offset_idx))
        for position, token in enumerate(sequence[curidx::-1],1):
            if token.postag == offset_postag:
                offset_idx-=1
            
            if offset_idx == 0:
                value_to_index = len(sequence[curidx::-1]) - position
                
                return value_to_index
                
        raise ValueError
      
      
    def _label2idx(self, sequence, idx, offset):
    
        offset_idx, offset_postag = offset.split("~")
        offset_idx = abs(int(offset_idx))
        for j, token in enumerate(sequence[idx::-1]):
            
            if offset_postag == token.postag:
                offset_idx-=1
            
            if offset_idx == 0:
                return idx - j     
            
        return "-1" 

