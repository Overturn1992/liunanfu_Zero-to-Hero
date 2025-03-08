import unicodedata

def get_stats(ids,counts=None):
    counts={} if counts is None else counts
    for pair in zip(ids,ids[1:]):
        counts[pair]=counts.get(pair,0)+1
    return counts

def merge(ids,pair,idx):
    new_ids=[]
    i=0
    while i<len(ids):
        if ids[i]==pair[0] and i!=len(ids)-1 and ids[i+1]==pair[1]:
            new_ids.append(idx)
            i+=2
        else:
            new_ids.append(ids[i])
            i+=1
    return new_ids

def replace_control_characters(s:str)->str:
    chars=[]
    for ch in s:
        if unicodedata.category(ch)[0]!="C":
            chars.append(ch)
        else:
            chars.append(f"\\u{ord(ch):04x}")
    return ''.join(chars)

def render_token(t:bytes)->str:
    s=t.decode('utf-8',errors='replace')
    s=replace_control_characters(s)
    return s

class Tokenizer:
    def __init__(self):
        self.merges={}
        self.pattern=""
        self.special_tokens={}
        self.vocab={}
    
    def train(self,text,vocab_size,verbose=False):
        raise NotImplementedError
    
    def encode(self,text):
        raise NotImplementedError
    
    def decode(self,ids):
        raise NotImplementedError
    
    def _build_vocad(self):
        vocab={idx:bytes([idx]) for idx in range(256)}
        for (p0,p1),idx in self.merges.items():
            vocab[idx]=vocab[p0]+vocab[p1]
        for special,idx in self.special_tokens.items():
            vocab[idx]=special.encode('utf-8')
        return vocab        
    
    def save(self,file_prefix):
        model_file=file_prefix+'.model'
        with open(model_file,'w') as f:
            f.write('minbpe v1\n')
            f.write(f"{self.pattern}\n")
            f.write(f'{len(self.special_tokens)}\n')
            for special,idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            for pair,idx in self.merges:
                f.write(f"{pair} {idx}\n")
        vocab_file=file_prefix+'.vocab'
        inverted_merges={idx:pair for pair,idx in self.merges.items()}
        with open(vocab_file,'w') as f:
            for idx,token in self.vocab.items():
                s=render_token(token)
                if idx in inverted_merges:
                    idx0,idx1=inverted_merges[idx]
                    s0=render_token(self.vocab[idx0])
                    s1=render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    f.write(f"{s} {idx}\n")
        
        def load(self,model_file):
            assert model_file.endswith('.model')
            merges={}
            special_tokens={}
            idx=256
            with open(model_file,'r') as f:
                version=f.readline().strip()
                assert version=='minbpe v1'
                self.pattern=f.readline().strip()
                num_special=int(f.readline().strip())
                for _ in range(num_special):
                    special,special_idx=f.readline().strip().split()
                    special_tokens[special]=int(special_idx)
                for line in f:
                    idx1,idx2=map(int,line.split())
                    merges[(idx1,idx2)]=idx
                    idx+=1
            self.merges=merges
            self.special_tokens=special_tokens
            self.vocab=self._build_vocad()