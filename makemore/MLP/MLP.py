import torch
import torch.nn.functional as F

words=open('names.txt','r').read().splitlines()
chars=sorted(list(set(''.join(words))))
stoi={s:i+1 for i,s in enumerate(chars)}
stoi['.']=0
itos={i:s for s,i in stoi.items()}

block_size=3
def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X,Y

import random
random.seed(42)
random.shuffle(words)

n1=int(len(words)*0.8)
n2=int(len(words)*0.9)

Xtr,Ytr=build_dataset(words[:n1])  #training
Xdev,Ydev=build_dataset(words[n1:n2])  #development
Xte,Yte=build_dataset(words[n2:])  #test

g=torch.Generator().manual_seed(2147483647)
C=torch.randn([27,10],generator=g)
W1=torch.randn([30,200],generator=g)
b1=torch.randn(200,generator=g)
W2=torch.randn([200,27],generator=g)
b2=torch.randn(27,generator=g)
parameters=[W1, b1, W2, b2]
for p in parameters:
    p.requires_grad=True
for i in range(200000):
    ix=torch.randint(0,Xtr.shape[0],[32])
    emb=C[Xtr[ix]]
    h=torch.tanh(emb.view(-1,30)@W1+b1)
    logits=h@W2+b2
    loss=F.cross_entropy(logits,Ytr[ix])
    for p in parameters:
        p.grad=None
    loss.backward()
    lr=0.1 if i<100000 else 0.01
    for p in parameters:
        p.data+=-lr*p.grad

emb=C[Xdev]
h=torch.tanh(emb.view(-1,30)@W1+b1)
logits=h@W2+b2
loss=F.cross_entropy(logits,Ydev)
print(loss.item())

g=torch.Generator().manual_seed(2147483647+10)
for _ in range(20):
    out=[]
    context=[0]*block_size
    while True:
        emb=C[torch.tensor(context)]
        h=torch.tanh(emb.view(1,-1)@W1+b1)
        logits=h@W2+b2
        probs=F.softmax(logits,dim=1)
        ix=torch.multinomial(probs,num_samples=1,generator=g)
        context=context[1:]+[ix]
        out.append(ix)
        if ix==0:
            break
    print(''.join(itos[i.item()] for i in out))










