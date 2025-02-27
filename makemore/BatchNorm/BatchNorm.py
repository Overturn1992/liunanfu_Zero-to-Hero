import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

g=torch.Generator().manual_seed(2147483647)

class Linear:
    def __init__(self,fan_in,fan_out,bias=True):
        self.weight=torch.randn([fan_in,fan_out],generator=g)/fan_in**0.5
        self.bias=torch.randn(fan_out) if bias else None
    
    def __call__(self, x):
        self.out=x@self.weight
        if self.bias is not None:
            self.out+=self.bias
        return self.out

    def parameters(self):
        return [self.weight]+([] if self.bias is None else [self.bias])
    
class BatchNormld:
    def __init__(self,dim,eps=1e-5,momentum=0.1):
        self.eps=eps
        self.momentum=momentum
        self.training=True
        self.gamma=torch.ones(dim)
        self.beta=torch.zeros(dim)
        self.running_mean=torch.zeros(dim)
        self.running_var=torch.ones(dim)

    def __call__(self, x):
        if self.training:
            xmean=x.mean(0,keepdim=True)
            xvar=x.var(0,keepdim=True)
        else:
            xmean=self.running_mean
            xvar=self.running_var
        xhat=(x-xmean)/torch.sqrt(xvar+self.eps)
        self.out=xhat*self.gamma+self.beta
        if self.training:
            self.running_mean=(1-self.momentum)*self.running_mean+self.momentum*xmean
            self.running_var=(1-self.momentum)*self.running_var+self.momentum*xvar
        return self.out
    
    def parameters(self):
        return [self.gamma,self.beta]
    
class Tanh:
    def __call__(self, x):
        self.out=torch.tanh(x)
        return self.out
    
    def parameters(self):
        return []

words=open('names.txt','r').read().splitlines()
chars=sorted(list(set(''.join(words))))
stoi={s:i+1 for i,s in enumerate(chars)}
stoi['.']=0
itos={i:s for s,i in stoi.items()}
vocab_size=len(itos)

block_size=3
def bulid_set(words):
    X,Y=[],[]
    for w in words:
        context=[0]*3
        for ch in w:
            X.append(context)
            Y.append(stoi[ch])
            context=context[1:]+[stoi[ch]]
    
    X=torch.tensor(X)
    Y=torch.tensor(Y)
    return X,Y

n1=int(0.8*len(words))
n2=int(0.9*len(words))
Xtr,Ytr=bulid_set(words[:n1])
Xdev,Ydev=bulid_set(words[n1:n2])
Xte,Yte=bulid_set(words[n2:])

n_emb=10
n_hidden=100

C=torch.randn([vocab_size,n_emb],generator=g)

Layers=[
    Linear(n_emb*block_size,n_hidden,bias=False),BatchNormld(n_hidden),Tanh(),
    Linear(n_hidden,n_hidden,bias=False),BatchNormld(n_hidden),Tanh(),
    Linear(n_hidden,n_hidden,bias=False),BatchNormld(n_hidden),Tanh(),
    Linear(n_hidden,n_hidden,bias=False),BatchNormld(n_hidden),Tanh(),
    Linear(n_hidden,n_hidden,bias=False),BatchNormld(n_hidden),Tanh(),
    Linear(n_hidden,vocab_size,bias=False),BatchNormld(vocab_size)
]

with torch.no_grad():
    Layers[-1].gamma*=0.1

parameters=[C]+[p for layer in Layers for p in layer.parameters()]
for p in parameters:
    p.requires_grad=True

max_steps=200000
batch_size=32
lossi=[]
ud=[]

for i in range(max_steps):
    ix=torch.randint(0,Xtr.shape[0],(batch_size,),generator=g)
    Xb,Yb=Xtr[ix],Ytr[ix]
    emb=C[Xb]
    x=emb.view(emb.shape[0],-1)
    for layer in Layers:
        x=layer(x)
    loss=F.cross_entropy(x,Yb)
    for p in parameters:
        p.grad=None
    loss.backward()
    lr=0.1 if i<150000 else 0.01
    for p in parameters:
        p.data+=-lr*p.grad
    
    if i%10000==0:
        print(f'{i:7d}/{max_steps:7d}:{loss.item():.4f}')
    lossi.append(loss.log10().item())
    with torch.no_grad():
        ud.append([((lr*p.grad).std() / p.data.std()).log10().item() for p in parameters])



        
    
        