import torch

words=open('names.txt','r').read().splitlines()
N=torch.zeros((27,27),dtype=torch.int32)
chars=sorted(list(set(''.join(words))))
stoi={s:i+1 for i,s in enumerate(chars)}
stoi['.']=0
itos={i:s for s,i in stoi.items()}
for w in words:
    chs=['.']+list(w)+['.']
    for ch1,ch2 in zip(chs,chs[1:]):
        ix1=stoi[ch1]
        ix2=stoi[ch2]
        N[ix1,ix2]+=1

prob=(N+1).float() #smoothing
prob/=prob.sum(1,keepdim=True)

g=torch.Generator().manual_seed(2147483647)
for i in range(5):
    out=[]
    index=0
    while True:
        p=prob[index]
        index=torch.multinomial(p,num_samples=1,replacement=True,generator=g).item()
        out.append(itos[index])
        if index==0:
            break
    print(''.join(out))

