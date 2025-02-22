import torch

words = open('names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
xs, ys = [], []
for w in words:
    for ch1, ch2 in zip(w, w[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)
xs = torch.tensor(xs)
ys = torch.tensor(ys)
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)

import torch.nn.functional as F

final_prob = torch.tensor([])

for i in range(20):
    xenc = F.one_hot(xs, num_classes=27).float()
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)
    loss = -probs[torch.arange(len(xs)), ys].log().mean() + 0.01 * (W ** 2).mean()
    W.grad = None
    loss.backward()
    W.data += -50 * W.grad
    if i == 19:
        # final_prob = probs
        print(loss.item())

# nlls = torch.zeros(len(xs))
# # for i in range(len(xs)):
# #     x = xs[i].item()
# #     y = ys[i].item()
# #     choice=torch.multinomial(final_prob[i],num_samples=1,replacement=True,generator=g)
# #     print('--------')
# #     print(f'bigram example {i + 1}: {itos[x]}{itos[y]}(indexes{x},{y})')
# #     print('input to the neural net:', itos[x])
# #     print('output probabilities from the neural net:', final_prob[i])
# #     print('label (actual next character):', itos[y])
# #     print('neural net choose:',itos[choice.item()])
# #     p = final_prob[i, y]
# #     print('probability assigned by the net to the correct character:', p.item())
# #     logp = torch.log(p)
# #     print('log likelihood:', logp.item())
# #     print('negative log likelihood:', -logp.item())
# #     nlls[i] = -logp.item()
# # print('=========')
# # print('average negative log likelihood, i.e. loss =', nlls.mean().item())

for i in range(5):
    output = []
    index = 0
    while True:
        xenc = F.one_hot(torch.tensor([index]), num_classes=27).float()
        logits = xenc @ W  # predict log-counts
        counts = logits.exp()  # counts, equivalent to N
        p = counts / counts.sum(1, keepdims=True)
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g)
        output.append(itos[ix.item()])
        if ix == 0:
            break
    print(''.join(output))

