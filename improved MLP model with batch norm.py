import torch
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
g = torch.Generator().manual_seed(2147483647)
#%%
class Layer:
    def __init__(self, fan_in, fan_out, bias=True):
      self.weights = torch.randn((fan_in, fan_out), generator = g) / fan_in**0.5
      self.bias = torch.zeros_like(fan_out) if bias else None 
    def __call__(self, inputs):
        self.out = inputs @ self.weights
        if self.bias is not None:
            self.out += self.bias 
        return self.out
    def parameters(self):
        return [self.weights] + ([] if self.bias is None else [self.bias])
    
class BatchNorm1D:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)
    def __call__(self, inputs):
        if self.training:
            mean = inputs.mean(0, keepdim=True)
            var = inputs.var(0, keepdim=True)
        else:
            mean = self.running_mean
            var = self.running_var
        self.out = self.gamma * ((inputs - mean) / torch.sqrt(var + self.eps)) + self.beta
        if self.training:
            with torch.no_grad():
                self.running_mean = self.running_mean * (1-self.momentum) + mean * self.momentum
                self.running_var = self.running_var * (1-self.momentum) + var * self.momentum
        return self.out
    def parameters(self):
        return [self.gamma, self.beta]
    
class Tanh:
    def __call__(self, inputs):
        self.out = torch.tanh(inputs)
        return self.out
    def parameters(self):
        return []
    
class Embedding:
    def __init__(self, number, length):
        self.C = torch.randn((number, length), generator = g)
    def __call__(self, X):
        self.out = self.C[X]
        return self.out 
    def parameters(self):
        return [self.C]
    
class Flatten:
    def __call__(self, emb):
        self.out = emb.view(emb.shape[0], -1)
        return self.out
    def parameters(self):
        return []
    
class Container:
    def __init__(self, layers):
        self.layers = layers
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
#%%
context_size = 3
vector_length = 10
batch_size = 32
layer_number = 100
words = open('names.txt', 'r').read().splitlines() # read text file of names
chars = "".join(words)
stoi = {s:i+1 for i, s in enumerate(sorted(set(chars)))} # create string to integer dictonary for indexing
stoi['.'] = 0 # adding start and end token
itos = {i:s for s, i in stoi.items()} # create integer to string ditonary for later use
def build_dataset(words):  
  X, Y = [], []
  for w in words:
    context = [0] * context_size
    for ch in w + '.':
      ix = stoi[ch]
      X.append(context)
      Y.append(ix)
      context = context[1:] + [ix] # crop and append

  X = torch.tensor(X)
  Y = torch.tensor(Y)
  return X, Y

random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr,  Ytr  = build_dataset(words[:n1])     # 80%
Xdev, Ydev = build_dataset(words[n1:n2])   # 10%
Xte,  Yte  = build_dataset(words[n2:])     # 10%
#C = torch.randn((27, vector_length), generator = g) # creating embedding vectorization of characters
#%%
model = Container([Embedding(27, vector_length), Flatten(),
          Layer(context_size * vector_length, layer_number, bias = False), BatchNorm1D(layer_number), Tanh(), 
          Layer(layer_number, layer_number, bias = False), BatchNorm1D(layer_number), Tanh(), 
          Layer(layer_number, layer_number, bias = False), BatchNorm1D(layer_number), Tanh(), 
          Layer(layer_number, layer_number, bias = False), BatchNorm1D(layer_number), Tanh(), 
          Layer(layer_number, layer_number, bias = False), BatchNorm1D(layer_number), Tanh(), 
          Layer(layer_number, 27, bias = False), BatchNorm1D(27)])
# layers = [
#   Linear(n_embd * block_size, n_hidden), Tanh(),
#   Linear(           n_hidden, n_hidden), Tanh(),
#   Linear(           n_hidden, n_hidden), Tanh(),
#   Linear(           n_hidden, n_hidden), Tanh(),
#   Linear(           n_hidden, n_hidden), Tanh(),
#   Linear(           n_hidden, vocab_size),
# ]

with torch.no_grad():
    model.layers[-1].gamma *= 0.1 # to fix early incorrect confidence and we use gamma here as last layer is batch normalization 
    # layers[-1].weights *= 0.1 if batch norm was not used
    for layer in model.layers[:-1]:
        if isinstance(layer, Layer):
            layer.weights *= 5/3 # regularization value for tanh

print(sum(p.nelement() for p in model.parameters())) # number of parameters in total
for p in model.parameters():
  p.requires_grad = True
#%%
max_steps = 200000
lossi = []
ud = []
for i in range(max_steps):
    batch = torch.randint(0, Xtr.shape[0], (batch_size,), generator = g)
    Xbatch, Ybatch = Xtr[batch], Ytr[batch]
    # emb = C[Xbatch]
    # inputs = emb.view(emb.shape[0], context_size * vector_length)
    logits = model(Xbatch)
    loss = F.cross_entropy(logits, Ybatch)
    # for layer in layers:
    #     layer.out.retain_grad()
    for p in model.parameters():
        p.grad = None
    loss.backward()
    rate = 0.1 if i < 150000 else 0.01
    for p in model.parameters():
        p.data += -rate * p.grad
    if i % 1000 == 0: # print every once in a while 
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
    lossi.append(loss.log10().item())
    with torch.no_grad():
        ud.append([((rate*p.grad).std() / p.data.std()).log10().item() for p in model.parameters()])
    # if i == 999:
    #     break
    #break
#%%
lossi = torch.tensor(lossi)
loss_plot = lossi.view(-1, 1000).mean(1)


plt.plot(loss_plot)
#%%
for layer in model.layers:
    if isinstance(layer, BatchNorm1D):
        layer.training = False
inputs = Xdev
dev_logits = model(Xdev)
dev_loss = F.cross_entropy(dev_logits, Ydev)
for p in model.parameters():
    p.grad = None
dev_loss.backward()
print(dev_loss.item())
#%%
plt.figure(figsize=(20, 4)) # width and height of the plot
legends = []
for i, layer in enumerate(model.layers[:-1]): # note: exclude the output layer
  if isinstance(layer, Tanh):
    t = layer.out
    print('layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%' % (i, layer.__class__.__name__, t.mean(), t.std(), (t.abs() > 0.97).float().mean()*100))
    hy, hx = torch.histogram(t, density=True)
    plt.plot(hx[:-1].detach(), hy.detach())
    legends.append(f'layer {i} ({layer.__class__.__name__}')
plt.legend(legends);
plt.title('activation distribution')
#%%
plt.figure(figsize=(20, 4)) # width and height of the plot
legends = []
for i, layer in enumerate(model.layers[:-1]): # note: exclude the output layer
  if isinstance(layer, Tanh):
    t = layer.out.grad
    print('layer %d (%10s): mean %+f, std %e' % (i, layer.__class__.__name__, t.mean(), t.std()))
    hy, hx = torch.histogram(t, density=True)
    plt.plot(hx[:-1].detach(), hy.detach())
    legends.append(f'layer {i} ({layer.__class__.__name__}')
plt.legend(legends);
plt.title('gradient distribution')
#%%
# visualize histograms
plt.figure(figsize=(20, 4)) # width and height of the plot
legends = []
for i,p in enumerate(model.parameters()):
  t = p.grad
  if p.ndim == 2:
    print('weight %10s | mean %+f | std %e | grad:data ratio %e' % (tuple(p.shape), t.mean(), t.std(), t.std() / p.std()))
    hy, hx = torch.histogram(t, density=True)
    plt.plot(hx[:-1].detach(), hy.detach())
    legends.append(f'{i} {tuple(p.shape)}')
plt.legend(legends)
plt.title('weights gradient distribution');
#%%
plt.figure(figsize=(20, 4))
legends = []
for i,p in enumerate(model.parameters()):
  if p.ndim == 2:
    plt.plot([ud[j][i] for j in range(len(ud))])
    legends.append('param %d' % i)
plt.plot([0, len(ud)], [-3, -3], 'k') # these ratios should be ~1e-3, indicate on plot
plt.legend(legends);
