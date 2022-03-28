import random
import torch
from d2l import torch as d2l

#生成数据集的过程
def synthetic_data(w,b,num_examples):
    X = torch.normal(0,1,(num_examples,len(w)))
    y = torch.matmul(X,w) + b
    y += torch.normal(0,0.01,y.shape)
    return X, y.reshape((-1,1))

true_w = torch.tensor([2,-3.4])
true_b = 4.2
features,labels = synthetic_data(true_w,true_b,1000)

print('features:',features[0],'\nlabel:',labels[0])

d2l.set_figsize()
d2l.plt.scatter(features[:,1].detach().numpy(),labels.detach().numpy(),1)

def data_iter(batch_size,features,labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    #这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0,num_examples,batch_size):
        batch_indices = torch.tensor(indices[i:min(i + batch_size,num_examples)])

        yield features[batch_indices], labels[batch_indices]
batch_size = 10
for X, y in data_iter(batch_size,features,labels):
    print(X,'\n',y)
    break
w = torch.normal(0,0.01,size = (2,1),requires_grad=True)
b = torch.zeros(1,requires_grad=True)
def linreg(X,w,b):
    #线性回归模型
    return torch.matmul(X,w) + b
def squared_loss(y_hat,y):
    #均方损失
    return (y_hat - y.reshape(y_hat.shape) ** 2) / 2
def sgd(params,lr,batch_size):
    #小批量随机梯度下降
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
lr = 0.03
num_ephochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_ephochs):
    for X,y in data_iter(batch_size,features,labels):
        l = loss(net(X,w,b),y)
        #X,y的小批量损失
        l.sum().backward()
        sgd([w,b],lr,batch_size) #使用参数的梯度更新
    with torch.no_grad():
        train_l = loss(net(features,w,b),labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')