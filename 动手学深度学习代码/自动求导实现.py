import torch
x = torch.arange(4.0)
x.requires_grad_(True)
y = 2 * torch.dot(x,x)
y.backward()
x.grad.zero_()
y = x.sum()
y.backward()
x.grad.zero_()
y = x * x
y.sum().backward()
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()

x.grad.zero_()
y.sum().backward()
#print(x.grad == 2 * x)

def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
a = torch.randn(size = (), requires_grad= True)
d = f(a)
d.backward()
print(a.grad == d / a)
