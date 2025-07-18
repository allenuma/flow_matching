import torch

print(torch.__version__)
print(torch.cuda.is_available())  # should be False

x = torch.randn(512, 4)
j = 20

x_rep = x.unsqueeze(1).repeat(1, j, 1)
print(x_rep.shape)
