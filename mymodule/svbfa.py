#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from math import pi
import matplotlib.pyplot as plt

def svbfa(y, nl, nem):
  ## Input data expected size=(data points, dimension)
  ## This code transpose y to (dimension, data points) like math form.
  y = y.transpose(0, 1)


  M = y.size(0) ## dimension
  K = y.size(1) ## data points

  # initialization
  # A is initialized using the svd of the data
  U, S, _ = torch.svd(y)
  sd = torch.diag(S[:nl])
  A = torch.matmul(U[:, :nl], sd)

  Ryy = torch.matmul(y, y.t()) ##covariance matrix
  lam = 1 / torch.diag(Ryy/K) ## noise precision vector
  Lam = torch.diag(lam) ## noise precision matrix

  iPsi = torch.eye(nl)
  A_L_A = A.t().matmul(Lam).matmul(A)
  Alpha = torch.diag(1 / torch.diag(A_L_A + iPsi))


  for i in range(nem):
    ##e_step
    Gamma = A.t().matmul(Lam).matmul(A) \
          + torch.eye(nl, nl) \
          + M * iPsi
    iGam = torch.inverse(Gamma)
    ubar = iGam.matmul(A.t()).matmul(Lam).matmul(y)


    Ruu = torch.matmul(ubar, ubar.t()) + K * iGam
    Ruy = ubar.matmul(y.t())
    Ryu = Ruy.t()

    ##m_step
    Psi = Ruu + Alpha
    iPsi = torch.inverse(Psi)
    A = Ryu.matmul(iPsi)

    ilam = torch.diag(Ryy - A.matmul(Ruy)) / K
    lam = 1 / ilam
    Lam = torch.diag(lam)

    ialf = torch.diag(A.t().matmul(Lam).matmul(A) / M + iPsi)
    Alpha = torch.diag(1 / ialf)

  yclean = A.matmul(ubar)

  return A.t(), Lam.t(), yclean.t(), ubar.t()



t = torch.linspace(-1, 1, steps=2000)
u = torch.zeros((2000, 3))
u_i = [torch.sin(2 * pi * 0.3 * t),
       torch.sin(2 * pi * 1 * t),
       torch.sin(2 * pi * 2* t)]
for i, ui in enumerate(u_i):
  u[:, i] = ui


A = torch.randn((3,6))

y = torch.matmul(u, A)
y += 0.5 * torch.randn(y.size())




W, _, y_c, ubar = svbfa(y, nl=6, nem=100000)


plt.figure()
plt.plot(t.numpy(), u.numpy())
plt.title('Latent data')

plt.figure()
plt.plot(t.numpy(), ubar.numpy())
plt.title('predicted Latent data')

plt.figure()
plt.plot(t.numpy(), y.numpy())
plt.title('Observed data')

plt.figure()
plt.plot(t.numpy(), y_c.numpy())
plt.title('noise detected Observed data')

plt.show()
