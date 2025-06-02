from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np
import random as r
import torch
import torch.nn as nn
import torch.nn.functional as F


class Predictor(nn.Module):
  """
  PyTorch Implementation of MLP for Predicting |p-q| for pq = n
  """
  def __init__(self, input_size, output_size=1, arch=[100, 100, 100]):
    super(Predictor).__init__()
    self.first = nn.Linear(input_size, arch[0])
    self.layers = nn.ModuleList([nn.Linear(arch[i], arch[i+1]) for i in range(len(arch))])
    self.last = nn.Linear(arch[-1], output_size)

  def forward(self, x):
    x = F.relu(self.first(x))
    for layer in self.layers:
      x = F.relu(self.layer(x))
    x = self.last(x)


def gen_primes(limit=1000):
  """Generate primes up to largest value 'limit'"""
  res = [2,3,5]
  for i in range(7, limit):
    j = 0
    isPrime = True
    while(res[j] <= (i ** 0.5)):
      if i % res[j] == 0:
        isPrime = False
      j += 1
    if isPrime: res.append(i)
  
  return res


def approx_m():
  """Machine Learning to Predict |p - q| (distance between p and q) for pq=n"""
  primes = gen_primes()

  # Generate 10000 samples of synthetic dataset
  X = []
  y = []
  for _ in range(10000):
    p = primes[r.randint(0,len(primes)-1)]
    q = primes[r.randint(0,len(primes)-1)]
    while (p == q):
      q = primes[r.randint(0,len(primes)-1)]
    
    if not p*q in X:
      X.append(p*q)
      y.append(abs(p - q))

  # Generate 10 test samples
  yTe = []
  xTe = []
  for _ in range(10):
    p = primes[r.randint(0,len(primes)-1)]
    q = primes[r.randint(0,len(primes)-1)]
    while (p == q):
      q = primes[r.randint(0,len(primes)-1)]
    
    if not p*q in X:
      xTe.append(p*q)
      yTe.append(abs(p - q))

  X = np.array(X).reshape(-1,1)
  y = np.array(y).reshape(-1,1)

  model = KernelRidge(kernel = 'rbf')
  # model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(24,100,100,100,100), random_state=1)
  model.fit(X,y)

  y_preds = model.predict(np.array(xTe).reshape(-1,1))

  print(y_preds)
  print(yTe)

  plt.scatter(X,y)
  plt.show()

  return model

def approx_m2(validate_num, primes, num_bits):
  """Second Version Using ML to Predict |p - q| (distance between p and q) 
  for pq=n (features are bits of n rather than n itself)"""
  # primes = gen_primes()
  X = []
  y = []
  for _ in range(10000):
    p = primes[r.randint(0,len(primes)-1)]
    q = primes[r.randint(0,len(primes)-1)]
    while (p == q):
      q = primes[r.randint(0,len(primes)-1)]
  
    if not p*q in X:
      val = [0]*num_bits
      bin_num = bin(p*q)[2:]
      try:
        #print(p,q)
        #print(bin(p*q)[2:])
        for i in range(len(bin_num)):
          val[-(i+1)] = int(bin_num[i])
        X.append(val)
        y.append(abs(p - q))
      except:
        print(p,q)
        print(bin_num)
        exit()

  xTe = []
  yTe = []
  nums = []
  for _ in range(validate_num):
    p = primes[r.randint(0,len(primes)-1)]
    q = primes[r.randint(0,len(primes)-1)]
    while (p == q):
      q = primes[r.randint(0,len(primes)-1)]

    if not p*q in xTe:
      val = [0]*num_bits
      bin_num = bin(p*q)[2:]
      for i in range(len(bin_num)):
        val[-(i+1)] = int(bin_num[i])
      xTe.append(val)
      yTe.append(abs(p - q))
      nums.append(p*q)

  X = np.array(X)
  y = np.array(y).reshape(-1,1)

  # print(X)
  # print(y)
  # print(xTe)
  # print(yTe)

  # model = KernelRidge(kernel = 'rbf')
  model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100,100,100), random_state=1)
  model.fit(X,y)

  y_preds = model.predict(np.array(xTe))

  error = 0
  num_close = 0
  epsilon = 20
  for i in range(validate_num):
    error += abs(y_preds[i] - yTe[i])
    if abs(y_preds[i] - yTe[i]) <= epsilon:
      num_close += 1


  # plt.xlim(0,1000)
  # plt.ylim(0,1000)
  # plt.scatter(y_preds.T.reshape(len(y_preds)),yTe)
  # plt.show()

  # print(y_preds.T.reshape(len(y_preds)))

  # avg = sum(y_preds.T.reshape(len(y_preds))) / len(y_preds.T.reshape(len(y_preds)))
  # # SCALE THE PREDICTED VALUES 
  # for i in range(len(y_preds)):
  #   # if y_preds[i] <= 140: y_preds[i] = 2
  #   # elif y_preds[i] >= 449: y_preds[i] = 998
  #   # else: #y_preds[i] = math.sinh((math.pi * y_preds[i]) / 100 - 10) + y_preds[i]
  #     # y_preds[i] = 100 * math.atan(math.pi * (y_preds[i] - avg)) + y_preds[i]
  #   #y_preds[i] =  (y_preds[i] / 40) + 150
  #   y_preds[i] /= 50
  
  # y_preds = y_preds.T.reshape(len(y_preds))
  # y_preds = np.exp(y_preds)
  # for i in range(len(y_preds)):
  #   if y_preds[i] >= 5000: 
  #     y_preds[i] = 5000

  # print(y_preds.T.reshape(len(y_preds)))
  # print(yTe)

  # PLOT PREDICTIONS COMPARED TO TRUE VALUES, SEE WHAT VALUES MODEL HAS SUCCESS
  # WITH AND ONES IT HAS TROUBLE
  plt.scatter(y_preds,yTe)
  # plt.plot(np.unique(y_preds), np.poly1d(np.polyfit(y_preds, yTe, 1))(np.unique(y_preds)))
  plt.show()

  yp_asort = np.argsort(y_preds)
  yTe_asort = np.argsort(yTe)

  #print(yp_asort)
  #print(yTe_asort)


  def cmp_asort_similarity(lst1, lst2):
    d = 0
    l_d = 0
    for i in range(len(lst1)):
      for j in range(len(lst2)):
        if lst1[i] == lst2[j]:
          d += abs(i - j)
          if abs(i - j) > l_d:
            l_d = abs(i - j)
          break
    return d, l_d

  avg_dist = 0
  for _ in range(200):
    lst1 = list(range(validate_num))
    lst2 = list(range(validate_num))
    r.shuffle(lst1)
    r.shuffle(lst2)

    dist_r, _ = cmp_asort_similarity(lst1, lst2)
    avg_dist += dist_r

  
  dist, largest_dist = cmp_asort_similarity(yp_asort, yTe_asort)
  
  def cmp_bits(y_preds, yTe):
    acc = 0
    total = 0
    accs = []

    for i, j in zip(y_preds, yTe):
      y_p = bin(i)[2:]
      y_t = bin(j)[2:]

      a = 0
      t = 0

      if len(y_p) > len(y_t):
        y_t = ('0'* (len(y_p) - len(y_t))) + y_t
      elif len(y_t) > len(y_p):
        y_p = ('0'* (len(y_t) - len(y_p))) + y_p  
      
      for v1, v2 in zip(y_t, y_p):
        acc += (v1 == v2)
        a += (v1 == v2)
        total += 1
        t += 1
      
      accs.append(a / t)
    
    return acc, total, accs
    
  acc, total, accs = cmp_bits(y_preds, yTe)

  plt.scatter(list(range(len(accs))), accs)
  plt.show()
    
  print('Expected Error: ' + str(error / validate_num))
  print('Average Distance: ' + str(sum(yTe) / validate_num))
  print('Probability of being close (within epsilon = ' + str(epsilon) + '): ' + str(num_close / validate_num))
  print('Argsort similarity: ' + str(dist / validate_num))
  print('Average random argsort similarity: ' + str((avg_dist / validate_num) / 200))
  print(('Largest argsort distance with validation set of size %d: ' + str(largest_dist)) % validate_num)
  print('Accuracy: ' + str(acc / total))
  # print(accs)

  return model


approx_m2(200, gen_primes(), 24)