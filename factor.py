import math
from decimal import *
import time

def factor(n, stdout=False):
  """
  VERSION 1 OF COMPUTING PRIME FACTORS
  Computes the prime factors p and q such that p*q=n

  Parameters:
  n: the number to compute the prime numbers
  stdout: prints out p and q once computed

  Returns: p and q, the two prime numbers, such that p <= q
  """
  i = 0
  while(True):
    x = math.sqrt(n + (i*i))
    if int(x) == x:
      if stdout:
        print(int(x-i), int(x+i))
      return int(x-i), int(x+i)
    i += 1


def factor2(n, stdout=False):
  """
  VERSION 2 OF COMPUTING PRIME FACTORS
  Computes the prime factors p and q such that p*q=n

  Parameters:
  n: the number to compute the prime numbers
  stdout: prints out p and q once computed

  Returns: p and q, the two prime numbers, such that p <= q
  """
  val = str(Decimal(n).sqrt())
  val = int(val[:val.index('.')]) + 1

  i = 0
  while(True):
    x = str(Decimal((val+i)**2 - n).sqrt())
    try:
      _ = x.index('.')
    except:
      if stdout:
        print(val+i-int(x), val+i+int(x))
      return val+i-int(x), val+i+int(x)
    
    i += 1


def factor3(n, stdout=False):
  """
  VERSION 3 OF COMPUTING PRIME FACTORS
  Computes the prime factors p and q such that p*q=n

  Parameters:
  n: the number to compute the prime numbers
  stdout: prints out p and q once computed

  Returns: p and q, the two prime numbers, such that p <= q
  """
  a = 1
  root_n = math.ceil(math.sqrt(n)) 
  b = 2 * root_n - 1
  S = (b + 1) * ((b + 1) / 2) / 2

  i = 1
  while(S != n):
    y = math.ceil((1-a) + math.sqrt((a-1)**2 + 4 * (S-n))) / 2
    S -= y*y + (a - 1) * y
    a = a + 2 * y

    while(S < n):
      b += 2
      S += b
    i += 1      
    
  p = (b - a) / 2 + 1
  q = (b + a) / 2

  if stdout:
    print(int(p), int(q))
  
  return int(p), int(q)


def factor4(n, stdout=False):
  """
  VERSION 4 OF COMPUTING PRIME FACTORS USING SLIDING WINDOW METHOD
  Computes the prime factors p and q such that p*q=n

  Parameters:
  n: the number to compute the prime numbers
  stdout: prints out p and q once computed

  Returns: p and q, the two prime numbers, such that p <= q
  """
  a = 1
  root_n = math.ceil(math.sqrt(n)) 
  b = 2 * root_n - 1
  S = (b*b - a*a + 2*b + 2*a) / 4

  while(S != n):
    if S < n:
      S += b + 2
      b += 2

    else:
      S -= a
      a += 2
      
  p = (b - a) / 2 + 1
  q = (b + a) / 2

  if stdout:
    print(int(p), int(q))

  return int(p), int(q)



# Sample Usage
# factor(83*431, True)
# factor2(83*431, True)
# factor3(83*431, True)
# factor4(83*431, True)


a = time.time()
# factor(23010067*23252729)
factor(83 * 98645273)
print('Time 1: ' + str(time.time() - a))

a = time.time()
# factor2(23010067*23252729)
factor2(83 * 98645273)
print('Time 2: ' + str(time.time() - a))

a = time.time()
# factor3(23010067*23252729)
factor3(83 * 98645273)
print('Time 3: ' + str(time.time() - a))

a = time.time()
# factor4(23010067*23252729)
factor4(83 * 98645273)
print('Time 4: ' + str(time.time() - a))