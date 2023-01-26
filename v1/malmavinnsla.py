import numpy as np
import matplotlib.pyplot as plt

skrá = "https://cs.hi.is/python/allir-malmar.txt"
A = np.loadtxt(skrá, skiprows=1, delimiter=';', dtype='str', encoding = 'utf-8').T
efnatákn    = A[0].tolist()
nafn        = A[1].tolist()
sætistala   = A[2].astype(int)
A3          = np.char.replace(A[3], ",", ".")
eðlisþyngd  = A3.astype(float)
bræðslumark = A[4].astype(int)
enskt_nafn  = A[5].tolist()


def noble_gas_atomic_number(k):
    if k == 0:
        return -1
    if k & 1:
        return ((k+1)*(k+2)*(k+3) // 6) - 2 
    return ((k+1)*(k+2)*(k+3) // 6) + k // 2 - 1 

def lota(s):
    assert s > 0 and s < 119
    for k in range(1,8):
        if s <= noble_gas_atomic_number(k):
            return k

def flokkur(s):
    if s == 1:
        return 1
    if s > 1 and s <= (noble_gas_atomic_number(lota(s) - 1) + 2):
        return s - noble_gas_atomic_number(lota(s) - 1)
    return max(3,18 + s - noble_gas_atomic_number(lota(s)))


def íslenska(s):
    a = list('0123456789aábcdðeéfghiíjklmnoópqrstuúvwxyýzþæö')
    k = dict(zip(a, range(1,len(a)+1)))
    return [k.get(c.lower(),0) for c in s]


d = dict(zip(nafn, enskt_nafn))

for n in sorted(nafn, key = íslenska):
    print(n.capitalize(), d[n])


x = sætistala
y = eðlisþyngd
colors = [lota(i) for i in x]
size = [(s + 2)**2 for s in y]
print(colors)
plt.scatter(x,y,s = size,c = colors)
plt.colorbar(label = "lota")
plt.xlabel("Sætistala")
plt.ylabel("Eðlisþyngd")
plt.show()