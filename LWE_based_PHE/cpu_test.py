import random
import time

n_lwe = 3000
s = 8
p = 2 ** 48 + 1
q = 2 ** 77
l = 2 ** 6

class PublicKey:
    def __init__(self, A, P, n_lwe, s):
        self.A = A
        self.P = P
        self.n_lwe = n_lwe
        self.s = s

    def __repr__(self):
        return 'PublicKey({}, {}, {}, {})'.format(self.A, self.P, self.n_lwe, self.s)

class Ciphertext:
    def __init__(self, c1, c2):
        self.c1 = c1
        self.c2 = c2

    def __repr__(self):
        return 'Ciphertext({}, {})'.format(self.c1, self.c2)

    def __add__(self, c):
        c1 = []
        for i in range(n_lwe):
            c1.append(self.c1[i] + c.c1[i])

        c2 = []
        for i in range(l):
            c2.append(self.c2[i] + c.c2[i])

        return Ciphertext(c1, c2)

def get_discrete_gaussian_random_matrix(m, n):
    sample = []
    for i in range(m):
        row_sample = []
        for i in range(n):
            row_sample.append(round(random.gauss(0, s)))
        sample.append(row_sample)
    return sample

def get_discrete_gaussian_random_vector(n):
    sample = []
    for i in range(n):
        sample.append(round(random.gauss(0, s)))
    return sample

def get_uniform_random_matrix(m, n):
    sample = []
    for i in range(m):
        row_sample = []
        for i in range(n):
            row_sample.append(random.randint(-q // 2 + 1, q // 2))
        sample.append(row_sample)
    return sample

def KeyGen():
    R = get_discrete_gaussian_random_matrix(n_lwe, l)
    S = get_discrete_gaussian_random_matrix(n_lwe, l)
    A = get_uniform_random_matrix(n_lwe, n_lwe)

    P = []
    for i in range(n_lwe):
        row_P = []
        for j in range(l):
            value = p * R[i][j]
            for tmp in range(n_lwe):
                value -= A[i][tmp] * S[tmp][j]
            row_P.append(value % p)
        P.append(row_P)
    return PublicKey(A, P, n_lwe, s), S

def Enc(pk, m):
    e1 = get_discrete_gaussian_random_vector(n_lwe)
    e2 = get_discrete_gaussian_random_vector(n_lwe)
    e3 = get_discrete_gaussian_random_vector(l)

    c1 = []
    for i in range(n_lwe):
        value = p * e2[i]
        for tmp in range(n_lwe):
            value += e1[tmp] * pk.A[tmp][i]
        c1.append(value)

    c2 = []
    for i in range(l):
        value = p * e3[i] + m[i]
        for tmp in range(n_lwe):
            value += e1[tmp] * pk.P[tmp][i]
        c2.append(value)
    return Ciphertext(c1, c2)

def Dec(S, c):
    m = []
    for i in range(l):
        value = c.c2[i]
        for tmp in range(n_lwe):
            value += c.c1[tmp] * S[tmp][i]
        m.append(value % p)
    return m

st = time.time()
pk, sk = KeyGen()
print("KeyGen Time: %.6f s" % (time.time() - st))

m1 = []
m2 = []
for i in range(l):
    m1.append(i*i)
    m2.append(i)

st = time.time()
c1 = Enc(pk, m1)
c2 = Enc(pk, m2)
print("Encrypt Time: %.6f ms/op" % ((time.time() - st) * 1000 / (2 * l)))

st = time.time()
c = c1 + c2
print("Add Time: %.6f ms/op" % ((time.time() - st) * 1000 / l))

st = time.time()
m = Dec(sk, c)
print("Decrypt Time: %.6f ms/op" % ((time.time() - st) * 1000 / l))
print(m)