import numpy as np

a = np.random.randint(0,2,(2,2,3))
print(a)
print("=======")
# print(a.reshape(-1,3))
a = a.reshape(-1,3)
for i in a:
    print()
print("=======")
b = np.random.randint(0,2,(2,3))
print(b)
print("=======")

print(np.argmin(np.sum(np.square(a-b), axis=2)))
print(np.argmin(np.sum(np.square(a-b), axis=2), axis=1))
