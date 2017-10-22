#!/user/bin/python
import numpy as np
f = open("dataFile.txt", 'w')

iterations = 1
for i in range(iterations):     #all 0
    for j in range(256):
        f.write(np.array2string(np.random.choice(np.array([0]))))
    f.write('\n')

for i in range(iterations):     #15/16 0
    for j in range(256):
        f.write(np.array2string(np.random.choice(np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]))))
    f.write('\n')

for i in range(iterations):     #14/16 0
    for j in range(256):
        f.write(np.array2string(np.random.choice(np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1]))))
    f.write('\n')

for i in range(iterations):     #13/16 0
    for j in range(256):
        f.write(np.array2string(np.random.choice(np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1]))))
    f.write('\n')

for i in range(iterations):     #12/16 0
    for j in range(256):
        f.write(np.array2string(np.random.choice(np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1]))))
    f.write('\n')

for i in range(iterations):     #11/16 0
    for j in range(256):
        f.write(np.array2string(np.random.choice(np.array([0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1]))))
    f.write('\n')

for i in range(iterations):     #10/16 0
    for j in range(256):
        f.write(np.array2string(np.random.choice(np.array([0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1]))))
    f.write('\n')

for i in range(iterations):     #9/16 0
    for j in range(256):
        f.write(np.array2string(np.random.choice(np.array([0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1]))))
    f.write('\n')

for i in range(iterations):     #8/16 0
    for j in range(256):
        f.write(np.array2string(np.random.choice(np.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]))))
    f.write('\n')

for i in range(iterations):     #7/16 0
    for j in range(256):
        f.write(np.array2string(np.random.choice(np.array([0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1]))))
    f.write('\n')

for i in range(iterations):     #6/16 0
    for j in range(256):
        f.write(np.array2string(np.random.choice(np.array([0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1]))))
    f.write('\n')

for i in range(iterations):     #5/16 0
    for j in range(256):
        f.write(np.array2string(np.random.choice(np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1]))))
    f.write('\n')

for i in range(iterations):     #4/16 0
    for j in range(256):
        f.write(np.array2string(np.random.choice(np.array([0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1]))))
    f.write('\n')

for i in range(iterations):     #3/16 0
    for j in range(256):
        f.write(np.array2string(np.random.choice(np.array([0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1]))))
    f.write('\n')

for i in range(iterations):     #2/16 0
    for j in range(256):
        f.write(np.array2string(np.random.choice(np.array([0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1]))))
    f.write('\n')

for i in range(iterations):     #1/16 0
    for j in range(256):
        f.write(np.array2string(np.random.choice(np.array([0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]))))
    f.write('\n')

for i in range(iterations):     #0/16 0
    for j in range(256):
        f.write(np.array2string(np.random.choice(np.array([1]))))
    f.write('\n')

ignore = 0
for m in range(16):
    matrix = np.zeros(16).astype(np.int)
    ignore = 0
    for j in range(256):
        if (j < ignore):
            # print(j , ignore)
            continue
        elif (((j + 2) % 8 == 0)and((j + 2) % 16 != 0)and((j + 2) != 0)):
            f.write("010")
            ignore = j + 3
        else:
            for g in range(m):
                matrix[g] = 1
            f.write(np.array2string(np.random.choice(matrix)))
    f.write('\n')


# for i in range(100):
#     for j in range(256):
#         if (((j + 1) % 8 == 0)and((j + 1) % 16 != 0)and((j + 1) != 0)):
#             f.write("1")
#         else:
#             f.write(np.array2string(np.random.choice(np.array([1,0]))))
#     f.write('\n')
for i in range(1):
    for j in range(256):
        if (((j + 1) % 8 == 0)and((j + 1) % 16 != 0)and((j + 1) != 0)):
            f.write("1")
        else:
            f.write("0")
    f.write('\n')



f.close
num_lines = sum(1 for line in open('datafile.txt'))

f = open("dataFile.txt", 'r')
for i in range (num_lines - 1):
    matrix = np.zeros((16,16)).astype(np.float32)
    for j in range(16):
        for p in range(16):
            matrix[j,p] = f.read(1)
    f.read(1)
    print(matrix)
