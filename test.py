# import torch
# a = list()
# # a = [(1, 2, 5, 6), (3, 4, 7, 10), (5, 6, 8, 19)]
# # b, c = zip(*list(map(lambda x: (x[0], x[3]), a)))
# # print(b, c)
# a = [(torch.rand(1,3), torch.rand(1, 3), torch.rand(1, 3)), (torch.rand(1,3), torch.rand(1, 3), torch.rand(1, 3)), (torch.rand(1,3), torch.rand(1, 3), torch.rand(1, 3))]
# print(a)
# b, c = zip(*list(map(lambda x: (x[0], x[2]), a)))
# print(b)
# print('==============')
# print(c)
# j = 0
# z = 0
# a = list()
# a = [1, 2, 3, 4, 5, 6]
# b = []
# for i in range(100):
#     b.append(i)
# from tqdm import tqdm
# for i in enumerate(a):
#     z += 1
#     for i, j in tqdm(enumerate(b) , desc='Training epoch ' + str(z) + ''):
#
#         n = 1000000
#         while n >= 0:
#             j += 1
#             n -= 1

a = {}
a['a'] = 1
a['b'] = 2
a['c'] = 3
print(type(a.keys()))
for key in a.keys():
    print(a[key])



