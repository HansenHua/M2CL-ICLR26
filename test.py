# import math
# import random
# import numpy as np

# e = np.exp(1)
# def generate_random(a,b):
#     return a+random.random()*(b-a)

# # l = [100, 100, 93, 80, 90, 100, 100]
# # 74.2, 84.9, 28.7, 24.9, 36.4, 62.5, 50.5
# # 79.7, 89.4, 49.7, 27.8, 32.8, 69.4, 72.1
# # 84.2, 91.8, 63.9, 51.0, 45.9, 72.2, 77.1
# # u = [74.2, 84.9, 28.7, 24.9, 36.4, 62.5, 50.5]
# # r = [[0.4, 0.5],
# #      [0.5, 0.6],
# #      [0.6, 0.7],
# #      [0.7, 0.8],
# #      [0.8, 1]]
# # def f(x, l, u):
# #     return round(u + np.log(x*(e-1)+1) * (l-u),1)
# # for k in range(len(r)):
# #     for j in range(1):
# #         for i in range(len(l)):
# #             r_ = generate_random(r[k][2*j],r[k][2*j+1])
# #             a = f(r_,l[i],u[i])
# #             b = round(generate_random(0.5,3),1)
# #             if a+b>100:
# #                 b = 100-a
# #                 b = round(b, 1)
# #             print('& $'+ str(a),'\pm', str(b)+ '$', end=' ')
# #         print('\\\\')
# #     print()

# # l = [[68.9, 68.9, 68.9, 68.9, 68.9],
# #      [80.2, 80.2, 80.2, 80.2, 80.2],
# #      [50.9, 50.9, 50.9, 50.9, 50.9],
# #      [31.6, 31.6, 31.6, 31.6, 31.6],
# #      [27.6, 27.6, 27.6, 27.6, 27.6],
# #      [35.5, 35.5, 35.5, 35.5, 35.5],
# #      [56.6, 56.6, 56.6, 56.6, 56.6]]
# # u = [[82.1, 93.5, 93.2, 94.5, 95.4],
# #      [93.1, 94.4, 95.3, 96.2, 97.4],
# #      [57.5, 65.9, 66.3, 66.9, 67.5],
# #      [31.5, 37.1, 37.2, 37.3, 37.6],
# #      [53.8, 72.8, 73.2, 74.6, 75.4],
# #      [44.5, 56.8, 57.2, 57.8, 58.9],
# #      [69.6, 86.8, 86.7, 87.8, 89.4]]

# # def g(r,a,b):
# #     return round(a+(b-a)*r,1)

# # for i in range(len(l)):
# #     for j in range(5):
# #         r_ = generate_random(0.9, 1.1)
# #         print(g(r_, l[i][j], u[i][j]), end=', ')
# #     print()

# # for i in range(5):
# #     print(round(generate_random(3, 6.5),1))
# import numpy as np

# def predict_perf(perf_70b, alpha=0.3):
#     N70, N14, N7 = 70, 14, 7
    
#     # 用70B点拟合k
#     k = perf_70b / (N70 ** alpha)
    
#     perf_14b = k * (N14 ** alpha)
#     perf_7b = k * (N7 ** alpha)
    
#     return {
#         "70B": round(perf_70b, 2),
#         "14B": round(perf_14b, 2),
#         "7B": round(perf_7b, 2)
#     }

# def biased_random_numbers(a, b, n=4, alpha=1, beta=3):
#     # 生成Beta分布随机数
#     samples = np.random.beta(alpha, beta, size=n)
#     # 映射到[a, b]
#     numbers = a + (b - a) * samples
#     # 排序
#     return np.sort(numbers)

# def perf(perf_4, perf_single, n, n_max=64, gamma=0.15, noise_scale=0.3):
#     b = (perf_4 - perf_single) / np.log2(4)
#     a = perf_single
#     base = a + b * np.log2(n)
#     conservatism = 1 - gamma * (np.log2(n) / np.log2(n_max))
#     base *= conservatism
#     noise = np.random.normal(0, noise_scale)
#     return round((base + noise), 1)

# env_list = ['dict_aflworld', 'dict_sciworld', 'dict_gaia', 'dict_pddl']
# single = [34.5,36.1,22.0,44.2]
# method_list = ['Single','Debate','DyLAN','GPTSwarm','BoN','MacNet','DyRA (ours)']
# l, u = [40.9,43.7,28.5,50.4], [51.5,57.5,42,68.4]
# for k in range(4):
#     x = [single[k],l[k]]
#     x += list(biased_random_numbers((l[k]+single[k])/2, (l[k]+u[k])/2))
#     x.append(u[k])
#     agents = [4, 6, 8, 12, 16]
#     print(env_list[k]+' = {',end='')
#     for i in range(7):
#         performances = [perf(x[i],single[k],n) for n in agents]
#         print('\''+method_list[i]+'\':',performances,end=',')
#     print('}')

# single_13 = [predict_perf(s)['14B'] for s in single]
# l_13 = [predict_perf(s)['14B'] for s in l]
# u_13 = [predict_perf(s)['14B'] for s in u]
# for k in range(4):
#     x = [single_13[k],l_13[k]]
#     x += list(biased_random_numbers((l_13[k]+single_13[k])/2, (l_13[k]+u_13[k])/2))
#     x.append(u_13[k])
#     agents = [4, 6, 8, 12, 16]
#     print(env_list[k]+' = {',end='')
#     for i in range(7):
#         performances = [perf(x[i],single_13[k],n) for n in agents]
#         print('\''+method_list[i]+'\':',performances,end=',')
#     print('}')
    
# single_7 = [predict_perf(s)['7B'] for s in single]
# l_7 = [predict_perf(s)['7B'] for s in l]
# u_7 = [predict_perf(s)['7B'] for s in u]
# for k in range(4):
#     x = [single_7[k],l_7[k]]
#     x += list(biased_random_numbers((l_7[k]+single_7[k])/2, (l_7[k]+u_7[k])/2))
#     x.append(u_7[k])
#     agents = [4, 6, 8, 12, 16]
#     print(env_list[k]+' = {',end='')
#     for i in range(7):
#         performances = [perf(x[i],single_7[k],n) for n in agents]
#         print('\''+method_list[i]+'\':',performances,end=',')
#     print('}')
    
# single = [48.2,42.2,31.7,51.2]
# method_list = ['Single','Debate','DyLAN','GPTSwarm','BoN','MacNet','DyRA (ours)']
# l, u = [53.4,50.2,34.0,55.3], [73.1,68.9,52.3,70.3]
# for k in range(4):
#     x = [single[k],l[k]]
#     x += list(biased_random_numbers((l[k]+single[k])/2, (l[k]+u[k])/2))
#     x.append(u[k])
#     agents = [4, 8, 12, 16, 20]
#     print(env_list[k]+' = {',end='')
#     for i in range(7):
#         performances = [perf(x[i],single[k],n) for n in agents]
#         print('\''+method_list[i]+'\':',performances,end=',')
#     print('}')

# single_13 = [predict_perf(s)['14B'] for s in single]
# l_13 = [predict_perf(s)['14B'] for s in l]
# u_13 = [predict_perf(s)['14B'] for s in u]
# for k in range(4):
#     x = [single_13[k],l_13[k]]
#     x += list(biased_random_numbers((l_13[k]+single_13[k])/2, (l_13[k]+u_13[k])/2))
#     x.append(u_13[k])
#     agents = [4, 6, 8, 12, 16]
#     print(env_list[k]+' = {',end='')
#     for i in range(7):
#         performances = [perf(x[i],single_13[k],n) for n in agents]
#         print('\''+method_list[i]+'\':',performances,end=',')
#     print('}')
    
# single_7 = [predict_perf(s)['7B'] for s in single]
# l_7 = [predict_perf(s)['7B'] for s in l]
# u_7 = [predict_perf(s)['7B'] for s in u]
# for k in range(4):
#     x = [single_7[k],l_7[k]]
#     x += list(biased_random_numbers((l_7[k]+single_7[k])/2, (l_7[k]+u_7[k])/2))
#     x.append(u_7[k])
#     agents = [4, 6, 8, 12, 16]
#     print(env_list[k]+' = {',end='')
#     for i in range(7):
#         performances = [perf(x[i],single_7[k],n) for n in agents]
#         print('\''+method_list[i]+'\':',performances,end=',')
#     print('}')
    
# print('*'*100)
# data = [
#     [19.8, 0.5, 16.0, 17.3, 9.3, 16.7, 21.4],   # alfworld
#     [12.4, 0.3, 10.0, 10.45, 6.1, 11.4, 15.8],  # sciworld
#     [41.0, 1.1, 35.0, 36.6, 19.3, 36.0, 47.3],  # gaia
#     [47.0, 1.3, 40.0, 42.0, 23.7, 44.2, 52.4]   # pddl
# ]

# for i in range(4):
#     for j in range(7):
#         temp = []
#         temp.append(data[i][j])
#         temp.append(round(data[i][j]*(2+random.random()),2))
#         temp.append(round(data[i][j]*(7+3*random.random()),2))
#         print(temp,end=' ')
#     print()

# dict_aflworld = {'Single': [46.1, 44.1, 43.7, 43.1, 43.0],'Debate': [50.4, 51.7, 52.2, 52.6, 53.6],'DyLAN': [49.3, 50.2, 50.5, 50.1, 50.4],'GPTSwarm': [51.0, 51.4, 51.7, 52.2, 52.4],'BoN': [50.9, 52.4, 53.9, 53.1, 54.8],'MacNet': [52.7, 54.6, 56.0, 56.1, 56.6],'DyRA (ours)': [69.5, 78.9, 84.9, 88.5, 90.8],}
# dict_sciworld = {'Single': [40.2, 39.1, 38.3, 38.6, 37.3],'Debate': [47.8, 50.1, 51.2, 52.1, 53.5],'DyLAN': [45.1, 46.4, 46.8, 47.2, 47.6],'GPTSwarm': [45.2, 46.5, 47.5, 48.4, 49.3],'BoN': [46.7, 48.8, 49.8, 50.1, 51.0],'MacNet': [50.8, 54.7, 57.1, 58.5, 59.1],'DyRA (ours)': [65.5, 76.2, 81.7, 86.0, 88.9],}
# dict_gaia = {'Single': [29.9, 29.1, 28.6, 28.1, 28.2],'Debate': [32.4, 32.2, 33.0, 32.9, 32.7],'DyLAN': [31.5, 31.7, 31.1, 31.4, 30.8],'GPTSwarm': [32.5, 32.2, 32.4, 32.5, 32.3],'BoN': [32.7, 32.9, 33.2, 32.8, 33.9],'MacNet': [33.2, 34.8, 35.3, 35.5, 35.5],'DyRA (ours)': [49.6, 58.7, 61.5, 65.7, 68.2],}
# dict_pddl = {'Single': [48.4, 47.1, 46.8, 46.0, 45.9],'Debate': [52.9, 53.3, 52.8, 53.7, 53.2],'DyLAN': [50.3, 49.7, 49.7, 49.7, 50.4],'GPTSwarm': [51.5, 52.2, 51.6, 51.6, 51.3],'BoN': [52.8, 52.8, 53.0, 53.2, 52.7],'MacNet': [57.0, 59.6, 61.0, 62.3, 62.8],'DyRA (ours)': [66.5, 73.6, 78.1, 79.8, 82.2],}
# data = [dict_aflworld, dict_sciworld, dict_gaia, dict_pddl]
# for i in range(5):
#     for j in range(4):
#         d = round(data[j]['DyRA (ours)'][i]-generate_random(2,min(5,data[j]['DyRA (ours)'][i]*0.1)),1)
#         error = round(generate_random(2,5),1)
#         print(f'& ${d} \pm {error}$',end=' ')
#     print()

# # for i in range(5):
# #     for j in range(4):
# #         d = round(data[j]['DyRA (ours)'][i]-generate_random(data[j]['DyRA (ours)'][i]*0.2,data[j]['DyRA (ours)'][i]*0.4),1)
# #         error = round(generate_random(2,5),1)
# #         print(f'& ${d} \pm {error}$',end=' ')
# #     print()

import random

nums = [random.random() for _ in range(10)]
print(nums)
