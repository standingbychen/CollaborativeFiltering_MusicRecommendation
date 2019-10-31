import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# alias_map = np.load("obj/alias_map.npy",allow_pickle=True).item()

"""
记录id到矩阵索引的映射
"""
user_id_index = {}
user_cnt = 0 # index 计数
artist_id_index = {}
artist_cnt = 0 # index 计数



# # 测试集保留
# test = {}  #（ (user,artist) : score )
# # 训练集记录
# train_map = {}  # ((user_index,artist_index) : score)

# # 构造训练集和测试集
# import random
# def add_item(user, artist, score):
#     """
#     需要 alias_map
#     20%几率将数据置0，作为测试集
#     """
#     # artist 别名规范化
#     artist = alias_map.get(artist,artist)
    
#     # 测试集：从训练集中随机抽取20%记录score作为测试，并置score=0写入训练集
#     import random
#     if random.random() < 0.2:
#         test[(user,artist)] = score
#         score = 0
        
#     # 训练集
#     '''
#     user in index ? index : new index, new row
#     artist in index ? index : new index, new col
#     '''
#     global user_cnt
#     user_index=user_cnt
#     if user in user_id_index:
#         user_index = user_id_index[user]
#     else:
#         user_id_index[user] = user_index
#         user_cnt +=1
        
#     global artist_cnt
#     artist_index=artist_cnt
#     if artist in artist_id_index:
#         artist_index = artist_id_index[artist]
#     else:
#         artist_id_index[artist] = artist_index
#         artist_cnt +=1
    
#     return (user_index,artist_index),score


# # 读取文件
# n=0
# eno =0

# with open("user_artist_data.txt",encoding="utf8") as f:
#     # 文件总计2400+w行
#     for l in f:
#         n=n+1
#         m = re.match(r"(\d+) (\d+) (\d+)",l)
#         eles = m.groups() if m is not None else () 
#         if len(eles) is 3:
#             key,value = add_item(eles[0], eles[1], int(eles[2]))
#             train_map[key] = value
#         else:
#             eno+=1
# #             print(str(n)+" - error: "+l)
#         if n%100000 is 0:
#             print(f"compete {n}", end="\r")

# print("error count : "+str(eno))
# print("lines count : "+str(n))





# index = pd.DataFrame( [ [_[0],_[1]] for _ in train_map.keys()] , columns=['user','item'])
# unpop_music = set( index["item"].value_counts()[ index["item"].value_counts() <= 2].index )
# train_map_pop = {key: value for key, value in train_map.items() if key[1] not in unpop_music}

# np.save("obj/train_map_pop.npy", train_map_pop)

train_map_pop = np.load("obj/train_map_pop.npy", allow_pickle=True).item()
user_cnt = len( set( [ _[0] for _ in train_map_pop.keys() ] ) ) 
artist_cnt_pop = len( set( [ _[1] for _ in train_map_pop.keys() ] ) ) 


K = 5  # 隐含主题个数，经验值

m,n = user_cnt,artist_cnt_pop

# 初始化 U，V 矩阵为1
global u,v
u = np.ones((m,K))
v = np.ones((K,n))


"""
计算 U
"""

# 减少重复计算，先存储分母
down_sum_vsj = [ sum( [ (v[s,j])**2 for j in range(n) ] ) for s in range(K) ]

for r in range(m):
    for s in range(K):
        # 遍历U每个位置
        print( f"processing {r},{s} in U .", end="\r" )
        up_sum = 0
        for j in range(n):
            value = train_map_pop.get( (r,j), 0 )
            if value > 0:
                uv_sum = sum([ u[r,k]* v[k,j] for k in range(K) if k!=s ])
                up_sum += v[s,j] * ( value - uv_sum )
        u[r,s] = up_sum / down_sum_vsj[s] if down_sum_vsj[s] >0 else 0
        
np.save("obj/u.npy",u)
print("processing U completed.")

# def cal_U(r):
#     for s in range(K):
#         # 遍历U每个位置
#         print( f"processing {r},{s} in U.\t", end="\r" )
#         up_sum = 0
#         for j in range(n):
#             value = train_map.get( (r,j), 0 )
#             if value > 0:
#                 uv_sum = sum([ u[r,k]* v[k,j] for k in range(K) if k!=s ])
#                 up_sum += v[s,j] * ( value - uv_sum )
#         u[r,s] = up_sum / down_sum_vsj[s] if down_sum_vsj[s] >0 else 0




"""
计算 V
"""

# 减少重复计算，先存储分母
down_sum_uir = [ sum( [ (u[i,r])**2 for i in range(m) ] ) for r in range(K) ]
for r in range(K):
    for s in range(n):
        # 遍历V每个位置
        print( f"processing {r},{s} in V ", end="\r" )
        up_sum = 0
        for i in range(m):
            value = train_map_pop.get( (i,s), 0 )
            if value > 0:
                uv_sum = sum([ u[i,k]* v[k,s] for k in range(K) if k!=r ])
                up_sum += u[i,r] * ( value - uv_sum )
        v[r,s] = up_sum / down_sum_uir[r] if down_sum_uir[r]>0 else 0
        
np.save("obj/v.npy",v)
print("processing V completed.")


# if __name__ == "__main__":
#     import multiprocessing

#     pool = multiprocessing.Pool(processes = 10)
#     for r in range(0,m,10):
#         tasks = pool.map(cal_U, range(r,r+10))
#         print( f"processing {r}~{r+20} in U completed.", end="\r" )
        