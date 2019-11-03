# Music Recommendation based on Collaborative Filtering

## 哈工大-数据挖掘课程实验（一）

## 1. 任务目标与系统背景

### 1.1 协同过滤系统（Collaborative filtering systems）

计算用户或者项之间的相似性来推荐项，与某用户相似的用户所喜欢的项会推荐给该用户。

### 1.2 效用矩阵（Utility Matrix）

* 两类元素：用户（user）、项（item）；
* 矩阵中每个“用户-项”所对应的值代表当前用户对此项的喜好程度；
* 通常该矩阵是稀疏的；
* 推荐系统的目标是预测矩阵的空白元素；
* 实践中，只需给出每一行中某些评级可能较高的元素即可。

### 1.3 任务目标

* 基于UV分解，建立协同过滤模型（矩阵分解的代码要自己编写）；
* 在user_artist_data中，预留20%的数据，作为验证集；
* 计算模型对验证集进行预测的结果的RMSE。

## 2. 数据集简介

* user_artist_data.txt：2420万条用户播放艺术家歌曲次数
* artist_data.txt: 160+万个艺术家的ID和名字
* artist_alias.txt: 拼写错误的艺术家ID（变体）到该艺术家的规范ID的映射关系（19万条记录）

## 3. 算法介绍

基于模型的协同过滤算法：

矩阵因子分解将项和用户都转化成相同的潜在空间，即用户偏好矩阵分解成一个用户-潜在因子矩阵乘以一个潜在因子-项矩阵，代表用户和项之间的潜相互作用。

$$
\left[\begin{array}{lllll}{5} & {2} & {4} & {4} & {3} \\ {3} & {1} & {2} & {4} & {1} \\ {2} & {?} & {3} & {1} & {4} \\ {2} & {5} & {4} & {3} & {5} \\ {4} & {4} & {5} & {4} & {?}\end{array}\right]=\left[\begin{array}{ll}{u_{11}} & {u_{12}} \\ {u_{21}} & {u_{22}} \\ {u_{31}} & {u_{32}} \\ {u_{41}} & {u_{42}} \\ {u_{51}} & {u_{52}}\end{array}\right] \times\left[\begin{array}{ccccc}{v_{11}} & {v_{12}} & {v_{13}} & {v_{14}} & {v_{15}} \\ {v_{21}} & {v_{22}} & {v_{23}} & {v_{24}} & {v_{25}} \end{array}\right]
$$

如上例，即将$5*5$矩阵分解为两个$5*2$矩阵和$2*5$矩阵,k=2即为潜在因子数。

评价指标：
均方根误差：$Root-mean-square error (RMSE)$
$$ \sqrt{  \frac1N \sum_{xi}(r_{xi} - \hat{r_{xi}} )^2  } $$

则矩阵因子分解算法的目标为最小化RMSE，或最小化 
$$ \sum{e^2} = \sum_{xi}(r_{xi} - \hat{r_{xi}} )^2$$

### 3.1 UV 分解（增量式计算）

将原始评分矩阵$M_{m*n}$分解为$U_{m*d}、V_{d*n}$，两个矩阵。即有：
$$ M_{m*n} \approx U_{m*d}、V_{d*n} = P_{m*n}  $$
其中d为潜在因子，为超参数，取值通常与项的种类数有关。

#### 求解UV

对$u_{rs}=x$进行变化，使得𝑀和𝑈𝑉间𝑅𝑀𝑆𝐸最小:
$$ p_{rj} = \sum^{d}_{k=1} u_{rk}v_{kj} $$
该元素对误差平方和贡献为：
$$ (m_{rj}-p_{rj})^2 = ( m_{rj} - \sum_{k\neq s}u_{rk}v_{kj} - xv_{sj} )^2 $$
对所有非空𝑚_𝑟𝑗在𝑗上求和：
$$ \sum_{j} ( m_{rj} - \sum_{k\neq s}u_{rk}v_{kj} - xv_{sj} )^2 $$
对上式求导并令其等于0，整理得：
$$ x = \frac{ \sum_{j} v_{sj}( m_{rj} - \sum_{k\neq s}u_{rk}v_{kj})  }
     { \sum_{j} v_{sj}^2 }  $$

类似的，对𝑣_𝑟𝑠=𝑦进行改变，则使RMSE最小的𝑦值：
$$ y=\frac{\sum_{i} u_{i r}\left(m_{i s}-\sum_{k \neq r} u_{i k} v_{k s}\right)}{\sum_{i} u_{i r}^{2}} $$

显然，对U、V对遍历修改复杂度很高。
当分别先后计算U、V矩阵时，单个矩阵内当不同行（列）间当计算互不影响，故可以考虑并行化加速计算。

### 3.2 梯度下降法(Gradient descent)

寻找 $P_{m*k}、Q_{k*n}$ 满足：
$$ M_{m*n} \approx P_{m*k}、Q_{k*n} = {\hat{M}}_{m*n}  $$
与3.1具体求解方式不同。

定义损失函数：
$$ e_{i j}^{2}=\left(r_{i j}-\hat{r}_{i j}\right)^{2}=\left(r_{i j}-\sum_{k=1}^{K} p_{i k} q_{k j}\right)^{2} $$

为了防止过拟合，加入（L2）正则化项：
$$ e_{i, j}^{2}=\left(r_{i, j}-\sum_{k=1}^{K} p_{i, k} q_{k, j}\right)^{2}+\frac{\beta}{2} \sum_{k=1}^{K}\left(p_{i, k}^{2}+q_{k, j}^{2}\right) $$

使用梯度下降法获得修正的p和q分量：

求解损失函数的负梯度：
$$ \begin{array}{l}{\frac{\partial}{\partial p_{i, k}} E_{i, j}^{2}=-2\left(r_{i, j}-\sum_{k=1}^{K} p_{i, k} q_{k, j}\right) q_{k, j}+\beta p_{i, k}=-2 e_{i, j} q_{k, j}+\beta p_{i, k}} \\ {\frac{\partial}{\partial q_{k, j}} E_{i, j}^{2}=-2\left(r_{i, j}-\sum_{k=1}^{K} p_{i, k} q_{k, j}\right) p_{i, k}+\beta q_{k, j}=-2 e_{i, j} p_{i, k}+\beta q_{k, j}}\end{array} $$

根据负梯度的方向更新变量：
$$ \begin{array}{l}{p_{i, k}^{\prime}=p_{i, k}-\alpha\left(\frac{\partial}{\partial p_{i, k}} e_{i, j}^{2}+\beta p_{i, k}\right)=p_{i, k}+\alpha\left(2 e_{i, j} q_{k, j}-\beta p_{i, k}\right)} \\ {q_{k, j}^{\prime}=q_{k, j}-\alpha\left(\frac{\partial}{\partial q_{k, j}} e_{i, j}^{2}+\beta q_{k, j}\right)=q_{k, j}+\alpha\left(2 e_{i, j} p_{i, k}-\beta q_{k, j}\right)}\end{array} $$
其中$p_{i, k}^{\prime}，q_{k, j}^{\prime}$即为更新后对值。

迭代直到算法最终收敛。

### 3.3 \* *ALS 算法*

## 4. 算法实现与过程优化

本项目分别实现了**UV分解算法**和**梯度下降法**。

### 4.0 \*数据预处理

1. 首先统一解析别名artist_alias中的对应关系，对训练数据进行预处理。
2. 考虑到训练集对应评分矩阵为10+w*100+w量级的巨大稀疏矩阵，故按照（坐标，评分）的形式key-value对存储为Map，以节约存储空间。（实验表明，按矩阵存储可能引起内存爆炸💥）

3. 评分标准化。数据集中的score即为用户对特定歌曲对播放次数。
   考虑到用户听音乐时可能发生的长时间单曲循环，或由于忘记关闭导致对大量播放对矩阵产生对影响；同时避免归一化降低音乐发烧友对音乐的评价影响力，算法中对用户听歌次数对数化作为用户评分。

### 4.1 UV分解增量计算算法实现

以计算矩阵U为例，根据公式，在计算$u_{rj}$时，会重复使用分母$\sum_{j} v_{sj}^2$，为了减少重复计算，可先计算出对不同s(s in K)。

```python
down_sum_vsj = [ sum( [ (v[s,j])**2 for j in range(n) ] ) for s in range(K) ]
```

此任务为计算密集型任务，为了提高计算效率，考虑多进程并行化计算（由于GIL限制，不推荐python多线程并发）。

```python
if __name__ == "__main__":
    import multiprocessing
    pal = 5
    pool = multiprocessing.Pool(processes = pal)

    # cal U
    for r in range(0,m,pal):
        pool.map(cal_U, range(r, min(r+pal,m)) )
        print( f"processing {r}~{r+10} in U completed.", end="\r" )
```

上例代码即为 5个进程同时并发计算，每次计算5行的代码事例，其中cal_U函数即为计算一行的函数。

### 4.2 梯度下降算法实现

根据3.2中迭代更新公式，对矩阵中有效位置进行遍历，计算与有效为相关的P、Q矩阵相关向量，核心迭代公式为：

```python
p[i,:] = p[i,:] + a*(eij*q[:,j] - b*p[i,:])
q[:,j] = q[:,j] + a*(eij*p[i,:] - b*q[:,j])
```

对模型学习率动态调整测试，由下图可见10轮迭代后收敛。
![损失 - 累计学习率](https://upload-images.jianshu.io/upload_images/15003357-7399f04c1bfecdfa.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/300)

## 5. 模型评价

### 4.1 UV分解增量计算法

由于算法计算复杂度过高，在有限时间内未完成训练。

### 4.2 梯度下降法

对测试集每一项，通过P、Q对应向量内积求得评分 s ，再求 $pre=2^s$ ，则pre即为预测听歌次数，对应于原始数据集。根据 3 中RMSE的计算，求得对20%测试集的RMSE为：
$$ RMSE = 75.39 $$

#### 推荐结果

以第100000位用户为例，该用户最喜欢的艺术家为：

```python
sample = { key[1]:value for key, value in test.items() if user_id_index[key[0]] == 100000 }

sample
{'1004226': 1, '1151014': 1, '1278': 13, '4267': 28, '5833': 8}
```

即id为4267的艺术家，根据artist_data，该艺术家为'Green Day'：
`绿日（Green Day），美国朋克乐队，2013年，获得第20届MTV欧洲音乐奖最佳摇滚艺人。2015年，入驻第30届美国摇滚名人堂。 -- 百度百科`
为知名摇滚乐队。

对第100000位用户的推荐结果为56275号艺术家'Kettcar'：
`Kettcar，德国摇滚乐队。 -- 酷狗音乐`

可见，该模型在一定程度上成功为用户推荐了喜欢的摇滚领域的艺术家。
