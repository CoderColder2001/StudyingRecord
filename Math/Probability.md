[TOC]

------
### 条件概率的一般形式
$P(A,B,C) = P(C|B,A) P(B,A) = P(C|B,A) P(B|A) P(A)$  
$P(B,C|A) = P(B|A) P(C|A,B)$

### 基于马尔可夫假设的条件概率
如果满足马尔可夫链关系 $A\rightarrow B \rightarrow C$：  
$P(A,B,C) = P(C|B,A) P(B,A) = P(C|B) P(B|A) P(A)$  
$P(B,C|A) = P(B|A) P(C|B)$

------
## 高斯分布
### 两个单一变量高斯分布间的KL散度
$KL(p,q) = log\frac{\sigma_1}{\sigma_2} + \frac{\sigma^2+(\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$  

<br>

### 参数重整化
若希望从高斯分布$N(\mu,\sigma)$中采样，可以先从标准分布$N(0,1)$采样出$z$，再得到 $\sigma*z+\mu$  
从而 <b>将随机性（采样）转移到 $z$ （常量）上，而 $\mu$ 和 $\sigma$ 作为仿射变换网络的一部分（保持采样出的值对于 $\mu$ 和 $\sigma$ 可以求导）</b>  

<br>

### 多元正态分布
由均值向量$\mu$和协方差矩阵$\sum$定义  
$p(x)=|2\pi\sum|^{-1/2}exp(-\frac{1}{2}(x-\mu)^T\sum^{-1}(x-\mu))$  

对协方差矩阵应用变换A，$\sum'=A\sum A^T$   
（任意一个协方差矩阵都可以视为线性变换的结果）  

多元正态分布的概率密度由协方差矩阵的特征向量控制旋转（rotation），特征值控制尺度（scale）  

