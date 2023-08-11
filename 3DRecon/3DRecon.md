# 三维重建
------
## Content
- SFM
- MVS

<br>

------
## MVS
<b>求深度图 -> 点云融合</b>  

对每一张去除畸变后的图像（ref），由sfm获取了一系列已排好序的近邻图像（src） ，算法可参考 `colmap2mvsnet_acm.py`   

先对ref图像随机初始化深度（范围在稀疏点云投影到这个相机的min_depth与max_depth间）  
小平面假设：`fp(n, d)`  

### patchmatch 寻找图像之间（特征空间上）的最近邻  
可参考：
- <a href = "https://zhuanlan.zhihu.com/p/377230002"> 解读PatchMatch: A Randomized Correspondence Algorithm for Structural Image Editing </a> 
- <a href = "https://zhuanlan.zhihu.com/p/562577568">78. 三维重建13-立体匹配9，经典算法PatchMatchStereo</a> 

PatchMatch与ADCensus等算法不一样的地方的地方在于，视差的计算不是直接进行的，而是通过平面参数计算而来的——立体匹配的过程；不是在一维的水平极线上进行搜索得到视差值，而是在3D空间中搜索最佳平面，再通过平面参数反算出视差。这意味着在匹配过程中就可以得到亚像素精度的视差值和正确的平面  

在深度计算上，利用匹配传递的思想 取邻居的`fp(n, d)`，看用邻居的和自己当前的来计算 哪个cost更小  