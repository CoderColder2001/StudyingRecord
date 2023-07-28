# Colmap3.8 编译与使用
<a href = "https://colmap.github.io/">Colmap 官方文档</a>    
## Content
- VS2019+CMAKE构建Colmap3.8
- 使用Colmap得到稀疏点云与相机参数

<br>

------
## VS2019 + CMAKE构建Colmap3.8
参考博客：
- <a href = "https://blog.csdn.net/qq_33765199/article/details/110452571">关于我想在windows上编译colmap这件事</a>
- <a href = "https://blog.csdn.net/cxy_hust/article/details/116892579">windows下编译colmap3.6[colmap][gmp][mfpr]</a>
- <a href = "https://blog.csdn.net/qq_41102371/article/details/115288530">用cmake在win10配置colmap</a>

ceres编译参考 <a href = "https://blog.csdn.net/qq_40957243/article/details/122902186">windows下编译、配置ceres库（保姆级教程</a>  

<br>

------
## 使用Colmap得到稀疏点云与相机参数
tips：注意不要在colmap的参数界面上用鼠标滚轮...  

在test_colmap文件夹下新建一个 `database.db`文件 与一个 `images`文件夹，放入一些图片；  
colmap界面下新建一个project  
Processing-》Feature extraction 进行特征点提取，这里先采用默认参数  
**TODO：参数解析**   
Processing-》Feature matching 进行特征点匹配，这里先采用默认参数

Reconstruction-》Start reconstruction 进行稀疏重建（SFM）  
<img src = ".\pic\colmap_1.png" width = "80%">

在test_colmap文件夹下新建一个 `sparse`文件夹  
Reconstruction-》Dense reconstruction &emsp; select 选择`sparse`文件夹，用以保存重建结果，依次点击Undistortion（图像去畸变）, Stereo（立体匹配 估计相机位姿？深度估计）； 不用Fusion（点云深度图融合 稠密重建？）  

File-》export model as text 选择`sparse\sparse`文件夹，以txt格式导出sfm相关的文件  
Q：去畸变后，如何导出相机为PINHOLE模型？
