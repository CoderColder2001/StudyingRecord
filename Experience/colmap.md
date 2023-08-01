# Colmap3.8 编译与使用
<a href = "https://colmap.github.io/">Colmap 官方文档</a>    
## Content
- VS2019+CMAKE构建Colmap3.8
- 使用Colmap得到稀疏点云与相机参数
- Colmap源码

<br>

------
## VS2019 + CMAKE构建Colmap3.8
参考博客：
- <a href = "https://blog.csdn.net/qq_33765199/article/details/110452571">关于我想在windows上编译colmap这件事</a>
- <a href = "https://blog.csdn.net/cxy_hust/article/details/116892579">windows下编译colmap3.6[colmap][gmp][mfpr]</a>
- <a href = "https://blog.csdn.net/qq_41102371/article/details/115288530">用cmake在win10配置colmap</a>

ceres编译参考 <a href = "https://blog.csdn.net/qq_40957243/article/details/122902186">windows下编译、配置ceres库（保姆级教程</a>  

TODO：vs编译后 各部分位置？  

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
undistortion结束后 需要先import加载 再导出  

<br>

------
## Colmap源码
### colmap_exe -> colmap.cc
main函数所在；进行命令行与调用函数的匹配  

### colmap_exe -> gui.cc
确认qt版本 调用gui（colmap_sources -> ui/mainwindow.cc）  
mainwindow的构造参数为`const OptionManager& options`   
mainwindow 有一个 `ThreadControlWidget` 好像是导入、导出模型相关的线程与窗口  
`CreateWidgets()`：创建工作流各步骤对应的一系列窗口，并把`ModelViewerWidget` 作为当前显示的widget   
每个widget构造都接受一个`const OptionManager& options`参数   
`options`也管理执行当前步骤的一些命令行参数  
关联信号槽 打开对应的窗口   
有一些action被加入到blocking_actions_里 在重建过程中这些action被置为disabled  
   
