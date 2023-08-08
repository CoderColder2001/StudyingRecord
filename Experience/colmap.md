# Colmap3.8 编译与使用
<a href = "https://colmap.github.io/">Colmap 官方文档</a>    
## Content
- VS2019+CMAKE构建Colmap3.8
  - Qt 5.15.2
  - miniconda + python3.9 
  - boost 编译 
  - GMP、MPFR、CGAL
  - ceres 编译
  - 其他依赖项
- 使用Colmap得到稀疏点云与相机参数
- Colmap源码

<br>

------
## VS2019 + CMAKE构建Colmap3.8
参考博客：
- <a href = "https://blog.csdn.net/qq_33765199/article/details/110452571">关于我想在windows上编译colmap这件事</a>
- <a href = "https://blog.csdn.net/cxy_hust/article/details/116892579">windows下编译colmap3.6[colmap][gmp][mfpr]</a>
- <a href = "https://blog.csdn.net/qq_41102371/article/details/115288530">用cmake在win10配置colmap</a>

按理来说使用bulid.py脚本编译colmap 指定一些库的路径 并会自动下载其他依赖的库  
```
python scripts/python/build.py --build_path F:/Workspace/colmap/build --colmap_path F:/Workspace/colmap --boost_path "F:/Workspace/ThirdParty/boost_1_78_0/msvc-142_64" --qt_path "F:/Qt/5.15.2/msvc2019_64" --cuda_path "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7" --cgal_path "F:/Workspace/ThirdParty/CGAL-5.4/CGAL-5.4" --cmake_generator "Visual Studio 16"
```
但我在py脚本编译时有问题 可能是出现在编译ceres或编译suitesparse的过程中，改为使用cmake来从每一个依赖开始逐个编译  

注意cmake完了要以管理员身份运行sln  
ALL_BUILD 生成所有项目，但不包括install和单元测试等。  
INSTALL 把cmake脚本里install指令指定的东西安装到CMAKE_INSTALL_DIR里面  

### Qt 5.15.2
QT切换清华镜像源安装  
命令行运行 
``` cmd
qt-unified-windows-x64-4.6.0-online.exe --mirror https://mirrors.tuna.tsinghua.edu.cn/qt  
```

### miniconda + python3.9
注意python版本与cuda、之后其他地方可能会用到的pytorch等的兼容性  

### boost 编译
参考 <a href = "https://zhuanlan.zhihu.com/p/85806857">Boost编译与使用</a>  
注意boost版本和vs版本的兼容性（发行日期）boost1.66在vs2019中不太兼容，后来换成了1.78   

解压后执行booststrap.bat脚本，之后会生成b2.exe和bjam.exe两个可执行文件，在编译时只用到了b2.exe  

``` cmd
b2.exe install --toolset=msvc-14.2 --prefix="F:\Workspace\ThirdParty\boost_1_78_0\msvc-142_64" link=static runtime-link=shared threading=multi architecture=x86 address-model=64 debug release --with-iostreams -s ZLIB_INCLUDE="F:\Workspace\ThirdParty\zlib\include" -s ZLIB_LIBPATH="F:\Workspace\ThirdParty\zlib\lib"
```

### GMP、MPFR、CGAL
参考 <a href = "https://blog.csdn.net/jialong_chen/article/details/115486598">CGAL-5.2.1的安装与编译</a>  
v5.0后的CGAL是header-only library 无需build&install  

### ceres 编译
ceres编译参考 <a href = "https://blog.csdn.net/qq_40957243/article/details/122902186">windows下编译、配置ceres库（保姆级教程）</a>   

### 其他依赖项
flann1.9.2需要pkg config  
参考<a href = "https://blog.csdn.net/LuckyHanMo/article/details/125471360">Windows环境下安装pkg-config</a>后依旧没有解决...  
改用1.9.1 参照<a href = "https://blog.csdn.net/m0_37829462/article/details/124897154">Win10系统VS2019+Cmake+flann_1.9.1环境配置</a>  

glew不要用带s的lib（静态链接版）  
 
cuda工具报错 参考<a href = "https://blog.csdn.net/weixin_43860261/article/details/123420492">【CUDA】No CUDA toolset found</a>  

metis配置 参考<a href = "https://blog.csdn.net/xuchanghui113/article/details/127581383">在Win10系统下使用与安装metis</a>  

vs报错glog找不到gflags： 拷贝include lib bin对应内容到glogLib文件夹  

sqlite3 参考<a href = "https://blog.csdn.net/m0_37909265/article/details/105102982">如何在Windows 10上构建SQLite3.lib文件</a>  
60kb那种扩展名.lib，叫做输入库，是静态库库的一种特殊形式。输入库不含代码，而是为链接程序提供信息，以便在.exe文件 中建立动态链接时要用到的重定位表。你的这种当然是有源码的静态库。两种库在windows下都叫静态库  

有一些工程文件是2017的 用vs2019打开并更新工具集  

`CGAL_BOOST_USE_STATIC_LIBS`  

预处理器定义加上`GOOGLE_GLOG_DLL_DECL=`   
glog编译添加`BULID_SHARED_LIBS`   

lz4链接错误 参考<a href = "https://github.com/lz4/lz4/issues/920">https://github.com/lz4/lz4/issues/920</a>   
用vs2019打开2017的工程 重新编译  

参数添加gui 在vs下调试运行 添加缺失的dll到colmap.exe所在目录F:\Workspace\colmap-3.8\build\src\exe\Release

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
   
