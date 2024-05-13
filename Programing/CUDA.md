# CUDA

<a href = "https://blog.csdn.net/m0_73409404/article/details/132187452">Windows安装Cuda和Cudnn教程</a>

---------
## Content
- CUDA/cudnn/CUDA Toolkit/NVCC区别简介

<br>

------
## CUDA/cudnn/CUDA Toolkit/NVCC区别简介
- **CUDA**：为“GPU通用计算”构建的运算平台。
- **cudnn**：为深度学习计算设计的软件库。
- **CUDA Toolkit (nvidia)**： CUDA完整的工具安装包，其中提供了 Nvidia 驱动程序、开发 CUDA 程序相关的开发工具包等可供安装的选项。包括 CUDA 程序的编译器、IDE、调试器等，CUDA 程序所对应的各式库文件以及它们的头文件。
- **CUDA Toolkit (Pytorch)**： CUDA不完整的工具安装包，其主要包含在使用 CUDA 相关的功能时所依赖的动态链接库。不会安装驱动程序。
- （NVCC 是CUDA的编译器，只是 CUDA Toolkit 中的一部分）

<br>

------

### cg::this_thread_block()
返回 当前线程所属的线程块（thread block）的句柄  
*线程块 是 CUDA 编程模型中的基本并行执行单元，一个线程块由多个线程组成，这些线程可以并行执行相同的代码，但它们共享相同的内存空间*  

线程块中的线程同步：  
``` c++
auto block = cg::this_thread_block();
...
block.sync(); //等待线程块中的所有线程都执行到这个点
```