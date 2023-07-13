# CUDA
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