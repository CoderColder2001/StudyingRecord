[TOC]

------
# LINUX & WSL2 & 服务器
基于WSL2或双系统  
记录LINUX的常用命令与有关知识  

<a href = "">ubuntu服务器安装桌面</a>  
windows安装winScp传文件到服务器  

<a href = "https://blog.csdn.net/qq_38048756/article/details/117935610">windows下的pycharm配置连接linux </a>  
右键工程文件夹目录upload  

------
## Content
- 双系统安装
- 软件安装links
- 常用命令
- vim

------
## 双系统安装
2023.11  主要参考了<https://zhuanlan.zhihu.com/p/363640824>  用师兄做好的启动盘  
注意U盘启动时用UEFI启动  

<br>

### 环境配置
<a href = "https://zhuanlan.zhihu.com/p/359354934">~/.bashrc-Linux环境变量配置超详细教程</a>

命令行安装  
gcc、make、git  

装机软件  
vscode  
WPS Office  


### 显卡驱动与CUDA
<https://blog.csdn.net/Kevin__47/article/details/131564415>  
<https://www.cnblogs.com/E-Dreamer-Blogs/p/13052655.html>  

<a href = "https://blog.csdn.net/Dove_Dan/article/details/130667793?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-130667793-blog-107234271.235%5Ev43%5Epc_blog_bottom_relevance_base6&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-130667793-blog-107234271.235%5Ev43%5Epc_blog_bottom_relevance_base6&utm_relevant_index=1">linux系统非root用户安装cuda和cudnn，不同版本cuda切换</a>

```sh
ubuntu-drivers devices   // 查询所有ubuntu推荐的驱动
sudo ubuntu-drivers autoinstall // 安装所有推荐的驱动程序
```
中途禁用了集显驱动，但还没有安装好nvidia驱动，导致开机黑屏  
Ctrl+Alt+F2 进入命令行  
```sh
sudo vi /etc/modprobe.d/blacklist.conf  # 删去此前在末尾新加的 
或
rm -f /etc/modprobe.d/blacklist-nouveau.conf  # 删除黑名单文件
update-initramfs -u
reboot
```

cuda  
```sh
sudo apt install nvidia-utils-470
```   

``` sh
vim ~/.bashrc

export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-11.3/bin:$PATH
export CUDA_HOME=/usr/local/cuda-11.3

source ~/.bashrc
```

<br>

### 卸载CUDA
```sh
cd /usr/local/cuda/bin
./cuda-uninstaller
sudo rm -rf /usr/local/cuda-xx.x # 删除残余cudnn
```
<a href="https://blog.csdn.net/weixin_45347379/article/details/120260072">Linux卸载重新安装cuda，cudnn和pytorch的误区和相关常用命令</a>

------
## 软件安装links
<https://archive.ubuntukylin.com/ubuntukylin/pool/partner/>     


------
## 常用命令
```sh
lsb_release -a // 查看ubuntu版本
uname -a // 查看linux内核版本号、操作系统版本号
nvcc --version // 查看已安装的cuda（通过/bin里的nvcc）

nvidia-smi // 查看显卡状态

ps aux # 查看运行的进程

sudo dpkg -i xxx.deb // 安装deb软件包
```

使用nohup后台运行进程  

------
## vim
dd 删除当前一行  


------
## Envrionment modules 模块配置工具
```sh
module avail  # 查看可用模块
module list  # 查看已加载模块 

module load | add  # 加载环境变量
module switch  # 改变环境变量的版本号

module unload | rm # 卸载环境变量

module swap # 替换环境变量
```

<br>

------
## slurm 任务调度工具
Linux集群的任务调度系统  

| command | Description |
|--|--|
|``` sbatch ``` | 向 SLURM 提交批处理脚本 |
|``` squeue ``` | 列出当前正在运行或在队列中的所有作业 |  
|``` scancel ``` | 取消提交的工作 | 
|``` sinfo ``` | 检查所有分区中节点的可用性 | 
|``` scontrol ``` | 查看特定节点的配置或有关作业的信息 |
|``` sacct ```| 显示所有作业的数据 |
|``` salloc ```| 预留交互节点 |

<br>

------
# Linux 开发

## core dump 文件
当一个进程因为某种原因（例如，非法内存访问、非法指令等）异常终止时，操作系统可以将进程的内存信息保存到一个core dump文件中，以便进行调试和分析  
主要包含了 用户空间的内存信息（用户空间栈、代码段、数据段和堆等）  

gdb 可以读取 core dump 文件，并提供了一系列命令来分析程序崩溃时的内存状态  
```sh
(gdb) bt # 查看程序崩溃时的堆栈信息
(gdb) p # 查看程序崩溃时的变量值
(gdb) info registers # 查看程序崩溃时的寄存器状态
(gdb) disassemble # 查看程序崩溃时的汇编代码
```

objdump（反汇编工具）可以将可执行文件和共享库文件反汇编成汇编代码，用于分析程序崩溃时的汇编代码   
```sh
objdump -d -j .text # 查看 core dump 文件中的程序代码段
objdump -s -j .data # 查看 core dump 文件中的程序数据段
objdump -t # 查看 core dump 文件中的程序符号表
```

readelf 用于查看可执行文件和共享库文件的 ELF 格式文件头，可以用于分析程序崩溃时的内存布局  
```sh
readelf -S <executable> # 查看 core dump 文件中的程序段
readelf -s <executable> # 查看 core dump 文件中的程序符号表
readelf -d <executable> # 查看 core dump 文件中的程序动态链接信息
```

<br>