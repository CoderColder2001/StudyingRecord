# LINUX & WSL2 & 服务器
基于WSL2或双系统  
记录LINUX的常用命令与有关知识  

<a href = "">ubuntu服务器安装桌面</a>  
windows安装winScp传文件到服务器  

<a href = "https://blog.csdn.net/qq_38048756/article/details/117935610">windows下的pycharm配置连接linux </a>  

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

### 环境配置
命令行安装  
gcc、make、git  

装机软件  
vscode  
WPS Office  


### 显卡驱动与CUDA
<https://blog.csdn.net/Kevin__47/article/details/131564415>  
<https://www.cnblogs.com/E-Dreamer-Blogs/p/13052655.html>  

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

sudo dpkg -i xxx.deb // 安装deb软件包
```

------
## vim
dd 删除当前一行  