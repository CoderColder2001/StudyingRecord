# LINUX & WSL2
基于WSL2或双系统  
记录LINUX的常用命令与有关知识

------
## Content
- 双系统安装

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

中途禁用了集显驱动，但还没有安装好nvidia驱动，导致开机黑屏  
Ctrl+Alt+F2 进入命令行  
```
sudo vi /etc/modprobe.d/blacklist.conf  //删去此前在末尾新加的 
或
rm -f /etc/modprobe.d/blacklist-nouveau.conf  //删除黑名单文件
update-initramfs -u
reboot
```

cuda  
```
sudo apt install nvidia-utils-470
```   

<br>

------
## 常用命令
```
uname -a // 查看linux内核版本号、操作系统版本号
```