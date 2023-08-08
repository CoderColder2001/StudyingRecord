## 记录使用 BOOST 的项目构建中遇到的问题

LNK1104找不到libboost_zlib, 进入zlib.hpp查看源码 发现通过宏定义控制链接库名  

If iostreams is built with a prebuilt zlib library, `BOOST_IOSTREAMS_NO_LIB` needs to be defined to avoid linking errors (missing libboost_zlib...). This should be documented somewhere.  

---
LNK1104找不到boost_log （我有的是libboost_log） 发现可能是预处理器定义里加了BOOST_LOG_DYN_LINK导致的   
但去掉之后会报错LNK2019 无法解析的外部符号 v2s_mt_nt62