# 深度探索C++对象模型 阅读笔记

本书关于<b>CPP的底层建设</b>；关于<b>编译器对于“面向对象”做了什么</b>  

C++对象模型：
- 1、语言中直接支持面向对象程序设计的部分
- 2、对于各种支持的底层实现机制

------
## Content
- 1、关于对象

------
## 1 关于对象
C++ 在布局以及存取时间上主要的额外负担是由virtual引起的（间接性 —— 内存读取&类型判断）  
虚函数的支持：  
- 每个class产生一堆指向virtual functions的指针，存放在虚函数表virtual table（vtbl）
- 每个class object维护一个指针 vptr 指向相关的virtual table，vptr的设定和重置都由每一个class的constructor、destructor和copy assignment运算符自动完成；同时每个class所关联的 type_info object（用以支持runtime type identification，RTTI）也由virtual table指出来，通常放在表格的第一个slot

C++最初采用的继承模型不运用任何间接性：base class subobject的data members被直接放在derived class object中（带来更改时的重新编译问题）  

C struct在C++中的一个合理用途是当要传递“一个复杂class object的全部或部分”到某个C函数时，struct声明可以将数据封装起来并保证有与C兼容的空间布局   

指针类型告诉编译器如何解释某个特定地址中的内存大小以及其内容  
（“类型转换”其实是一种编译器指令）  

