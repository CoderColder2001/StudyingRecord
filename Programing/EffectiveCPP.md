# Effective C++ 阅读笔记

本书关于 <b>“C++ 如何行为、为什么那样行为，以及如何运用其行为形成优势”</b>

------
## Content
- 导读部分
- Part1. 习惯C++
------
## 导读部分
一个函数的类型：参数&返回类型  
声明式：告诉编译器某个东西的名称和类型，但略去细节  
定义式：编译器为此对象拨发内存的地点  
初始化：“给予对象初值” 的过程，对用户自定义类型的对象而言，初始化由构造函数执行  

构造函数声明为explicit，阻止隐式类型转换的执行  
copy构造函数：“以同型对象初始化自我对象”  
copy操作符：“从另一个同型对象中拷贝其值到自我对象”  

<br>

---
## Part1. 习惯C++
### 视C++为一个语言联邦
- C
- 面向对象（类 构造函数&析构函数、封装、继承、对台、虚函数（动态绑定））
- 模板（泛型编程）
- STL及其相关规约

<br>

### 尽量以 const、enum、inline 替换 #define
以编译器替换预处理器   
define不能够提供任何封装性

常量定义式通常被放在头文件内（以便被不同的源码含入）  
为将常量的作用域限制于class内，且至多有一份实体：成为class的一个static成员  

`enum { NumTurns = 5}; `其行为更像 #define，一个属于枚举类型的数值可权充int被使用（如在数组声明时），不能取地址或引用  

使用 template inline函数
```c++
template<typename T>
inline void callWithMax(cosnt T& a, const T& b)
{
    f(a > b ? a : b);
}
```
<br>

### 尽可能使用 const
`const char * p` const data   
`char * const p` const pointer   

const 最关键的用法是在于面对函数声明时的应用  
const 可以与函数返回值、各参数、函数自身（若为成员函数）产生关联  

令函数返回一个常量值，往往可以降低因客户错误而造成的意外  
const 成员函数可以确认该成员函数可作用于const对象（得知哪个函数可以改动对象内容而哪个函数不行）  

两个成员函数如果只是常量性（constness）不同，可以被重载
```c++
const char& operator[](std::size_t position) const //const成员函数
{ return text[position]; } // const对象[] 返回const char的引用

char& operator[](std::size_t position)
{ return text[position]; }
```
<br>

### 确定对象被使用前已先被初始化
确保每一个构造函数都将对象的每一个成员初始化  
对象的成员变量的初始化动作发生在进入构造函数本体之前  
**使用初始化列表**  
如果成员变量是 const 或 references，它们就一定需要初值，不能被赋值  
base classes 更早于其 derived classes 被初始化，而 class 的成员变量总是以其声明次序被初始化  

C++ 对 “定义在不同编译单元内的 non-local static 对象” 的初始化次序没有明确定义  
解决方案：将每个 non-local staic 对象搬到自己的专属函数内（该对象在此函数中被声明为static），这些函数 **返回一个reference指向它所含的对象**，然后用户调用这些函数，而不直接指涉这些对象（用 local static 对象替换 non-local static 对象）即 **单例模式**。C++保证函数内的 local static 对象会在 “该函数被调用期间”“首次遇上该对象定义式”时被初始化   
这些函数 “内含static对象” 的事实使它们在多线程系统中带有不确定性。可以在程序的单线程启动阶段手工调用所有的reference-returning函数，避免在多线程环境下“等待”  
（单例模式的析构函数中不要释放（delete）单例对象，而是应该定义一个静态的嵌套类对象去析构单例对象）  