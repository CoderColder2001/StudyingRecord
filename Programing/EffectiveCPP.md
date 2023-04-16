# Effective C++ 阅读笔记

本书关于 <b>“C++ 如何行为、为什么那样行为，以及如何运用其行为形成优势”</b>

------
## Content
- 导读部分
- Part1. 习惯C++
- Part2. 构造/析构/赋值运算
------
## 导读部分
一个函数的类型：参数&返回类型  
声明式：告诉编译器某个东西的名称和类型，但略去细节  
定义式：编译器为此对象拨发内存的地点  
初始化：“给予对象初值” 的过程，对用户自定义类型的对象而言，初始化由构造函数执行  

构造函数声明为explicit，阻止参数隐式类型转换的执行  
copy构造函数：“以同型对象初始化自我对象”  
copy操作符：“从另一个同型对象中拷贝其值到自我对象”  

<br>

---
## Part1. 习惯C++
### 1. 视C++为一个语言联邦
- C
- 面向对象（类 构造函数&析构函数、封装、继承、多态、虚函数（动态绑定））
- 模板（泛型编程）
- STL及其相关规约

<br>

### 2. 尽量以 const、enum、inline 替换 #define
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

### 3. 尽可能使用 const
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

### 4. 确定对象被使用前已先被初始化
确保每一个构造函数都将对象的每一个成员初始化  
对象的成员变量的初始化动作发生在进入构造函数本体之前  
**使用初始化列表**  
如果成员变量是 const 或 references，它们就一定需要初值，不能被赋值  
base classes 更早于其 derived classes 被初始化，而 class 的成员变量总是以其声明次序被初始化  

C++ 对 “定义在不同编译单元内的 non-local static 对象” 的初始化次序没有明确定义  
解决方案：将每个 non-local static 对象搬到自己的专属函数内（该对象在此函数中被声明为static），这些函数 **返回一个reference指向它所含的对象**，然后用户调用这些函数，而不直接指涉这些对象（用 local static 对象替换 non-local static 对象）即 **单例模式**。C++保证函数内的 local static 对象会在 “该函数被调用期间”“首次遇上该对象定义式”时被初始化   
这些函数 “内含static对象” 的事实使它们在多线程系统中带有不确定性。可以在程序的单线程启动阶段手工调用所有的reference-returning函数，避免在多线程环境下“等待”  
（单例模式的析构函数中不要释放（delete）单例对象，而是应该定义一个静态的嵌套类对象去析构单例对象）  

<br>

------
## Part2. 构造/析构/赋值运算
### 5. 了解C++默默编写并调用哪些函数
如果未声明，编译器就会自动声明一个copy构造函数、一个copy操作符和一个析构函数；当未声明任何构造函数时，编译器会声明一个默认构造函数   

如果打算在一个 “内含reference成员” 或 “内含const成员” 的class内支持赋值操作，必须字节定义copy操作符  
如果某个 base classes将copy操作符声明为private，编译器也将拒绝为其derived classes生成copy操作符  
<br>

### 6. 若不想使用编译器自动生成的函数，就该明确拒绝
可以将相应的成员函数声明为private并不予实现；或继承一个这样做了的基类  
<br>

### 7. 为多态基类声明virtual析构函数
仅仅是指代多态性质（polymorphic）的base classes，这种基类的设计目的是用来 **通过base class接口处理derived class对象**  

当derived class对象经由一个base class指针被删除，而该base class带有一个non-virtual析构函数，实际执行时，对象的derived成分没有被销毁  
任何class只要带有virtual函数都几乎应该确定也有一个virtual析构函数（不带virtual函数通常意味着它并不用来做base class）  

注意：内含virtual函数会导致对象体积增大，在需要传递至（或接受自）其他语言所写函数时可能带来限制  

为抽象类（至少带有一个纯虚函数，不能被实体化）声明纯虚析构函数；必须为这个纯虚析构函数提供一份定义  
析构函数的运作方式：最深层派生most derived的类的析构函数最先调用，然后调用其每个基类的析构函数  
<br>

### 8. 别让异常逃离析构函数
如果析构函数必须执行一个动作，而该动作可能会在失败时抛出异常，该怎么办？  
——重新设计接口，将该动作暴露为一个新函数供客户使用（如 `void close()` 用来关闭数据库连接）；使用一个bool变量追踪该动作是否已被执行，未执行时再由析构函数执行  

如果某个操作可能在失败时抛出异常，而又存在某种需要，客户必须处理该异常，那么这个异常须来自析构函数以外的某个函数（class提供一个普通函数，而非在析构函数中操作）。  
如果一个被析构函数调用的函数可能抛出异常，析构函数应该捕捉任何异常，然后吞下它们（不传播）或结束程序。  

注：C++当有异常被抛出，调用栈(call stack)，即栈中用来储存函数调用信息的部分，会被按次序搜索，直到找到对应类型的处理程序(exception handler)；这个寻找异常相应类型处理器的过程就叫做栈展开。同时在这一过程中，当从f1返回到f2时，f1里局部变量的资源会被清空，即调用了对象的析构函数。 &emsp; 栈展开会自动调用函数本地对象的析构函数，如果这时对象的析构函数时又抛出一个异常，现在就同时有两个异常出现，但C++最多只能同时处理一个异常，因此程序这时会自动调用std::terminate()函数，导致闪退或者崩溃

<br>

### 9. 绝不在构造和析构过程中调用virtual函数
在base class构造期间，virtual函数不是virtual函数（而是base class的版本）  
根本原因：在derived class对象的base class构造期间，对象的类型是base class而不是derived class  

一旦derived class的析构函数开始执行，对象内的derived class成员变量便呈现未定义值   

解决方案：改为non-virtual函数，并要求derived class构造函数传递必要信息给base class构造函数，用以作为调用此non-virtual函数的参数
还可以通过一个static的函数用以构造要传给base class构造函数的信息    
<br>

### 10. 令 operator= 返回一个reference to *this
为了实现 “连锁赋值 x=y=z=...”
```c++
Widget& operator=(const Widget& rhs)
{
    ...
    return * this;
}
```
<br>

### 11. 在 operator= 中处理“自我赋值”
自我赋值安全性 & 异常安全性（应对`new XXX(*rhs.xxx); `抛出异常）
在复制东西之前不要删除  
<br>

### 12. 复制对象时勿忘其每一个成分
让derived class的copying函数调用相应的base class函数
```c++
PriorityCustomer::PriorityCustomer(const PriorityCustomer& rhs)
    : Customer(rhs),
      priority(rhs.priority)
{
    ...
}
PriorityCustomer& PriorityCustomer::operator=(const PriorityCustomer& rhs)
{
    ...
    Customer::operator=(rhs);
    priority = rhs.priority;
    return *this;
}
```
<br>

------
## 存疑列表
c++异常处理 栈展开