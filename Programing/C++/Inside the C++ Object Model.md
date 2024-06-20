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
- 每个class object维护一个指针 `vptr` 指向相关的virtual table，`vptr`的设定和重置都由每一个class的constructor、destructor和copy assignment运算符自动完成；同时每个class所关联的 type_info object（用以支持runtime type identification，RTTI）也由virtual table指出来，通常放在表格的第一个slot

C++最初采用的继承模型不运用任何间接性：base class subobject的data members被直接放在derived class object中（带来更改时的重新编译问题）  

C struct在C++中的一个合理用途是当要传递“一个复杂class object的全部或部分”到某个C函数时，struct声明可以将数据封装起来并保证有与C兼容的空间布局   

指针类型告诉编译器如何解释某个特定地址中的内存大小以及其内容  
（“类型转换”其实是一种编译器指令）  

<br>

------
## 2 构造函数语义学
四种造成编译器必须为未声明 constructor 的 class 合成 implicit nontrivial（不是什么都不干的）default constructor 的场合： 
- 包含带有default constructor的 member class object
- 继承带有default constructor的 base class
- 带有虚函数的class
- 带有虚基类的class

至于implict trivial的默认构造函数，实际上并不会被合成出来

编译器会把合成的 default constructor、copy constructor、destructor、assignment copy operator 都以 inline 方式完成；如果函数太复杂不适合inline，就会合成出一个 explicit non-inline static 实例  

被编译器合成的 default constructor 只满足编译器的需要（关于部分member objects或base class的初始化，或为每一个object初始化其虚函数机制与虚基类机制（成员访问）），而不是程序的需要   

编译器会扩张已存在的 constructors，使得在 user code 执行前，先调用必要的内含 member class objects 的默认构造函数（按照它们的声明顺序）；对于声明或继承虚函数或派生链中含有虚基类的class，同时还需要扩张代码已产生`vtbl`和`vptr`，以及改写有关的调用（以使用`vptr`和 `vtbl`中的条目）    

<br>

<b>默认的情况：bitwise copy</b>   

当一个 class 不展现“bitwise copy semantics”时，编译器会合成 copy constructor：  
- 此 class 内含有 member object 而后者的声明有一个 copy constructor
- 此 class 继承自一个 base class 而后者存在一个 copy constructor
- 此 class 声明了一个或多个虚函数（需要对`vptr`适当地初始化）
- 此 class 派生自一个继承串链，其中有一个或多个虚基类（安插一些代码以设定 virtual base class pointer / offset 的值）  

编译器对虚拟继承的支持代表着 **让派生类对象中的虚基类子对象的位置在执行期准备妥当**    

<br>

```c++
X xx0(1024); //被单一的constructor操作设定初值

X xx1 = X(1024);
X xx2 = (X)1024;
// 将临时object以拷贝构造的方式作为explicit object的初值
// 这两种情况下，会调用两个constructor，有一个临时的object产生并会针对该临时object调用deconstructor
```  

copy constructor 导致编译器多多少少会对程序代码做部分转化；尤其是一个函数以传值方式传回 class object 时，而该 class 有一个显示定义的或合成的copy constructor 时（NRV优化：隐式增加一个引用类型的参数，直接修改这个参数）  

当class需要大量memberwise初始化操作（如以传值方式作为参数或返回值）时，且编译器提供NRV优化时，提供一个copy constructor的explicit inline 函数实例是合理的：
```c++
Point3d::Point3d(const Point3d &rhs)
{
    memcpy(this, &rhs, sizeof(Point3d));
}
```
但尤其注意使用 `memcpy(...)` 和 `memset(...)` 时，都只能在 “class 不含任何由编译器产生的内部 members” 的前提下（如`vptr`）（否则会导致它们的初值被改写）   

<br>

### member initialization list
减少临时对象的创建及相应构造函数的隐式调用  

编译器会一一操作 initialization list，以在class中的声明顺序在 constructor 之内安插初始化操作，且在任何 explicit user code 之前  

<br>

------
## 3 Data语义学
<b>讨论class的对象成员 与 class的层级关系</b>

C++对象模型尽量以空间优化和存取速度优化的考虑来表现 nonstatic data members，并保持与C语言struct的兼容性；但并不强制定义其排列顺序。  
static data members 保持在程序的 global data segment 中，不影响类实例对象大小，且永远只存在一份（尽管没有实例对象时）  
程序员声明的同一 access section 内的 nonstatic data members 总是按其被声明的顺序排列（但不一定连续）  

对象大小的影响因素：
- 对 virtual base class 的支持：一个指针，指向该virtual base class的subobject 或 一个相关表格（存放subobject的地址或偏移位置）
- 编译器加上的额外的data members，用以支持某些语言特性（主要是各种virtual特性）
- 编译器对特殊情况（如不包含实际数据的virtual base class）提供的优化处理
- 对齐策略

<br>

### Data Member的存取
static data members 被视作全局变量（但只在 class 生命范围内可见），其存取不需要通过 class object；内部会转化为对该唯一 extern 实例的直接参考操作   

每一个 nonstatic data member 的偏移位置（即使它属于一个派生自单一或多重继承链的类）在编译时期即可确定  

通过对象访问数据成员 与 通过指针访问对象成员 的差异？  
当某个类的继承结构中包含虚基类，存取该虚基类继承而来的member时，通过指针访问不能确认指针属于哪一个class type，该存取操作需要延迟至执行期  

<br>