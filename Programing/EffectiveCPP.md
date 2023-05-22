# Effective C++ 阅读笔记

本书关于 <b>“C++ 如何行为、为什么那样行为，以及如何运用其行为形成优势”</b>

------
## Content
- 导读部分
- Part1. 习惯C++
- Part2. 构造/析构/赋值运算
- Part3. 资源管理
- Part4. 设计与声明
- Part5. 实现
- Part6. 继承与面向对象设计
- Part7. 模板与泛型编程
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

如果打算在一个 “内含reference成员” 或 “内含const成员” 的class内支持赋值操作，必须自己定义copy操作符  
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
## Part3. 资源管理
### 13. 以对象管理资源
为确保工厂函数`createXXX()`返回的资源总是被释放，需要将资源放进对象内，当控制流离开请求了资源的函数f时，该对象的析构函数回自动释放那些资源。  
<b>把资源放进对象内，依靠C++的 “析构函数自动调用机制” 确保资源被释放。</b>  
获得资源后应立刻放进管理对象内（资源取得时机便是初始化时机）  

对于heap-based资源：  
`auto_ptr`通过copy构造函数或copy操作符复制它们时，会变成null，而复制所得的指针取得资源唯一拥有权。（其复制行为导致`auto_ptr`不能用于STL中）  
`shared_ptr`引用计数  
两者的析构函数所做的都是`delete`而非`delete[]`，不能用在动态分配的array上  
<br>

### 14. 在资源管理类中小心copying行为
如何处理一个资源管理类对象被赋值的场合？  
- 禁止复制（见6.）
- 对底层资源使用引用计数
- 复制底部资源（深拷贝）
- 转移底部资源的拥有权（类似auto_ptr）
  
<br>

### 15. 在资源管理类中提供对原始资源的访问
<b>通过资源管理类处理与资源之间的所有互动</b>  

`shared_ptr`和`auto_ptr`都提供一个get成员函数，用来执行**显式转换**，返回智能指针内部的原始指针（的复件）   
同时，重载了`operator->`和`operator *`，允许**隐式转换**到底部原始指针  
<br>

### 16. 成对使用new和delete时要采取相同形式
使用`new`时：内存被分配；针对此内存有一个或多个构造函数被调用  
使用`delete`时：针对此内存有一个或多个析构函数被调用；内存被释放  
`delete`的最大问题在于：即将被删除的内存之内究竟存有多少对象？（有多少析构函数必须被调用？）  
即将被删除的指针所指的是单一对象还是对象数组？  

如果在`new`时使用`[]`，则`delete`时也使用`[]`  
<br>

### 17. 以独立语句将newed对象置入智能指针
资源一旦被创建，就要转换为资源管理对象  
反例如
```c++
processWidget(shared_ptr<Widget>(new Widget), priority());
```
在实际执行时，编译器可能将`priority()`部分语句插入到`new Widget`与`shared_ptr`的构造函数之间，这导致priority部分出现异常时，引发内存泄漏
应改为：
```c++
shared_ptr<Widget> pw(new Widget);
processWidget(pw, priority());
```
<br>

------
## Part4. 设计与声明
### 18. 让接口容易被正确使用，不易被误用
限制类内声明事可做，什么事不能做（常见的限制是加 const）  
提供行为一致的接口
“组织误用”的办法包括建立新类型、限制类型上的操作、束缚对象值，以及消除客户的资源管理责任  
<br>

### 19. 设计class犹如设计type
新增一个类也是在扩张类型系统  
重载函数和操作符、控制内存的分配和归还、定义对象的初始化和终结  

面对每一个class：
- 新type的对象应该如何被创建和销毁？（构造函数、析构函数、内存分配函数、释放函数）
- 对象的初始化和对象的赋值该有什么区别？
- 新type如果被passed by value，意味着什么？（拷贝构造函数相关）
- 什么是新type的“合法值”？
- 新type需要配合某个继承图系吗？（受到函数是virtual或non-virtual的影响）
- 新type需要什么样的转换？
- 什么样的操作符和函数对此新type是合理的？
- 什么样的标准函数应该驳回？（见 6.）
- 谁该取用新type的哪些成员？
- 什么是新type的“未声明接口”？
- 你的新type有多么一般化？（或许该是一系列types或class template）
- 真的需要定义一个新type吗？  

<br>

### 20. 宁以pass-by-reference-to-const替换pass-by-value
还可以避免对象切割问题：当一个derived class对象以by-value传递并被视作base class对象时，base class的copy构造函数被调用。  
<br>

### 21. 必须返回对象时，别妄想返回其reference
返回一个新对象  
<br>

### 22. 将成员变量声明为private
统一使用成员函数进行成员访问（同时实现精确的访问控制）  
如果通过函数访问成员变量，日后也可便于以某个计算替换这个成员变量（封装性）  

将成员变量隐藏在函数接口的背后，可以为“所有可能的实现”提供弹性。比如可以使得成员变量被读或被写时轻松通知其他对象、可以验证class的约束条件以及函数的前提或事后状态、可以在多线程环境中执行同步控制等...  

对客户隐藏成员变量（封装它们），可以确保class的约束条件总是得到维护（只有成员函数可以影响它们）  
public意味着不封装，而不封装几乎意味着不可改变   

protect变量改变会影响使用它的derived class   
<br>

### 23. 宁以non-member、non-friend替换member函数 
保证封装性，non-member non-friend函数并不增加“能够访问private成分”的函数数量  
（可以是另一个类的member）  
在c++中，让该函数成为一个non-member函数，并位于该类的同一个namespace内（不同函数可以跨越多个源码文件）   
<br>

### 24. 若所有参数皆需类型转换，请为此采用non-member函数
只有当参数被列于参数列内，这个参数才是隐式类型转换的合格参与者  
（作为成员函数时，被this所指向的隐式参数绝不是隐式转换的合格参与者）  
<br>

### 25. 考虑写出一个不抛异常的swap函数
对于 “以指针指向一个对象，内含真正数据” 的情形   
此时应只置换内部指针   
```c++
class Widget {
public:
    ...
    void swap(Widget& other)
    {
        using std::swap; // 让std::swap在函数体内曝光
        swap(pImpl, other.pImpl);
    }
    ...
};
namespace std {
    template<>  // 特化版本
    void swap<Widget>(Widget& a, Widget& b)
    {
        a.swap(b);
    }
}
```  

注意类模板的情形，C++只允许对class templates，而不允许对function templates（如swap）进行偏特化；此时，声明一个non-member swap调用member swap，但不将non-member swap声明为std::swap的特化版本或重载版本   
```c++
namespace WidgetStuff {
    ...
    template<typename T>
    class Widget {...};
    ...
    template<typename T>
    void swap(Widget<T>&a, Widget<T>&b)
    {
        a.swap(b);
    }
}
```

1. 提供一个public swap成员函数，实现高效置换
2. 在该class或template所在命名空间内提供一个non-member swap，用以调用上述swap成员函数
3. 如果正编写一个class（而非 class template），为其特化std::swap，并令它调用swap成员函数


在编写一个需要用到swap的函数or函数模板时，调用前使用 `using std::swap` 曝光   
<u>成员版的swap绝不可抛出异常</u>! swap的一个重大应用是帮助classes和class templates提供强烈的异常安全性保障  
<br>

------
## Part5. 实现
### 26. 尽可能延后变量定义式的出现时间
不要定义一个不使用的变量，在确实要用到它的时候再定义   
甚至可以延后至它可以获得初值时，避免无意义的default构造  
<br>

### 27. 尽量少做转型动作
C++提供的四种新式转型：  
- `const_cast<T>(expression)`：通常用来将对象的常量性移除（唯一能将const转为non-const）
- `dynamic_cast<T>(expression)`：“安全向下转型”，用来决定某对象是否归属继承体系中的某个类型（唯一可能耗费重大运行成本的转型动作） base -> derived
- `reinterpret_cast<T>(expression)`：基本不会用到。意图执行低级转型，实际动作及结果可能取决于编译器（不可移植）
- `static_cast<T>(expression)`：强迫隐式转换，如将non-const对象转换为const对象，或将int转为double等

C++中，单一对象可能拥有一个以上的地址（以base* 指向它时与以derived* 指向它时）  
对象的布局方式和它们的地址计算方式随编译器的不同而不同  

易犯错误：派生类成员函数中执行 `static_cast<Base>(*this).onResize();` 调用的是转型动作所创建的另一个“*this对象的base class成分”的临时副本上的onResize  
解决方案：使用`Window::onResize();`  

避免dynamic_cast，替代方案：
- 使用容器并在其中存储直接指向derived class对象的指针（通常是智能指针）
- 在base class中提供virtual函数与一份“什么都不做”的缺省实现（基于virtual函数调用，将virtual函数往继承体系上方移动）

<br>

### 28. 避免返回handles指向对象内部成分
reference、指针和迭代器都是所谓的handles（用来取得某个对象），而返回一个 “代表对象内部数据” 的handle带来降低对象封装性的风险   
返回 `const XXX&`  

风险：handles的生命周期困难长于其所指对象  
<br>

### 29. 为“异常安全”而努力是值得的
当异常被抛出时，带有异常安全性的函数应：
- 不泄露任何资源 （见13.如何以对象管理资源；见14.确保如互斥器等被释放）
- 不允许数据败坏 （空指针；破坏一致性；违背class约束等） 提供三个保证中的一个：
  - 基本承诺：如果异常被抛出，程序内任何事物仍处于一个有效状态下（但不确定具体哪一个）
  - 强烈保证：如预期般达到函数成功执行后的状态，或返回到函数被调用前的状态
  - 不抛掷保证：绝不抛出异常（作用于内置类型上的所有操作）

任何使用动态内存的东西（例如所有的STL容器）如果无法找到足够内存以满足需求，通常便会抛出一个`bad_alloc`异常   

不要为了表示某件事情发生而改变对象状态，除非那件事情针对已经发生了（注意代码顺序）  

使用copy and swap策略进行强烈保证：为打算修改的对象（原件）做一个副本，然后在副本上做修改，待所有改变都成功后，再将修改过的副本和原对象通过一个不抛除异常的swap函数（25.）中置换 （额外的时间和空间开销）  
但这种方法之只能在操作局部性状态（local state）时提供强烈保证；若函数内调用了其他异常安全性比“强烈保证”低的函数，或当函数对“非局部性数据”有连带影响时，很难进行强烈保证   
<br>

### 30. 透彻了解inlining的里里外外  
会导致目标代码增大（程序体重用更多内存，可能导致额外换页，降低cache命中率）  
inline函数依然有可能存在一个函数本体（用以取地址以及通过函数指针调用）  
不要对构造函数和析构函数inline  
<br>

### 31. 将文件间的编译依存关系降至最低
<b>handle class 内含一个指针成员指向其实现类</b>  
object references、object pointers 依赖类型声明式，而 objects 依赖类型定义式   
尽可能依赖的是其他文件内的声明式而非定义式   

或编写基类接口，通过工厂函数（通常在基类内被声明为static）创建对象  
<br>

------
## Part6. 继承与面向对象设计
### 32. 确定你的public继承塑膜出is-a（是一种）关系
当`class D` 以public继承 `class B`，意味着每一个D对象也是一种B对象（D是B的特殊化），在B对象可用的任何地方，D对象一样也可用  
注意考虑 `class B` 的所有行为，以及程序中可以施行在 `class B` 上的所有行为   
<br>

### 33. 避免遮掩继承而来的名称
derived class 作用域被嵌套在 base class 作用域内  
base class内存在函数重载时，注意derived内的同名函数会遮掩同名但类型不同的base函数   

若要继承同名的重载函数：derived class中使用`using Base::f1;` 让 base class 内名为f1的所有东西在derived作用域内都可见  
*实际上，使用public继承时必须继承这些重载函数，否则违反了 is-a 关系*  

只需要部分继承时（而非继承全部重载）时，使用简单的转交函数（forwarding function）：  
```c++
    virtual void f1()
    { Base::f1(); }
```
<br>

### 34. 区分接口继承和实现继承
public继承意味着成员函数的接口总是会被继承  

纯虚函数：必须被任何继承了它们的具象类重新声明，而且它们在抽象类中通常没有定义  
<b>声明一个纯虚函数的目的时为例让derived class只继承函数接口</b>（不干涉实现）  
但依然可以在基类为纯虚函数提供定义；调用它的唯一途径是通过明确指出类名`xxx->Base::f_purevirtual();`  

<b>声明普通虚函数让derived class继承它们的接口和缺省实现</b>   
为普通的虚函数提供一份更平常更安全的缺省实现  
让接口与缺省实现分开：声明为纯虚函数并提供一份定义，需要缺省实现时显示调用这份定义  

<b>声明non-virtual函数的目的是为了令derived class继承它们的接口以及一份强制性实现</b>（不打算在derived class内有不同的行为）  
<br>

### 35. 考虑virtual函数以外的其他选择
<b>Template Method 设计模式</b>  
改为 Non-Virtual Interface，间接调用  
让public成员函数为non-virtual（作为一个wrapper），在其中调用一个private virtual函数   
这个wrapper确保可以在一个virtual函数被调用前设定好适当场景（如上锁、写日志、验证class约束条件、验证函数先决条件等），并在调用结束后清理场景（如解锁、验证函数事后条件、再次验证class约束条件等）  
*NVI手法允许 derived class 重新定义virtual函数，从而赋予它们对 “ 如何实现机能 ” 的控制能力，但 base class 保留诉说 “ 函数何时被调用 ” 的权利*   
<br>

<b>Strategy 设计模式</b>  
1 . 由函数指针实现   
*构造函数接受一个指针，指向一个具体的函数实现*   
```c++
class GameCharacter; // 前置声明
int defaultHealthCalc(const GameCharacter& gc);
class GameCharacter {
public:
    typedef int (*HealthCalcFunc)(const GameCharacter&); //

    explicit GameCharacter(HeathCalcFunc hcf = defaultHealthCalc)
        : healthFunc(hcf)
    {}

    int heathValue() const
    { return healthFunc(*this); } //
    ...
private:
    HealthCalcFunc healthFunc; //
}
```
优点：
- 同一类型实体可以有不同的函数实现  
- 某实体的函数可以在运行期变更

缺点：
- 该函数不再是继承体系内的成员函数，不能访问对象内部成分  
- 除非弱化该类的封装（如声明该函数为friend，或提供某一部分的public访问函数）

<br>

2 . 由函数对象实现
```c++
typedef std::tr1::function<int (const GameCharacter&)> HealthCalcFunc;
```
`tr1::function` 类型的对象可以持有任何与模板签名式相兼容（参数类型、返回值类型可以隐式转换）的可调用物  
甚至可以接受另一个类的成员函数（通过bind绑定一个该类实体作为this参数）  
```c++
GameLevel currentLevel; // ebg2的health计算函数总以currentLevel作为GameLevel对象
EvilBadGuy ebg2(std::tr1::bind(&GameLevel::health, currentLevel), _1); // 绑定一个GameLevel的成员函数
``` 

3 . 将函数通过另一个分离的继承体系中维护  
声明为另一个继承体系中的virtual成员函数  
再通过类内维护一个指向该继承体系的基类指针  
<br>

### 36. 绝不重新定义继承而来的non-virtual函数
non-virtual函数是静态绑定的，根据指针类型调用  
且违反了 32.  
<br>

### 37. 绝不重新定义继承而来的缺省参数值
virtual函数是动态绑定，而缺省参数值是静态绑定   
静态类型：在程序中被声明时所采用的类型  
动态类型：目前所指对象的类型   

需要在虚函数中指定缺省参数的替代方案：用 35 中的NVI手法；在non-virtual wrapper中指定原来的缺省参数  
<br>

### 38. 通过复合塑模出has-a或“根据某物实现出”
复合composition：类型之间的一种关系；某种类型的对象内含其他类型的对象  
<br>

### 39. 明智而审慎地使用private继承
如果继承关系是private，编译器不会自动将一个derived class对象转换为一个base class对象  
由private base继承而来的所有成员，在derived class中都会变成private属性  

private继承意味着：<b>"根据某物实现出"</b>  
如果class D 以private继承class B，用意应是为了采用class B内已经备妥的某些特性，而不是B对象和D对象有任何观念上的联系  
（只继承实现，不继承接口）  

尽可能使用复合，必要时（涉及访问protected成员或需要重定义virtual函数时）才使用 private 继承   
需要重定义虚函数时，另一种方案：*在class D内部声明一个嵌套式的private class DB，让这个class DB 以public继承 class B并重新定义虚函数；* 这种方案可以实现class D的派生类无法再度重定义这个虚函数。  
同时，如果将DB移出D外，通过D内含一个DB的指针，D所在文件可以只带着一个DB的声明式（而非必须看到定义式，需要include），从而减小了程序的编译依存性

尽管不带任何数据（没有non-static成员变量、没有virtual函数、也没有virtual base classes），独立对象的大小也不一定为0（通常会被安插占位符，或被要求对齐等）。但在private继承中，作为一个base class成分时除外！
<br>

### 40. 明智而审慎地使用多重继承
不同base class中可能存在相同名称  
继承体系中在只需要一份共同base class数据时，声明为virtual继承  
使用virtual继承会引入额外的体积和访问时间；且vitual base的初始化责任由most derived class负责（若使用vitual继承，避免在vitual base classes中放置数据）  
<br> 

------
## Part7. 模板与泛型编程
### 41. 了解隐式接口和编译期多态
显示接口通常由函数的签名式（函数名称、参数类型、返回类型）构成；隐式接口而是由有效表达式组成（表达式决定约束条件）  
对于一个模板类对象 w：
- 隐式接口：w 必须支持哪一种接口，系由template 中执行于 w 的操作来决定  
- 编译期多态：凡涉及 w 的任何函数调用，例如 `operator >` 和 `operator !=`，有可能造成在编译期的template的具现化；“以不同的template参数具现化function templates” 会导致调用不同的函数  

注意隐式接口类型中的表达式内可能的 **隐式类型转换** （类型兼容性）  
<br>

### 42. 了解typename的双重意义
声明template参数时，前缀关键字`class`和`typename`无差别  
typename告诉C++解析器 一个嵌套从属名称（如`T::const_iterator`）是一个类型  
但是typename不可以出现在base classes list 内的嵌套从属名称之前，也不可在成员初值列中作base class修饰符  
<br>

### 43. 学习处理模板化基类内的名称
为避免全特化的模板类不提供和一般性template相同的接口，C++编译器往往不会在 templatized base class（模板化基类）中寻找继承而来的名称  
在定义式中，编译器并不知道一个 class template Derived继承自具体什么样的类（不到具现化时，不知模板参数）  
三种方法让编译器进入templatized base class中寻找名称：  
- 在base class 函数调用动作前加 `this->`
- 使用语句 `using XXX<ABC>::dosomething;` （类似于33. ）
- 调用动作加`XXX<ABC>::` 明确指出被调用函数位于base class内；*然而，这种方法会关闭virtual绑定*

<br>

### 44. 将与参数无关的代码抽离templates
当template被具现化很多次时可能发生的重复  

如，对一个`template<typename T, std::size_t n>`，让其继承一个`template<typename T>`，`n` 作为参数传递，inline调用base class版本的方法  
```c++
template<typename T, std::size_t n>
class SquareMatrix: private SquareMatrixBase<T> {
private:
    using SquareMatrixBase<T>:: invert; // 避免名称遮掩，见33.
public:
    ...
    void invert() { this->invert(n) }; // this-> 保证进入模板化基类寻找名称
}
```
private继承表明 基类只是为了帮助派生类实现  

基类如何知道该操作哪些数据？如何知道数据存放位置？  
令Base存一个指针指向数据所在内存位置与大小；这允许Derived决定内存分配方式，然后在其构造函数中再通知Base  
```c++
SquareMatrix()
    : SquareMatrix<T>(n, 0), // 基类维护大小与指针
      pData(new T[n*n])
{
    this->setDataPtr(pData.get); // 基类中方法
}
```

优劣：这样抽离出具有共性的代码，减少了可执行文件大小（强化了指令cache的引用集中化），但往往会降低了编译优化率，并增加了对象的大小  
<br>

### 45. 运用成员函数模板接受所有兼容类型
以自定义智能指针模仿原始指针在同一继承体系下的类型转化为例：  
```c++
template<typename T>
class SmartPtr {
public:
    template<typename U>
    SmartPtr(const SmartPtr<U>& other) // 为不同的类型 生成相应构造函数
        : heldPtr(other.get()) { ... } // 利用原始指针间的类型转化约束U的类型
    T* get() const {return heldPtr};
private:
    T* heldPtr
}
```
对满足转化要求的任何类型 T 和 U，都可以根据`SmartPtr<U>` 生成一个 `SmartPtr<T>`  

声明泛化的copy构造函数并不会阻止编译器生成non-template的copy构造函数，对赋值操作也相同  
<br>

### 46. 需要类型转换时请为模板定义非成员函数
条款24. 对模板类要支持隐式类型转换的运算时，定义为 “class template 内部的friend 函数” ，使编译器总能在`class XXX<T>`具现化时得知 `T`，并具现化该模板函数  
具现化后，作为一个函数而非函数模板，编译器可以在调用它时执行隐式转换函数  

*为了让类型转换可能发生于所有实参上，需要是non-member函数（条款24.）；为了令这个函数模板被自动具现化，需要将它声明在class内部；而在class内部声明non-member函数的唯一办法就是 —— 让它成为一个friend*  
<br>

### 47. 请使用traits classes 表现类型信息
input迭代器和output迭代器都只能一次一步地向前移动，且只能读or写其所指物最多一次，它们只适合“一次性操作算法”  

如何取得类型信息？  
traits技术 在编译期间取得某些类型信息  
因为我们无法将信息嵌套于原始指针内，因此类型的traits信息必须位于类型自身之外。标准技术是把它放进一个template及其一个或多个特化版本中  
```c++
template<typename IterT>
struct iterator_traits {
    typedef typename IterT::iterator_category iterator_category;
    ...
};
```
`iterator_traits`的运作方式：针对每一个类型`IterT`，在结构内一定声明某个`typedef`名为 `iterator_category`，这个`typedef`用来确认迭代器分类，如：  
```c++
template<...>
class myDeque {
public:
    class iterator {
    public:
        typedef random_access_iterator_tag iterator_category; // xxx_tag是一个空struct
        ...
    };
    ...
};
```   
为了支持指针迭代器，针对指针类型提供偏特化版本：  
```c++
template<typename IterT>
struct iterator_traits<IterT*> {
    typedef random_access_iterator_tag iterator_category;
    ...
};
```

使用 **函数重载** 在编译期间完成条件分支（函数重载是一个针对类型而发生的“编译期条件语句”）   
建立一个统一的控制函数或函数模板，再建立一组重载函数或函数模板；控制函数通过额外传递一个不同的类型参数对象，调用相应版本的重载函数  
```c++
template<typename IterT, typename DistT>
void advance(IterT& iter, DistT d)
{
    doAdvance(iter, d,
              typename std::iterator_traits<IterT>::iterator_category());
}
```

<br>

### 48. 认识template元编程
可以在编译期间找到错误  
<br>

------
## 存疑列表
c++异常处理 栈展开  
类型转换语句  
template相关   

