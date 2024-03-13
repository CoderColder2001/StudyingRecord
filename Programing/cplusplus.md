# C++ 学习笔记
记录整理《C++ Primer》与网上资料
[TOC] 

---------
## Content  
* C++内存管理  
* 变量与类型 
* 智能指针
* 迭代器
* 函数
* STL
* 异常处理
* 多线程

<br>

------
  
## c++内存管理
---
### 对象模型  
非静态数据成员被置于每一个类对象之内，静态数据成员则被存放在类对象之外；静态和非静态成员函数也被存放在类对象外。  
在c++继承模型中，一个子类的内存大小是其基类的数据成员加上其自己数据成员大小的总和。大多数编译器对子类的内存布局是先安排基类的数据成员，然后是本身的数据成员。  

声明了virtual函数后，类将产生一系列指向虚函数的指针存放在虚表中；并且类的实例将安插一个指向虚表的指针（大多数编译器放在类实例的开始处）。  

类成员函数的this指针，是编译器将类实例的地址以第一个参数的形式传递进去的  

非virtual析构函数版本，决定继承体系中析构函数链调用的因素是指针的声明类型，virtual版本决定的因素是指针的实际类型（实际指向）；从该类开始依次调用父类析构函数  

<u>**引入virtual可能带来的副作用**</u> : &emsp; virtual会带来构造函数的强制合成（需要生成虚函数表 安排虚表指针）  
* 每个类额外增加一个指针大小的内存占用  
* 函数调用多一层间接性   
<br>  

## 智能指针（C++11） todo
---
RAII（Resource Acquisition Is Initialization） <b>将资源的生命周期绑定到对象的生命周期上</b>   
<b>用对象管理内存</b>   
通过makexxx创建  
&emsp; &emsp; 在RAII中，资源的获取和释放被绑定到对象的构造函数和析构函数上。当对象被创建时，资源被获取并初始化，当对象离开作用域时，析构函数被调用，资源被释放  

* ### shared_ptr

`shared_ptr`和他人共享资源，持有者仅可以显式地释放自己共享的份额（`use_count`）但是不能释放资源；只有持有者已经是最后一个持有人地时候，释放自己份额地时候也会释放资源
- 不要使用原始指针初始化多个`shared_ptr`（它们之间不知情）
- 不要从栈的内存中（指向栈内存的指针）创建`shared_ptr`对象

`reset`可以传参也可以不传参，不传参时和当前资源脱钩并减一引用计数，传参时绑定到新资源
```c++
std::shared_ptr<int> sp1(new int(10)); //资源被sp1共享，sp1引用计数 1
std::shared_ptr<int> sp2 = sp1; //资源被sp2共享，sp1、2引用计数都是2
sp1.reset();//sp1放弃共享资源，sp1自身置空，sp2引用计数变为1
sp2.reset();//sp2放弃共享资源，sp2自身置空，计数归零，资源被释放
//sp2.reset(new int(5))  sp2放弃共享资源，计数归零，资源被释放,sp2指向了新资源
```

* ### weak_ptr 
对象的一种弱引用，<b>不增加对象的`use_count`</b>  
为了避免 `shared_ptr` 带来的环状引用  

`shared_ptr`可以直接赋值给`weak_ptr`，`weak_ptr`也可以通过调用`lock()`函数来获得`shared_ptr`   
`weak_ptr`并没有重载`operator ->`和`operator *`操作符，因此不可直接通过`weak_ptr`使用对象；典型的用法是调用其`lock()`函数来获得`shared_ptr`实例，进而访问原始对象  

* ### unique_ptr 
<b>独占资源（该块内存只能通过这个指针访问）</b> 

开销比 `shared_ptr` 小许多  
`unique_ptr`可以显示地删除资源，也可以分享独占资源或者将独占资源移交给另外地独占者

使用`std::make_unique`创建：  
```c++
std::unique_ptr<int> up1 = std::make_unique<int>(1111);
std::unique_ptr<int> up3 = std::move(up1);
```

---
### 智能指针作为函数参数
**当函数涉及到智能指针的生存期语义的时候才使用智能指针作为传参**，否则使用普通指针或者引用  
```c++
foo(std::unique_ptr<Widget> widget);  // foo转移了资源的所有权
foo(std::unique_ptr<Widget>& widget); // foo将要对widget进行修改，指向另外的内容
foo(std::shared_ptr<Widget> widget);  // foo参与了资源的共享
foo(std::shared_ptr<Widget>& widget); // foo可能要对widget进行修改，指向另外的内容
foo(const std::shared_ptr<Widget>& widget); // foo可能要持有一份引用计数
```

<br>

------
 
## 变量与类型
---  
### 常量引用 & 引用常量 & 常量指针 & 指针常量  
| code | explain |
|--|--|
|``` const int &r ``` | 对const的引用 |
|``` int &const r ``` | 对const的引用 |  
|``` const int *p ``` | 指向常量的指针 | 
|``` int const *p ``` | 指向常量的指针 | 
|``` int *const p ``` | 指针本身是常量（顶层const） | 

在执行拷贝操作时，确保拷入和拷出的对象必须具有 <b>相同的底层const资格</b>  
（不能将一个const int * 赋值给 int *）

---
### 左值和右值 & std::move（C++11）
左值是可以放在赋值号左边可以被赋值的值；左值必须要在内存中有实体；一个左值表达式表示的是一个对象的身份。（变量是左值）  
右值当在赋值号右边取出值赋给其他变量的值；右值可以在内存也可以在CPU寄存器；一个右值表达式表示的是一个对象的值。  
*右值引用只能绑定到一个将要销毁的对象*  

一个对象被用作右值时，使用的是它的内容(值)，被当作左值时，使用的是它的地址。  
- 赋值运算符需要一个（非常量）左值作为其左侧运算对象，得到的结果也仍然是一个左值
- 取地址符作用于一个左值运算对象，返回一个指向该运算对象的指针，这个指针是一个右值
- 内置解引用运算符、下标运算符、迭代器解引用运算符、string和vector的下标运算符的求值结果都是左值
- 内置类型和迭代器的递增递减运算符作用于左值运算对象，其前置版本所得的结果也是左值

左值引用绑定到返回左值的表达式   
右值引用（或 const 左值引用）绑定到返回右值的表达式  

**左值持久，右值短暂**；左值有持久的状态，而右值要么是字面常量，要么是表达式求值过程中创建的临时对象。  
我们可以自由地将一个右值引用的资源“移动”到另一个对象中（使用右值引用的代码可以自由地接管所引用的对象的资源）   

<b>通过 `std::move` 将左值强制转化为右值引用，获得一个绑定到左值上的右值引用</b>   
`std::move`  **将对象的状态或者内存所有权从一个对象转移到另一个对象**（`push_back(move(str))` 后，原str会变为空），只是转移，没有内存的搬迁或者内存拷贝  

使用移动而非拷贝的另一个原因源于IO类或unique_ptr 这样的类，这些类都包含不能被共享的资源（如指针或IO缓冲），因此这些对象不能拷贝但可以移动   

使用移动构造函数接管内存  
```c++
StrVec::StrVec(StrVec &&s) noexcept // 移动操作（通常不分配任何资源）不应抛出任何异常 noexcept通知标准库（在标准库维护相应承诺保障时，标准库与自定义类型间的交互）
    : elements(s.elements), first_free(s.first_free), cap(s.cap)
{
    // 令s进入这样的状态，对其运行析构函数才是安全的
    s.elements = s.first_free = s.cap = nullptr;
}
```
移动构造函数不分配任何新内存，它接管给定的源对象中的内存  

移动赋值运算符：
```c++
StrVec &StrVec::operator=(StrVec &&rhs) noexcept
{
    if(this != &rhs)
    {
        free(); // 释放已有元素
        elements = rhs.elements(); // 从rhs接管资源
        first_free = rhs.first_free;
        cap = rhs.cap;

        rhs.elements = rhs.first_free = rhs.cap = nullptr; // 置rhs于可析构状态
    }
    return *this;
}
```

---
### auto（C++11）  
主要用法  
* 代替冗长复杂的变量声明
* 定义模板参数时，用于声明依赖模板参数的变量
* 模板函数依赖于模板参数的返回值   

当引用被作为初始值时，真正参与初始化的是引用对象的值  
此时编译器以引用对象的类型作为auto的类型  

auto一般会忽略顶层const &emsp; 希望auto是顶层const时，需要明确指出 ```const auto f = xxx; ```  
```auto &f = xxx``` ：将引用的类型设为auto  

还可以用于这样取出元素
```c++
    queue<tuple<int, int, int>> q;
    auto [x, y, keyState] = q.front();
```
  
---
### decltype (C++11) todo
推导类型（如容器的模板参数未知时）  

---
### enum class (C++11)
enum class 将 { } 内的变量，加上 class 限制其在 { } 作用域内可见，是"域内枚举" (scoped enums)，可以防止命名空间污染  
```auto c = Color::yellow;```
域内的枚举成员，不能隐式的转换为广义整型；需使用类型转换符如`static_cast<>()`  

---
### size_t
<b>不同平台移植性问题 &emsp; 表示字节大小或数组索引</b>  
参数中带有size_t的函数通常会含有局部变量用来对数组的大小或者索引进行计算，在这种情况下，size_t是个不错的选择


size_t是 sizeof 关键字（注：sizeof是关键字，并非运算符）运算结果的类型  

---
### volatile（与多线程不相关）
关闭编译器优化，系统总是重新从它所在的内存中读取数据（保证对特殊地址的稳定访问）  
一般用于与外部硬件交流的时候，存储器映射的硬件寄存器通常也要加`volatile`说明，因为每次对它的读写都可能由不同意义

---
### lambda表达式（C++11）
*源于函数式编程 &emsp; 可以就地匿名定义目标函数或函数对象，不需要额外写一个命名函数或者函数对象*   

当定义一个lambda时，编译器生成一个与lambda对应的新的（未命名的）类类型；  
向一个函数传递一个lambda时，同时定义了一个新的类类型和该类型的一个对象  

lambda表达式表示一个可调用的代码单元，定义了一个匿名函数（代替函数对象），并且可以捕获一定范围内的变量  
`[ capture 捕获列表 ] ( params 参数列表 ) opt 函数选项-> ret 返回值类型 { body; 函数体 };`   
* [] 不捕获任何变量。
* [&] 捕获外部作用域中所有变量，并作为引用在函数体中使用（按引用捕获）。*必须确保被引用对象在lambda执行时是存在的*
* [=] 捕获外部作用域中所有变量，并作为副本在函数体中使用（按值捕获）。*值捕获是在lambda创建时拷贝的*
* [=，&foo] 按值捕获外部作用域中所有变量，并按引用捕获 foo 变量。
* [bar] 按值捕获 bar 变量，同时不捕获其他变量。
* [this] 捕获当前类中的 this 指针，让 lambda 表达式拥有和当前类成员函数同样的访问权限。如果已经使用了 & 或者 =，就默认添加此选项。捕获 this 的目的是可以在 lamda 中使用当前类的成员函数和成员变量  

lambda 表达式的类型在 C++11 中被称为 <b>“闭包类型（Closure Type）”</b> 。它是一个特殊的，匿名的非 nunion 的类类型。可以认为它是 <b>一个带有 operator() 的类，即仿函数</b>。 因此，我们可以使用 `std::function` 和 `std::bind` 来存储和操作 lambda 表达式：  

```c++
std::function<int(int)>  f1 = [](int a){ return a; };
std::function<int(void)> f2 = std::bind([](int a){ return a; }, 123);
```

对于没有捕获任何变量的 lambda 表达式，还可以被转换成一个普通的函数指针：
```c++
using func_t = int(*)(int);
func_t f = [](int a){ return a; };
f(123);
```
需要注意的是，没有捕获变量（没有状态）的 lambda 表达式可以直接转换为函数指针，而捕获变量（有状态）的 lambda 表达式则不能转换为函数指针。
lambda 表达式可以说是就地定义仿函数闭包的“语法糖”。它的捕获列表捕获住的任何外部变量，最终均会变为闭包类型的成员变量。而一个使用了成员变量的类的 operator()，如果能直接被转换为普通的函数指针，那么 lambda 表达式本身的 this 指针就丢失掉了。而没有捕获任何外部变量的 lambda 表达式则不存在这个问题。 
<br>  

按照 C++ 标准，lambda 表达式的 operator() 默认是 const 的。按值捕获时，一个 const 成员函数无法修改成员变量的值。而 `mutable` 的作用，就在于取消 operator() 的 const  
`auto f2 = [=]() mutable { return a++; };`

· &emsp; 在priority_queue中使用lambda表达式时：  
因为在初始化priority_queue时，三个参数必须是类型名，而cmp是一个对象，因此必须通过`decltype()`来转为类型名； 因为lambda这种特殊的class没有默认构造函数，而 pq 内部排序比较的时候要使用的是一个实例化的lambda对象，通过lambda的copy构造进行实例化（pq 构造函数的时候传入这个lambda对象）  
<br>

lambda表达式还可以用于 *构造一个复杂的const对象并进行初始化*
<br>

------
## 迭代器
<b>迭代器令算法不依赖于容器</b>  
访问元素：`(*iter) `   

使用 `find(...)` 查找：
```c++
vector<int>::iterator num = find(num_list.begin(), num_list.end(), find_num);  //返回一个迭代器指针
```
<br>

### 范围库 ranges（C++ 20）
操作具有 `begin()` 和 `end()` 的范围  
提供了描述范围和对范围的操作的统一接口  

ranges中的算法是惰性的（懒求值）  
使用视图views作为在范围上应用并执行某些操作的东西  
可以将各种view的关系转化用符号 `|` 串联起来

some example：
```c++
ranges::count(str, '1');
```
<br>

------
## 函数
  
### explicit关键字
指定构造函数或转换函数 （C++11起）为显式, 即它**不能用于隐式转换和复制初始化**   
explicit 指定符可以与常量表达式一同使用，函数若且唯若 该常量表达式求值为 `true` 时 才为显式. （C++20起）    

### const
函数前const：普通函数或成员函数（非静态成员函数）前均可加const修饰，表示函数的返回值为const，不可修改  
函数后加const：只有类的非静态成员函数后可以加const修饰，表示该类的this指针为const类型，不能改变类的成员变量的值  

---
### 函数的返回
返回的值用于初始化调用位置的一个临时量，该临时量就是函数调用的结果  
  
如果函数返回引用，则该引用指示它所引对象的一个别名（不会真正拷贝对象） 没有复制返回值 而是返回的是对象本身（不要返回局部对象的引用）     
`const xx &`返回，用`xx p`接收（调用构造函数初始化p）或`const xx &p`接收

<br> 

------
## STL
- datastructure
  - multiset
  - priority_queue
  - array
  - vector
  - string
  - unordered_map
- algorithm

------
## DataStructure
### multiset
允许重复元素（允许多个元素具有相同关键字key）  
与`unordered_xxx`不同 会对内部元素进行排序 基于红黑树  
`*s.rbegin()` 访问最大元素  
`*s.begin()` 访问最小元素

`set::lower_bound(key)` 返回指向大于等于key的下一个元素的迭代器；超过最大值时返回`s.end()`    

---  
### priority_queue  

"<" (`less<int>`) 构造大顶堆&emsp; ">"（`greater<int>`）构造小顶堆
``` c++
struct cmp {
    bool operator()(const Tweet *a, const Tweet *b) {
        return a->time < b->time;
    }
};
// 构造大顶堆 时间最大的排在最上面
priority_queue<Tweet*, vector<Tweet*>, cmp> q;
```
---
### array(C++ 11)
固定长度的数组  
相较于内置数组而言可以支持迭代器访问、对象赋值、拷贝、获取容量等操作，并也可以获取原始指针    

---
### vector

TODO：大小改变的过程？？  

将哈希表内容存到vector中，并根据value降序排序，按值重复存放到string中：  
``` c++
vector<pair<char, int>> vec;
for (auto &it : mp)
    vec.emplace_back(it);

sort(vec.begin(), 
    vec.end(), 
    [](const pair<char, int> &a, const pair<char, int> &b) {
        return a.second > b.second; // 降序
    }
);

string ret;
for (auto &[ch, num] : vec) {
    for (int i = 0; i < num; i++) {
        ret.push_back(ch);
    }
}
```

`vector.erase(const_iterator position);` 删除一个元素  
`vector.erase(const_iterator first, const_iterator last);` 删除范围元素  
返回一个迭代器指向下一元素  

`std::remove` 不会改变输入vector的长度。其过程相当于去除指定的字符，剩余字符往前靠；  
返回新范围的末尾迭代器的下一个（指向第一个无效值）；同`erase`搭配使用删除指定条件的元素   
如`xxx.erase(remove(xxx.begin(), xxx.end(), 0), xxx.end());` 删除0  

---
### string
查找：  
`find("xxx")` 返回首次出现位置（size_t类型） 若没有找到则返回 `string::npos`  
`find("xxx", n)` 从n开始查找  
`rfind("xxx")` 逆向查找  

修改：  
`erase(n)` 去掉从n开始所有字符  
`resize(n, 'x')` 改变长度，如果超过了原有长度，后面补充x  
`insert(2,"xxx")` 下标n处插入   

---
### unordered_map、unordered_set
无序容器在存储上组织为一系列桶，性能依赖于哈希函数质量以及桶的数量和大小。 理想状态下哈希函数将每个特定值映射到唯一的桶，但当一个桶保存多个元素时，需要顺序查找。  

默认情况下，无序容器使用关键字类型的 `==` 运算符来比较元素，使用 `hash<key_type>` 类型的对象生成每个元素的哈希值  
STL为内置类型（包括指针）和一些标准库类型（string、智能指针等）提供了hash模板    
对于自定义类型，需要提供函数来代替 `==` 运算符和 hash计算函数  
```c++
// 自定义pair的哈希函数
struct pairHash
{
    template<class T1, class T2>
    size_t operator() (pair<T1, T2> const &pair) const
    {
        size_t h1 = hash<T1>()(pair.first); // 用默认hash分别处理
        size_t h2 = hash<T2>()(pair.second);
        return h1^h2;
    }
};
unordered_set<pair<int, int>, pairHash> visited;
```

<br>

---
## Algorithm
### accumulate
第三个参数是和的初值，其类型决定了函数中使用哪个加法运算符以及返回值的类型  

### sort
```c++
sort(idx.begin(), idx.end(), [&](int i, int j) {
    return heights[i] > heights[j]; // 按heights降序排序
});
```

### unique
去除相邻重复元素（故使用前一般要先sort），将重复的元素放到容器的末尾，返回值是去重之后的尾地址  
容器实现原地去重：
```c++
sort(xxx.begin(), xxx.end());
xxx.erase(unique(xxx.begin(), xxx.end()), xxx.end());
```

### lower_bound
底层为二分查找  
有序容器中，返回 `[first,last)` 中指向第一个值不小于val的位置（第一个 >= val 的位置）  
```c++
// comp可以传仿函数对象，也可以传函数指针
// 找第一个不符合 comp 规则的元素   模板中无此项时为 "<"
template <class ForwardIterator, class T, class Compare>
ForwardIterator lower_bound (ForwardIterator first, ForwardIterator last, const T& val, Compare comp);
```

<br>

---
## C++ 异常处理
异常处理机制允许程序中独立开发的部分能够在运行时就出现的问题进行通信并做出相应的处理；将问题的检测与解决过程分离开来   

C++中通过 **抛出** 一条表达式来 **引发** 一个异常；被抛出的表达式类型以及当前的调用链共同决定了哪段**处理代码（handler）** 来处理该异常（在调用链中与抛出对象类型匹配的最近的处理代码）  
执行一个`throw`时，程序的控制权从`throw`转移到与之匹配的`catch`模块；该`catch`可能是同一函数中的局部`catch`，也可能位于直接或间接调用了发生异常函数的另一个函数中，这意味着：
- 沿着调用链的函数可能会提早退出
- 一旦程序开始执行handler，沿着调用链创建的局部对象将被销毁

### 栈展开
沿着嵌套函数的调用链不断查找  
找到一个匹配的`catch`子句后执行完后，从与当前`catch`子句关联`try块`的最后一个`catch`子句后开始执行  

由于栈展开可能使用析构函数，析构函数不应该抛出不能被它自身处理的异常（*析构函数若要执行某个可能抛出异常的操作，则该操作应被置于try语句块中并在析构函数内部得到处理*）  

### 异常对象
编译器使用异常抛出表达式对异常对象进行<b>拷贝初始化</b>  
因此`throw`语句中的表达式必须拥有完全类型（类型被声明且定义）  

异常对象位于由编译器管理的空间中，编译器确保无论最终调用哪个catch子句都能访问该空间；异常处理完毕后，异常对象被销毁  

### catch
`catch`子句中的异常声明看起来像是只包含一个形参的函数形参列表；这个类型必须是完全类型，可以是左值引用，但不能是右值引用  
进入`catch`语句后，通过异常对象初始化异常声明中的参数  
类似于将派生类对象以值传递形式给一个接受基类对象的普通函数，`catch`的参数也可以使用派生类类型的异常对象进行初始化（对象会被切掉一部分）  
异常声明的静态类型将决定`catch`语句所能执行的操作  

`catch`语句中，通过`throw;`重新抛出异常。如果在改变了异常参数后重新抛出，则只有当catch异常声明是引用类型时，所做的改变才会被保留并继续传播  
`catch(...)`捕获所有异常   

### noexcept（C++11）
`noexcept`指定某个函数不会抛出异常  

<br>

------
## C++ 多线程 todo
`std::call_once`、`std::call_flag`（C++ 11）：确保某个函数或可调用对象在程序执行期间只被调用一次  
```c++
KSQLiteManager * KSQLiteManager::m_pInstance = nullptr; // 单例对象初始化
std::once_flag KSQLiteManager::m_flag; // once_flag 初始化
KSQLiteManager* KSQLiteManager::getInstance()
{
    std::call_once(m_flag, []() 
    {
            m_pInstance = new KSQLiteManager(nullptr);
    });
    return m_pInstance;
}
```

<br>

### std::thread

### std::mutex

### atomic

### shared_lock & unique_lock（C++14）