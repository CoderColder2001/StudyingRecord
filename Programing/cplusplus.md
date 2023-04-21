## Content  
* C++内存管理  
* 变量与类型
* 函数
* STL

<br>

------
  
* ## c++内存管理
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

------
##  
* ## 变量与类型
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
### 迭代器
访问元素：```(*iter) ```

---
### size_t
<b>不同平台移植性问题 &emsp; 表示字节大小或数组索引</b>  
参数中带有size_t的函数通常会含有局部变量用来对数组的大小或者索引进行计算，在这种情况下，size_t是个不错的选择


size_t是 sizeof 关键字（注：sizeof是关键字，并非运算符）运算结果的类型  

---
### lambda表达式（C++11）
*源于函数式编程 &emsp; 可以就地匿名定义目标函数或函数对象，不需要额外写一个命名函数或者函数对象*  

lambda表达式定义了一个匿名函数（代替函数对象），并且可以捕获一定范围内的变量  
`[ capture 捕获列表 ] ( params 参数列表 ) opt 函数选项-> ret 返回值类型 { body; 函数体 };`   
* [] 不捕获任何变量。
* [&] 捕获外部作用域中所有变量，并作为引用在函数体中使用（按引用捕获）。
* [=] 捕获外部作用域中所有变量，并作为副本在函数体中使用（按值捕获）。
* [=，&foo] 按值捕获外部作用域中所有变量，并按引用捕获 foo 变量。
* [bar] 按值捕获 bar 变量，同时不捕获其他变量。
* [this] 捕获当前类中的 this 指针，让 lambda 表达式拥有和当前类成员函数同样的访问权限。如果已经使用了 & 或者 =，就默认添加此选项。捕获 this 的目的是可以在 lamda 中使用当前类的成员函数和成员变量  

lambda 表达式的类型在 C++11 中被称为“闭包类型（Closure Type）”。它是一个特殊的，匿名的非 nunion 的类类型。可以认为它是一个带有 operator() 的类，即仿函数。因此，我们可以使用 std::function 和 std::bind 来存储和操作 lambda 表达式：  

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
<br>

需要注意的是，没有捕获变量（没有状态）的 lambda 表达式可以直接转换为函数指针，而捕获变量（有状态）的 lambda 表达式则不能转换为函数指针。  
lambda 表达式可以说是就地定义仿函数闭包的“语法糖”。它的捕获列表捕获住的任何外部变量，最终均会变为闭包类型的成员变量。而一个使用了成员变量的类的 operator()，如果能直接被转换为普通的函数指针，那么 lambda 表达式本身的 this 指针就丢失掉了。而没有捕获任何外部变量的 lambda 表达式则不存在这个问题。 

按照 C++ 标准，lambda 表达式的 operator() 默认是 const 的。按值捕获时，一个 const 成员函数是无法修改成员变量的值。而 mutable 的作用，就在于取消 operator() 的 const  
`auto f2 = [=]() mutable { return a++; };`

· &emsp; 在priority_queue中使用lambda表达式时：  
因为在初始化priority_queue时，三个参数必须是类型名，而cmp是一个对象，因此必须通过decltype()来转为类型名； 因为lambda这种特殊的class没有默认构造函数，pq内部排序比较的时候要使用的是一个实例化的lambda对象，通过lambda的copy构造进行实例化（pq构造函数的时候传入这个lambda对象）  
<br>

------
* ## 函数
  
函数前const：普通函数或成员函数（非静态成员函数）前均可加const修饰，表示函数的返回值为const，不可修改  
函数后加const：只有类的非静态成员函数后可以加const修饰，表示该类的this指针为const类型，不能改变类的成员变量的值  

---
### 函数的返回
返回的值用于初始化调用位置的一个临时量，该临时量就是函数调用的结果  
  
如果函数返回引用，则该引用指示它所引对象的一个别名（不会真正拷贝对象） 没有复制返回值 而是返回的是对象本身（不要返回局部对象的引用）     
`const xx &`返回，用`xx p`接收（调用构造函数初始化p）或`const xx &p`接收

<br> 

------
* ## STL
- datastructure
  - multiset
  - priority_queue
  - vector
  - string
- algorithm

------
## DataStructure
### multiset
允许重复元素  
与`unordered_xxx`不同 会对内部元素进行排序 基于红黑树  
`*s.rbegin()` 访问最大元素  
`*s.begin()` 访问最小元素

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
### vector

将哈希表内容存到vector中，并根据value降序排序，按值重复存放到string中：  
``` c++
vector<pair<char, int>> vec;
for (auto &it : mp)
    vec.emplace_back(it);

sort(vec.begin(), 
    vec.end(), 
    [](const pair<char, int> &a, const pair<char, int> &b) {
        return a.second > b.second;
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

<br>

---
## Algorithm
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
