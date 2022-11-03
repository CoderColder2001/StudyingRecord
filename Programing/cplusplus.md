## Content  
* C++内存管理  
* 变量与类型
* STL

<br>

----
  
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

----
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

在执行拷贝操作时，确保拷入和拷出的对象必须具有相同的底层const资格  
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

---
### 迭代器
访问元素：```(*iter) ```

<br>
  
----
* ## STL
  - priority_queue
  - vector

---  
### priority_queue  

"<"构造大顶堆 &emsp; ">"（`greater<int>`）构造小顶堆
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
