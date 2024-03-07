[TOC]

# Python
------
## 常用语法
`map(function, iterable, ...)`：对指定序列进行映射；以参数序列中的每一个元素调用 function 函数，返回包含每次 function 函数返回值的新列表    

`chr(i)`转化成字符，`ord(c)`转化成编码   

满足条件的元素对统计：
```python
sum(x >= y for x, y in pairwise(word))
```
`itertool.pairwise()`（python3.10+）：获取对象中重叠的连续对的迭代器（如"abcd"中"ab""bc""cd"）  

`str[::-1]` 字符串反转  

`[..., None]` 在最后增加一个维度  
<br>

------
## Conda

常用命令
```
conda info --envs // 查看环境列表
conda list // 查看当前环境下的包
```

<br>

------
### python中一切皆对象  
参数的对象绑定 &emsp; 调用时绑定参数对象 &emsp; 定义时绑定参数默认值  
注意参数对象为可变对象的时候可能带来问题， 如 `func(a=[ ])`   

### 方便函数接口的定义  
`func(a, b, *args)`  
`*args` &emsp; 可变参数列表 &emsp; 类型为元组  
`*` 在函数调用时进行打包和解包操作 &emsp; 如 `args=("a", 1)` &emsp; func(*args)

`func(a, b, *args, debug = true)` ==> `func(a, b, *args, **kwargs)`  
`**kwargs` &emsp; 任意关键字参数 &emsp; 类型为字典  
用于扩展关键字参数  
`*args` 、 `**kwargs` 可以接受任意的入参 并透传给另一个函数  

<br>

------
## Python语法糖 简洁优雅的代码
### 1、列表推导式  
``` python
numbers = [1, 2, 3, 4, 5]
squares = [x**2 for x in numbers]
```
### 2、字典推导式
```py
numbers = [1, 2, 3, 4, 5]
squares = {x: x**2 for x in numbers}
```
### 3、集合推导式
```py
numbers = [1, 2, 3, 4, 5]
squares = {x**2 for x in numbers}
```
### 4、条件表达式
``` py
x = 10
even_odd = "even" if x % 2 == 0 else "odd"
```
### 5、with语句（自动管理文件等资源的打开和关闭，无需手动调用 `file.close()`）
```py
with open("file.txt", "r") as file:
data = file.read()
```
### 6、函数装饰器（使用 `@` 将装饰器函数应用到另一个函数上，以便在函数调用前后执行额外的逻辑）
在我们调用这个函数的时候，第一件事并不是执行这个函数，而是将这个函数做为参数传入装饰器  
```py
def decorator(func):
    def wrapper():
        print("Before function call")
        func()
        print("After function call")
        return wrapper

@decorator
def hello():
    print("Hello, world!")
```
### 7、解构赋值（一次给多个变量赋值或函数一次返回多个值）
```py
x, y = 10, 20

def get_name():
    return "John", "Doe"
first_name, last_name = get_name()
```
### 8、内置的 `enumerate()` 函数（在迭代列表时同时获取索引和元素值）
```py
names = ["Alice", "Bob", "Charlie"]
for index, name in enumerate(names):
    print(index, name)
```
### 9、内置的 `zip()` 函数（将多个可迭代对象进行配对，用于同时迭代多个列表）
```py
names = ["Alice", "Bob", "Charlie"]
ages = [25, 30, 35]

for name, age in zip(names, ages):
    print(name, age)
```
### 10、扩展with语句 创建自定义的上下文管理器
通过使用 contextlib 模块中的 `@contextmanager` 装饰器将函数转换为上下文管理器  
```py
from contextlib import contextmanager

@contextmanager
def open_file(filename):
    file = open(filename)
    yield file
    file.close()

with open_file("file.txt") as file:
    data = file.read()
```
上下文管理器本质上是能够支持`with`操作的类或对象，实现了`__init__()`、`__enter__()`（上文方法）和`__exit__()`（下文方法）  
### 11、可变参数与关键字参数
```py
def func(*args, **kwargs):
    # 处理可变位置参数 args
    # 处理可变关键字参数 kwargs
    pass

func(1, 2, name="Alice", age=25)
```
### 12、列表切片
```py
numbers = [1, 2, 3, 4, 5]
subset = numbers[2:4]
```
### 13、else语句与循环
```py
for item in items:
    if condition:
        # 满足条件时执行
        break
else:
    # 循环完成且未触发 break 时执行
    pass
```
### 14、try-except 块与异常处理
```py
try:
    # 可能引发异常的代码块
    pass
except SomeException:
    # 处理某个特定异常
    pass
except AnotherException:
    # 处理另一个特定异常
    pass
else:
    # 当没有引发任何异常时执行
    pass
finally:
    # 无论是否引发异常都会执行
    pass
```
### 15、上下文管理器类
通过定义 `__enter__()` 和 `__exit__()` 方法，实现自己的上下文管理器，用于资源的安全管理
```py
class MyContextManager:
    def __enter__(self):
        # 在进入 with 代码块前执行的操作
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 在离开 with 代码块后执行的操作
        pass

with MyContextManager() as obj:
# 在 with 代码块中可以安全地使用 obj 对象
    pass
```
### 16、链式比较
```py
age = 25
if 18 <= age < 30:
    print("年龄在 18 到 30 之间")
```
### 17、`any()` 和 `all()` 函数（对可迭代对象进行判断）
返回一个布尔值，表示对于可迭代对象中的元素是否满足某种条件  
```py
numbers = [10, -5, 20, -15] 
is_positive = all(num > 0 for num in numbers) 
has_negative = any(num < 0 for num in numbers) 
```

<br>

------
# Pytorch

`torch.stack(...)`:  
沿着一个新维度对输入张量序列进行连接(如把二维堆成三维)   

`xxx.detach()`:  
切断一些分支的反向传播  
返回一个新tensor，与原始张量共享数据，但不再参与任何梯度计算  

`@torch.no_grad():`   
上下文管理器中执行张量操作时，PyTorch 不会为这些操作计算梯度  