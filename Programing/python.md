[TOC]

# Python
------
## 常用语法
`map(function, iterable, ...)`：对指定序列进行映射；以参数序列中的每一个元素调用 function 函数，返回包含每次 function 函数返回值的新列表    
返回一个 `map` 对象（一个惰性计算的迭代器）

`chr(i)`转化成字符，`ord(c)`转化成编码   

满足条件的元素对统计：
```python
sum(x >= y for x, y in pairwise(word))
```
`itertool.pairwise()`（python3.10+）：获取对象中重叠的连续对的迭代器（如"abcd"中"ab""bc""cd"）  

`str[::-1]` 字符串反转  

`[..., None]` （在最后）增加一个维度  

<br>

---
### yield
<https://blog.csdn.net/mieleizhi0522/article/details/82142856>  

（一种特殊的“return”）  
带yield语句的函数是一个生成器，而不是一个函数，用于生成迭代器对象（在每一次执行 yield语句 时执行）  

```py
def f():
    while True:
        res = yield 99
        print(res)

g = f()
print(next())
print(g.send(100))
```
执行语句`g = f() # f 中含有yield` 时，函数并不会真的执行，而是先得到一个生成器g（相当于一个对象） 
调用next时，`f`才真正执行  
send方法会将100传给res并调用next  

**返回迭代器对象可以节省内存开销（在需要返回大量有连续性的数据时）**  

<br>

------
## Conda

### 常用命令
```sh
conda info --envs # 查看环境列表
conda list # 查看当前环境下的包

conda create --name=xxx python=x.x.x # 创建环境
conda env remove --name xxx # 删除环境

conda create --name new_env --clone source_env # 本机复制环境

conda install -c conda-forge cudatoolkit=x.x # 安装某一版本cuda（安装pytorch前）

## 
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu118
```

（服务器）安装前加载cuda模块、gcc模块  
记得检查是否为gpu版本pytorch  
``` python
import torch

print(torch.__version__)  # 打印 PyTorch 版本
print(torch.cuda.is_available())  # 如果返回 True，表示 PyTorch 可以使用 GPU
print(torch.cuda.current_device())  # 打印当前设备的ID
print(torch.cuda.get_device_name(0))  # 打印 GPU 名称
```

### pycharm + conda
配置conda环境中 选择 '...\Anaconda\condabin\conda.bat'  
再选择相应的虚拟环境  

### 环境配置的经验
对于一些子项目，可以先git clone下来，到对应目录`pip install .`（凭借setup.py）  

`~/.condarc`  
```sh
conda config --remove-key channels #

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple some-package # 临时使用清华源安装some-package
```

<br>

### conda install 和 pip install
安装位置：
- conda install 安装的库都会放在anaconda3/pkgs目录下  
- 在conda虚拟环境下使用 pip install 安装的库： 如果使用系统的的python，则库会被保存在 ~/.local/lib/python3.x/site-packages 文件夹中；如果使用的是conda内置的python，则会被保存到 anaconda3/envs/current_env/lib/site-packages中

pip 从PyPI（Python Package Index）上拉取数据；conda 从 Anaconda.org 上拉取数据（比较少）  

conda会检查当前环境下所有包之间的依赖关系，保证当前环境里的所有包的所有依赖都会被满足   
（但有时候会乱改？）  

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

# 遍历计算二维数组grid的列之合
col_sum = [sum(col) for col in zip(*grid)]
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
# Numpy
`np.column_stack(...)`：接收一系列数组或类似的序列对象（例如列表或元组），然后将它们按列堆叠  
```python
xys = np.column_stack([tuple(map(float, elems[0::3])),
                        tuple(map(float, elems[1::3]))])
```

------
# Pytorch

## torch 相关配置

```sh
python -c "import torch; print(torch.__version__, torch.version.cuda )" # 检测pytorch版本和对应的cuda支持
```
```python
torch.cuda.is_available()
```

<br>

---
## tensor
tensor（张量）类似于numpy，但特点是可以在GPU上运行，支持自动微分    
可以由 列表、np_array或另一个tensor创建  
tensor和numpy可以共享同一块内存   
```python
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

a = torch.ones_like(data)
b = torch.zeros_like(data)
c = torch.rand((2, 2))
d = torch.eye(3) # 3*3 对角线1 其他0
e = torch.full((2,2), 5) # 2*2 值为5
if torch.cuda.is_available():
    c.to('cuda') # 移动开销

# 查看信息
a.dtype
a.shape
a.device

torch.arange(start=0, end, step=1, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) # 创建一维张量

torch.range(start=0, end, step=1, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) # 创建一维张量（元素数量比arange多一个）

for i in torch.range(10):
    print("epoch: ", i)
```

<br>

## torch 相关方法
### tensor 操作
`xxx.half()`：  
转换为半精度浮点数，减少内存占用、提升计算速度  

---
`torch.cat(tensors, dim=0, *, out=None)`：  
将各`tensors`在第`dim`维上连接，要求除`dim`外其他维度形状相同   
*一个`2*2`的tensor 与一个`2*3`的tensor在 第`1`维上cat，得到`2*5`的 tensor*   

---
`torch.chunk(tensor, chunks, dim=0)`：  
将`tensor`在第`dim`维上分割成`chunks`个 tensors  

---
`torch.split(tensor, split_size_or_sections, dim = 0)`：
划分tensor；`split_size_or_sections`为整形时指示划分块的大小，为列表时指示各划分的长度  

---
`torch.stack(tensors, dim=0, *, out=None)`：  
沿着一个新维度对输入张量序列进行连接(如把二维堆成三维)   
所有张量大小一致  

---
`torch.unbind(input, dim=0)`:  
删去一个维度，返回沿着这个元素对应的所有元组  

---
`torch.reshape(tensor, shape)`：  
保持张量元素的顺序，改变形状  
`shape`为`(-1,)`或`[-1]`时，压缩成一维  

---
`torch.squeeze(tensor, dim=None, out=None)`：  
压缩tensor，删去大小为`1`的维   

---
`torch.unsqueeze(tensor, dim)`：
在`dim`维上新增一个大小为`1`的维  

---
`torch.gather(tensor, dim, index, *, sparse_grad=False, out=None)`：  
沿着第`dim`维（index的值作用于第几维），按照index索引的结果收集`tensor`上的值  
```python
out[i][j][k] = input[index[i][j][k]][j][k] # dim == 0
out[i][j][k] = input[i][index[i][j][k]][k] # dim == 1
out[i][j][k] = input[i][j][index[i][j][k]] # dim == 2
```
input和index的维度（dim）相同，而形状（shape）不必一致  

---
`xxx.scatter_(dim, index, src)`：  
沿着第`dim`维（index的值作用于第几维），按照index将`src`对应元素写入`xxx`   
```python
self[index[i][j][k]][j][k] = src[i][j][k] # dim == 0
self[i][index[i][j][k]][k] = src[i][j][k] # dim == 1
self[i][j][index[i][j][k]] = src[i][j][k] # dim == 2
```
`scatter() `不会直接修改原来的 tensor，而 `scatter_()` 会修改原先的 tensor  

---
`torch.tile(input, dims)`：  
在`i`维上重复`dims[i]`份input以构造新的tensor  

---
`torch.transpose(input, dim0, dim1)`：  
`dim0`维和`dim1`维 转置   

---
`torch.where(condition, tensorX, tensorY)`：  
`tensorX`条件成立时的地方为原值，否则为`tensorY`对应位置的值   

---
`xxx.detach()`:  
切断一些分支的反向传播  
返回一个新tensor，与原始张量共享数据，但不再参与任何梯度计算  

---
`torch.normal(mean, std, size， *, out=None)`： 
根据正态分布构造tensor   
或者不指定size参数，而mean或std为列表  

---
`xxx.cuda(device=None, non_blocking=False, memory_format=torch.preserve_format) -> Tensor`：  
拷贝到cuda内存中，返回对象在cuda内存中的拷贝  

---
`xxx.type(dtype=None, non_blocking=False, **kwarg) -> Str or Tensor`：  
无`dtype`参数时，返回类型；否则将tensor转化为对应类型  

---
`@torch.no_grad()`:     
上下文管理器中执行张量操作时，PyTorch 不会为这些操作计算梯度  

---
`ctx.save_for_backward(...)`:  
在自定义forward函数时，保存张量以便在反向传播过程中使用  

<br>

------
### Dataset & DataLoader
`Dataset`处理单个数据样本   
继承`Dataset`类，构建自己的数据集类，实现`__init__(...)`（*通常需要传入自己数据集的目录；`transform`和`target_transform`参数表示对特征和标签进行预处理的方法*）、`__getitem__(self, index)`、`__len__(self)`   

`DataLoader`对多个样本（minibatch），同时利用python多进程读取数据   
```python 
DataLoader(dataset, batch_size=1, shuffle=False, batch_sampler, num_workers=0) # 常用参数 
```
通过`Dataloader`遍历数据集，一次得到一个 batch 的特征和标签  

<br>

------
### nn.Parameter
`Pararmeter(data=None, requires_grad=True) ` 
Tensor的子类，作为模块参数的张量  
在 Module 定义 Pararmeter 实例时，会自动加入到 Module 的参数列表中  

<br>

------
### nn.Module
所有神经网络模块的基类  
通过创建一个继承自`nn.Module`的类，定义了一个可以包含其他模块、执行前向传播、以及包含参数和缓冲区的神经网络模块  
重写 `__init__ `方法来定义层（内部子模块）和其他属性，重写 `forward` 方法来定义前向传播的计算  

Module成员：
- `register.buffer('xxx', torch.zeros(num_features))`：用于保存不是模块参数但在模块训练中需要缓存的量（如 BatchNorm 中的`running_mean`），由`persistent`（默认`True`）决定该量是否持久化
- `register.parameter('xxx', param)`：将paremeter添加到模型参数列表（一个字典）中，以名称`'xxx'`为键  

<br>

---
`xxx._apply(fn)`：  
对所有的子模块（递归`_apply(fn)`）、参数、buffers调用`fn`   

---
`xxx.apply(fn)`：  
递归地对所有子模块调用`apply(fn)`  
一般用于模块参数初始化时  

<br>

---
### torch.autograd.Function
用于定义 自定义的<b>自动微分操作</b> 的 抽象类  
通过继承 `torch.autograd.Function` 并实现`forward`和`backward`方法来在PyTorch的自动微分系统中创建一个自定义函数  

`ctx` 对象常用于在前向传播和反向传播之间传递信息，例如保存中间结果或梯度  

<br>