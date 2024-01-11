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