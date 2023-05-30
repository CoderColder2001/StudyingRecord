# QT 5
## Content
- 事件循环
- QObject
- UI、元对象系统
- 信号槽机制

<br>

------
## QObject
通过对象树 自动、有效地组织和管理继承自QObject的Qt对象  

<br>

------
## UI、元对象系统
moc读取一个c++头文件。如果它找到包含Q_OBJECT宏的一个或多个类声明，它会生成一个包含这些类的元对象代码的c++源文件，并且以moc_作为前缀  

信号和槽机制、运行时类型信息和动态属性系统 需要元对象代码   
信号槽，属性系统，运行时类信息都存储在静态对象`staticMetaObject`中  
<br>

------
## 信号槽机制
用于<b>多个对象通信</b>，代替函数回调   
当特定事件发生时，会发出一个信号；槽是响应特定信号的函数  

- 类型安全：信号的参数必须与槽函数的参数相匹配。(实际上，槽的参数可以比它接收到的信号参数更少，因为槽可以忽略额外的参数) &emsp; 由于参数是兼容的，所以在使用基于函数指针语法的信号与槽关联机制时，编译器可以帮助检测类型是否匹配，从而可以检测出在开发中信号和槽函数关联时出现的问题  
- 松耦合

信号和槽函数机制完全独立于GUI事件循环  
除非使用`Qt::QueuedConnection`连接，`emit`语句之后的代码将在所有槽函数都返回之后才执行  

使用`connect()`时可以指定的连接类型：  
|序号| 类型 | 含义 |
|---|------|-----------|
|1|`Qt::AutoConnection`|【默认】如果接收者生活在发出信号的线程中，使用`Qt::DirectConnection`。否则，使用`Qt::QueuedConnection`。连接类型是在信号发出时确定。|
|2|`Qt::DirectConnection`|当信号发出时，槽函数立即被调用。槽函数在发送信号的线程中执行|
|3|`Qt::QueuedConnection`|当控制返回到接收方线程的事件循环时，将调用槽函数。槽函数在接收方的线程中执行|
|4|`Qt::BlockingQueuedConnection`|与`Qt::QueuedConnection`相同，只是在槽函数返回之前线程会阻塞。如果接收方存在于发送信号的线程中，则不能使用此连接（否则死锁）|
|5|`Qt::UniqueConnection`|一个标志，可以使用按位OR与上述的连接类型进行组合。当`Qt::UniqueConnection`被设置时，如果连接已经存在，`QObject::connect()`将失败(例如，如果相同的信号已经连接到同一对对象的相同槽位)|

推荐使用函数指针连接： 许编译器检查信号的参数是否与槽的参数兼容。当然，编译器还可以隐式地转换参数  
```c++
connect(sender, &QObject::destroyed, this, &MyObject::objectDestroyed, Qt::ConnectionType type = Qt::AutoConnection);
```

声明信号 `signals:` :
- 所有的信号声明都是公有的，所以Qt规定不能在signals前面加public,private, protected  
- 所有的信号都没有返回值，所以返回值都用void  
- 所有的信号都不需要定义
- 必须直接或间接继承自`QObject`类，并且开头私有声明包含`Q_OBJECT`

声明槽 `public/protected/private slots:` :
- 槽其实就是普通的C++函数，它可以是虚函数，static函数
- 必须直接或间接继承自`QObject`类，并且开头私有声明包含`Q_OBJECT`

------
## 存疑列表
C++对象通信的其他解决方案（设计模式）  