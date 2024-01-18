[TOC]
# QT 5
## Content
- 事件循环
- QObject
- UI、元对象系统
- 信号槽机制
- QT MVD

<br>

------
## Qt 事件
事件驱动架构Event Driving Architecture（程序的执行流由事件决定）：
- 事件队列Event Queue：接受事件的入口，存储待处理事件
- 分发器Event Mediator：将不同的事件分发到不同的业务逻辑单元
- 事件通道Event Channel：分发器与处理器之间的联系渠道
- 事件处理器Event Processor：实现业务逻辑，处理完成后发出事件，触发下一步操作

用户操作首先被OS捕捉，转化为系统内核的消息，再传递进入GUI程序的事件处理框架   
Qt的事件处理框架由事件循环（Event Loop）实现；当应用程序启动后，通过QApplication的`exec()`函数启动事件循环  

在Qt中，任意的`QObject`对象都具有事件处理能力   

用户操作 => 系统内核 => Qt App（转换成事件对象QEvent） => `QApplication::notify(QObject *receiver, QEvent *e)`事件分发 => `QObject::event(QEvent *e)`事件处理逻辑 => 事件处理函数（可以进行信号的发送） => 槽函数  

**一般在子类重写父类的事件处理函数时，要调用父类的事件处理函数以保证默认的事件处理！！**

Qt用户自定义处理事件的方式：
- （*）重写特定事件处理函数（如`mousePressEvent()`）
- 重写实现`QObject::event`
- （*）安装事件过滤器
- 在QApplication上安装事件过滤器
- 重写`QApplication::notify`方法


在事件处理函数中调用：  
```c++  
void QEvent::ignore(); // 表示当前对象忽略该事件；传给父对象
void QEvent::accept(); // 表示当前对象接受并处理事件；事件不会继续传到父对象
```  
<br>

### Qt 事件过滤器
在目标对象接收到事件前，进行拦截或处理  
可以对目标对象的事件进行修改、过滤、转发或记录；并可以与目标对象代码分离  

```c++
void QObject::installEventFilter(QObject *filterObj); // 为指定对象安装过滤器
```
在事件过滤器对象中实现事件过滤器逻辑（重写`eventFilter`函数）
```c++
// 事件过滤器处理函数接口
virtual bool QObject::eventFilter(QObject *watched, QObject *event) override; // 事件目标对象 & 传递的事件
```

------
## QObject
通过对象树 自动、有效地组织和管理继承自QObject的Qt对象  

<br>

------
## UI、元对象系统
### QWidget、QMainWindow、QDialog
- `QWidget`：是所有用户界面对象的基类；它封装了基本的应用程序窗口功能，可以用作单独的窗口，也可以作为其他窗口部件的容器
- `QMainWindow`：主窗口类，用于 创建和管理应用程序的主窗口；提供了一个菜单栏、工具栏、状态栏和一个中心窗口部件，以及一个可选的停靠窗口和工具栏区域
- `QDialog`：是一个对话框类，用于 创建和管理应用程序的模态对话框（在显示时会阻塞用户与其它窗口的交互）或非模态对话框（允许用户在与对话框交互的同时与其他窗口交互）

### 元对象系统
moc读取一个c++头文件。如果它找到包含`Q_OBJECT`宏的一个或多个类声明，它会生成一个包含这些类的元对象代码的c++源文件，并且以moc_作为前缀  

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

<br>

------
## QT MVD设计模式
### MVC框架
将应用程序的数据、逻辑和界面分离  
- 模型（Model）：负责存储和处理数据，通常与数据库进行交互
- 视图（View）：负责显示数据，将模型中的数据可视化；是用户与系统交互的接口
- 控制器（Controller）：负责接收用户输入，处理用户请求，并更新模型和视图

在Qt中，Model-View是一种设计模式，用于将数据（Model）与其可视化表示（View）分离，主要包括以下三个组件：  
- Model：数据的抽象表示，负责存储和管理数据；通常从底层数据源（如数据库、文件或数据结构）获取数据，并将其呈现给视图
- View：数据的可视化表示，它负责呈现模型中的数据
- Delegate：负责处理数据的显示和编辑；Delegate（委托）可以自定义单元格的渲染方式，以及将用户输入的数据写回模型

`QXxxxWidget` 是 `QXxxxView` 的子类，封装了Model/View框架  

### Qt Model
主要负责数据存储和处理；所有Model的父类都是 `QAbstractItemModel`
（如，`QTableWidget` 中的单元格数据是通过 `QTableWidgetItem` 来进行描述）  
模型索引由 `QModelIndex` 类来表示，包含了一个数据项的行、列、有效性、父索引信息、一个指向其所属模型的指针；通过模型索引,可以访问和操作模型中的数据   

Qt的标准模型主要支持字符串与图标(`QIcon`),对于其他类型支持能力较弱；如果需要显示自定义数据结构，则更好的方式是采用 <b>自定义Model</b>；同时，对于大量数据的处理，自定义数据模型可以实现数据的按需加载、缓存等策略，以提高视图的性能  
自定义 Model 类，根据数据结构的特点，继承相应的基类，实现相应接口的中的方法（包括必须要实现的与可选方法）：
- 表格 `QAbstractTableModel`
- 列表 `QAbstractListModel`
- 通用 `QAbstractItemModel`

**QVariant**：
- 可以存储各种类型数据的通用容器的类；支持多种内置的 C++ 数据结构，如 `int`，`bool`，`double`，`QString`，`QByteArray`
- 支持存储自定义的数据类型
- 提供了一种高效、便捷的方法来 在函数和对象之间传递数据

**role 数据角色**：
- 描述 Model 中存储的数据在 View 中的用途和表现形式；每个数据项可以有多个不同的数据角色，例如文本内容、文本颜色、背景颜色等
- View 通过 `role` 来获取数据项的相关属性，并根据这些属性来绘制数据项
- Qt 内置了预定义的数据角色，见 `Qt::ItemDataRole` 中的常量



### Qt Delegate
Qt Delegate 负责 <b>在View上对数据的编辑和渲染</b>。当用户需要编辑数据时，Delegate负责提供编辑器，同时负责将编辑后的数据写回Model   
默认委托是由 `QStyledItemDelegate` 类来进行过描述，继承于`QAbstractItemDelegate`  

当遇到单元格的内容非文本，数字和图像等基本数据类型，则应该考虑自定义 Delegate *（一般选择 `QStyledItemDelegate` 作为基类）*   
对不想要自定义绘制的类型，调用基类的实现   

Model 与 Delegate通过 `QModelIndex` 实现数据交互

------
## 存疑列表
C++对象通信的其他解决方案（设计模式）  