[TOC]
# 设计模式
把“变化”限制在局部的地方（管理变化、提高可复用性）  
手段：分解 + 抽象  

八大原则：
- 依赖倒置原则
- 开放封闭原则
- 单一职责原则
- Liskov替换原则
- 接口隔离原则
- 对象组合优于类继承
- 封装变化点
- 面向接口编程

利用组合和多态（指针指向多态对象）：  
```c++
class A {
    B* pb; // 松耦合 
    // ...
}
```

关注 抽象类&接口  
理清变化点和稳定点、审视依赖关系、framework 和 application（留什么扩展点）   

<br>

## Content
- 组件协作模式
- 单一职责模式
- 对象创建模式
- 对象性能模式
- 接口隔离模式
- 状态变化模式
- 数据结构模式
- 行为变化模式
- 领域规则模式


------
## 组件协作模式
- Template Method
- Strategy
- Observer/Event

通过虚函数晚绑定机制，实现 <b>框架与具体应用的松耦合</b>  

<br>

------
## 对象创建模式  
- Factory Method
- Abstract Factory
- Prototype
- Builder

动机：由于需求的变化，需要创建的对象具体类型经常变化  
绕开`new`，避免`new`带来的与具体类的紧耦合（依赖具体类）  
面向接口的编程要求：对象的类型应该声明成抽象类或接口，而非具体的类（导致后续难以变化）  

<b>使用方法来返回需要的对象</b>  
使用虚函数 **将创建对象带来的依赖关系延迟到运行时**  

<br>

------
## 对象性能模式
- Singleton
- Flyweight

对于 “对象带来的成本”（抽象带来的代价）   
确保某些类在系统中只有一个实例  
*如何绕过常规的构造器，提供一种机制来保证一个类仅有一个实例？*  

C++11 避免汇编指令重排带来的双检查锁失效（指针已被赋值而构造器还没执行完）：  
```c++
std::atomic<Singleton*> Singleton::m_instance;
std::mutex Singletom::m_mutex;

Singleton* Singleton::get_instance() {
    Singleton* tmp;
    std::atomic_thread_fence(std::memory_order_require); // 获取内存fence
    if(tmp == nullptr) {
        std::lock_guard<std::mutex> lock(m_mutex);
        tmp = m_instance.load(std::memory_order_relaxed);
        if(tmp == nullptr) {
            tmp = new Singleton;
            std::atomic_thread_fence(std::memory_order_release); // 释放内存fence
            m_instance.store(tmp, std::memory_order_relaxed);
        }
    }
    return tmp;
}
```

---
### Flyweight 享元模式
在软件系统中应用纯粹对象方案会导致 大量的细粒度对象带来高昂的运行时代价   
采用 <b>“共享技术”</b> 解决性能问题   

维护一个对象池；根据key映射查询  
内部状态一般是不变的（只读的）  
用户对象在访问flyweight对象时提供外部状态（如需要）  

<br>

------
## 接口隔离模式
- Facade
- Proxy
- Adapter
- Mediator

采用一层间接接口（稳定）回避接口间的直接依赖  

---
### Facade 外观模式
定义一个高层接口，为子系统中的一组接口提供一个一致的界面  
（用一个稳定的接口隔离变化体）  

*（Facade 更注重从架构的层次看待整个系统，而不是在单个类的层次）*

<br>

---
### Proxy 代理模式
<b>控制对某些对象的访问</b>  
某些对象由于某些原因（如对象创建开销大、某些操作需要安全控制、需要进程外访问 等）不宜被使用者直接访问  
（增加一层间接层，以在不生气透明操作对象的同时来管理、控制这些对象固有的复杂性）  

为这些对象（接口类`ISubject`）设计一个代理类  
代理类和实际类型采用一致的接口  
`SubjectProxy`内部进行的操作取决于具体任务（安全控制、性能提升、分布式 等）  

<br>

---
### Adapter 适配器模式
应对于 <b>应用环境的变化</b>（迁移）  
将一个类的接口转换为客户希望的另一个接口（应用到新的环境中），使得原本由于接口不兼容不能一起工作的类可以一起工作   

<br>

---
### Mediator 中介者模式 （不常用）
多个对象相互关联，对象之间呈现复杂的引用关系时，面对需求更改会导致直接引用的关系不断变化   

数据绑定模块 用一个新对象解耦依赖关系  
使用一个“中介对象”管理对象之间的关联关系：用一个中介对象封装一系列的对象交互（封装变化），使各对象不需要显示地相互引用（使 编译时依赖 转为 运行时依赖）   

需要定义和实现 <b>调用通知的消息规范</b>   

<br>

------
## 状态变化模式
- State
- Memento

管理对象状态的变化，维持高层模块的稳定  
（组织与特定状态相关的代码）

--- 
### State 状态模式
如何在运行时根据对象的状态改变对象的行为？  

把状态的枚举类型（enum）转换为抽象基类；把与状态有关的操作转换为状态对象的行为（虚函数）  
将所有与状态相关的请求都委托给状态对象   

同时，若状态对象不含除下一状态之外的实例变量，它可以被多个上下文共享（使用Singleton模式实现单例、Flyweight模式实现共享）  

<br>

---
### Memento 备忘录模式 （不常用）
用于支持对象状态的回溯  
如何在不破坏对象封装性的前提下，捕获其内部状态，实现 对象状态的良好保存与恢复？  

用一个备忘录对象（memento）存储目标对象（originator）在某个瞬间的内部状态  

Memento模式的核心是信息隐藏，即 originator 既要向外界隐藏信息以保持封装性，又需要将信息保存到外界  
具体实现时，采用不同的对象序列化方案  

<br>

------
## 数据结构模式
- Composite
- Iterator
- Chain of Responsibility

封装组件内部的数据结构，在外部实现统一的接口，提供与特定数据结构无关的访问  

---
### Composite 组合模式
将对象组合成<b>树形结构</b> 以表示 <b>“部分-整体”的层次结构</b>，同时使得用户对单个对象和组合对象的使用具有一致性   
`Composite`和`Leaf`都继承`Component`的接口  

<br>

---
### Iterator 迭代器模式 （不常用）
集合对象内部结构各异；满足在不暴露内部结构的同时，让外部客户代码透明地访问其中包含的元素（而不用关系内部结构）  
提供<b>独立于对象内部实现的顺序访问</b>  

隔离算法和容器（容器内部的变化）  

现代C++基于 template泛型编程 实现 Iterator，而非通过面向对象的方式（避免运行时多态带来的开销）  

<br>

---
### Chain of Responsibility 职责链模式 （不常用）
一个请求可能被多个对象处理，但每个请求只能对应一个接收者   
如何不显示指定接收者？让候选接收者在运行时自动决定处理请求  
<b>避免请求的发送者和接收者之间的耦合关系</b>  

沿着职责链传递请求  
```c++
class ChainHandler {
    ChainHandeler* nextChain; //基类指针 实现一个多态的链表
    void sendRequestToNextHandler(const Request &req)
    {
        if(nextChain != nullptr)
            nextChain->handle(req);
    }
protected:
    virtual bool canHandleRequest(const Request &req) = 0; // 是否能处理
    virtual void handleRequest(cosnt Request &req) = 0;
public:
    ChainHandler() 
    {
        nextChain = nullptr;
    }
    void setNextChain(ChainHandler *next)
    {
        nextChain = next;
    }
    void handle(const Request &req) // 处理逻辑pipeline
    {
        if(canHandleRequest(req))
            handleRequest(req);
        else
            sendRequestToNextHandler(req);
    }
}

```

<br>

------
## 行为变化模式
- 命令模式
- 访问器模式

将组件行为与组件本身解耦，支持组件行为的变化  

---
### Command 命令模式 （不常用）
将一个请求（行为）封装成对象，从而使得 可以用不同的请求对客户进行参数化，将“行为请求者”与“行为实现者”解耦  

基类中虚函数：  
```
virtual void execute() = 0;
```
子类中存放请求的接收者（行为实现者），`execute`调用接收者的一个或多个操作   

<b>用对象表征行为</b>  
（变成“对象”后提升了灵活度）  
（对象的意义：可以当作参数来传递、可以序列化进行存储（从而实现redo、undo））  

可以通过Composite组合模式将多个命令封装成一个复合命令   

Command模式以面向对象中的“接口-实现”定义行为规范（运行时绑定），有性能损失； C++中还可以使用 **函数对象**（编译时绑定），以函数签名定义接口规范   

<br>

---
### Visitor 访问器模式 （很难满足）
如何避免当需要对基类增加新的操作时带来的修改与变动？如何在不改变类层次的前提下动态地添加新操作？  

利用 “二次多态辨析”  
定义基类 `Visitor`，表示作用域于对象结构（基类及各个子类）的操作；通过`Visitor`的子类实现扩展操作    
在基类 `Element` 中添加一个虚函数：  
```c++
virtual void accept(Visitor &visitor) = 0;
// 子类实现中调用visitor的相应方法 如 visitor.visitElementA(*this)
```

缺陷：需要`Element`的子类个数稳定  
适用于`Element`类层次结构稳定，但其操作频繁变化的场合  

<br>

------
## 领域规则模式
- 解析器模式

可以把问题抽象成某种规则

---
### Interpreter 解释器模型 （不常用）
把特定领域的问题表达为（抽象为） <b>某种语法规则下的句子</b>  
构建语法树（以及自底向上的释放）  

表达式基类`Expression`声明一个抽象的`Interperter`接口  
每个具体表达式类节点对应具体的`Interpreter`方法实现  

<br>