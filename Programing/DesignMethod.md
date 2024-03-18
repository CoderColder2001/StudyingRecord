# 设计模式
把“变化”限制在局部的地方  

## Content
- 组件协作模式
- 对象创建模式
- 对象性能模式
- 接口隔离模式


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
使用虚函数将创建对象带来的依赖关系延迟到运行时  

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