## Content
- Introduction
- 窗口 & 渲染

<br>

------
* ## Introduction 
---
### OpenGL 状态机
OpenGL自身是一个巨大的状态机(State Machine)：一系列的变量描述OpenGL此刻应当如何运行。OpenGL的状态通常被称为 <b>OpenGL上下文(Context)</b>。我们通常使用如下途径去更改OpenGL状态：`设置选项`，`操作缓冲`。最后，我们`使用当前OpenGL上下文来渲染`  
状态设置函数 与 状态使用函数  

--- 
### OpenGL 对象
在OpenGL中一个对象是指<b>一些选项的集合</b>，它代表 <b>OpenGL状态的一个子集</b>。  
比如，我们可以用一个对象来代表绘图窗口的设置，之后我们就可以设置它的大小、支持的颜色位数等等。  
可以把对象看做一个C风格的结构体(Struct)。

<br>

我们首先创建一个对象，然后`用一个id保存它的引用`（实际数据被储存在后台）。然后我们`将对象绑定至上下文的目标位置`（例子中窗口对象目标的位置被定义成`GL_WINDOW_TARGET`） 
* 使用对象的一个好处是在程序中，我们不止可以定义一个对象，并设置它们的选项，每个对象都可以是不同的设置。在我们执行一个使用OpenGL状态的操作的时候，只需要绑定含有需要的设置的对象即可

---
### GLFW库与GLAD库
GLFW是一个专门针对OpenGL的C语言库，它提供了一些渲染物体所需的最低限度的接口。它允许用户创建OpenGL上下文、定义窗口参数以及处理用户输入  

因为OpenGL只是一个标准/规范，具体的实现是由驱动开发商针对特定显卡实现的。由于OpenGL驱动版本众多，它大多数函数的位置都无法在编译时确定下来，需要在运行时查询。开发者需要在运行时获取函数地址并将其保存在一个函数指针中供以后使用  &emsp; 通过GLAD提供的封装实现这些重复操作 ( <u><b>管理OpenGL函数指针</b></u> )  

<details>
<summary> &emsp; <b>code</b> </summary>

```c++
	GLFWwindow* window = glfwCreateWindow(800, 600, "LearnOpenGL", NULL, NULL);
	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window); 
    // 通知GLFW 将window的上下文设置为当前线程的主上下文
```
```c++
	//在调用任何OpenGL的函数前 需要初始化GLAD
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}
```
</details>

------
* ## 窗口&渲染
---
`glViewport(0, 0, 800, 600)`告诉OpenGL渲染窗口的尺寸大小，前两个参数控制窗口左下角的位置。第三个和第四个参数控制渲染窗口的宽度和高度（像素）。  
幕后使用glViewport中定义的位置和宽高进行2D坐标的转换  
当用户改变窗口的大小的时候，视口也应该被调整。可以对窗口注册一个回调函数`glfwSetFramebufferSizeCallback(window, framebuffer_size_callback)`  

---
双缓冲解决像素非同时绘制导致的问题：  
前缓冲保存着最终输出的图像，它会在屏幕上显示；而所有的的渲染指令都会在后缓冲上绘制。当所有的渲染指令执行完毕后，交换(Swap)前缓冲和后缓冲，这样图像就立即呈显出来了

---
### OpenGL 图形渲染管线
图形渲染管线可以被划分为两个主要部分：第一部分把3D坐标转换为2D坐标，第二部分把2D坐标转变为实际的有颜色的像素  

<img src = "./pic/OpenGLpic1.png" width = "100%">

* 顶点着色器：主要的目的是把3D坐标转为另一种3D坐标，同时顶点着色器允许我们对顶点属性进行一些基本处理
* 图元装配阶段(Primitive Assembly)：将顶点着色器输出的所有顶点作为输入（如果是GL_POINTS，那么就是一个顶点），并所有的点装配成指定图元的形状  
* 几何着色器：把图元形式的一系列顶点的集合作为输入，它可以`通过产生新顶点构造出新的（或是其它的）图元`来生成其他形状。
* 光栅化阶段(Rasterization Stage)：`把图元映射为最终屏幕上相应的像素`，生成供片段着色器使用的片段(Fragment)。在片段着色器运行前会执行`裁切(Clipping)`，丢弃超出视图以外的所有像素。
* 片段着色器：计算一个像素的最终颜色，这也是所有OpenGL高级效果产生的地方。通常，片段着色器包含3D场景的数据（比如光照、阴影、光的颜色等等）。
* Alpha测试和混合(Blending)阶段：检测片段的对应的深度

---
### <b>VBO & VAO & EBO</b> 

在GPU上创建内存用于储存我们的顶点数据，还要`配置OpenGL如何解释这些内存`，并且指定其如何发送给显卡。   

<b>使用VBO（顶点缓冲对象）管理这个内存</b>  
`glBindBuffer(GL_ARRAY_BUFFER, VBO)`  把新创建的缓冲绑定到GL_ARRAY_BUFFER目标上，从这一刻起，使用的任何（在GL_ARRAY_BUFFER目标上的）缓冲调用都会用来配置当前绑定的缓冲(VBO)  
`glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW)`把用户定义的数据复制到当前绑定缓冲。  

<br>

`glVertexAttribPointer`函数设置顶点属性指针 告诉OpenGL<b> 如何解析顶点数据（应用到逐个顶点属性上）</b>  
<b>输入数据与顶点着色器中顶点属性的对应</b>  
VAO（顶点数组对象）把这些状态配置储存在一个对象中，在绘制物体前简单地把VAO绑定到希望使用的设定上。  

<br>

元素缓冲对象EBO（也称索引缓冲对象IBO）<b>存储 OpenGL 用来决定要绘制哪些顶点的索引</b>  
在绑定VAO时，绑定的最后一个元素缓冲区对象存储为VAO的元素缓冲区对象。绑定到VAO也会自动绑定该EBO  

<img src = "./pic/OpenGLpic2.png" width = "100%">

---
标准化设备坐标：一个 <b>x、y和z值在-1.0到1.0内</b> 的一小段空间。  
任何落在范围外的坐标都会被丢弃/裁剪。  
一旦顶点坐标已经在顶点着色器中处理过，它们就应该是标准化设备坐标了

---
### &emsp; Shader
当今大多数显卡都有成千上万的小处理核心，它们在GPU上为每一个（渲染管线）阶段运行各自的小程序，从而在图形渲染管线中快速处理你的数据。这些小程序即着色器( <u>Shader</u> )  

### &emsp; o 顶点着色器  
处理我们在内存中指定数量的顶点  转换为标准化设备坐标。

<br>

### &emsp; o 几何着色器  
 

<br>

### &emsp; o 片段着色器  
计算像素最后的颜色输出