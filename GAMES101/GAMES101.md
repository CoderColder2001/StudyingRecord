# GAMES101
wsl2配置环境：  
<https://www.bilibili.com/read/cv11143517>  
<https://zhuanlan.zhihu.com/p/371080057>  

------
## Content
- 1 Transformation
- 2 Rasterization
- 3 Shading

------
## 1 Transformation
- 模型变换
- 视图变换

用矩阵描述变换 变换前后的坐标关系   
### 二维变换、三维变换  
平移、缩放、旋转、切边  
齐次坐标：增加一个维度，为了将平移变换用矩阵表示   
新的维度 点为1，向量为0（向量具有平移不变性）  

旋转矩阵的逆就是旋转矩阵的转置（*旋转矩阵是正交矩阵*）  
三维空间中的旋转   
<img src = "./pic/c4_1.png" width = "80%">  
绕任意轴的选择（分解为x、y、z轴上的选择）：  
<img src = "./pic/c4_2.png" width = "80%">  
默认起点在原点上；若不在原点上，先平移到原点上 再平移回来  

旋转矩阵不适合做插值 四元数适合作旋转与旋转之间的插值    
<br>

---
模型变换M => 视图变换V => 投影变换P => `[-1,1]^3` 立方体    
### 视图变换 View/Camera Transformation
相机的定义（观测矩阵）：一个向量表示位置、一个向量表示观测方向、一个向量表示向上方向   
约定假设相机是固定的（at Origin, up at Y, look at -Z） 而是去移动场景（模型变换）  
<img src = "./pic/c4_3.png" width = "80%">  
（模型变换与视图变换常常是一起的 都作用于物体）  

### 投影变换 Projection Transformation  
- 正交投影 Orthographic Projection
- 透视投影 Perspective Projection

### **正交投影**：映射一个cuboid（[l, r] * [b, t] * [f, n]）到 标准canonical cube（[-1, 1]^3） 
<img src = "./pic/c4_4.png" width = "80%">  

先平移、再缩放  
look at -z（右手系）导致了 `n > f` （near > far）  

### **透视投影（视锥）**：映射一个视锥到标准canonical cube（[-1, 1]^3）
先**保持近平面不变，缩放远平面及中间平面**（z不变，收缩x、y），得到一个cuboid；再作**正交投影**（两个矩阵相乘）  
<img src = "./pic/c4_5.png" width = "80%">   
利用 *近平面上的点不变、远平面的中心点不变*， 求解出透视投影矩阵   
``` c++
Mpersp2otho << zNear,0,0,0,
               0,zNear,0,0,
               0,0,zNear + zFar,-zNear*zFar,
               0,0,1,0;
```
处于视锥中间的点，在透视投影变换后会被推向远平面  
可参考<https://zhuanlan.zhihu.com/p/445801392>   
<br>

------
## 2 Rasterization
屏幕：一个二维数组（元素对应像素、数组大小对应分辨率）  
多边形映射到像素  

三角形的性质：
- 任意多边形都能拆分成三角形
- 三角形内部一定是一个平面
- 三角形内外定义清晰
- 定义顶点属性后，三角形内部可以通过插值计算（通过三角形<b>重心坐标</b>）  

光栅化的关键问题：**判断一个像素（的中心点）与三角形的位置关系**  
采样：离散化一个函数  
**利用像素中心对屏幕空间进行采样**  

光栅化加速：
- 只遍历三角形包围盒（轴向AABB包围盒）内像素

采样带来的锯齿问题（采样率对于信号来说不够高）  

### 反走样（频域）  
走样产生的原因：同样的采样频率去采样两种不同函数却能得到相同的结果  
在采样前先做一次模糊（滤波）  
滤波：去除特定频段  
  
傅里叶变换（展开）：将任意函数分解成不同的频率（从时域到频域）   
<img src = "./pic/c6_1.png" width = "80%">   
时域的卷积对应频域的乘积  

### 可见性（遮挡）问题
画家算法：先画远的； 但无法解决几个互相遮挡的三角形的绘制，故不可用  

Z-Buffer深度缓存：对每个像素（或采样点），记录最近的深度  
光栅化时，同时生成绘制图（frame-buffer）与深度图（z-buffer）  

<br>

------
## 3 Shading
对不同的物体应用不同的材质（和光线相互作用的不同方法）  
把着色分为三个部分（Bling-Phong着色模型）：镜面反射 + 漫反射 + 环境光照  

Shading is local（对于一个shading point 视作一个小平面）
<img src = "./pic/c7_1.png" width = "80%">   

shading的局部性：shading并不考虑阴影（不考虑其他物体的存在，只考虑自己）  

对于一个点单位面积接收到的光的能量与传播距离成反比   
Lambertian(Diffuse) Shading <b>漫反射项</b> 与观察方向无关（光线被均匀地反射出去）    
<img src = "./pic/c7_2.png" width = "80%">   
kd漫反射系数 定义对光线的吸收（等于0时为黑）  

<b>高光项</b> 观察方向和镜面反射方向接近，即法向量方向与半程向量方向接近  
<img src = "./pic/c8_1.png" width = "80%">   
<b>通过向量点乘衡量两个向量是否接近</b>  
夹角余弦加一个指数幂操作  

<b>环境光项</b> 本质是一个常数（的颜色）   

### 着色频率  
<img src = "./pic/c8_2.png" width = "80%">   

着色应用在每一个像素上   
插值计算三角形内部点的颜色  
法线在三角形内部进行插值（每一个像素都有自己的法线）  

### Real-time Rendering Pipeline
<img src = "./pic/c8_3.png" width = "80%">   

Shader对每一个顶点或每一个fragment（如像素）都会执行一次  
- Program vertex and fragment processing stages
- Describe operation on a single vertex（or fragment）

<https://www.shadertoy.com/>  

### 纹理映射
定义不同物体表面不同位置的任意点的属性  
空间中的三角形（顶点）映射到纹理（二维坐标`(u, v)`）上  
不同位置可以映射到相同纹理上  

如果纹理相对于分辨率太小了怎么办？（在查询纹理时，如何处理计算结果为非整数坐标所对应的值？若直接取整，纹理太小时会导致模糊）  
<b>双线性插值</b> 查找这个 “非整数点” 周边的四个点  
<img src = "./pic/c9_3.png" width = "80%">   

如果纹理过大（一个像素覆盖纹理区域过大） 会产生走样问题   
每一个像素内使用多个采样点（超采样）计算代价高昂  

### 重心坐标
为什么需要插值？ 得到平滑的过度  
插值的应用场景：各类逐顶点定义的属性，如纹理坐标、颜色、法线等  

也可以用重心坐标是否都大于0判断点是否在三角形内   
<a href = "https://zhuanlan.zhihu.com/p/144360079">计算机图形学补充1：重心坐标(barycentric coordinates)详解及其作用</a>  
<img src = "./pic/c9_1.png" width = "80%">   

重心坐标也对应于这个点将三角形分割成三个小三角形的面积  
<img src = "./pic/c9_2.png" width = "80%">   

重心坐标的问题：重心坐标在投影作用下不一定不变  
需要在三维空间中计算重心坐标和插值（如计算深度时），而不能在投影后再做  