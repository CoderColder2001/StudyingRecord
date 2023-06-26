# GAMES101

## Content
- 1 Transformation

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
旋转矩阵不适合做插值  
<br>

---
模型变换 => 视图变换 => 投影变换  
### 视图变换 View/Camera Transformation
相机的定义（观测矩阵）：一个向量表示位置、一个向量表示观测方向、一个向量表示向上方向   
假设相机是固定的（at Origin, up at Y, look at -Z） 而是去移动场景（模型变换）  
<img src = "./pic/c4_1.png" width = "80%">  

### 投影变换 Projection Transformation  
- 正交投影 Orthographic Projection
- 透视投影 Perspective Projection

正交投影：映射一个cuboid（[l,r] * [b,t] * [f, n]）到 标准canonical cube（[-1, 1]^3）  
<img src = "./pic/c4_2.png" width = "80%">   

透视投影：先保持近平面不变，缩放远平面及中间平面（z不变，收缩x、y），得到一个cuboid；再作正交投影（两个矩阵相乘）  
利用 *近平面上的点不变、远平面的中心点不变*， 求解出透视投影矩阵   
<br>

------