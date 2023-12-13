# Paper Reading Record
## Content
-

------
## GaussianEditor: Swift and Controllable 3D Editing with Gaussian Splatting

arxiv202311  
keywords：3D编辑；GS；  

通过自然语言编辑GS场景；利用GS的显式表示的典型特性来提升3D编辑的效果；  
采用2D diffusion model进行编辑    

### Background：  
基于高维MLP的方法对场景进行了隐式编码，很难进行inpaint和decompose  
（某种观点认为）与隐式表示的方法有神经网络作为缓冲不同，GS在训练时直接受到损失的随机性影响，gaussian的属性在训练中直接改变，导致训练不稳定  

3D编辑的两种思路：
- 将3D模型的噪声渲染作为输入；由扩散模型生成的分数来指导模型更新
- 根据prompt对3D模型的多个视图进行2D编辑

### 问题：  
1、如何识别需要编辑的gaussian？（作者提出了 **gaussian semantic trace**）  
&emsp; 同时，gaussian semantic trace 可以视为一种动态的mask  
2、如何解决编辑时随机性（具有高度随机性的生成引导）带来的问题？（作者提出了 **HGS**）  
3、如何解决物体删除带来的边缘空洞or填入物体？ （作者提出了一种 **3D inpainting** 的方法）   
&emsp; image to 3Dmesh，再转为GS  

### Gassian semantic tracing
*确保只有目标相关区域被修改，使得编辑精准、可控*  
将二维分割mask投影到三维高斯，并为每个高斯分配一个语义属性（j类语义标签）  
编辑时只更新目标高斯  
致密化过程中，新致密的高斯继承其父高斯的语义属性  
对于一组3D高斯，生成多个渲染图并运用2D分割，再将2D分割语义标签投影回GS模型   
为每个高斯维护一个权重和一个计数器   
`wij`表示第i个高斯对第j个标签的权重；根据一个高斯分布的平均权重是否超过一个手动设置的阈值来确定它是否属于第j个语义类    

### Hierarchical Gaussian splatting (HGS)
*对于生成式任务，使能够连续优化出更好的结果，从而模拟了通过神经网络隐式表示实现的缓冲函数*   
基于在编辑任务的训练过程的多个致密化过程中的序列，将GS组织成代  
早期致密化中形成的高斯具有更严格的约束（保持原始状态）  
引入Anchor Loss；在各属性的锚点状态和当前状态之间分别计算MSE损失，确保高斯不会偏离各自的锚点状态太远  

### 3D inpainting algorithm
*对于去除对象后的局部修复算法，以及提供prompt和2Dmask的对象添加算法*  
对于要添加的物体，先使用2D inpainting，再使用image to 3D转换成粗糙网格，再转成HGS并精细化   

---