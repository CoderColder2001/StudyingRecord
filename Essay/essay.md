### GaussianEditor: Swift and Controllable 3D Editing with Gaussian Splatting

arxiv202311  
通过自然语言编辑GS场景  

Background：  
基于MLP的方法对场景进行了隐式编码，很难进行inpaint和decompose  
（某种观点认为）与隐式表示的方法有神经网络作为缓冲不同，GS在训练时直接受到损失的随机性影响，gaussian的属性在训练中直接改变，导致训练不稳定  

问题：  
如何识别需要编辑的gaussian？（作者提出了 gaussian semantic trace）  
&emsp; 同时，gaussian semantic trace 可以视为一种动态的mask  
如何解决编辑时随机性带来的问题？（作者提出了 HGS）  
如何解决物体删除带来的边缘空洞or填入物体？ （作者提出了一种3D inpainting的方法）   
&emsp; image to 3Dmesh，再转为GS  

