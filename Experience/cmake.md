# CMake
记录一些与理解cmake有关的文档、博客

------
<a href = "https://bangbang.blog.csdn.net/article/details/131158601?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-131158601-blog-108830901.235%5Ev38%5Epc_relevant_default_base&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-131158601-blog-108830901.235%5Ev38%5Epc_relevant_default_base&utm_relevant_index=1">CMake学习(7): CMake的嵌套</a>

<a href = "https://subingwen.cn/cmake/CMake-primer/">CMake 保姆级教程（上）</a>

------
`link_libraries`用来链接静态库，`target_link_libraries`用来链接导入库，即按照header file + .lib + .dll方式隐式调用动态库的.lib库   

`target_include_directories()` 的功能完全可以使用 `include_directories()` 实现。但是还是建议使用`target_include_directories()` 以保持清晰  

`include_directories(header-dir)` 是一个全局包含，向下传递。如果某个目录的 CMakeLists.txt 中使用了该指令，其下所有的子目录默认也包含了`header-dir`目录  
