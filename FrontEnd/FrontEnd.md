# Content
- BASIC
- Vue 3


---
# BASIC
## HTML
本质是是传达内容  
标签标识语义

## CSS
定义页面元素的样式  
- 设置字体和颜色
- 设置位置和大小
- 添加动画效果

```css
选择器Selector {
    选择器 Property属性：属性值;
}
```

## JavaScript
类型：除了基本类型外，其他都是Object  
函数也是Object
```javascript
function Person(first, last) {
    this.first = first;
    this.last = last;
}
Person.prototype.fullName = function() {
    return this.first + ' ' + this.last;
}
Person.prototype.fullNameReversed = function() {
    return this.last + ', ' + this.first;
}
```
`Person.prototype` 是一个可以被 `Person` 的所有实例共享的对象。它是一个名叫原型链（prototype chain）的查询链的一部分：当你试图访问 `Person` 某个实例一个没有定义的属性时，解释器会首先检查这个 `Person.prototype` 来判断是否存在这样一个属性   
JavaScript 允许你在程序中的任何时候修改原型（prototype）中的一些东西，也就是说你可以在运行时 (runtime) 给已存在的对象添加额外的方法  

<br>

**JavaScript中的嵌套函数可以访问父函数作用域中的变量**     
*如果某个函数依赖于其他的一两个函数，而这一两个函数对你其余的代码没有用处，你可以将它们嵌套在会被调用的那个函数内部，这样做可以减少全局作用域下的函数的数量，这有利于编写易于维护的代码*  

<br>

------
## Vue 3
**声明式的、组件化的编程模型**  

Vue的两个核心功能：
- 声明式渲染：Vue 基于标准 HTML 拓展了一套模板语法，使得我们可以声明式地描述最终输出的 HTML 和 JavaScript 状态之间的关系。
- 响应性：Vue 会自动跟踪 JavaScript 状态并在其发生变化时响应式地更新 DOM

单文件组件 `***.vue` 将一个组件的逻辑 (JavaScript)，模板 (HTML) 和样式 (CSS) 封装在同一个文件里  
 
API风格：  
- 选项式API  
    用包含多个选项的对象来描述组件的逻辑，例如 `data`、`methods` 和 `mounted`。选项所定义的属性都会暴露在函数内部的 `this` 上，它会指向当前的组件实例
- **组合式API** （推荐）  
    使用导入的 `API函数` 来描述组件逻辑。单文件组件中，组合式 API 通常会与 `<script setup>` 搭配使用  
    **直接在函数作用域内定义响应式状态变量，并将从多个函数中得到的状态组合起来处理复杂问题**

<br>

---
### 模板语法
指令由 `v-` 作为前缀，表明它们是一些由 Vue 提供的特殊 attribute，它们将为渲染的 DOM 应用特殊的响应式行为  
模板中的表达式仅能够访问到有限的全局对象列表   
指令 attribute 的期望值为一个 JavaScript 表达式 （除了 v-for、v-on 和 v-slot）。一个指令的任务是 *在其表达式的值变化时响应式地更新 DOM*   

`v-bind` 指令指示 Vue 将元素的 `id` attribute 与组件的 `dynamicId属性`保持一致（响应式地更新一个 HTML attribute）。如果绑定的值是 null 或者 undefined，那么该 attribute 将会从渲染的元素上移除  

`v-on` 指令（缩写为 `@`） 将监听 DOM 事件   

在指令参数上也可以使用一个 JavaScript 表达式，需要包含在方括号 [ ] 内

可以在绑定的表达式中使用一个组件暴露的方法，但绑定在表达式中的方法在组件每次更新时都会被重新调用，因此**不应该产生任何副作用**，比如改变数据或触发异步操作

<br>

---
### 响应式
使用 `reactive()函数` 创建一个响应式对象或数组  
响应式对象其实是 JavaScript Proxy，其行为表现与一般对象相似。不同之处在于 Vue 能够跟踪对响应式对象属性的访问与更改操作    
状态是默认深层响应式的  
`reactive()` 返回的是一个原始对象的 Proxy，只有代理对象是响应式的，更改原始对象不会触发更新   
为保证访问代理的一致性，对同一个原始对象调用 `reactive()` 会总是返回同样的代理对象，而对一个已存在的代理对象调用 `reactive()` 会返回其本身  
`reactive` API 的局限性：
- **仅对对象类型有效**（对象、数组和 Map、Set 这样的集合类型），而对 string、number 和 boolean 这样的 **原始类型** 无效
- 不可以随意地“替换”一个响应式对象，这将导致对初始引用的响应性连接丢失；将响应式对象的属性赋值或解构至本地变量时，或是将该属性传入一个函数时，会失去响应性

`ref()` 方法来创建可以使用任何值类型的响应式 ref，将传入参数的值包装为一个带 `.value` 属性的 ref 对象  
`ref()` 能创造一种对任意值的 “引用”，并能够在不丢失响应性的前提下传递这些引用。 

<br>

---
### 计算属性
使用计算属性来描述<b>依赖响应式状态的复杂逻辑</b>  
计算属性默认是只读的
```html
<script setup>
import { reactive, computed } from 'vue'

const author = reactive({
  name: 'John Doe',
  books: [
    'Vue 2 - Advanced Guide',
    'Vue 3 - Basic Guide',
    'Vue 4 - The Mystery'
  ]
})

// computed() 方法期望接收一个 getter 函数，返回值为一个计算属性 ref
const publishedBooksMessage = computed(() => {
  return author.books.length > 0 ? 'Yes' : 'No'
})
</script>

<template>
  <p>Has published books:</p>
  <span>{{ publishedBooksMessage }}</span>
</template>

```
若将同样的函数定义为一个方法而不是计算属性，两种方式在结果上确实是完全相同的，然而，不同之处在于 **计算属性值会基于其响应式依赖被缓存。一个计算属性仅会在其响应式依赖更新时才重新计算**，而方法调用总是会在重渲染发生时再次执行函数  

<br>

---
### 类与样式绑定
为 `class` 和 `style` 的 `v-bind` 用法提供了特殊的功能增强；除了字符串外，表达式的值也可以是对象或数组   

可以直接绑定一个样式对象  
```javascript
const styleObject = reactive({
  color: 'red',
  fontSize: '13px'
})
```
```html
<div :style="styleObject"></div>
```

如果样式对象需要更复杂的逻辑，也可以使用返回样式对象的计算属性  