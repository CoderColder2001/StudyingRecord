# Content
- BASIC
- Vue 3


---
# BASIC
## HTML
标签 + 元素 + 属性  
本质是是传达内容  
标签：标识<b>语义</b>  

| 标签名 | 定义     | 说明 |
|:---:|---|---|
| `<html></html>` | HTML标签 | 文页面中最大的标签，我们成为根标签 |
| `<head></head>` | 文档的头部 | *包含字符集、关键词、页面描述、页面标题、IE适配、视口、iPhone小图标等；* 注意在`head`标签中我们必须要设置的标签是 `title` |
| `<title></title>`| 文档的标题 | 网页标题 |
| `<body></body>` | 文档的主体 | 元素包含文档的所有 **内容**，页面内容基本都是放到body里面的 |

HTML只在乎标签的嵌套结构，对换行不敏感，~~对tab不敏感~~；所有文字间，如果有空格、换行、tab都将被折叠为一个空格显示  

`<!-- 我是 html 注释  -->`  
标题使用`<h1>`至`<h6>`标签进行定义。`<h1>`定义最大的标题，`<h6>`定义最小的标题。具有`align`属性，属性值可以是：`left`、`center`、`right`  
`<p>`段落标签  
`<div>`默认换行 `<span>`不换行  

html元素：  
- 块级元素
- 行内元素
- inline-block：如form表单元素。对外的表现是行内元素（不会独占一行），对内的表现是块级元素（可以设置宽高）


<br>

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

<br>

---
### 条件渲染
`v-if` &emsp; `v-else-if` &emsp; `v-else`  
可以在一个 `<template>` 元素（不可见的包装多个元素的包装器）上使用 `v-if`  

`v-show` 会在 DOM 渲染中保留该元素；`v-show` 仅切换了该元素上名为 display 的 CSS 属性

<br>

---
### 列表渲染
基于一个数组来渲染一个列表  
也可以在 `<template>` 标签上使用 `v-for` 来渲染一个包含多个元素的块  

```javascript
const parentMessage = ref('Parent')
const items = ref([{ message: 'Foo' }, { message: 'Bar' }])
```
```html
<li v-for="(item, index) in items">
  {{ parentMessage }} - {{ index }} - {{ item.message }}
</li>
```

也可以使用 `v-for` 来遍历一个对象的所有属性   
```html
<script setup>
import { reactive } from 'vue'

const myObject = reactive({
  title: 'How to do lists in Vue',
  author: 'Jane Doe',
  publishedAt: '2016-04-10'
})
</script>

<template>
	<ul>
    <li v-for="(value, key, index) in myObject">
		  {{ index }}. {{ key }}: {{ value }}
		</li>
  </ul>
</template>
```

<br>

---
### 侦听器
使用 `watch` 函数在每次响应式状态发生变化时触发回调函数    
第一个参数可以是不同形式的“数据源”：它可以是一个 **ref (包括计算属性)**、一个**响应式对象**、一个 **getter 函数:** `() => obj.count`、或 **多个数据源组成的数组**  

```html
<script setup>
import { ref, watch } from 'vue'

const question = ref('')
const answer = ref('Questions usually contain a question mark. ;-)')

watch(question, async (newQuestion, oldQuestion) => {
  if (newQuestion.indexOf('?') > -1) {
    answer.value = 'Thinking...'
    try {
      const res = await fetch('https://yesno.wtf/api')
      answer.value = (await res.json()).answer
    } catch (error) {
      answer.value = 'Error! Could not reach the API. ' + error
    }
  }
})
</script>

<template>
  <p>
    Ask a yes/no question:
    <input v-model="question" />
  </p>
  <p>{{ answer }}</p>
</template>
```

通过传入 `immediate: true` 选项来强制侦听器的回调在创建侦听器时立即执行  

可以用 `watchEffect` 函数自动跟踪回调的响应式依赖  

```javascript
watchEffect(async () => {
  const response = await fetch(
    `https://jsonplaceholder.typicode.com/todos/${todoId.value}`
  )
  data.value = await response.json()
})
```
回调会立即执行，不需要指定 `immediate: true`; 在执行期间，它会自动追踪 `todoId.value` 作为依赖（和计算属性类似）  

当更改了响应式状态，它可能会同时触发 Vue 组件更新和侦听器回调。默认情况下，用户创建的侦听器回调，都会在 Vue 组件更新之前被调用。这意味着你在侦听器回调中访问的 DOM 将是被 Vue 更新之前的状态；如果想在侦听器回调中能访问被 Vue 更新之后的 DOM，你需要指明 `flush: 'post'` 选项  

在 `setup()` 或 `<script setup>` 中用**同步**语句创建的侦听器，会自动绑定到宿主组件实例上，并且会在宿主组件卸载时自动停止； 如果用异步回调创建一个侦听器，那么它不会绑定到当前组件上，必须调用 `watch` 或 `watchEffect` 返回的函数（） 手动停止它，以防内存泄漏
```javascript
const unwatch = watchEffect(() => {})
unwatch() // ...当该侦听器不再需要时
```

如果需要等待一些异步数据，你可以使用条件式的侦听逻辑：
```javascript
// 需要异步请求得到的数据
const data = ref(null)

watchEffect(() => {
  if (data.value) {
    // 数据加载后执行某些操作...
  }
})
```

<br>

---
### 模板引用  
<b>直接访问底层 DOM 元素</b>  
多数情况下，应该首先使用标准的 `props` 和 `emit` 接口来实现父子组件交互  
