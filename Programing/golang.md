[TOC]
# Golang

------
## Goroutine
概念上（使用上）是线程，实际上是线程+协程  
每个CPU上有一个 Go Worker，运行协程  
协程执行blocking API（如sleep、read）时：创建一个“事件”放在队列里，然后直接执行`yield()`切换到另一个协程    
（把所有同步IO变成异步IO）  

使用管道channels实现线程/协程之间的同步与通信（避免通过共享内存）  

<br>

------
## zerolog

------
## Hertz
<a herf = "https://www.cloudwego.io/zh/docs/hertz/"> Hertz官方文档</a>   

路由分组，并将中间件注册到路由组上：  
```go
auth := hertz.Group("/douyin", jwt.JwtMiddleware.MiddlewareFunc())
auth.POST("/favorite/action/", controller.FavoriteAction)
```

## Hertz 常用封装
### RequestContext 请求上下文
请求上下文 `RequestContext` 是用于保存 HTTP 请求和设置 HTTP 响应的上下文，它提供 API 接口帮助用户开发  
Hertz handler/middleware 函数签名：
```go
type HandlerFunc func(c context.Context, ctx *RequestContext)
```

`context.Context` 与 `RequestContext` 都有存储值的能力，具体选择使用哪一个上下文依据：所储存值的生命周期和所选择的上下文要匹配   
- `ctx` 主要用来存储请求级别的变量，请求结束就回收了，特点是查询效率高（底层是 map），协程不安全，且未实现 `context.Context` 接口
- `c` 作为上下文在中间件 `/handler` 之间传递，协程安全。所有需要 `context.Context` 接口作为入参的地方，直接传递 `c` 即可

除此之外，如果存在 *异步传递 `ctx`* 或 *并发使用 `ctx`* 的场景，hertz 也提供了 `ctx.Copy()` 接口，方便业务能够获取到一个协程安全的副本  
<br>

```go
// BindAndValidate binds data from *RequestContext to obj and validates them if needed.
// NOTE: obj should be a pointer.
func (ctx *RequestContext) BindAndValidate(obj interface{}) error {
	return ctx.getBinder().BindAndValidate(&ctx.Request, obj, ctx.Params)
}
```