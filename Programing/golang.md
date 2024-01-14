# Golang

------
## zerolog

## Hertz
<a herf = "https://www.cloudwego.io/zh/docs/hertz/"> Hertz官方文档</a>   

路由分组并将中间件注册到路由组上：  
```go
auth := hertz.Group("/douyin", jwt.JwtMiddleware.MiddlewareFunc())
auth.POST("/favorite/action/", controller.FavoriteAction)
```

## 常用封装

### RequestContext
```go
// BindAndValidate binds data from *RequestContext to obj and validates them if needed.
// NOTE: obj should be a pointer.
func (ctx *RequestContext) BindAndValidate(obj interface{}) error {
	return ctx.getBinder().BindAndValidate(&ctx.Request, obj, ctx.Params)
}
```