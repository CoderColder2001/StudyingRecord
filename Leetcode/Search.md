## Content
- BFS/DFS
- 状态压缩 + BFS/DFS
- 图论
  
<br>

------
## BFS/DFS
### 概念
---
---
### 题目
---
### &emsp; 1263. 推箱子 :rage: HARD
关键思路：
- <b>使用BFS寻找最短路</b>
- 人的单点可达性问题：对人BFS，再判断人这次移动是否推动箱子
- 因为是对人BFS而要求箱子最短路，<b>使用优先队列存状态</b>，比较箱子移动的距离
- 使用`set` 记录已经访问过的 { 人的位置，箱子的位置 }

<details> 
<summary> <b>C++ Code</b> </summary>

``` c++
class Solution {
public:
    int minPushBox(vector<vector<char>>& grid) {
        // [0]最小步数 [1][2]人坐标 [3][4]箱子坐标
        priority_queue<vector<size_t>, vector<vector<size_t>>, greater<vector<size_t>>> pq;
        size_t m = grid.size();
        size_t n = grid[0].size();

        vector<size_t> start (5, 0);
        for(size_t x = 0; x < m; x++)
        {
            for(size_t y = 0; y < n; y++)
            {
                if(grid[x][y] == 'S')
                {
                    start[1] = x;
                    start[2] = y;
                    grid[x][y] = '.';
                }
                else if(grid[x][y] == 'B')
                {
                    start[3] = x;
                    start[4] = y;
                    grid[x][y] = '.';
                }
            }
        }
        pq.push(start);
        set<vector<size_t>> dist; // 记录走过的位置
        dist.insert({start[1], start[2], start[3], start[4]});

        int dx[4] = {0, 0, 1, -1};
        int dy[4] = {1, -1, 0, 0};
        while(!pq.empty()) // BFS人的移动 并判断是否能推动箱子
        {
            auto v = pq.top();
            pq.pop();
            for(int i = 0; i < 4; i++)
            {
                vector<size_t> next_s = {v[1] + dx[i], v[2] + dy[i]};
                if (next_s[0] >= m || next_s[1] >= n || grid[next_s[0]][next_s[1]] == '#')
				    continue;
                vector<size_t> next_b = { v[3], v[4] };
			    size_t dis = v[0];
                if (next_s == next_b) // 推动箱子
			    {
				    next_b[0] += dx[i];
				    next_b[1] += dy[i];
				    if (next_b[0] >= m || next_b[1] >= n || grid[next_b[0]][next_b[1]] == '#')
					    continue;
				    dis++;
                    if (grid[next_b[0]][next_b[1]] == 'T') // 是否到达终点
				        return (int)dis;
			    }

                if (dist.find({next_s[0], next_s[1], next_b[0], next_b[1]}) != dist.end())
				    continue;
                
                dist.insert({next_s[0], next_s[1], next_b[0], next_b[1]});
			    pq.push({dis, next_s[0], next_s[1], next_b[0], next_b[1]});
            }
        }
        return -1;
    }
};
```
</details>
<br>

---
### &emsp; 1377. T秒后青蛙的位置 :rage: HARD
关键思路：
- <b>DFS 从根节点出发搜索target</b>
- 如何避免浮点数精度丢失：由于答案由若干分子为1的分数相乘得到，改为计算分母的乘积，最后再求倒数
- 两种DFS的实现：
  - 自顶向下：“递”；一边计算乘积，一边看是否到达target
  - 自底向上： 在“归”的过程中做乘法；在不含target的子树中搜索时，不会盲目地做乘法
- 把根节点1添加一个0号邻居，避免了特判根节点的情况。
- DFS 中的时间不是从 0 开始增加到 t，而是从 `leftT=t` 开始减小到 0，这样代码中只需和 0 比较，无需和 t 比较，从而减少一个 DFS 之外变量的引入。

<details> 
<summary> <b>C++ Code</b> </summary>

``` c++
class Solution {
public:
    double frogPosition(int n, vector<vector<int>>& edges, int t, int target) {
        vector<vector<int>> g(n+1); // 邻接表
        g[1] = {0}; //把节点1添加一个0号邻居 从而避免判断当前节点为根节点1，也避免了特判n=1 的情况。
        for(auto &e : edges) {
            int x = e[0], y = e[1];
            g[x].push_back(y);
            g[y].push_back(x);
        }

        function<long long(int, int, int)> dfs = [&](int x, int fa, int left_t) -> long long {
            if(left_t == 0)
                return x == target;
            if(x == target)
                return g[x].size() == 1; // 恰好t秒后到达 或叶节点留在原地
            
            for(int y : g[x])
            {
                if(y != fa)
                {
                    auto prod = dfs(y, x, left_t - 1);
                    if(prod != 0)
                        return prod * (g[x].size() - 1); // 乘上孩子个数返回
                }
            }
            return 0;
        };
        auto prod = dfs(1, 0, t);
        return prod ? 1.0/prod : 0;
    }
};
```
</details>
<br>

------
## 状态压缩 + BFS/DFS
### 概念
---
关键在于考虑 <b>抽象状态的定义与转移</b>

<br>

---
### 题目
---
### &emsp; 864. 获取所有钥匙的最短路径 :rage: HARD
关键思路：
- `int state` 存储二进制状态 表示当前是否持有某个钥匙  
- 最终状态定义 <b>(x,y,keys)</b> &emsp; 即当前坐标与当前所持有的钥匙   
- <b>状态转移</b>：从队列头取出一个点，可获得此处的位置和携带钥匙，向四个方向查看是否能走。  
- “ 能走 ” 等价于： 下个状态不越界 && 没访问过下个状态 && （下个状态是通路 ||（下个状态是大写字母 && 当前我有下个状态需要的钥匙））

<details> 
<summary> <b>C++ Code</b> </summary>

``` c++
class Solution {
public:
    const static int N = 35, K = 7, INF = 0x3f3f3f3f; 
    bool states[N][N][1<<K];
    int dx[4] = {0, 1, 0, -1};
    int dy[4] = {-1, 0, 1, 0};

    int shortestPathAllKeys(vector<string>& grid) {
        int m = grid.size();
        int n = grid[0].size();
        int k = 0; // 钥匙&锁数量
        int start_x = -1, start_y = -1;

        // init
        for(int i = 0; i < grid.size(); i++)
        {
            for(int j = 0; j < grid[i].size(); j++)
            {
                char c = grid[i][j];
                if(c == '@')
                {
                    start_x = i;
                    start_y = j;
                }
                else if(c <= 'z' && c >= 'a')
                    k++;
            }
        }

        //BFS
        int step = 0;
        queue<tuple<int, int, int>> q;
        q.emplace(start_x, start_y, 0);
        while(!q.empty())
        {
            int qSize = q.size(); // 注意size()不能写在循环判断内
            for(int i = 0; i < qSize; i++)
            {
                auto [x, y, keyState] = q.front();
                q.pop();
                if(keyState == (1 << k) - 1) // 找到了所有钥匙
                    return step;
                    
                for(int move = 0; move < 4; move++)
                {
                    int _x = x + dx[move];
                    int _y = y + dy[move];
                    if(_x >= 0 && _x < m && _y >= 0 && _y < n) // 边界范围内移动
                    {
                        char c = grid[_x][_y];
                        int _keyState = keyState;
                        if(c == '#') // 墙
                            continue;
                        else if(c <= 'Z' && c >= 'A' && !(_keyState>>(c-'A') & 1)) // 还没有钥匙
                            continue;
                        else if(c <= 'z' && c >= 'a') // 得到钥匙
                            _keyState = _keyState | 1<<(c-'a');

                        if(!states[_x][_y][_keyState]) // 未访问过这个状态
                        {
                            states[_x][_y][_keyState] = true;
                            q.emplace(_x, _y, _keyState);
                        }
                    }
                }
            }
            step++;
        }

        return -1;
    }
};
```
</details>

<br>

------
## 图论
### 概念
---
### <b>存图方式</b>  
### 1. 邻接矩阵
使用二维矩阵，适用于边数较多的稠密图时（边数约达到点数的平方） 

### 2. 邻接表（链式前向星）
适用于边数较少的稀疏图  
`he[N]`：存放某个节点对应边链表的头节点  
`e[M]`：存放某条边指向的节点  
`ne[M]`：访问对应点的边链表上的下一条边  
`w[M]`：存放边的权重值  

### 3. 创建一个Edge类

<br>

### <b>算法</b>
- ###  Dijkstra O(n^2)
BFS思想 求单源最短路 不能有负边  
`dis[N]`：保存当前最短路长度  
`vis[N]`：保存已经找到最短路的节点  
每一轮最小`dis`的顶点在`vis`中标记 &emsp; 再用它的边来更新`dis`

堆优化Dijkstra（优先队列存放） O(mlogn)  

<br>

- ### BellmanFord O(n*m)
DP思想 求单源最短路 可以有负边 可以判断存在负环（最短路不存在）  
迭代超过 V-1 次，就说明存在负权回路  
使用邻接表或类存边  
每一轮对所有边`e(a,b,w)`进行松弛操作，若`dis[b] > dis[a]+w` 则更新dis[b]  

队列优化BellmanFord —— SPFA（使用邻接表）

<br>

---
### 题目
---
### &emsp; 787. K站中转内最便宜的航班 MID
关键思路：
- 有边数限制的最短路 使用 <b>BellmanFord</b>
- 需要注意的是，在遍历所有的“点对/边”进行松弛操作前，需要先对 dis 进行备份，否则会出现「本次松弛操作所使用到的边，也是在同一次迭代所更新的」  

<details>
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    int findCheapestPrice(int n, vector<vector<int>>& flights, int src, int dst, int k) {
        int m = flights.size();
        const int inf = INT_MAX>>1;
        vector<int> dis(n, inf);
        vector<int> _dis;
        dis[src] = 0;
        for(int limit = 0; limit < k + 1; limit++)
        {
            _dis = dis;
            for(const auto &edge : flights)
            {
                int from = edge[0], to = edge[1], cost = edge[2];
                dis[to] = min(dis[to], _dis[from] + cost);// 使用上一轮的_dis更新
            }
        }
        return dis[dst] < inf ? dis[dst]:-1;
    }
};
```
</details>
<br>

---
### &emsp; 2699. 修改图中的边权 :rage: HARD
关键思路：
- 坑：改变边权后 最短路的选择可能改变
- <b>两遍 Dijkstra</b>，第一遍求出差值`delta`，第二遍修改边
- 由于 <b>Dijkstra 算法保证每轮拿到的点的最短路就已经是最终的最短路</b>；按照 Dijkstra 算法遍历点/边的顺序去修改 就不会对已确定的最短路产生影响
- 为使 `dis[x][1] + wt + (dis[des][0] - dis[y][0]) = target`，故修改 `wt = delta + dis[y][0] - dis[x][1]`
- 在计算权值时，无需考虑最短路还走不走这条边（不考虑这条边增大到多少时开始对 `dis[des]`不产生影响）
- 如果第二遍 Dijkstra 跑完后，从起点到终点的最短路仍然小于`target`，那么就说明无法修改，返回空数组
- 注意在建图的数据结构定义（如链式前向星、邻接表）上，如何在修改边权的时候更新图上双向边权（如何快速找到反向边的存放位置）

<details>
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    vector<vector<int>> modifiedGraphEdges(int n, vector<vector<int>>& edges, int source, int destination, int target) {
        vector<pair<int, int>> g[n]; // 邻接表 {to, edgeID}
        for(int i = 0; i < edges.size(); i++)
        {
            int x = edges[i][0], y = edges[i][1];
            g[x].emplace_back(y, i); // 额外记录边的ID 用于访问or修改权值
            g[y].emplace_back(x, i);
        }

        int dis[n][2]; // 两次dijkstra
        int delta = -1; // 第一次最短路与target的差值
        int vis[n];
        memset(dis, 0x3f3f3f3f, sizeof(dis));
        dis[source][0] = dis[source][1] = 0;

        auto dijkstra = [&](int k) {
            memset(vis, 0, sizeof(vis));
            for(int i = 0; i < n; i++)
            {
                // 找到当前节点最短路 去更新其邻居的最短路
                int x = -1;
                for(int i = 0; i < n; i++) // 找一个尚未vis中已求dis最近的节点
                {
                    if(!vis[i] && (x < 0 || dis[i][k] < dis[x][k]))
                        x = i;
                }

                if(x == destination)
                    return;

                vis[x] = true;
                for(auto [y, eid] : g[x]) // 边xy
                {
                    int wt = edges[eid][2];
                    if(wt == -1)
                        wt = 1;
                    if(k == 1 && edges[eid][2] == -1)
                    {
                        // 第二次 Dijkstra，改成 w
                        int w = delta + dis[y][0] - dis[x][1];
                        // y后面的边还没修改 用第一遍dijkstra的结果
                        if (w > wt)
                            edges[eid][2] = wt = w; // 直接在 edges 上修改
                    }
                    dis[y][k] = min(dis[y][k], dis[x][k] + wt); // 更新y的最短路
                }
            }
        };

        dijkstra(0);
        delta = target - dis[destination][0];
        if(delta < 0)
            return {}; // 全改成1也大于target
        
        dijkstra(1);
        if(dis[destination][1] < target)
            return {}; // 最短路无法再变大
        
        for (auto &e: edges)
            if (e[2] == -1) // 剩余没修改的边全部改成 1
                e[2] = 1;
        return edges;
    }
};
```
</details>
<br>