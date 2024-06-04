# 搜索 & 图论
[TOC]
## Content
- BFS/DFS
- 状态压缩 + BFS/DFS
- 图论
- 连通性问题
  
<br>

------
## BFS/DFS
### 概念
关于 路径 / “路径上的状态”    

---
### 题目
---
### &emsp; 365. 水壶问题 MID
关键思路：
- “状态转移” 实际上也是一个图
- 定义状态 `(a, b)`：当前A、B水壶的水量
- <b>使用BFS遍历状态空间</b>，unordered_set存储以及达到过的状态

<details> 
<summary> <b>C++ Code</b> </summary>

``` c++
class Solution {
public:
    // 自定义pair的哈希函数
    struct pairHash
    {
        template<class T1, class T2>
        size_t operator() (pair<T1, T2> const &pair) const
        {
            size_t h1 = hash<T1>()(pair.first); // 用默认hash分别处理
            size_t h2 = hash<T2>()(pair.second);
            return h1^h2;
        }
    };

    bool canMeasureWater(int jug1Capacity, int jug2Capacity, int targetCapacity) {
        if(targetCapacity == 0)
            return true;
        if(jug1Capacity + jug2Capacity < targetCapacity)
            return false;
        
        pair<int, int> start(0, 0);
        queue<pair<int, int>> q;
        unordered_set<pair<int, int>, pairHash> visited;
        q.push(start);
        visited.insert(start);
        
        while(!q.empty()) // bfs
        {
            pair<int, int> now = q.front();
            visited.insert(now);
            q.pop();
            int cur1 = now.first, cur2 = now.second;
            if(cur1 == targetCapacity || cur2 == targetCapacity || cur1 + cur2 == targetCapacity)
                return true;
            
            // 下一状态
            if(visited.find({jug1Capacity, cur2}) == visited.end())
                q.push({jug1Capacity, cur2});
            if(visited.find({cur1, jug2Capacity}) == visited.end())
                q.push({cur1, jug2Capacity});
            if(visited.find({0, cur2}) == visited.end())
                q.push({0, cur2});
            if(visited.find({cur1, 0}) == visited.end())
                q.push({cur1, 0});
            if(visited.find({cur1 - min(cur1, jug2Capacity - cur2), cur2 + min(cur1, jug2Capacity - cur2)}) == visited.end())
                q.push({cur1 - min(cur1, jug2Capacity - cur2), cur2 + min(cur1, jug2Capacity - cur2)});
            if(visited.find({cur1 + min(cur2, jug1Capacity - cur1), cur2 - min(cur2, jug1Capacity - cur1)}) == visited.end())
                q.push({cur1 + min(cur2, jug1Capacity - cur1), cur2 - min(cur2, jug1Capacity - cur1)});
        }

        return false;
    }
};
```
</details>
<br>

---
### &emsp; 514. 自由之路 :rage: HARD
关键思路：  
- 定义状态： `(key拼到哪一位, ring指向哪一位)`
- <b>BFS 寻找由`(0, 0)` 到 `(m, i)`的最短路</b>
- 如果`ring[i] == key[j]`， 拼写`key[j]`并移动状态至`(j+1, i)`
- 否则向左转移动至状态`((i−1+n) mod n, j)` 或向右转移动至状态`((i+1) mod n,j)`

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    int findRotateSteps(string s, string t) {
        int n = s.length(), m = t.length();

        vector<vector<int>> vis(n, vector<int>(m + 1));
        vis[0][0] = true;

        vector<pair<int, int>> q = {{0, 0}}; // 当前BFS层的状态
        for(int step = 0; ; step++) 
        {
            vector<pair<int, int>> nxt;
            for(auto [i, j] : q)
            {
                if(j == m)
                    return step;
                
                if(s[i] == t[j])
                {
                    if(!vis[i][j + 1]) 
                    {
                        vis[i][j + 1] = true;
                        nxt.emplace_back(i, j + 1);
                    }
                    continue;
                }
                for(int i2 : {(i - 1 + n) % n, (i + 1) % n}) 
                {
                    if(!vis[i2][j])
                    {
                        vis[i2][j] = true;
                        nxt.emplace_back(i2, j);
                    }
                }
            }
            q = move(nxt); // 用move
        }
    }
};
```
</details> 
<br>

---
### &emsp; 994. 腐烂的橘子 MID
关键思路：  
- <b>多源BFS</b>

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
    int DIRECTIONS[4][2] = {
        {-1, 0}, {1, 0}, {0, -1}, {0, 1}
    }; // 四方向

public:
    int orangesRotting(vector<vector<int>>& grid) {
        int m = grid.size(), n = grid[0].size();
        int fresh = 0;
        vector<pair<int, int>> q;
        for(int i = 0; i < m; i++) 
        {
            for(int j = 0; j < n; j++) 
            {
                if(grid[i][j] == 1) 
                    fresh++; // 统计新鲜橘子个数
                else if(grid[i][j] == 2)
                    q.emplace_back(i, j); // 一开始就腐烂的橘子
            }
        }

        int ans = -1;
        while(!q.empty()) 
        {
            ans++; // 经过一分钟
            vector<pair<int, int>> nxt;
            for(auto& [x, y] : q) // 上一轮腐烂的橘子
            {
                for (auto d : DIRECTIONS) 
                {
                    int i = x + d[0], j = y + d[1];
                    if(0 <= i && i < m && 0 <= j && j < n && grid[i][j] == 1) // 新鲜橘子 
                    {
                        fresh--;
                        grid[i][j] = 2; // 变成腐烂橘子
                        nxt.emplace_back(i, j);
                    }
                }
            }
            q = move(nxt); // 更新q
        }

        return fresh ? -1 : max(ans, 0);
    }
};
```
</details> 
<br>

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

---
### &emsp; 1553. 吃掉N个橘子的最少天数 :rage: HARD
关键思路：
- 用 <b>求最短路</b> 的方式思考状态的转移；可以执行的操作 对应 节点间连接的边
- 对于“减一”操作，只有在不能整除2或3的时候才会做；故可以设置修改边权（节点间距离）为`x % d + 1`，合并“除`d`”与此前的“减一”操作
- Dijkstra，堆中存放`（最短路值，节点id）`；答案对应 n 到 0 的最短路
- 代码实现时，无需建图，根据出堆的数字 `x` 计算出对应的邻居和边权

<details> 
<summary> <b>C++ Code</b> </summary>

``` c++
class Solution {
public:
    int minDays(int n) {
        unordered_map<int, int> dis;
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq;
        pq.emplace(0, n);
        while(true)
        {
            auto [dx, x] = pq.top();
            pq.pop();
            if(x <= 1)
                return dx + x;
            if(dx > dis[x])
                continue;

            for(int d = 2; d <= 3; d++)
            {
                int y = x / d;
                int dy = dx + x % d + 1;
                if(!dis.contains(y) || dy < dis[y])
                {
                    dis[y] = dy;
                    pq.emplace(dy, y);
                }
            }
        }
    }
};
```
</details>
<br>

---
### &emsp; 1654. 到家的最少跳跃次数 MID
关键思路：
- <b>BFS记忆化搜索最短路径</b>
- 搜索边界可进一步缩小为 `max(f + a + b, x + b)`，其中 `f` 为最远的禁止点；
  - 当 a > b 时，搜索边界为 x + b
  - 当 a <= b 是，搜索边界为 `max(f + a + b, x)`， 这是因为任何超出该边界的路径都可以通过调换操作顺序“平移”到 `(f, max(f + a + b, x))` 中  

<details> 
<summary> <b>C++ Code</b> </summary>

``` c++
class Solution {
public:
    int minimumJumps(vector<int>& forbidden, int a, int b, int x) {
        unordered_set<int> s(forbidden.begin(), forbidden.end());
        queue<pair<int, int>> q;
        q.emplace(0, 1);
        const int n = 6000;
        bool vis[n][2];
        memset(vis, false, sizeof(vis));
        for(int ans = 0; q.size(); ans++)
        {
            for(int t = q.size(); t; t--)
            {
                auto [i, k] = q.front();
                q.pop();
                if(i == x)
                    return ans;
                vector<pair<int, int>> nxts = {{i + a, 1}};
                if(k & 1)
                    nxts.emplace_back(i - b, 0);
                for(auto [j, l] : nxts)
                {
                    if(j >= 0 && j < n && !s.count(j) && !vis[j][l])
                    {
                        q.emplace(j, l);
                        vis[j][l] = true;
                    }
                }
            }
        }
        return -1;
    }
};
```
</details>
<br>

---
### &emsp; 2581. 统计可能的树根数目 :rage: HARD
关键思路：
- 如何判断是否猜对？<b>DFS向下过程中 查询是否有关于当前父子节点对的猜测</b>
- <b>用 哈希表 压缩存储所有猜测</b>
- 如果x和y相邻，那么根从x变为y时，只有x和y的父子关系改变（只会影响到`[x, y]`和`[y, x]`两个猜测的正确性）
- 在计算完以0为根的猜对次数`cnt0`后，再次从0出发，<b>在DFS的过程中 “换根” 并统计</b>

<details> 
<summary> <b>C++ Code</b> </summary>

``` c++
using LL = long long;

class Solution {
public:
    int rootCount(vector<vector<int>>& edges, vector<vector<int>>& guesses, int k) {
        vector<vector<int>> g(edges.size() + 1);
        for(auto &e : edges) // 邻接表建图
        {
            int x = e[0], y = e[1];
            g[x].push_back(y);
            g[y].push_back(x);
        }

        unordered_set<LL> s;
        for(auto &e : guesses) // guesses转成哈希表 两个4字节数字压缩存放
            s.insert((LL) e[0] << 32 | e[1]);

        int ans = 0, cnt0 = 0;
        function<void(int, int)> dfs = [&](int x, int fa) {
            for(int y : g[x])
            {
                if(y != fa)
                {
                    cnt0 += s.count((LL) x << 32 | y); // 统计以0为根时猜对的个数
                    dfs(y, x);
                }
            }
        };
        dfs(0, -1); // 以0为根dfs

        // 在dfs的过程中换根
        // cnt的转移根据 换根DP
        function<void(int, int, int)> reroot = [&](int x, int fa, int cnt) {
            ans += cnt >= k; // 上一结果是否满足猜对个数不少于k
            for(int y : g[x])
            {
                if(y != fa)
                {
                    reroot(y, x, cnt - s.count((LL) x << 32 | y) + s.count((LL) y << 32 | x));
                    // 原本对的现在错了，原本错的现在对了
                }
            }
        };
        reroot(0, -1 ,cnt0);
        return ans;
    }
};
```
</details>
<br>

---
### &emsp; 2684. 矩阵中移动的最大次数 MID
关键思路：
- DFS or BFS 向矩阵右边遍历，记录能达到的最大列号
- DFS可以把`grid[i][j]`置为 0，标记已访问过的格子
- <b>标记优化BFS 通过`grid[i][j] *= -1`记录已入队的元素</b> 从而无需队列和vis数组

<details> 
<summary> <b>C++ Code</b> </summary>

``` c++
class Solution {
public:
    int maxMoves(vector<vector<int>>& grid) {
        int m = grid.size(), n = grid[0].size();
        for(auto &row : grid)
            row[0] *= -1; // 入队标记
        
        for(int j = 0; j < n - 1; j++) // BFS 一次向前一列
        {
            bool ok = false;
            for(int i = 0; i < m; i++)
            {
                if(grid[i][j] > 0) // 不在队列中
                    continue;
                for(int k = max(i - 1, 0); k < min(i + 2, m); k++)
                {
                    if(grid[k][j + 1] > -grid[i][j])
                    {
                        grid[k][j + 1] *= -1;
                        ok = true;
                    }
                }
            }
            if(!ok) // 不能再往右了
                return j;
        }
        return n - 1;
    }
};
```
</details>
<br>

---
### &emsp; 2867. 统计树中的合法路径数目 :rage: HARD
关键思路：
- <b>质数节点把这棵树分成了若干个连通块</b>
- 使用 <b>DFS 预处理</b> 得到每个非质数节点所在连通块的总节点数
- 枚举路径上的这个质数 以及它可以连接到的连通块

<details> 
<summary> <b>C++ Code</b> </summary>

``` c++
const int MX = 1e5;
bool np[MX + 1]; // 质数false 非质数true
int init = []() { // 预处理出质数
    np[1] = true;
    for(int i = 2; i * i <= MX; i++)
    {
        if(!np[i])
        {
            for(int j = i * i; j <= MX; j += i)
                np[j] = true;
        }
    }
    return 0;
}();

class Solution {
public:
    long long countPaths(int n, vector<vector<int>>& edges) {
        vector<vector<int>> g(n + 1); // 邻接表
        for(auto &e : edges)
        {
            int x = e[0], y = e[1];
            g[x].push_back(y);
            g[y].push_back(x);
        }

        vector<int> size(n + 1); // 节点所在连通块的总节点数
        vector<int> nodes; // buffer
        function<void(int, int)> dfs = [&](int x, int fa) {
            nodes.push_back(x); // 统计连通块非质数节点
            for(int y : g[x])
            {
                if(y != fa && np[y])
                    dfs(y, x);
            }
        };

        long long ans = 0;
        for(int x = 1; x <= n; x++) // 枚举1-n的质数
        {
            if(np[x]) // 跳过非质数
                continue;
            int sum = 0; // 统计以这个质数作为中间点的路径数
            for(int y : g[x])
            {
                if(!np[y])
                    continue;
                if(size[y] == 0) // 尚未计算过
                {
                    nodes.clear();
                    dfs(y, -1); // 遍历 y 所在连通块
                    for(int z : nodes)
                    {
                        size[z] = nodes.size();
                    }
                }
                // 连通块中size[y]个非质数 与之前遍历到的sum个非质数 构造出包含一个质数的路径
                ans += (long long ) size[y] * sum;
                sum += size[y];
            }
            ans += sum;
        }
        return ans;
    }
};
```
</details>
<br>

---
### &emsp; 3067. 在带权树网络中统计可连接服务器对数目 MID
关键思路：
- 枚举根 DFS
- 看从每个相邻点出发各能达到多少满足要求的节点

<details> 
<summary> <b>C++ Code</b> </summary>

``` c++
class Solution {
public:
    vector<int> countPairsOfConnectableServers(vector<vector<int>>& edges, int signalSpeed) {
        int n = edges.size() + 1;
        vector<vector<pair<int, int>>> g(n);
        for(auto &e : edges)
        {
            int x = e[0], y = e[1], wt = e[2];
            g[x].push_back({y, wt});
            g[y].push_back({x, wt});
        }

        function<int(int, int, int)> dfs = [&](int x, int fa, int sum) -> int {
            int cnt = sum % signalSpeed == 0;
            for(auto &[y, wt] : g[x])
            {
                if(y != fa)
                    cnt += dfs(y, x, sum + wt);
            }
            return cnt;
        };

        vector<int> ans(n);
        for(int i = 0; i < n; i++)
        {
            int sum = 0;
            for(auto &[y, wt] : g[i])
            {
                int cnt = dfs(y, i, wt);
                ans[i] += cnt * sum;
                sum += cnt;
            }
        }
        return ans;
    }
};
```
</details>
<br>

---
### &emsp; LCP41. 黑白翻转棋 MID
关键思路：
- 枚举空余位置放置黑棋
- 放置后，沿八个方向上BFS扩展，看是否能翻转
- 成功翻转后的新黑棋位置入队列，继续BFS

<details> 
<summary> <b>C++ Code</b> </summary>

``` c++
class Solution {
public:
    int flipChess(vector<string>& chessboard) {
        int m = chessboard.size(), n = chessboard[0].size();
        auto bfs = [&](int i, int j) -> int {
            queue<pair<int, int>> q;
            q.emplace(i, j);
            auto g = chessboard;
            g[i][j] = 'X';
            int cnt = 0;
            while(!q.empty())
            {
                auto p = q.front();
                int i = p.first, j = p.second;
                q.pop();
                // 遍历八个方向 看是否满足翻转条件
                for(int h = -1; h <= 1; h++)
                {
                    for(int v = -1; v <= 1; v++)
                    {
                        if(h == 0 && v == 0)
                            continue;
                        int x = i + v, y = j + h;
                        while(x >= 0 && x < m && y >= 0 && y < n && g[x][y] == 'O') // 沿着已有的白棋前行
                        {
                            x += v;
                            y += h;
                        }
                        if(x >= 0 && x < m && y >= 0 && y < n && g[x][y] == 'X') // 以黑棋结尾
                        {
                            x -= v;
                            y -= h;
                            cnt += max(abs(x - i), abs(y - j));
                            while(x != i || y != j)
                            {
                                g[x][y] = 'X';
                                q.emplace(x, y); // 相当于新落一个黑子
                                x -= v;
                                y -= h;
                            }
                        }
                    }
                }
            }
            return cnt;
        };

        int ans = 0;
        for(int i = 0; i < m; i++)
        {
            for(int j = 0; j < n; j++)
            {
                if(chessboard[i][j] == '.')
                    ans = max(ans, bfs(i, j));
            }
        }
        return ans;
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

---
### &emsp; 980. 不同路径III :rage: HARD
关键思路：
- <b>DFS</b> `dfs(grid, x, y, left)` `left`表示当前还剩下`left`个格子要走
- 对于走过的格子 修改 `grid[x][y] = -1`， 回溯时再复原
- 到达终点时判断 `left == 0`
- 向上下左右移动，累加四个方向递归的返回值

<details> 
<summary> <b>C++ Code</b> </summary>

``` c++
class Solution {
public:
    int dfs(vector<vector<int>> &grid, int x, int y, int left)
    {
        if(x < 0 || x >= grid.size() || y < 0 || y >= grid[x].size() || grid[x][y] < 0)
            return 0;
        if(grid[x][y] == 2)
            return left == 0;
        grid[x][y] = -1; // dfs子树中 标记为访问过
        int ans = dfs(grid, x - 1, y, left - 1) + dfs(grid, x, y - 1, left - 1) +
                  dfs(grid, x + 1, y, left - 1) + dfs(grid, x, y + 1, left - 1);
        grid[x][y] = 0; // 回溯时恢复
        return ans;
    }
    int uniquePathsIII(vector<vector<int>>& grid) {
        int m = grid.size(), n = grid[0].size(), cnt0 = 0, sx, sy;
        for(int i = 0; i < m; i++)
        {
            for(int j = 0; j < n; j++)
            {
                if(grid[i][j] == 0)
                    cnt0++;
                else if(grid[i][j] == 1)
                    sx = i, sy = j;
            }
        }
        return dfs(grid, sx, sy, cnt0 + 1); // +1是把起点也算上
    }
};
```
</details>

修改为状态压缩：
- 使用位运算进行状态压缩 用`二进制数vis`表示访问过的坐标集合
- 在递归中去修改 `vis`，不去修改 `grid[x][y]`，做到无后效性，从而可以使用记忆化搜索
- 二维坐标`(x, y)` 映射为 `nx + y`
- 把障碍方格也加到 `vis` 中，这样递归到终点时，只需要判断 `vis` 是否为全集，即可知道是否已访问所有格子
- 由于有大量状态是无法访问到的，相比数组，用哈希表记忆化更好 
- （实际上，并没有太多重复递归调用，使用哈希表反而拖慢了速度，方法一可能更快）

<details> 
<summary> <b>C++ Code</b> </summary>

``` c++
class Solution {
public:
    int uniquePathsIII(vector<vector<int>>& grid) {
        int m = grid.size(), n = grid[0].size(), vis = 0, sx, sy;
        for(int i = 0; i < m; i++)
        {
            for(int j = 0; j < n; j++)
            {
                if(grid[i][j] < 0)
                    vis |= 1 << (i * n + j); // 障碍
                else if(grid[i][j] == 1)
                    sx = i, sy = j;
            }
        }

        int all = (1 << m * n) - 1;
        unordered_map<int, int> memo;
        function<int(int, int, int)> dfs = [&](int x, int y, int vis) -> int {
            int p = x * n + y;
            if(x < 0 || x >= m || y < 0 || y >= n || vis >> p&1)
                return 0;
            vis |= 1 << p;
            if(grid[x][y] == 2)
                return vis == all;
            int key = (p << m * n) | vis; // 压缩dfs的参数 左移m*n是因为vis至多有m*n个bit
            if(memo.count(key))
                return memo[key];
            return memo[key] = dfs(x - 1, y, vis) + dfs(x, y - 1, vis) +
                               dfs(x + 1, y, vis) + dfs(x, y + 1, vis);
        };
        return dfs(sx, sy, vis);
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
`vis[N]`：保存已经找到最短路的节点集合（划分节点集）  
每一轮 <b>最小`dis`的顶点</b> 在`vis`中标记（加入集合） &emsp; 再用它的边来更新`dis`

堆优化Dijkstra（优先队列存放）（适用于稀疏图） O(mlogn) ； 当 边数 m < n^2 （相对稀疏时）优先采用堆优化  
（同一点可能多次入队，m轮出队才能保证所有节点计算完毕）  

<br>

- ### BellmanFord O(n*m)
DP思想 求单源最短路 可以有负边 可以判断存在负环（最短路不存在）  
迭代超过 V-1 次，就说明存在负权回路  
使用邻接表或类存边  
每一轮对所有边`e(a,b,w)`进行松弛操作，若`dis[b] > dis[a]+w` 则更新dis[b]  

队列优化BellmanFord —— SPFA（使用邻接表）

- ### Floyd O(n^3) 
贪心 + DP 求所有节点对间的最短路  
以每个点为「中转站」，刷新所有「入度」和「出度」的距离   
外层枚举中转节点（能用于作为中转节点的编号）：枚举到第 `i` 轮时，可以理解为“已经计算完可将节点 `0` ~ `i-1` 作为中间节点”的情况  

注：不关心起止点的多源最短路问题可以通过 <b>建立虚拟源点和虚拟汇点</b> 转换为单源最短路问题  

<details>
<summary> <b>C++ Code</b> </summary>

```c++
vector<vector<int>> w(n, vector<int>(n, INT_MAX / 2)); // 邻接矩阵 防止加法溢出
for(auto &e : edges)
{
    int x = e[0], y = e[1], wt = e[2];
    w[x][y] = w[y][x] = wt;
}

vector<vector<vector<int>>> f(n + 1, vector<vector<int>>(n, vector<int>(n)));
f[0] = w; // 不经过任何点

// 最外层枚举k：
// 要算f[k+1][i][j]，必须先把 f[k][i][j]、f[k][i][k]和f[k][k][j]算出来。
// 由于我们不知道 k 和 i,j 的大小关系，只有把 k 放在最外层枚举，才能保证先把 f[k][i][j]、f[k][i][k]和 f[k][k][j] 算出来。
// 并且对于 i 和 j来说，由于在计算 f[k+1][i][j]的时候，f[k][⋅][⋅] 已经全部计算完毕，所以 i 和 j 按照正序/逆序枚举都可以

for(int k = 0; k < n; k++) // k这一维可以空间优化
{
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
            f[k+1][i][j] = min(f[k][i][j], f[k][i][k] + f[k][k][j]);
    }
}
```
</details>
<br>

<br>

---
### 题目
---
### &emsp; 310. 最小高度树 MID
关键思路：
- <b>拓扑排序</b>，从外向内剥离叶子节点
- 从所有叶子节点（出度为1）出发BFS
- 最后一层则对应可作为最小高度树的根节点

<details>
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    vector<int> findMinHeightTrees(int n, vector<vector<int>>& edges) {
        if(n == 1)
            return {0};
        
        vector<vector<int>> g(n);
        vector<int> degree(n, 0);
        for(auto& e : edges)
        {
            int a = e[0], b = e[1];
            g[a].push_back(b);
            g[b].push_back(a);
            degree[a]++;
            degree[b]++;
        }

        queue<int> q;
        vector<int> ans;
        for(int i = 0; i < n; i++)
        {
            if(degree[i] == 1)
                q.emplace(i);
        }
        while(!q.empty())
        {
            ans.clear();
            for(int i = q.size(); i > 0; i--) // 一次“推进”一层
            {
                int cur = q.front();
                q.pop();
                ans.push_back(cur);
                for(auto& v : g[cur])
                {
                    if(--degree[v] == 1)
                        q.emplace(v);
                }
            }
        }
        return ans;
    }
};
```
</details>
<br>

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
### &emsp; 1462. 课程表IV MID
关键思路：
- 在 <b>拓扑排序问题</b> 的基础上 维护先后关系
- 使用二维数组记录节点间可达性，用邻接表建图，并计算各节点入度
- 入度为0，代表已经计算完毕了所有至它的可达性  
- 队头元素（入度为0）作为中间节点，根据邻接表与当前计算的可达性，更新至其下一节点的可达性

<details>
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    vector<bool> checkIfPrerequisite(int numCourses, vector<vector<int>>& prerequisites, vector<vector<int>>& queries) {
        int n = numCourses;
        bool f[n][n]; // 可达性
        memset(f, false, sizeof(f));
        vector<int> g[n]; // 二维邻接表
        vector<int> indeg(n); // 入度
        for(auto& p : prerequisites)
        {
            g[p[0]].push_back(p[1]);
            indeg[p[1]]++;
        }
        queue<int> q; // 一个节点进入队列时入度为0，代表已经计算完毕了所有至它的可达性
        for(int i = 0; i < n; i++)
        {
            if(indeg[i] == 0)
                q.push(i);
        }

        while(!q.empty())
        {
            int i = q.front();
            q.pop();
            for(int j : g[i])
            {
                f[i][j] = true;
                for(int h = 0; h < n; h++) // i 作为中间节点
                {
                    f[h][j] |= f[h][i];
                }
                if(--indeg[j] == 0)
                    q.push(j);
            } 
        }
        vector<bool> ans;
        for(auto& qry : queries)
            ans.push_back(f[qry[0]][qry[1]]);
        return ans;
    }
};
```
</details>
<br>

---
### &emsp; 2976. 到达目的地的方案数 MID
关键思路：
- Dijkstra更新节点最短路距离的过程中DP，统计到每个节点最短路的个数

<details>
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    const int MOD = 1e9 + 7;
    int countPaths(int n, vector<vector<int>>& roads) {
        vector<vector<pair<int, int>>> g(n);
        for(auto &r : roads)
        {
            int from = r[0], to =r[1], cost = r[2];
            g[from].emplace_back(to, cost);
            g[to].emplace_back(from, cost);
        }
        vector<long long> dis(n, LLONG_MAX);
        dis[0] = 0;
        vector<int> dp(n); // 到节点的最短路个数
        dp[0] = 1;
        priority_queue<pair<long long, int>, vector<pair<long long, int>>, greater<>> pq; // 堆优化
        pq.emplace(0, 0);
        while(true)
        {
            auto [dx, x] = pq.top();
            pq.pop();
            if(x == n-1)
            {
                return dp[n-1]; // 边权都是正数 不可能找到更短的了
            }
            if(dx > dis[x])
                continue;
            for(auto &[y, d] : g[x])
            {
                long long new_dis = dx + d;
                if(new_dis < dis[y])
                {
                    dis[y] = new_dis;
                    dp[y] = dp[x]; // 最短路经过x
                    pq.emplace(new_dis, y);
                }
                else if (new_dis == dis[y])
                {
                    dp[y] = (dp[y] + dp[x]) % MOD;
                }
            }
        }
        return -1;
    }
};
```
</details>
<br>

---
### &emsp; 2127. 参加会议的最多员工数 :rage: HARD
关键思路：
- 内向基环树：具有 n个点 n条边 的联通块；内向指每个点只有一条出边
- 从节点 i 出发根据 favourite[i] 连接有向边
- 对于基环长度大于 2 的情况，圆桌的最大员工数目即为最大的基环长度
- 同时还可以放置多个长度2的环 + 其左右最长链；使用<b>拓扑排序 + 动态规划</b>的方法寻找最长游走路径
- 通过一次拓扑排序，可以「剪掉」所有树枝。因为拓扑排序后，树枝节点的入度均为 0，基环节点的入度均为 1
- 遍历deg为1的节点即遍历基环；再以基环与树枝的连接处为起点，顺着反图来遍历树枝，从而将问题转化成一个树形问题

<details>
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    int maximumInvitations(vector<int>& favorite) {
        int n = favorite.size();
        vector<int> deg(n);
        for(int f : favorite)
        {
            deg[f]++; // 统计基环树每个节点的入度 便于进行拓扑排序
        }

        vector<vector<int>> rg(n); // 拓扑排序建立反图 被喜欢
        queue<int> q;
        for(int i = 0; i < n; i++)
        {
            if(deg[i] == 0)
                q.push(i);
        }
        while(!q.empty())
        {
            int x = q.front();
            q.pop();
            int y = favorite[x];
            rg[y].push_back(x);
            if(--deg[y] == 0)
                q.push(y);
        }

        // 通过dfs反图rg 寻找树枝上最深的链
        function<int(int)> rdfs = [&](int x) -> int {
            int max_depth = 1;
            for(int son : rg[x])
            {
                max_depth = max(max_depth, rdfs(son) + 1);
            }
            return max_depth;
        };

        int size1 = 0, size2 = 0; // 一个2以上的环 或 多个2环及其两端最长链
        for(int i = 0; i < n; i++)
        {
            if(deg[i] == 0)
                continue;

            // 遍历基环上的点
            deg[i] = 0; // 将基环上的点的入度标记为0，避免重复访问
            int ring_size = 1;
            for(int x = favorite[i]; x != i; x = favorite[x])
            {
                deg[x] = 0;
                ring_size++;
            }

            if(ring_size == 2)
            {
                size2 += rdfs(i) + rdfs(favorite[i]); // 最长链
            }
            else
            {
                size1 = max(size1, ring_size);
            }
        }
        return max(size1, size2);
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

------
## 连通性问题
### 概念
由连通性定义 “图上的状态” / 图的划分  

---
### 题目
---
### &emsp; 924. 尽量减少恶意软件的传播 :rage: HARD
关键思路： 
- <b>图结构中的连通性</b>
- 问题转化：寻找只包含一个被感染节点的最大连通块
- 如何表达“连通块内有一个或多个被感染的节点”（对应连通块的不同状态）
- 通过`vis`记录已访问节点，以免重复遍历连通块

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    int minMalwareSpread(vector<vector<int>>& graph, vector<int>& initial) {
        set<int> st(initial.begin(), initial.end());
        int n = graph.size();
        vector<int> vis(n);
        int node_id, size;
        function<void(int)> dfs = [&](int x) {
            vis[x] = true;
            size++;
            if(node_id != -2 && st.contains(x)) // 更新连通块状态
                node_id = node_id == -1 ? x : -2;
            for(int y = 0; y < n; y++)
            {
                if(graph[x][y] && !vis[y])
                    dfs(y);
            }
        };

        int ans = -1, max_size = 0;
        for(int x : initial)
        {
            if(vis[x])
                continue;

            node_id = -1;
            size = 0;
            dfs(x); // 寻找连通块
            if(node_id >= 0)
            {
                if(size > max_size || (size == max_size && node_id < ans))
                {
                    ans = node_id;
                    max_size = size;
                }
            }
        }
        return ans < 0 ? ranges::min(initial) : ans;
    }
};
```
</details> 
<br>

---
### &emsp; 928. 尽量减少恶意软件的传播II :rage: HARD
关键思路： 
- 与 T924.的不同 删除节点会改变图的结构（从而导致DFS遍历的结构变化）
- 先求出initial节点对图的划分
- 从不在 initial 中的点 `v` 出发 DFS，在不经过 initial中的节点的前提下，看看 `v` 是只能被一个点感染到，还是能被多个点感染到
- 如果 `v` 只能被点 `x=initial[i]` 感染到，那么在本次 DFS 过程中访问到的其它节点，也只能被点 `x` 感染到

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    int minMalwareSpread(vector<vector<int>>& graph, vector<int>& initial) {
        unordered_set<int> st(initial.begin(), initial.end());
        vector<int> vis(graph.size());
        int node_id, size;
        function<void(int)> dfs = [&](int x) {
            vis[x] = true;
            size++;
            for(int y = 0; y < graph[x].size(); y++)
            {
                if(graph[x][y] == 0)
                    continue;

                if(st.contains(y))
                {
                    if(node_id != -2 && node_id != y) // 更新连通块状态
                        node_id = node_id == -1 ? y : -2;
                }
                else if(!vis[y]) // dfs只递归访问不在initial的节点
                {
                    dfs(y);
                }
            }
        };

        unordered_map<int, int> cnt; // 统计某一initial作为唯一感染节点感染的连通块节点数
        for(int i = 0; i < graph.size(); i++)
        {
            if(vis[i] || st.contains(i))
                continue;

            node_id = -1;
            size = 0;
            dfs(i);
            if(node_id >= 0) // 这个连通块只连接一个在intial中的节点
                cnt[node_id] += size;
        }

        int max_cnt = 0;
        int min_node_id = 0;
        for(auto [node_id, c] : cnt)
        {
            if(c > max_cnt || (c == max_cnt && node_id < min_node_id))
            {
                max_cnt = c;
                min_node_id = node_id;
            }
        }
        return cnt.empty() ? ranges::min(initial) : min_node_id;
    }
};
```
</details> 
<br>
