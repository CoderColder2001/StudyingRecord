[TOC]
## 并查集
---
`核心思想`：每一个集合选择一个 **代表元**，集合的其他元素指向这个代表元  
通过代表元维护一个集合的共有性质  

路径压缩：每个节点直接指向代表元  
按秩合并：把简单的集合（树）往复杂的集合（树）上合并  

可以通过一个节点的邻居查询相应连通块的性质  

可以在条件单调变化的场景下动态地增加新的边  

---
还可以使用并查集快速找到节点的下一个未访问可达节点（访问到一个节点时，将该节点的 fa 指针指向相邻的下一个节点，这样下次访问到这个节点时，可以跳到下一个未访问的节点）  

### **模板**
<details>
<summary> <b> C++ Code</b> </summary>

``` c++
class UnionFind {
public:
    vector<int> father; // 代表元
    vector<int> size; // 秩
    int n; // 节点数
    int comp_cnt; // 集合数

    UnionFind(int _n) : n(_n), comp_cnt(_n), father(_n), size(_n, 1){
        iota(father.begin(), father.end(), 0);
    }
    int find(int x) // 更新并返回father[x]
    {
        if(x != father[x])
            father[x] = find(father[x]);
        return father[x];
    }
    bool unite(int x, int y)
    {
        x = find(x);
        y = find(y);
        cout<<"unite "<<x<<" "<<y<<endl;
        if(x == y)
            return false;
        if(size[x] < size[y])
            swap(x, y);
        father[y] = x;
        size[x] += size[y];
        comp_cnt--;
        return true;
    }
};
```
</details>

---
## Leetcode中利用并查集的题目

---
### &emsp; 765. 情侣牵手 :rage: HARD
关键思路：
- 考虑 “构成错误的集合” ：如果有两对情侣互相坐错了位置，需要一次交换，回到正确的位置 => 如果有 n 对情侣互相坐错了位置，需要 n-1 次交换，回到正确的位置
- <b>使用 并查集 维护这些集合</b>；（总是以“情侣组”为单位考虑问题）
- 对于被分开的情侣，连接被分至的两个组

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class UnionFind {
    vector<int> father;
    vector<int> size;
public:
    int compCnt;
    UnionFind(int _n) : compCnt(_n), father(_n), size(_n, 1)
    {
        iota(father.begin(), father.end(), 0);
    }

    int find(int x)
    {
        if(x != father[x])
            father[x] = find(father[x]);
        return father[x];
    }

    bool unite(int x, int y)
    {
        x = find(x);
        y = find(y);
        if(x == y)
            return false;
        if(size[x] < size[y])
            swap(x, y);
        father[y] = x;
        size[x] += size[y];
        compCnt--;
        return true;
    }
};

class Solution {
public:
    int minSwapsCouples(vector<int>& row) {
        int n = row.size();
        int m = n / 2;
        UnionFind uf(m);
        for(int i = 0; i < n; i += 2)
        {
            // 连接一对情侣被拆散到的两个组
            uf.unite(row[i] / 2, row[i + 1] / 2); // 组id
        }
        return m - uf.compCnt;
    }
};
```
</details>
<br>

---
### &emsp; 1697. 检查边长度限制的路径是否存在 :rage: HARD
关键思路：
- queries数组 离线查询
- 根据对边长度的限制 从小到大排序queries_id
- 对边集根据边长排序 随着当前遍历到的查询（限制长度满足单调增）依次添加边
- <b>使用并查集维护图的连通性</b> &emsp; 在并查集中添加`judgeConnection`方法

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class UnionFind {
public:
    vector<int> father; // 代表元
    vector<int> size; // 秩
    int n; // 节点树
    int comp_cnt; // 集合数

    UnionFind(int _n) : n(_n), comp_cnt(_n), father(_n), size(_n, 1){
        iota(father.begin(), father.end(), 0);
    }
    int find(int x) // 更新并返回father[x]
    {
        if(x != father[x])
            father[x] = find(father[x]);
        return father[x];
    }
    bool unite(int x, int y)
    {
        x = find(x);
        y = find(y);
        cout<<"unite "<<x<<" "<<y<<endl;
        if(x == y)
            return false;
        if(size[x] < size[y])
            swap(x, y);
        father[y] = x;
        size[x] += size[y];
        comp_cnt--;
        return true;
    }
    bool judgeConnection(int x, int y) // 使用并查集判断连通性
    {
        x = find(x);
        y = find(y);
        return x == y;
    }
};
class Solution {
public:
    vector<bool> distanceLimitedPathsExist(int n, vector<vector<int>>& edgeList, vector<vector<int>>& queries) {
        vector<int> q_id(queries.size());
        iota(q_id.begin(), q_id.end(), 0);
        // 对queries的id依limit增序排序
        sort(q_id.begin(), q_id.end(), [&](int i, int j){
            return queries[i][2] < queries[j][2];
        });

        // 将edgeList依dis增序排序
        sort(edgeList.begin(), edgeList.end(), [](const auto& e1, const auto& e2){
            return e1[2] < e2[2];
        });

        UnionFind uf(n);
        int i = 0; // edgeList的指针
        vector<bool> ans(queries.size());
        for(int query : q_id)
        {
            while(i < edgeList.size() && edgeList[i][2] < queries[query][2]) // 根据当前query 添加满足条件的边 连接图
            {
                uf.unite(edgeList[i][0], edgeList[i][1]);
                i++;
            }
            ans[query] = uf.judgeConnection(queries[query][0], queries[query][1]);
        }
        return ans;
    }
};
```
</details>
<br>

---
### &emsp; 2617. 网格图中最少访问的格子数 :rage: HARD
关键思路：
- 暴力BFS中需要遍历每个节点的下一个可达节点 n ，这些可达节点中可能已经经由其它节点访问过，此时再访问也不可能得到节点 n 的更优解。因此，应该跳过访问过的节点
- *如何快速找到一个节点的下一个未访问可达节点？*
- <b>用并查集合并访问过的节点</b>（用 应该遍历的下一个节点 作这些节点的“代表”）
- 访问到一个节点时，将该节点的 fa 指针指向相邻的下一个节点，这样下次访问到这个节点时，可以跳到下一个未访问的节点
- 法二：单调栈优化DP，见 `DynamicPrograming.md`

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    using Node = tuple<int, int, int>; //step x y

    int find(vector<int>& fa, int x)
    {
        return x == fa[x] ? x : fa[x] = find(fa, fa[x]);
    }
    void merge(vector<int>& fa, int x) // 标记fa[x]指向下一节点
    {
        fa[x] = x + 1;
    }

    int minimumVisitedCells(vector<vector<int>>& grid) {
        int m = grid.size(), n = grid[0].size();

        vector<vector<int>> row_fas(m, vector<int>(n+1));
        for(int i = 0; i < m; i++)
        {
            iota(row_fas[i].begin(), row_fas[i].end(), 0); // 从0开始递增填充
        }
        vector<vector<int>> col_fas(n, vector<int>(m+1));
        for(int i = 0; i < n; i++)
        {
            iota(col_fas[i].begin(), col_fas[i].end(), 0);
        }

        queue<Node> q;
        q.emplace(1, 0, 0);

        while(!q.empty())
        {
            auto[d, x, y] = q.front();
            q.pop();
            if(x == m-1 && y == n-1)
                return d;

            int g = grid[x][y];
            
            // 使用find遍历下一访问节点
            // right
            for(int ny = find(row_fas[x], y + 1);
                ny < min(y + g + 1, n);
                ny = find(row_fas[x], ny + 1))
            {
                merge(row_fas[x], ny);
                q.emplace(d + 1, x, ny);
            }
            // down
            for(int nx = find(col_fas[y], x + 1);
                nx < min(x + g + 1, m);
                nx = find(col_fas[y], nx + 1))
            {
                merge(col_fas[y], nx);
                q.emplace(d + 1, nx, y);
            }
        }
        return -1;
    }
};
```
</details>
<br>

---
### &emsp; 2812. 找出最安全路径 MID
关键思路：
- 并查集优点：可以（在枚举答案过程中）动态地连边 动态地判断连通性
- <b>从大到小倒序枚举答案</b> 枚举答案的过程中<b>动态连边</b>（连边的条件对于枚举过程是单调的）
- <b>多源BFS预处理</b> 从小偷单元格出发 记录到每一个格子的步数
- 使用滚动数组进行分组 数组长度即为当前走了多少步（当前格子的安全系数）

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
    static constexpr int dirs[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
public:
    int maximumSafenessFactor(vector<vector<int>>& grid) {
        int n = grid.size();
        vector<pair<int, int>> q;
        vector<vector<int>> dis(n, vector<int>(n, -1));
        for(int i = 0; i < n; i++) 
        {
            for(int j = 0; j < n; j++) 
            {
                if(grid[i][j])
                {
                    q.emplace_back(i, j);
                    dis[i][j] = 0;
                }
            }
        }

        vector<vector<pair<int, int>>> groups = {q}; // step0
        while(!q.empty()) // 多源BFS
        {
            vector<pair<int, int>> nq; // 对应step+1的分组
            for(auto &[i, j]: q) 
            {
                for(auto &d: dirs)
                {
                    int x = i + d[0], y = j + d[1];
                    if(0 <= x && x < n && 0 <= y && y < n && dis[x][y] < 0) // dis为-1时还没有标记
                    {
                        nq.emplace_back(x, y);
                        dis[x][y] = groups.size();
                    }
                }
            }
            groups.push_back(nq); // 相同dis分组记录
            q = move(nq);
        }

        // 并查集模板
        vector<int> fa(n * n);
        iota(fa.begin(), fa.end(), 0);
        function<int(int)> find = [&](int x) -> int { 
            return fa[x] == x ? x : fa[x] = find(fa[x]);
        };

        for(int ans = (int) groups.size() - 2; ans > 0; ans--) // 枚举答案
        {
            for(auto &[i, j]: groups[ans]) // 访问该dis的分组
            {
                for(auto &d: dirs) // 相邻点
                {
                    int x = i + d[0], y = j + d[1];
                    if(0 <= x && x < n && 0 <= y && y < n && dis[x][y] >= ans) // 连接满足条件的边
                        fa[find(x * n + y)] = find(i * n + j);
                }
            }
            if(find(0) == find(n * n - 1)) // 写这里判断更快些
                return ans;
        }
        return 0;
    }
};
```
</details>
<br>
