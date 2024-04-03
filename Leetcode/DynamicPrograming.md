[TOC]  
# DynamicPrograming
------

## Content
- 贪心
- 线性DP
- 树形DP
- 序列DP
- 区间DP
- 状压DP
- 数位DP
- 计数DP

求解 <b>最优化问题</b>  
有重叠子问题 -> 用 DP 优化  
通过组合子问题的解来求解原问题  
<br>

---
<b>选或不选</b> 与 <b>枚举选哪个</b>  
在DP状态转移过程中是否需要精确的信息  
枚举选哪个 适用于完全需要序列（转移的相邻状态）精确的信息  
如 最长递增子序列问题需要知道子序列相邻数字的具体大小  

------
## 贪心
### 概念
贪心其实就是不用考虑所有状态的DP；  
贪心要达到最优解同样要求问题具有 <b>最优子结构</b>  
并要求可以证明当前贪心解构成最优解的一个划分（每一步贪心选择在最优解中）

<br>

---

### 题目
---
### &emsp; 765. 情侣牵手 :rage:HARD
关键思路：
- 贪心：从情侣组考虑，每次遇到不匹配的情侣组，就去找到另一半的位置并交换  
- 问题其实相当于考虑： *当前遍历到第 k 组位置，且前 k-1 组位置都已是情侣，接下来怎么做才能使交换次数最低*
- 使用vector维护 *数组 row从值到索引的映射* （位置buffer）
- `lover = a ^ 1` 取相邻的奇偶数

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    int minSwapsCouples(vector<int>& row) {
        int n = row.size();
        vector<int> loc(n, 0); // 位置buffer 从编号值映射到索引i
        for(int i = 0; i < n; i++)
        {
            loc[row[i]] = i;
        } 
        int ans = 0;
        for(int i = 0; i < n-1; i += 2)
        {
            int a = row[i]; // ID值
            int lover = a ^ 1; // 取相邻的奇偶数
            if(row[i+1] != lover)
            {
                loc[row[i+1]] = loc[lover]; // 更新位置buffer
                swap(row[i+1], row[loc[lover]]); // 交换 更新row
                loc[lover] = i+1; // 更新位置buffer
                ans++;
            }
        }
        return ans;
    }
};
```
</details> 
<br>

---
### &emsp; 2952. 需要添加的硬币的最小数量 MID
关键思路：
- 基于贪心策略进行递推构造
- 由归纳法思考，假设现在可以得到区间 `[0, s−1]` 中的所有整数 如何添加？
- 添加后，可以得到 `[0, 2s−1]` 中的所有整数

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    int minimumAddedCoins(vector<int>& coins, int target) {
        ranges::sort(coins);
        int ans = 0, s = 1, i = 0; // s 指示下一步要构造出的数
        while(s <= target)
        {
            if(i < coins.size() && coins[i] <= s)
                s += coins[i++]; // 更新区间 可以得到 [0, s+x−1]
            else // 无法得到s
            {
                s *= 2; // 添加coins[i] 更新区间
                ans++;
            }
        }
        return ans;
    }
};
```
</details> 
<br>

------
## 线性DP
### 概念
---
* ### 线性DP
    基于给定数组 描述状态和状态转移

<br>

---

### 题目
---
### &emsp; 514. 自由之路 :rage: HARD
关键思路：  
- 定义状态： `(key拼到哪一位, ring指向哪一位)`
- <b>状态转移：每次转到 左边最近的 或 右边最近的</b>
- 预处理出 对于ring每一位来说，左边与右边a-z最近的下标
- 使用一个数组`pos`缓存每个字母最后出现的位置，遍历过程中更新`pos`并直接赋值给`left[i]`、`right[i]`
- *如何在遍历中表达 环形、顺时针、逆时针？*

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    int findRotateSteps(string ring, string key) {
        int n = ring.length(), m = key.length();

        array<int, 26> pos; // 缓存在当前方向上每个字母最近的位置

        // 对于每个ring[i] 左边（逆时针） a-z最近的下标（左边没有就从 n-1 往左找）
        vector<array<int, 26>> left(n);
        for(int i = 0; i < n; i++) // 第0位左边最近的a-z的下标
        {
            ring[i] -= 'a';
            pos[ring[i]] = i; // 每个字母最后出现的位置
        }
        for(int i = 0; i < n; i++) // 向右（逆时针）
        {
            left[i] = pos;
            pos[ring[i]] = i; // 左边出来了一位 更新下标
        }

        vector<array<int, 26>> right(n);
        for(int i = n - 1; i >= 0; i--) // 第n-1位右边最近的a-z的下标
        {
            pos[ring[i]] = i; // 每个字母首次出现的位置
        }
        for(int i = n - 1; i >= 0; i--) // 向左（顺时针）
        {
            right[i] = pos;
            pos[ring[i]] = i; // 右边出来了一位 更新下标
        }

        vector<vector<int>> dp(m + 1, vector<int>(n)); // 边界：dp[m][i] = 0;
        for(int j = m - 1; j >= 0; j--)
        {
            char c = key[j] - 'a';
            for(int i = 0; i < n; i++)
            {
                if(ring[i] == c) // 无需旋转
                    dp[j][i] = dp[j + 1][i];
                else
                {
                    int l = left[i][c], r = right[i][c];
                    dp[j][i] = min(dp[j + 1][l] + (l > i ? n - l + i : i - l),
                                   dp[j + 1][r] + (r < i ? n - i + r : r - i));
                }
            }
        }
        return dp[0][0] + m; // m次拼写
    }
};
```
</details> 
<br>

---
### &emsp; 518. 零钱兑换II MID
关键思路：  
- 背包类<b>构造型问题</b> 求解方案个数
- <b>枚举能用的，去更新各个结果的方案数</b>
- 状态转移式子第一维只与上一层有关，可以进一步空间优化

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    int change(int amount, vector<int>& coins) {
        int n = coins.size();
        vector<vector<int>> dp(n + 1, vector<int>(amount + 1));
        dp[0][0] = 1;
        for(int i = 0; i < n; i++) // 枚举能用的硬币
        {
            for(int c = coins[i]; c <= amount; c++) // 影响各个金额的方案
            {
                if(c < coins[i])
                    dp[i + 1][c] = dp[i][c];
                else
                    dp[i + 1][c] = dp[i][c] + dp[i + 1][c - coins[i]];
            }
        }
        return dp[n][amount];
    }
};
```
</details> 
<br>

---
### &emsp; 978. 最长湍流子数组 MID
关键思路：  
- 定义两个状态量`up`，`down` 
- 分别描述以arr[i]结尾上一动作是升高or降低的最长湍流子数组

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    int maxTurbulenceSize(vector<int>& arr) {
        int up = 1, down = 1; 
        int res = 1;
        for(int i = 1; i < arr.size(); i++)
        {
            if(arr[i] > arr[i - 1])
            {
                up = down + 1;
                down = 1;
            }
            else if(arr[i] < arr[i - 1])
            {
                down = up + 1;
                up = 1;
            }
            else
            {
                up = 1;
                down = 1;
            }
            res = max(res, up);
            res = max(res, down);
        }
        return res;
    }
};
```
</details> 
<br>

---
### &emsp; 1262. 可被三整除的最大和 MID
关键思路：  
- 考虑 <b>选或不选</b>
- 选择了之后 mod的条件`j'`将变成 `x + j mod 3 == j'`

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    int maxTurbulenceSize(vector<int>& arr) {
        int up = 1, down = 1; 
        int res = 1;
        for(int i = 1; i < arr.size(); i++)
        {
            if(arr[i] > arr[i - 1])
            {
                up = down + 1;
                down = 1;
            }
            else if(arr[i] < arr[i - 1])
            {
                down = up + 1;
                up = 1;
            }
            else
            {
                up = 1;
                down = 1;
            }
            res = max(res, up);
            res = max(res, down);
        }
        return res;
    }
};
```
</details> 
<br>

---
### &emsp; 1335. 工作计划的最低难度 :rage:HARD
关键思路：  
- <b>区间分段最大值之和的最小值</b>
- `f[d][n]`： `d`为剩余天数，控制区间遍历的范围
- 根据`d`的控制 枚举最后一天进行工作的开始下标`k` 并求最后一天工作难度的最大值`mx`，再更新`f[i][j]`
- dp数组更新顺序为斜向下，只用到上一行的数据；但由于斜向下，在空间优化省略第一维时，第二维要倒序遍历

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    int minDifficulty(vector<int>& jobDifficulty, int d) {
        int n = jobDifficulty.size();
        if(n < d)
            return -1;

        int f[d][n]; // d天 [0,n]
        f[0][0] = jobDifficulty[0];
        for(int j = 1; j < n; j++)
        {
            f[0][j] = max(f[0][j-1], jobDifficulty[j]);
        }
        for(int i = 1; i < d; i++)
        {
            for(int j = n-1; j >= i; j--) // 倒序遍历j 可以空间优化省去第一维 但若不倒序遍历f[k-1]会被覆盖
            {
                f[i][j] = INT_MAX;
                int mx = 0;
                // 枚举最后一段工作的开始下标k
                for(int k = j; k >= i; k--)
                {
                    mx = max(mx, jobDifficulty[k]); // 从 a[k] 到 a[j] 的最大值
                    f[i][j] = min(f[i][j], f[i-1][k-1] + mx);
                }
            }
        }
        return f[d-1][n-1];
    }
};
```
</details> 
<br>

---
### &emsp; 1388. 3n块披萨 :rage:HARD
关键思路：  
- 注意 `n >= 2` 时 要选择的数中一定存在一个数x有一侧至少有连续两个数没有被选择（假设所有选中不相邻数的间隔为 `1`，即中间只有一个数没有被选择，那么总数为 `2n`，与总数为 `3n` 矛盾
- *问题转化为 在一个长度3n的环形数组中选择n个不相邻的数使得和最大*
- 选择一个数字后 两侧的数也一起删去 变为子问题
- 对于问题描述 可以表示为定义函数 `g(nums)` 表示在 `nums` 中选取 `n` 个不相邻的数 使得其和最大
- 对于环形数组 展开；如果选择了第一个数，那么最后一个数就不能选择，如果选择了最后一个数，那么第一个数就不能选择
- 因此可以 <b>将环形数组拆成两个数组</b> ，一个是去掉第一个数的，一个是去掉最后一个数的，然后分别求解这两个数组的g的最大值
- 定义 `dp[i][j]` 表示在 `nums` 前 `i` 个数中选择 `j` 个不相邻的数的最大和

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    int maxSizeSlices(vector<int>& slices) {
        int n = slices.size() / 3;
        auto g = [&](vector<int> &nums) -> int {
            int m = nums.size();
            int dp[m + 1][n + 1];
            memset(dp, 0, sizeof dp);
            for(int i = 1; i <= m; i++)
            {
                for(int j = 1; j <= n; j++)
                    dp[i][j] = max(dp[i - 1][j], (i >= 2 ? dp[i - 2][j - 1] : 0) + nums[i - 1]);
            }
            return dp[m][n];
        };

        vector<int> nums(slices.begin(), slices.end() - 1);
        int a = g(nums);
        nums = vector<int>(slices.begin() + 1, slices.end());
        int b = g(nums);
        return max(a, b);
    }
};
```
</details> 
<br>

---
### &emsp; 1444. 切披萨的方案数 :rage:HARD
关键思路：  
- 注意到 “切一刀” 这个动作 代表从原问题到一个规模更小的子问题
- 无论怎么切 右下角始终为`(m-1, n-1)`
- 二维前缀和 预处理子矩形中苹果数
- `dp[c][i][j]` 表示左上角在`(i, j)` 右下角在`(m-1, n-1)`的子矩形切 `c` 刀，每块都包含至少一个苹果的方案数

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class MatrixSum { // 二维前缀和模板
private:
    vector<vector<int>> sum;
public:
    MatrixSum(vector<string> &matrix) {
        int m = matrix.size(), n = matrix[0].length();
        sum = vector<vector<int>>(m + 1, vector<int>(n + 1));
        for(int i = 0; i < m; i++)
        {
            for(int j = 0; j < n; j++)
                sum[i + 1][j + 1] = sum[i + 1][j] + sum[i][j + 1] - sum[i][j] + (matrix[i][j] & 1);
        }
    }

    // 返回左上角在 (r1,c1) 右下角在 (r2-1,c2-1) 的子矩阵元素和（类似前缀和的左闭右开）
    int query(int r1, int c1, int r2, int c2) {
        return sum[r2][c2] - sum[r2][c1] -sum[r1][c2] + sum[r1][c1];
    }
};

class Solution {
public:
    int ways(vector<string>& pizza, int k) {
        const int MOD = 1e9 + 7;
        MatrixSum ms(pizza);
        int m = pizza.size(), n = pizza[0].length();
        int dp[k][m][n];
        for(int c = 0; c < k; c++) // 从切0刀开始
        {
            for(int i = 0; i < m; i++)
            {
                for(int j = 0; j < n; j++)
                {
                    if(c == 0)
                    {
                        dp[c][i][j] = ms.query(i, j, m, n) ? 1 : 0; // 是否有苹果
                        continue;
                    }
                    int res = 0; // 累加方案数 枚举切的位置
                    for(int j2 = j + 1; j2 < n; j2++) // 垂直切
                    {
                        if(ms.query(i, j, m, j2))
                            res = (res + dp[c - 1][i][j2]) % MOD;
                    }
                    for(int i2 = i + 1; i2 < m; i2++) // 水平切
                    {
                        if(ms.query(i, j, i2, n))
                            res = (res + dp[c - 1][i2][j]) % MOD;
                    }
                    dp[c][i][j] = res;
                }
            }
        }
        return dp[k - 1][0][0];
    }
};
```
</details> 

优化思路：  
- 若左边界没有苹果，`sum[i][j]=sum[i][j+1]`；若上边界没有苹果，`sum[i][j]=sum[i+1][j]`
- 观察状态转移方程；计算的就是 `dp[c−1]` 第 `i` 行的后缀和，以及第 `dp[c−1]` 第 `j` 列的后缀和；可以预处理出来或一边计算 `dp[c][i][j]` 一边更新
- 此外，将二维前缀和改为二维后缀和
- 由于`dp[c]` 只依赖于 `dp[c-1]`，还可以空间压缩

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    int ways(vector<string>& pizza, int k) {
        const int MOD = 1e9 + 7;
        int m = pizza.size(), n = pizza[0].size();
        vector<vector<int>> sum(m + 1, vector<int>(n + 1)); // 二维后缀和
        vector<vector<int>> dp(m + 1, vector<int>(n + 1));
        for(int i = m-1; i >= 0; i--)
        {
            for(int j = n-1; j >= 0; j--)
            {
                sum[i][j] = sum[i][j + 1] + sum[i + 1][j] - sum[i + 1][j + 1] + (pizza[i][j] & 1);
                if(sum[i][j])
                    dp[i][j] = 1; // 在这初始化
            }
        }

        while(--k)
        {
            vector<int> col_s(n, 0); // dp数组第j列的后缀和

            // 倒序遍历的过程中 更新后缀和
            for(int i = m-1; i >= 0; i--)
            {
                int row_s = 0; // dp[i]的后缀和
                for(int j = n-1; j >= 0; j--)
                {
                    int temp = dp[i][j];
                    if(sum[i][j] == sum[i][j + 1]) // 左边界没有苹果
                        dp[i][j] = dp[i][j + 1];
                    else if(sum[i][j] == sum[i + 1][j]) // 上边界没有苹果
                        dp[i][j] = dp[i + 1][j];
                    else 
                        dp[i][j] = (row_s + col_s[j]) % MOD;

                    row_s = (row_s + temp) % MOD;
                    col_s[j] = (col_s[j] + temp) % MOD;
                }
            }
        }
        return dp[0][0];
    }
};
```
</details> 
<br>

---
### &emsp; 1483. 树节点的第K个祖先 :rage:HARD
关键思路：  
- <b>倍增</b>支持快速查询：使用DP预处理 存储 node 节点距离为 `2^i` 的祖先是谁
- 第k个祖先 将k拆解成二进制（2的n次项 之和）
- 查询时 取二进制数位 根据这个跳步查询

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class TreeAncestor {
public:
    vector<vector<int>> pa_dp;
    TreeAncestor(int n, vector<int>& parent) {
        int m = 32 - __builtin_clz(n); // n 的二进制长度
        pa_dp.resize(n, vector<int>(m, -1));

        for(int i = 0; i < n; i++)
            pa_dp[i][0] = parent[i];
        for(int i = 0; i < m-1; i++) // dp 二进制倍增
        {
            for(int x = 0; x < n; x++)
            {
                if(int p = pa_dp[x][i]; p != -1)
                    pa_dp[x][i+1] = pa_dp[p][i];
            }
        }
    }
    
    int getKthAncestor(int node, int k) {
        int m = 32 - __builtin_clz(k); // k 的二进制长度
        for(int i = 0; i < m; i++)
        {
            // 取二进制数位 根据这个跳步查询
            if((k >> i) & 1) 
            { // k 的二进制从低到高第 i 位是 1
                node = pa_dp[node][i];
                if(node < 0)
                    break;
            }
        }
        return node;
    }
};
```
</details> 
<br>

---
### &emsp; 2304. 网格中的最小路径代价 MID
关键思路：  
- 先寻找子问题；当前节点的最小代价如何得到？
- 还可以空间优化，将dp数组直接存在grid中

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    int minPathCost(vector<vector<int>>& grid, vector<vector<int>>& moveCost) {
        int m = grid.size(), n = grid[0].size();
        vector<vector<int>> dp(m, vector<int>(n, INT_MAX));
        dp[m-1] = grid[m-1];
        for(int i = m - 2; i >= 0; i--) // 倒序枚举行
        {
            for(int j = 0; j < n; j++)
            {
                for(int k = 0; k < n; k++) // 枚举下一行的第k列
                    dp[i][j] = min(dp[i][j], dp[i+1][k] + moveCost[grid[i][j]][k]);
                dp[i][j] += grid[i][j];
            }
        }
        return *min_element(dp[0].begin(), dp[0].end());
    }
};
```
</details> 
<br>

### 扩展：LCA问题（求xy最近公共祖先）的解法模板  
- DFS预处理各节点深度
- 假设`depth[x] < depth[y]`（否则交换两点），先将 y 更新为 y 的第`depth[y] - depth[x]`个祖先节点，使 x、y 处于同一深度
- 如果此时`x == y`，得解；否则一起往上跳
- 先尝试大步跳，再尝试小步跳；设 `i=⌊log2(n)⌋`，循环直到 `i == 0`（*类似于二分*）：
  - 若`2^i`个祖先不存在；跳大了，`i--`
  - 若存在，但依然`pa[x][i] != pa[y][i]`，更新x、y，`i--`
  - 若存在且`pa[x][i] == pa[y][i]`，为防止跳大了，不更新x、y，`i--`
  - 循环最终`lca = pa[x][0]`

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class TreeAncestor {
    vector<int> depth;
    vector<vector<int>> pa;
public:
    TreeAncestor(vector<pair<int, int>> &edges) {
        int n = edges.size() + 1;
        int m = 32 - __builtin_clz(n); // n 的二进制长度
        vector<vector<int>> g(n);
        for(auto [x, y]: edges)
        { // 节点编号从 0 开始
            g[x].push_back(y);
            g[y].push_back(x);
        }

        depth.resize(n);
        pa.resize(n, vector<int>(m, -1));
        function<void(int, int)> dfs = [&](int x, int fa) {
            pa[x][0] = fa;
            for(int y: g[x]) {
                if(y != fa) {
                    depth[y] = depth[x] + 1;
                    dfs(y, x);
                }
            }
        };
        dfs(0, -1);

        for(int i = 0; i < m - 1; i++)
            for(int x = 0; x < n; x++)
                if(int p = pa[x][i]; p != -1)
                    pa[x][i + 1] = pa[p][i];
    }

    int get_kth_ancestor(int node, int k) {
        for(; k; k &= k - 1)
            node = pa[node][__builtin_ctz(k)];
        return node;
    }

    // 返回 x 和 y 的最近公共祖先（节点编号从 0 开始）
    int get_lca(int x, int y) {
        if(depth[x] > depth[y])
            swap(x, y);
        // 使 y 和 x 在同一深度
        y = get_kth_ancestor(y, depth[y] - depth[x]);
        if (y == x)
            return x;
        for(int i = pa[x].size() - 1; i >= 0; i--) 
        {
            int px = pa[x][i], py = pa[y][i];
            if(px != py) 
            {
                x = px;
                y = py;
            }
        }
        return pa[x][0];
    }
};
```
</details> 
<br>

---
### &emsp; 2369. 检查数组是否存在有效划分 MID
关键思路：  
- 状态定义：数组前i位是否存在有效划分

<details> 
<summary> <b>C++ Code</b> </summary>  

```c++
class Solution {
public:
    bool validPartition(vector<int>& nums) {
        int n = nums.size();
        vector<bool> dp(n + 1);
        dp[0] = true;
        for(int i = 1; i <= n; i++)
        {
            if(i >= 2 && nums[i-1] == nums[i-2])
                dp[i] = dp[i] | dp[i-2];
            if(i >= 3 && nums[i-1] == nums[i-2] && nums[i-1] == nums[i-3])
                dp[i] = dp[i] | dp[i-3];
            if(i >= 3 && nums[i-1] == nums[i-2] + 1 && nums[i-2] == nums[i-3] + 1)
                dp[i] = dp[i] | dp[i-3];
        }
        return dp[n];
    }
};
```
</details> 
<br>

---
### &emsp; 2617. 网格图中最少访问的格子数 :rage:HARD
关键思路：  
- 定义`f[i][j]`：从`(i, j)` 到 `(m-1, n-1)`经过的最少格子数
- 状态转移：可以从行转`f[k][j]`也可以从列转`f[i][k]`，取能够作为转移来源的最小`f`
- $f[i][j]=min(min_{k=j+1}^{j+g}f[i][k], min_{k=i+1}^{i+g}f[k][j]) + 1$
- 注意状态转移过程的单调性（枚举遍历的过程）：从f[i][j]到f[i][k]，倒序枚举 `j` 时，`k` 的左边界 `j+1` 单调减小，右边界无单调性
- 使用一个 <b>根据`f`值单调递增的栈</b> 维护 `f[i][j]` 以及 `j`，由于是在倒序枚举 `j` 的过程，对于 `j` 来说在栈中是单调递减的
- 因此<b>在单调栈上二分查找（利用有序性）最大的不超过 `j+g` 的下标`k`，对应 `f[i][k]` 就是最小的 `f[i][k]`</b>
- 由于按照先行再列的顺序遍历，只需要一个行单调栈维护当前行的信息
- 法二：用并查集合并访问过的节点，见Union-find.md

<details> 
<summary> <b>C++ Code</b> </summary>  

```c++
class Solution {
public:
    int minimumVisitedCells(vector<vector<int>>& grid) {
        int m = grid.size(), n = grid[0].size();
        vector<vector<pair<int, int>>> col_stacks(n); // 每列的单调栈
        vector<pair<int, int>> row_st; // 行单调栈
        // 栈内保存 (f[id1][id2], id)
        int mn; // 缓存f[i][j] 
        for(int i = m - 1; i >= 0; i--)
        {
            row_st.clear();
            for(int j = n - 1; j >= 0; j--)
            {
                int g = grid[i][j];
                auto &col_st = col_stacks[j];
                mn = i < m - 1 || j < n - 1 ? INT_MAX : 1;
                if(g != 0) // 可以向右/向下
                {
                    // 在单调栈上二分查找最优转移的来源
                    // 对于k来说是单调减栈（逆序遍历过程）
                    auto it = lower_bound(row_st.begin(), row_st.end(), j + g, 
                        [](const auto &a, const int b) {
                            return a.second > b;
                        });
                    if(it < row_st.end())
                        mn = it->first + 1;

                    it = lower_bound(col_st.begin(), col_st.end(), i + g,
                        [](const auto &a, const int b) {
                            return a.second > b;
                        });
                    if(it < col_st.end())
                        mn = min(mn, it->first + 1);
                }
                if(mn < INT_MAX) // 插入单调栈中
                {
                    // f值单调增
                    while(!row_st.empty() && mn <= row_st.back().first)
                        row_st.pop_back();
                    row_st.emplace_back(mn, j);

                    while(!col_st.empty() && mn <= col_st.back().first)
                        col_st.pop_back();
                    col_st.emplace_back(mn, i);
                }
            }
        }
        return mn < INT_MAX ? mn : -1; // 最后一个算出的 mn 就是 f[0][0]
    }
};
```
</details> 
<br>

---
### &emsp; 2707. 字符串中的额外字符 MID
关键思路：  
- 将字典中的字符串存入set中
- 遍历字符串s，考虑选或不选当前字符作为多余字符；若不选，枚举 <b>能选哪个以当前遍历 i 结尾子字符串</b> 的起点 j

<details> 
<summary> <b>C++ Code</b> </summary>  

```c++
class Solution {
public:
    int minExtraChar(string s, vector<string>& dictionary) {
        unordered_set<string> set(dictionary.begin(), dictionary.end());
        int n = s.size();
        vector<int> dp(n+1);
        for(int i = 0; i < n; i++)
        {
            dp[i+1] = dp[i] + 1; // 不选
            for(int j = 0; j <= i; j++) // 往前，枚举能选哪个
            {
                if(set.count(s.substr(j, i-j+1)))
                {
                    dp[i+1] = min(dp[i+1], dp[j]);
                }
            }
        }
        return dp[n];
    }
};
```
</details> 
<br>

------
## 树形DP
### 概念
---
* ### 树形DP
    基于左右子树描述状态 在树上状态转移（一般递归进行）  
    树形DP的出发点：思考 **如何通过递归去计算，如何由子问题算出原问题** （选或不选、枚举选哪个 等）
    <br>

* ### 换根DP
    树形 DP 中的换根 DP 问题又被称为 <b>二次扫描</b>，通常不会指定根结点；并且根结点的变化会对一些值，例如子结点深度和、点权和等产生影响
    **通常需要两次 DFS**，第一次 DFS 预处理诸如深度，点权和之类的信息，在第二次 DFS 开始运行换根动态规划
<br>

---

### 题目
---
### &emsp; 337. 打家劫舍 MID
关键思路：  
- <b>树上最大独立集问题的变形</b>
- 对当前节点 考虑选或不选
- <b>把 “选/不选” 作为两个状态</b> 来影响状态转移
- 选当前节点，则左右孩子都不能选；不选当前节点，左右孩子可选可不选
- 提炼状态为：选当前节点时，子树最大点权和 & 不选当前节点时，子树最大点权和
- 最终取 `max（选根节点， 不选根节点）`

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
    pair<int, int> dfs(TreeNode* node) // 抢 or 不抢
    {
        if(node == nullptr)
            return {0,0};
        auto[l_rob, l_not_rob] = dfs(node->left);
        auto[r_rob, r_not_rob] = dfs(node->right);
        
        // 状态转移方程
        int rob = l_not_rob + r_not_rob + node->val;
        int not_rob = max(l_rob, l_not_rob) + max(r_rob, r_not_rob);
        return {rob, not_rob};
    }
public:
    int rob(TreeNode* root) {
        auto [root_rob, root_not_rob] = dfs(root);
        return max(root_rob, root_not_rob);
    }
};
```
</details> 
<br>

---
### &emsp; 834. 树中距离之和 :rage: HARD
关键思路：  
- <b>换根DP</b>（状态转移过程对应为 “换根”） 在DFS的过程中“换根”
- 从 0 出发DFS 计算 0 到每个点的距离（得到ans[0]） 并计算出每个子树的大小（通过后序遍历计算每棵子树的大小）
- “换根”时 如从当前root换到右孩子时，导致右子树所有节点距离-1，左子树节点及root的距离+1
- 故由 x 换到 y 时 $ans[y] = ans[x] - size[y] + (n - size[y])$ （注意对于一条节点链上换根的过程，这个状态转移方程是递推形式的）


<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    vector<int> sumOfDistancesInTree(int n, vector<vector<int>>& edges) {
        vector<vector<int>> g(n); // 邻接表建图（无向树）
        for(auto & e : edges)
        {
            int x = e[0], y = e[1];
            g[x].push_back(y);
            g[y].push_back(x);
        }

        vector<int> ans(n);
        vector<int> size(n, 1); // 统计子树大小
        // 无向树dfs参数记得传递父节点fa
        function<void(int, int, int)> dfs = [&](int x, int fa, int depth) {
            ans[0] += depth; // depth 为 0 到 x的距离
            for(int y : g[x])
            {
                if(y != fa)
                {
                    dfs(y, x, depth + 1);
                    size[x] += size[y];
                }
            }
        };
        dfs(0, -1, 0); // 0 没有父节点

        // "换根" dfs
        function<void(int, int)> reroot = [&](int x, int fa) {
            for(int y : g[x])
            {
                if(y != fa)
                {
                    ans[y] = ans[x] + n - 2*size[y];
                    reroot(y, x);
                }
            }
        };
        reroot(0, -1);
        return ans;
    }
};
```
</details> 
<br>

---
### &emsp; 894. 所有可能的真二叉树 MID
关键思路：  
- 分析问题性质：真二叉树一定是奇数个节点；且每增加两个叶子，整棵树就会多一个叶子，故一个n节点真二叉树恰好有 `(n+1)/2` 个叶子
- <b>DP：枚举左子树有多少个叶子，划分子问题</b>
- `dp[i]`：有i个叶子的所有真二叉树的列表  
- （为什么从叶子节点数入手思考？因为叶子节点数每一个值都有意义，但总节点数如果不是奇数就没有意义；其实二者是等价的）


<details> 
<summary> <b>C++ Code</b> </summary>

```c++
vector<TreeNode*> dp[11];
auto init = [] {
    dp[1] = {new TreeNode()};
    for(int i = 2; i < 11; i++) // 计算dp[1]
    {
        for(int j = 1; j < i; j++) // 枚举左子树叶子树
        {
            for(auto left : dp[j]) // 枚举左子树
                for(auto right : dp[i - j]) // 枚举右子树
                    dp[i].push_back(new TreeNode(0, left, right));
        }
    }
    return 0;
}();

class Solution {
public:
    vector<TreeNode*> allPossibleFBT(int n) {
        return dp[n % 2 ? (n + 1)/2 : 0];
    }
};
```
</details> 
<br>

------
## 序列DP
### 概念
---
* ### 序列DP
    线性 DP 通常强调「状态转移所依赖的前驱状态」是由给定数组所提供的，即拓扑序是由原数组直接给出；而序列 DP 通常需要 <b>结合题意来寻找前驱状态</b>，即需要 <b>自身寻找拓扑序关系</b>   
    *状态转移依赖于与可能的前驱状态的关系*   
    对于最长递增子序列问题（或者一般的序列DP问题），通常都可以用「选或不选」和「枚举选哪个」来启发思考
---

* ### 最长递增子序列问题 LIS
  在一个给定的数值序列中，找到一个子序列，使得这个子序列元素的数值依次递增，并且这个子序列的长度尽可能地大。最长递增子序列中的元素在原序列中不一定是连续的  
  —— 最快 O(nlogn)  

`DP作为求解的辅助工具`  
序列DP + 二分寻找右边界：   
通过一个数组`dp[k]`来缓存长度为k的递增子序列的最末元素值，若有多个长度为k的递增子序列，则记录最小的末元素值

- 首先`len = 1`， `dp[0] = seq[0]`  
- <b>遍历seq</b> 看是否能更新`dp[]` &emsp; 对`seq[i]`： 
- &emsp;&emsp;若`seq[i] > dp[len]`，那么`len++`，`dp[len] = seq[i]`（增长链）  
- &emsp;&emsp;否则，从`dp[0]`到`dp[len]`中找到一个`j`，满足 `dp[j-1] < seq[i] < dp[j]` 然后更新`dp[j]=seq[i]`  
- 最终`len`即为最长递增子序列LIS的长度  
- 因为在dp中插入数据<b>有序</b>且<b>只需替换不用挪动</b>，因此我们可以使用<b>二分查找</b>，将每一个数字的插入时间优化为O(logn)  算法的时间复杂度从使用 排序+LCS 的O(n^2)降低到了O(nlogn)

<br>

---

### 题目
---
### &emsp; 646. 最长数对链 MID
关键思路：  
- 将pairs <b>按第一个数升序排序</b>  
- 定义`f[i]`：以`pairs[i]`为结尾的 <b>最长数对链长度</b>，所有`f[i]`中的最大值为答案  
- 由贪心思想确定状态转移方程：从`j=i-1`往回找到第一个`pairs[j][1] < pairs[i][0]`，`f[i] = f[j]+1`  
- &emsp;&emsp;证明贪心的正确性：假设还存在`j'< j`满足`f[j'] > f[j]`，由于`pairs[j][0] > pairs[j'][0]`，`pairs[j]`可以替换`f[j']`对应的路径中最后的`pairs[j']`，故假设不成立  
- &emsp;&emsp;亦即，对于一个特定`pairs[i]`而言，其所有合法（满足条件`pairs[j][1] < pairs[i][0]`）的前驱状态 `f[j]` 必然是非单调递增的  

- &emsp;&emsp;根据LIS问题的贪心解的思路，可以<b>额外使用一个数组记录下特定长度数链的最小结尾值</b>，从而实现 <b>二分找前驱状态</b>  
- 二分的关键：<b>确定搜索空间 & 确定循环不变量（搜索区间的性质）</b>  

- &emsp;&emsp;具体地，创建`g[ ]`，其中`g[len]=x` 代表数链长度为len时结尾元素的第二维最小值为x

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    int findLongestChain(vector<vector<int>>& pairs) {
        sort(pairs.begin(), pairs.end());
        int n = pairs.size();
        int ans = 0;
        vector<int> g(n+1, INT_MAX); // DP
        for(int i = 0; i < n; i++)
        {
            // 以pairs[i]结尾的链 更新g[]
            int left = 1, right = i + 1; 
            // 对 i而言  len可能的取值范围 [1,i+1)
            // 二分寻找满足 g[len] < pairs[i][0]的最大len 
            while(left < right)
            {
                int mid = (left + right) >> 1;
                if(g[mid] >= pairs[i][0]) // 这个mid对应的链不满足作为前驱状态的条件
                    right = mid;
                else // 满足 继续向右边找 看看能不能更右
                    left = mid + 1;
            } 
            // right - 1 是找到的右边界  此时 right == left
            // 结束时 right == mid(不能增长链) || right == mid + 1(可以增长链)

            // pairs[i]对应链的长度为 right-1+1 即right
            g[right] = min(g[right], pairs[i][1]); // 更新g[len] (检查是否替换原先的末尾数对)
            ans = max(ans, right);
        }
        return ans;
    }

    //---------------------
    // 思路二：贪心
    // int findLongestChain(vector<vector<int>>& pairs) {
    //     int cur = INT_MIN;
    //     int res = 0;
    //     sort(pairs.begin(), pairs.end(), [](const vector<int> &a, const vector<int> &b){
    //         return a[1] < b[1];
    //     });
    //     for(auto &p: pairs)
    //     {
    //         if(cur < p[0])
    //         {
    //             cur = p[1];
    //             res++;
    //         }
    //     }
    //     return res;
    // }
};
```
</details> 
<br>

---
### &emsp; 1043. 分隔数组以得到最大和 MID
关键思路：  
- 枚举最后一段子数组的开始下标

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    int maxSumAfterPartitioning(vector<int>& arr, int k) {
        int n = arr.size();
        int dp[n+1];
        memset(dp, -1, sizeof(dp));
        dp[0] = 0;
        dp[1] = arr[0];
        for(int i = 2; i <= n; i++)
        {
            int maxV = arr[i-1];
            for(int j = 1; j <= k && i - j >= 0; j++) // 最后一个子数组大小
            {
                maxV = max(maxV, arr[i-j]);              
                dp[i] = max(dp[i], dp[i-j] + j*maxV);
            } 
        }
        return dp[n];
    }
};
```
</details> 
<br>

---
### &emsp; 1092. 最短公共超序列 :rage: HARD
关键思路：  
- <b>dp数组记录描述状态的转移（状态间的关系）</b> 用dp数组递推构造出结果字符串

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    string shortestCommonSupersequence(string str1, string str2) {
        int n = str1.length(), m = str2.length();
        int dp[n+1][m+1]; // 以下标 i j 结束的子串的最短公共超序列长度
        // 递归边界 -- 记忆化数组初始值
        for(int i = 0; i < n; i++)
            dp[i][0] = i;
        for(int j = 0; j < m; j++)
            dp[0][j] = j;

        // 记忆化搜索改成dp 每个参数对应一层循环
        for(int i = 0; i < n; i++)
        {
            for(int j = 0; j < m; j++)
            {
                if(str1[i] == str2[j]) // 此时最短公共子序列一定包含当前相同的末尾
                    dp[i+1][j+1] = dp[i][j] + 1;
                else
                    dp[i+1][j+1] = min(dp[i][j+1], dp[i+1][j]) + 1;
            }
        }

        string ans;
        int i = n-1, j = m-1;
        while(i >= 0 && j >= 0) // 找前驱状态
        {
            if(str1[i] == str2[j]) // 相当于继续递归 make_ans(i-1, j-1)
            {
                ans += str1[i];
                i--;
                j--;
            }
            else if(dp[i+1][j+1] == dp[i][j+1] + 1) // 相当于继续递归 make_ans(i-1, j)
            {
                ans += str1[i];
                i--;
            }
            else // 相当于继续递归 make_ans(i, j-1)
            {
                ans += str2[j];
                j--;
            }
        }
        reverse(ans.begin(), ans.end());
        return str1.substr(0, i+1) + str2.substr(0, j+1) + ans; // 记得加上剩余子串 相当于make_ans的边界返回值
    }
};
```
</details> 
<br>

---
### &emsp; 1105. 填充书架 MID
关键思路：  
- 观察题目条件，要求摆放书的顺序与整理好的顺序相同
- 故枚举最后一层的第一本书的下标 j 即可

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    int minHeightShelves(vector<vector<int>>& books, int shelfWidth) {
        int n = books.size();
        int dp[n+1];
        dp[0] = 0;
        for(int i = 1; i <= n; i++)
        {
            dp[i] = INT_MAX;
            int width = 0;
            int height = 0;
            for(int j = i; j >= 1 && width < shelfWidth; j--)
            {
                width += books[j-1][0];
                if(width > shelfWidth)
                    break;
                height = max(height, books[j-1][1]);

                dp[i] = min(dp[j-1] + height, dp[i]);
            }
        }
        return dp[n];
    }
};
```
</details> 
<br>

---
### &emsp; 1186. 删除一次得到子数组最大和 MID
关键思路：  
- 问题拆分成两种情况（状态）：【1】.子数组不能删除数字 & 【2】.子数组必须删除一个数字
- 枚举子数组右端点`[0, i)` 考虑左边的数选或不选（继承之前的子数组 还是 新开始一段子数组）
- 对于【2】的状态转移，讨论是否删除当前数字
- 边界条件：子数组为空时，`INT_MIN`表示不合法，递归中通过取`max`自然会去到合法的情况
- （可以进一步优化空间复杂度）

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    int maximumSum(vector<int>& arr) {
        int ans = INT_MIN;
        int n = arr.size();
        vector<vector<int>> dp(n + 1, vector<int>(2, INT_MIN / 2)); // 防止负数相加溢出
        for(int i = 1; i <= n; i++)
        {
            dp[i][0] = max(dp[i-1][0] + arr[i-1], arr[i-1]); // 不删数字的状态
            dp[i][1] = max(dp[i-1][1] + arr[i-1], dp[i-1][0]); // 只删一个数字的状态
            ans = max(ans, max(dp[i][0], dp[i][1]));
        }
        return ans;
    }
};
```
</details> 
<br>

---
### &emsp; 1187. 使数组严格递增 :rage: HARD
关键思路：  
- <b>对于最长递增子序列问题（或者一般的序列DP问题），通常都可以用「选或不选」和「枚举选哪个」来启发思考</b> 
- 本题解法为 “枚举选哪个”
- 最终要寻找一个符合条件（不在其中的元素可以被替换）的LIS；把重点放在 LIS 上，关注哪些 a[i] 没有被替换，那么答案就是 n − length(lis)
- <b>dp求a[i]能链上的最长的LIS链</b>
- b[k]为 大于等于 a[i]的最小元素 则b[k-1]为 小于 a[i]的最大元素
- dp[i] 表示以 a[i] 结尾的 LIS 的长度

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    int makeArrayIncreasing(vector<int>& arr1, vector<int>& arr2) {
        vector<int> a(arr1), b(arr2);
        a.push_back(INT_MAX);
        sort(b.begin(), b.end());
        b.erase(unique(b.begin(), b.end()), b.end()); // 原地去重

        int n = a.size();
        int dp[n];
        for(int i = 0; i < n; i++)
        {
            int k = lower_bound(b.begin(), b.end(), a[i]) - b.begin(); // b[k]为>= a[i]的最小元素 则b[k-1]为 < a[i]的最大元素
            // 考察以a[i]作为lis的开始
            int res = k < i ? INT_MIN : 0; // 不足以替换a[i]前所有数时，初始化为INT_MIN
            
            // 考察a[i]是否能链在某个lis末尾
            if(i && a[i-1] < a[i]) // 递增 无需替换
            {
                res = max(dp[i-1], res);
            }
            for(int j = i - 2; j >= i-k-1 && j >= 0; j--)
            {
                if(b[k - (i-j-1)] > a[j]) // 往前找i-(j+1)个元素
                {
                    // a[j+1] 到 a[i-1] 可以替换成 b[k-(i-j-1)] 到 b[k-1]
                    res = max(dp[j], res);
                }
            }
            dp[i] = res + 1; // 取能链上的最长的lis 长度+1
        }
        return dp[n-1] > 0 ? n - dp[n-1] : -1;
    }
};
```
</details> 
<br>

---
### &emsp; 1911. 最大子序列交替和 MID
关键思路：  
- <b>选或不选</b> 描述状态转移
- `f[i]`：从前i个元素选 最后一个下标为奇数时的最大交替和
- `g[i]`：从前i个元素选 最后一个下标为偶数时的最大交替和
- 注意当前状态`i`只与上一步状态`i-1`有关，可空间优化

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    long long maxAlternatingSum(vector<int>& nums) {
        int n = nums.size();
        vector<long long> f(n + 1), g(n + 1);
        for(int i = 1; i <= n; i++)
        {
            f[i] = max(g[i - 1] - nums[i - 1], f[i - 1]);
            g[i] = max(f[i - 1] + nums[i - 1], g[i - 1]);
        }
        return max(f[n], g[n]);
    }
};
```
</details> 
<br>

---
### &emsp; 2809. 使数组和小于等于k的最小时间 :rage: HARD
关键思路：  
- 对每个元素，至多操作一次（否则只操作最后一次就可以更优了）；总体最多操作 n 次
- <b>枚举最小时间 `t`</b>
- 不进行操作`t`秒后元素和 `s1 + s2*t`
- 对于一个元素，如果在第`k`秒操作 则最后减少了`nums1[i] + nums2[i]* k`
- 对于一个 选定要操作的元素集 ，先操作nums2小的、再操作nums2大的，可以使收益最大化（使大的`k`分配给大的`nums2[i]`）；故先根据nums2进行排序
- 问题转化为： **从`[0, n-1]`选总共 `t`个下标作为 子序列 并进行 依次操作（问题建模中保证了对这个元素集来说收益最大化），如何使最终收益最大（找到一个最大收益最大的 t-元素集）**
- 类似0-1背包问题 定义`dp[i+1][j]`： 表示 *从`[0, i]`选择`j`个元素 可以达到的 最大减少量* （DP状态值对应最大收益）
- <b>状态转移考虑 选或不选</b>
- `dp[i+1][j]` 的前置状态为 `dp[i][j]`、`dp[i][j-1]` 因此可以压缩第一维


<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    int minimumTime(vector<int>& nums1, vector<int>& nums2, int x) {
        int n = nums1.size();
        vector<int> ids(n); // 下标数组
        iota(ids.begin(), ids.end(), 0);
        sort(ids.begin(), ids.end(), [&](const int i, const int j) {
            return nums2[i] < nums2[j];
        }); // 根据nums2排序下标数组

        vector<int> dp(n + 1); // 最多操作n次
        for(int i = 0; i < n; i++)
        {
            int a = nums1[ids[i]], b = nums2[ids[i]];
            for(int j = i + 1; j > 0; j--)
            {
                dp[j] = max(dp[j], dp[j - 1] + a + b*j);
            }
        }

        int s1 = accumulate(nums1.begin(), nums1.end(), 0);
        int s2 = accumulate(nums2.begin(), nums2.end(), 0);
        for(int t = 0; t <= n; t++) // 枚举需要的操作次数t
        {
            if(s1 + s2 * t - dp[t] <= x)
                return t;
        }
        return -1;
    }
};
```
</details> 
<br>

------
## 区间DP
### 概念
---
* ### 区间DP
    线性DP一般再前缀/后缀上转移，而区间DP<b>从小区间转移到大区间</b>

<br>

---

### 题目
---
### &emsp; 1000. 合并石头的最低成本 :rage: HARD
关键思路： 
- 将一个区间的合并的问题划分为两个区间分别合并，枚举划分点
- `dfs(i, j, p)`： 把`[i,j]` 合并成 `p` 堆的最低成本 （1 <= p <= k-1）
- 枚举划分点时，保证`m-i`是`k-1`的倍数（`[i,m]`合并成一堆）
- `p > 1`时，`j-i`不是`k-1`的倍数，否则可以合成一堆
- 因此，可以省略对参数p的存储空间

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    int mergeStones(vector<int>& stones, int k) {
        int n = stones.size();
        if((n-1) % (k-1)) // 无法合并成一堆
            return -1;
        
        int s[n+1];
        s[0] = 0;
        for(int i = 0; i < n; i++)
            s[i+1] = s[i] + stones[i]; // 前缀和

        int dp[n][n];
        for(int i = n - 1; i >= 0; i--)
        {
            dp[i][i] = 0; // 仅此一堆无需再合并
            for(int j = i + 1; j < n; j++)
            {
                dp[i][j] = INT_MAX;
                for(int m = i; m < j; m += k-1) // 枚举划分点 [i, m]合为一堆
                {
                    dp[i][j] = min(dp[i][j], dp[i][m] + dp[m+1][j]); 
                }
                if((j-i)%(k-1) == 0) // 可以合并成一堆
                    dp[i][j] += s[j+1] - s[i];
            }
        }
        return dp[0][n-1];
    }
};
```
</details> 
<br>

---
### &emsp; 1039. 多边形三角形剖分的最低得分 MID
关键思路： 
- <b>对于一条边，枚举另一个顶点</b> 等于枚举一个三角形 <b>（枚举划分点k）</b>
- 定义 <b>从i到j</b> 区间，表示沿着顶点i顺指针到顶点j，再加上边ji组成的多边形
- 由于需要从下标比`i`更大的`k`转移到`dp[i][]`，`i`需要倒序枚举
- 由于需要从下标比`j`更小的`k`转移到`dp[][j]`，`j`需要正序枚举

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    int minScoreTriangulation(vector<int>& values) {
        int n = values.size();
        int dp[n][n];
        for(int i = 0; i < n - 1; i++)
            dp[i][i+1] = 0;

        for(int i = n - 3; i >= 0; i--)
        {
            for(int j = i + 2; j < n; j++)
            {
                dp[i][j] = INT_MAX;
                for(int k = i + 1; k < j;k++)
                    dp[i][j] = min(dp[i][j], dp[i][k] + dp[k][j] + values[i]*values[j]*values[k]);
            }
        }
        return dp[0][n-1];
    }
};
```
</details> 
<br>

---
### &emsp; 1130. 叶值的最小代价生成树 MID
关键思路： 
- 考虑 树的“结构”与“生成”
- 将数组（当前区间）划分为左右两个非空子数组（子区间），分别对应树的左右子树
- 用一个二维数组`g[][]`记录数组区间最大值

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    int mctFromLeafValues(vector<int>& arr) {
        int n = arr.size();
        int g[n][n];
        int f[n][n];
        memset(f, 0, sizeof(f));
        for(int i = n - 1; i >= 0; i--)
        {
            g[i][i] = arr[i];
            for(int j = i + 1; j < n; j++) // 区间[i, j]
            {
                g[i][j] = max(g[i][j-1], arr[j]);
                f[i][j] = 0x3f3f3f3f;
                for(int k = i; k < j; k++) // 枚举左右子树划分点 生成一个新的非叶节点
                {
                    f[i][j] = min(f[i][j], f[i][k] + f[k+1][j] + g[i][k]*g[k+1][j]);
                }
            }
        }
        return f[0][n-1];
    }
};
```
</details> 
<br>

------
## 状压DP
### 概念
---
* ### 状压DP
    

<br>

---

### 题目
---
### &emsp; 1494. 并行课程II :rage: HARD
关键思路： 
- <b>集合论&位运算</b> 使用bit位状态表示集合 标识已修课程的状态
- <b>子集状压DP</b> &emsp; `dp[i]`：修完`i`中所有课程需要的最少学期数
- 定义`pre[j]` 为课程`j` 的先修课程的并集
- 枚举`i` 的大小不超过`k`的非空子集，作为一个学期内需要学完的课程，进行 <b>集合间的状态转移</b>
- 注意这些子集中的课程`j` 的先修课程必须要在`i`的补集内，表示之前学期中已经学完，才可以在当前轮次（当前学期）进行状态转移
- 即得到了枚举可选课程的条件：`j属于i && pre[j]在i的补集中`
- 遍历可选课程的子集：<b>`for(int j = lessons; j; j = (j-1) & lessons)`</b>

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    int minNumberOfSemesters(int n, vector<vector<int>>& relations, int k) {
        int pre[n]; // 各课程的先修课程集合
        memset(pre, 0, sizeof(pre));
        for(auto &r : relations)
            pre[r[1] - 1] |= 1 << (r[0] - 1);

        int u = (1<<n) -1; // 全集
        int dp[1<<n];
        dp[0] = 0;
        for(int i = 1; i < 1<<n; i++)
        {
            dp[i] = INT_MAX;
            int lessons = 0; // 可选课程的集合
            int ci = u ^ i; // i的补集
            for(int j = 0; j < n; j++) // 枚举课程
            {
                if(((i >> j) & 1) && (pre[j] | ci) == ci) // j属于i && pre[j]在i的补集 可以学
                {
                    lessons |= 1 << j;
                }
            }
            if(__builtin_popcount(lessons) <= k) // 如果可学课程少于k 则可以全部学习 不再枚举子集
            {
                dp[i] = dp[i ^ lessons] + 1;
                continue;
            }
            for(int j = lessons; j; j = (j-1) & lessons) // 枚举lessons的子集
            {
                if(__builtin_popcount(j) == k)
                    dp[i] = min(dp[i], dp[i ^ j] + 1); // 顺序枚举状态i 已经保证了此时dp[i^j]是计算完毕的
            }
        }
        return dp[u];
    }
};
```
</details> 
<br>

---
### &emsp; 1595. 连通两组点的最小成本 :rage: HARD
关键思路： 
- 由 <b>枚举选哪个</b> 切入思考，枚举第一组的每个点连接第二组的所有情况 再用贪心的思路处理第二组剩余没有连的点
- 定义<b>状态</b>：第二组中点的连接情况
- DP的边界条件：第二组还有子集j中的点没连上
- 状态转移：当第一组的点`i`连上第二组的点`k`，第二组剩余子集`j ` 的前驱状态为`j & ~(1 << k)`（`j∖{k}`，从集合`j` 中去掉元素`k`）
- 遍历第二组点对应的所有状态，寻找 *在第一组的每种枚举连接情况下* 的前驱状态

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    int connectTwoGroups(vector<vector<int>>& cost) {
        int n = cost.size(), m = cost[0].size(); // 注意 n >= m
        vector<int> min_cost(m, INT_MAX); // 连接第二组最后还没连上的点
        for(int j = 0; j < m; j++) // 计算第二组每个点的最小连接成本
        {
            for(auto &c : cost)
            {
                min_cost[j] = min(min_cost[j], c[j]);
            }
        }

        vector<vector<int>> dp(n + 1, vector<int>(1 << m));// 第二维描述第二组的状态
        // 计算dp边界条件 第二组还有集合j的点没连上
        for(int j = 0; j < 1 << m; j++) // 第二组的所有子集
        {
            for(int k = 0; k < m; k++)
            {
                if(j >> k & 1)
                    dp[0][j] += min_cost[k];
            }
        }

        for(int i = 0; i < n; i++) // 顺序遍历第一组 枚举第一组的点如何连接
        {
            for(int j = 0; j < 1 << m; j++) // 遍历第二组的状态
            {
                int res = INT_MAX;
                for(int k = 0; k < m; k++) // 枚举第一组i与第二组k连接
                {
                    res = min(res, dp[i][j & ~(1 << k)] + cost[i][k]); // j的第k位置零
                }
                dp[i+1][j] = res;
            }
        }
        return dp[n][(1 << m) - 1];
    }
};
```
</details> 
<br>

---
### &emsp; 1681. 最小不兼容性 :rage: HARD
关键思路： 
- 要使同一个子集里面没有两个相同的元素
- 枚举所有的子集`i`，若`i` 对应状态有m个1 且没有重复元素 则计算不兼容性 
- `f[i]`：当前已经划分的集合状态为`i` 时，其子集不兼容性之和 的最小值
- 找出所有未被划分且不重复的元素 用一个状态`mask`表示
- DP把不同的合法子集状态拼成“更大的”状态

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    int minimumIncompatibility(vector<int>& nums, int k) {
        int n = nums.size();
        int m = n / k; // 子集大小

        int g[1 << n]; // 预处理计算所有合法子集的不兼容性
        memset(g, -1, sizeof(g));
        for(int i = 1; i < 1<<n; i++)
        {
            if(__builtin_popcount(i) != m)
                continue;
            
            unordered_set<int> s; // set用于判断重复元素
            int mi = 0x3f3f3f3f, mx = 0;
            for(int j = 0; j < n; j++) // 子集元素
            {
                if(i >> j & 1)
                {
                    if(s.count(nums[j])) // 重复
                        break;
                    s.insert(nums[j]);
                    mx = max(mx, nums[j]);
                    mi = min(mi, nums[j]);
                }
            }
            if(s.size() == m) // 合法子集
                g[i] = mx - mi;
        }

        int dp[1 << n];
        memset(dp, 0x3f3f3f3f, sizeof(dp));
        dp[0] = 0;
        for(int i = 0; i < 1 << n; i++)
        {
            if(dp[i] == 0x3f3f3f3f)
                continue;
            
            unordered_set<int> s;
            int mask = 0;
            for(int j = 0; j < n; j++) // 不在i中（未被划分）且没有与其重复的元素
            {
                if((i >> j & 1) == 0 && !s.count(nums[j]))
                {
                    s.insert(nums[j]);
                    mask |= 1 << j; // 加入mask
                }
            }
            if(s.size() < m)
                continue;
            for(int j = mask; j; j = (j - 1) & mask) // mask的所有子集
            {
                if(g[j] != -1) // 合法子集
                    dp[i|j] = min(dp[i|j], dp[i] + g[j]); // 拼起来 计算一个“更大的”状态的结果
            }
        }
        return dp[(1 << n) - 1] == 0x3f3f3f3f? -1: dp[(1 << n) - 1];
    }
};
```
</details> 
<br>

---
### &emsp; 1799. N次操作后的最大分数和 :rage: HARD
关键思路： 
- 设计状态：指示 <b>当前有哪些元素已经参与了计算</b>
- 预处理出所有数对的最大公约数
- 计算状态对应二进制1的个数`cnt` 同时也得到这是第`cnt/2`次操作

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    int maxScore(vector<int>& nums) {
        int n = nums.size();
        // 预处理得所有数对的gcd值
        vector<vector<int>> g(n, vector<int>(n, 0));
        for(int i = 0; i < n; i++)
        {
            for(int j = 0; j < n; j++)
                g[i][j] = gcd(nums[i], nums[j]); 
        }

        vector<int> dp(1 << n , 0); // 状压dp
        for(int k = 0; k < 1<<n; k++) // 枚举状态
        {
            int cnt = __builtin_popcount(k); // 计算1的个数 也由此得是第几次操作
            if(cnt % 2 == 0) // cnt为偶数才是有效状态
            {
                for(int i = 0; i < n; i++)
                {
                    if(k >> i & 1)
                    {
                        for(int j = i + 1; j < n; j++)
                        {
                            if(k >> j & 1)
                                dp[k] = max(dp[k], dp[k^(1<<i)^(1<<j)] + (cnt / 2) * g[i][j]);
                        }
                    }
                }
            }
        }
        return dp[(1 << n) - 1];
    }
};
```
</details> 
<br>

------
## 数位DP
### 概念
---
* ### 数位DP
    某范围内满足某条件的数的个数  
    基于位运算描述状态 <b>按位遍历数字</b> （构造） 
    以每一 “位” 为单位（转化为字符串）  
    “数位” 与 “状态” 将带来限制（如何建模表达限制？）


<br>

---

### 题目
---
### &emsp; 1012. 至少有一位重复的数字 :rage: HARD
关键思路：  
- 转换为求无重复数字的个数
- 存储状态 <b>当前遍历到第几位数字 * 使用过的数字mask</b>
- 终结状态的返回值也标示着一条合法搜索路径的结束
- 注意 <b>`is_limit` 与 `is_num` 的变化在调用过程中都是单向的</b> 不需要再dp里存储状态

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    // 转换成求无重复数字的个数
    int numDupDigitsAtMostN(int n) {
        string s = to_string(n);
        int len = s.length();
        int dp[len][1 << 10]; // 存储状态 到第几位数字 * mask
        memset(dp, -1, sizeof(dp));

        // i:正在处理第i位 
        // mask:标记已经用过的数字 
        // is_limit:前i-1位的数字是否与n的前i-1位相同 当前填的位数字是否有上界约束
        // is_num:是否已经开始填数字了 为false时 当前位可以继续跳过
        function<int(int, int, bool, bool)> f = [&](int i, int mask, bool is_limit, bool is_num) -> int {
            // is_limit 与 is_num 的变化在调用过程中都是单向的 不需要再dp里存储状态
            if(i == len)
                return is_num; // 为true时得到了一个合法数字 (一条搜索路径到结尾)
            if(!is_limit && is_num && dp[i][mask] != -1) // 已经算过了
                return dp[i][mask];
            
            int res = 0;
            if(!is_num) // 还能继续跳过
                res = f(i+1, mask, false, false); // 先统计继续跳过的res
            int up = is_limit ? s[i] - '0' : 9;
            for(int d = 1 - is_num; d <= up; d++) // 前面已有数字时 可以考虑填0
            {
                if((mask>>d & 1) == 0)
                    res += f(i+1, mask | (1<<d), is_limit && (d == up), true);
            }
            if(!is_limit && is_num)
                dp[i][mask] = res;
            return res;
        };

        return n - f(0, 0, true, false);
    }
};
```
</details> 
<br>

---
### &emsp; 2719. 统计整数数目 :rage: HARD
关键思路：  
- 构造答案的过程：如何遍历可能的数字？如何建模限制状态转移的条件？
- 状态：<b>当前遍历到第 i 位，数位和为 sum</b> 
- `max_sum` 和 `min_sum` 转化为递归中的剪枝条件
- 可以将问题转化为 `“ <= num2 的个数” - “ <= num1 的个数”`，再特判一下num1的数位和是否满足（两次记忆化搜索）
- 也可以同时传入两个参数`limit_low`和`limit_high`，只进行一次记忆化搜索
- 注意`limit`只是限制状态转移的条件（而不是问题的状态） 在数位的枚举过程中只会由`true`到`false`一次

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    const int MOD = 1e9+7;
    int count(string num1, string num2, int min_sum, int max_sum) {
        int n = num2.length();
        num1 = string(n - num1.length(), '0') + num1; // 前导补0 和num2对齐

        vector<vector<int>> memo(n, vector<int>(min(9*n, max_sum)+1, -1));
        function<int(int, int, bool, bool)> dfs = [&](int i, int sum, bool limit_low, bool limit_high) -> int {
            // limit 限制状态转移 在数位枚举过程中只会由true到false一次
            // max_sum 和 min_sum 转化为递归中的剪枝条件
            if(sum > max_sum)
                return 0;
            if(i == n)
                return sum >= min_sum;
            if(!limit_low && !limit_high && memo[i][sum] != -1)
                return memo[i][sum];

            int low = limit_low? num1[i] - '0' : 0;
            int high = limit_high? num2[i] - '0' : 9;

            int res = 0;
            for(int d = low; d <= high; d++) // 枚举这一位填什么
            {
                res = (res + dfs(i + 1, sum + d, limit_low & (d == low), limit_high & (d == high))) % MOD;
            }
            if(!limit_low && !limit_high)
                memo[i][sum] = res;
            return res;
        };
        return dfs(0, 0, true, true);
    }
};
```
</details> 
<br>

------
## 计数DP
### 概念
---
* ### 计数DP
    统计选择项（用于“构造”的素材）的数目，枚举选或不选   
    <b>DP数组描述构造方式</b>  

<br>

---

### 题目
---
### &emsp; 分割一个集合，使得两个子集的和尽可能接近
- 本质为 <b>枚举一个集合可以构造出的子集的和 找到最接近`sum/2`的结果</b>
- 开一个dp数组 `dp[sum/2 + 1]` 表示：子集可以达到的和，初始只有`dp[0] = true`
- 遍历集合成员 遍历所有状态并更新状态（注意逆序，不然会重复更新）：
```c++
for(int j = sum; j >= nums[i]; j--)
    dp[j] |= dp[j - nums[i]];
```

<br>

---
### &emsp; 1079. 活字印刷 MID
关键思路：  
- 统计各字符（材料）的个数
- `dp[i][j]`表示用前`i`种字符构造长度为`j`的序列的方案数
- 遍历各类字符，并枚举每个字符选`k`个去构造长度为`j`的序列（方案数相当于乘上组合数 `C(j, k)`）
- 对于组合数的预处理计算，本质是考虑`j`个数中选`k`个，可以考虑其状态转移为当前数选or不选

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
const int MX = 8;
int c[MX][MX];
int init = []() { // 预处理 计算组合数c
    for(int i = 0; i < MX; i++)
    {
        c[i][0] = c[i][i] = 1;
        for(int j = 1; j < i; j++)
            c[i][j] = c[i-1][j-1] + c[i-1][j]; // 第i个数选或不选 
    }
    return 0;
}();

class Solution {
public:
    int numTilePossibilities(string tiles) {
        unordered_map<char, int> counts;
        for(char c : tiles)
            counts[c]++;
        int n = tiles.size(); // 序列最大长度
        int m = counts.size();// 字母数
        int dp[m+1][n+1];
        memset(dp, 0 ,sizeof(dp));
        dp[0][0] = 1; // 构造空序列的方案数
        int i = 1;
        for(auto &[_, cnt] : counts) // 枚举第i种字母
        {
            for(int j = 0; j <= n; j++) // 枚举序列长度
            {
                for(int k = 0; k <= j && k <= cnt; k++) // 枚举这个字母选多少个
                    dp[i][j] += dp[i-1][j-k] * c[j][k];
            }
            i++;
        }
        return accumulate(dp[m] + 1, dp[m] + n + 1, 0);
    }
};
```
</details> 

<br>
