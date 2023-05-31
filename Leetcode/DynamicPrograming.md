## Content
- 线性DP
- 树形DP
- 序列DP
- 区间DP
- 状压DP
- 数位DP
- 计数DP

有重叠子问题 -> 用 DP 优化

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

------
## 树形DP
### 概念
---
* ### 树形DP
    基于左右子树描述状态 在树上状态转移  
    树形DP的出发点：思考 **如何通过递归去计算，如何由子问题算出原问题** （选或不选、枚举选哪个 等）

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

------
## 序列DP
### 概念
---
* ### 序列DP
    线性 DP 通常强调「状态转移所依赖的前驱状态」是由给定数组所提供的，即拓扑序是由原数组直接给出；而序列 DP 通常需要 <b>结合题意来寻找前驱状态</b>，即需要 <b>自身寻找拓扑序关系</b>
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
### &emsp; 1043. 分隔数组以得到最大和
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
### &emsp; 1105. 填充书架
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
    基于位运算描述状态 <b>按位遍历数字</b>  
    以每一 “位” 为单位（转化为字符串）  
    “数位” 与 “状态” 将带来限制


<br>

---

### 题目
---
### &emsp; 1012. 至少有一位重复的数字 :rage: HARD
关键思路：  
- 转换为求无重复数字的个数
- 存储状态 <b>当前遍历到第几位数字 * 使用过的数字mask</b>
- 终结状态的返回值也标示着一条合法搜索路径的结束

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
