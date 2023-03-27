## Content
- 线性DP
- 序列DP
- 状压DP
- 数位DP

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

------
## 序列DP
### 概念
---
* ### 序列DP
    线性 DP 通常强调「状态转移所依赖的前驱状态」是由给定数组所提供的，即拓扑序是由原数组直接给出；而序列 DP 通常需要 <b>结合题意来寻找前驱状态</b>，即需要 <b>自身寻找拓扑序关系</b>
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
