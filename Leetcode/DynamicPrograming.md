## Content
- 序列DP


<br>

---
## 序列DP
### 概念
---
* ### 序列DP
    线性 DP 通常强调「状态转移所依赖的前驱状态」是由给定数组所提供的，即拓扑序是由原数组直接给出；而序列 DP 通常需要<b>结合题意来寻找前驱状态</b>，即需要<b>自身寻找拓扑序关系</b>
---

* ### 最长递增子序列问题 LIS
  在一个给定的数值序列中，找到一个子序列，使得这个子序列元素的数值依次递增，并且这个子序列的长度尽可能地大。最长递增子序列中的元素在原序列中不一定是连续的  
  —— 最快 O(nlogn)  

`DP作为求解的辅助工具`  
序列DP + 二分寻找右边界：   
通过一个数组dp[k]来缓存长度为k的递增子序列的最末元素，若有多个长度为k的递增子序列，则记录最小的

首先`len=1`, `dp[0]=seq[0]`  
遍历seq 看是否能更新`dp[]`  
&emsp;&emsp;对`seq[i]`：若`seq[i] > dp[len]`，那么`len++`，`dp[len]=seq[i]`（增长链）  
&emsp;&emsp;否则，从`dp[0]`到`dp[len-1]`中找到一个`j`，满足 `dp[j-1] < seq[i] < d[j]` 然后更新`dp[j]=seq[i]`  
最终len即为最长递增子序列LIS的长度  
因为在dp中插入数据有序且只需替换不用挪动，因此我们可以使用二分查找，将每一个数字的插入时间优化为O(logn)  算法的时间复杂度从使用 排序+LCS 的O(n^2)降低到了O(nlogn)

<br>

---

### 题目
---
### &emsp; 646. 最长数对链 MID
将pairs按第一个数升序排序  
定义`f[i]`：以`pairs[i]`为结尾的最长数对链长度，所有`f[i]`中的最大值为答案  
&emsp;&emsp;由贪心思想确定状态转移方程：从`j=i-1`往回找 第一个`pairs[j][1] < pairs[i][0]`的`f[j]+1`  

&emsp;&emsp;证明贪心的正确性：假设还存在`j' < j`满足`f[j'] > f[j]`，由于`pairs[j][0] > pairs[j'][0]`，`pairs[j]`可以替换`f[j']`对应的路径中最后的`pairs[j']`，故假设不成立  
&emsp;&emsp;亦即，对于一个特定`pairs[i]`而言，其所有合法（满足条件`pairs[j][1] < pairs[i][0]`）的前驱状态 `f[j]` 必然是非单调递增的  

&emsp;&emsp;根据LIS问题的贪心解的思路，可以额外使用一个数组记录下特定长度数链的最小结尾值，从而实现二分找前驱状态   
&emsp;&emsp;具体地，创建`g[ ]`，其中`g[len]=x` 代表数链长度为len时结尾元素的第二维最小值为x

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
            // 寻找满足 g[len] < pairs[i][0]的最大len 
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