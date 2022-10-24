### 概念
---
* ### 最长递增子序列问题 LIS
  在一个给定的数值序列中，找到一个子序列，使得这个子序列元素的数值依次递增，并且这个子序列的长度尽可能地大。最长递增子序列中的元素在原序列中不一定是连续的  
  最快O(nlogn)  

DP + 二分寻找右边界：   
通过一个数组dp[k]来缓存长度为k的递增子序列的最末元素，若有多个长度为k的递增子序列，则记录最小的

首先len=1, dp[0]=seq[0]  
对seq[i]：若seq[i] > dp[len]，那么len++，dp[len] = seq[i]  
否则，从dp[0]到dp[len-1]中找到一个j，满足 dp[j-1] < seq[i] < d[j] 然后更新dp[j]=seq[i]  
最终len即为最长递增子序列LIS的长度  
因为在dp中插入数据有序且只需替换不用挪动，因此我们可以使用二分查找，将每一个数字的插入时间优化为O(logn)  算法的时间复杂度从使用排序+LCS的O(n^2)降低到了O(nlogn)

## 题目
---
### &emsp; 646. 最长数对链
将pairs按第一个数升序排序  
定义f[i]：以pairs[i]为结尾的最长数对链长度，所有f[i]中的最大值为答案  

由贪心思想确定状态转移方程：从j=i-1往回找 第一个pairs[j][1] < pairs[i][0]的f[j]+1  
证明贪心的正确性：假设还存在j' < j满足f[j'] > f[j]，由于pairs[j][0] > pairs[j'][0]，pairs[j]可以替换f[j']对应的路径中最后的pairs[j']，故假设不成立  
亦即，对于一个特定pairs[i]而言，其所有合法（满足条件pairs[j][1] < pairs[i][0]）的前驱状态 f[j] 必然是非单调递增的  
根据LIS问题的贪心解的思路，可以额外使用一个数组记录下特定长度数链的最小结尾值，从而实现二分找前驱状态   
具体的，创建g[ ]，其中g[len]=x 代表数链长度为len时结尾元素的第二维最小值为x

```c++
class Solution {
public:
    int findLongestChain(vector<vector<int>>& pairs) {
        sort(pairs.begin(), pairs.end());
        int n = pairs.size();
        int ans = 0;
        vector<int> g(n+1, INT_MAX);
        for(int i = 0; i < n; i++)
        {
            int left = 1, right = i + 1; 
            // 对i而言 len可能的取值范围 [1,i+1)
            //寻找满足g[len] < pairs[i][0]的最大len 
            while(left < right)
            {
                int mid = (left + right) >> 1;
                if(g[mid] >= pairs[i][0]) // 这个mid对应的链不满足作为前驱状态的条件
                    right = mid;
                else 
                    left = mid + 1;
            } 
            // right - 1 是找到的有边界
            // 结束时 right == mid(不能增长链) || right == mid + 1(可以增长链)

            g[right] = min(g[right], pairs[i][1]); // left == right 更新g[len] (检查是否替换原先的末尾数对)
            ans = max(ans, right);
        }
        return ans;
    }

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