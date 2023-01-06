### 概念
* 滑动窗口
* 前缀和
* 差分
* 单调队列 / 单调栈

---
* ### 滑动窗口： 
O(n) 处理序列的连续子序列  
滑动窗口的关键是问题对于序列具有<b>单调性</b>(类似于二分性质)  
一般可以理解为，随着左端点位置的增加，其最优决策的右端点位置单调不减   
<b>当一个指针位置固定时，能够确定另一个指针不需要再继续移动的位置</b>   
（即能够知道 “什么时候移动右指针 什么时候移动左指针” ）   


## 题目
--- 
### &emsp; 395. 至少有K个重复字符的最长子串 MID
关键思路：
- 假设有长度 t 的区间满足，无法确定长度 t+1 的区间是否满足，不能直接滑动窗口
- <b>枚举</b> 窗口的字符类型数量
- 当确定了窗口包含的字符种类数量时，<b>区间重新具有了二分性质</b>
- <b>通过枚举 对问题条件加了一层约束</b>
- 此时，右指针向右移动必然会导致字符类型数量增加（或不变），左指针往右移动必然会导致字符类型数量减少（或不变）  
- 记录有多少字符符合要求（出现次数不少于 k），当区间内所有字符都符合时更新答案

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    int longestSubstring(string s, int k) {
        int ans = 0;
        int n = s.length();
        vector<int> cnt(26, 0);
        for(int limit = 1; limit <= 26; limit++) // 枚举窗口内的字符数量
        {
            cnt.assign(26, 0);
            for(int right = 0, left = 0, cnt_1 = 0, cnt_k = 0; right < n; right++)
            {
                int r = s[right] - 'a';
                cnt[r]++;
                if(cnt[r] == 1)
                    cnt_1++; // 窗口内字符数量
                if(cnt[r] == k)
                    cnt_k++; // 窗口内满足条件的字符数量
                while(cnt_1 > limit) // 移动左指针的情形
                {
                    int l = s[left] - 'a';
                    cnt[l]--;
                    if(cnt[l] == 0)
                        cnt_1--;
                    if(cnt[l] == k-1)
                        cnt_k--;
                    left++;
                }
                if(cnt_1 == cnt_k)
                    ans = max(ans, right - left + 1);
            }
        }
        return ans;
    }
};
```
</details>

---
### &emsp; 424. 替换后的最长重复字符 MID
关键思路：
- 题目限定替换次数`k` 直观上指区间内有多少个字母与区间内出现次数最高的字母不同
- 转化为 `区间长度len - 区间内出现次数最高字母出现次数maxCnt > k`
- 计算某字母出现在某窗口中的最大次数`maxCnt`
- 题目要 <b>寻找最长窗口，仅考虑窗口长度增大或者不变的转移情况</b>
- 只有 <b>maxCnt增加</b> 的情况，窗口才会变长，`len`才可能取到最大值
- 不满足条件的情况下，`left`和`right`一起移动，`len`不变（`left`只移动了0-1次）
- 因为长度小于`right - left`的区间就没必要考虑了，所以`right - left`一直保持为当前的最大值（right会多走一步 不用 +1 了）

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    int characterReplacement(string s, int k) {
        // 不同的字符 转化为考虑区间内出现最多的字符
        vector<int> cnt(26, 0);
        int maxCnt = 0;// 窗口中出现最多的字符出现次数
        int left = 0, right = 0;
        for(; right < s.length(); right++)
        {
            cnt[s[right] - 'A']++;
            maxCnt = max(maxCnt, cnt[s[right] - 'A']);
            while(right - left + 1 - maxCnt > k)
            {
                cnt[s[left] - 'A']--;
                left++;
            }
        }
        return right - left;// 因为最后right会多走一步 不用 +1 了
    }
};
```
</details>

---
### &emsp; 1052. 爱生气的书店老板 MID
关键思路：
- 首先预处理customers数组 <b>统计原本就满意的顾客数并置相应 `customers[i] = 0`</b>
- 转化为滑动窗口求解customers中长度为minutes的最大子数组

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    int maxSatisfied(vector<int>& customers, vector<int>& grumpy, int minutes) {
        int n = customers.size();
        int ans = 0;
        for(int i = 0; i < n; i++)
        {
            if(grumpy[i] == 0)
            {
                ans += customers[i];
                customers[i] = 0;
            }
        }
        int maxCnt = 0, currentCnt = 0;
        for(int i = 0; i < n; i++)
        {
            currentCnt += customers[i];
            if(i >= minutes)
            {
                currentCnt -= customers[i - minutes];
            }
            maxCnt = max(maxCnt, currentCnt);
        }
        return ans + maxCnt;
    }
};
```
</details>

<br>

---
* ### 前缀和： 
可以快速求子数组的和（转换为两个前缀和的差）  
为方便计算，常用左闭右开区间 `[left,right)` 来表示子数组（`s[0] = 0`），此时子数组 `[left,right]` 的和为 `s[right+1]−s[left]`  


## 题目
--- 
### &emsp; 1703. 得到连续K个1的最少相邻交换次数 :rage: HARD
关键思路：
- 分析区间性质
- 枚举包含k个1且左右端点为1的 <b>滑动窗口</b>
- 每个0往左边/右边移动 交换次数cost即为：左边/右边1的个数
- 预处理出两个数组 `zeros`：两个1间0的个数 & `cost`：这些0对应的cost值
- 在nums上窗口长度是变化的 但<b>转为在zeros上后，窗口长度是固定的</b> 此时窗口长度为`k-1（[0, k-2]）`
- 对于每个窗口的cost 两端是1 中间依次递增
- 利用滑动窗口的特性 <b>下一窗口的结果值可以由上一窗口快速得到</b>
- 可以找到一个中点mid，它的cost是不变的，它左边的cost都减少了1，而右边的cost都增加了1 分k为奇偶讨论mid
- 利用前缀和求zeros的区间和
  
<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:    
    vector<int> zeros;
    vector<int> presum{0}; //前缀和

    void generateZeros(const vector<int> &nums)
    {
        int n = nums.size();
        int i = 0;
        while(i < n && nums[i] == 0)
            i++;
        while(i < n)
        {
            int j = i + 1; // 统计0个数的指针
            while(j < n && nums[j] == 0)
                j++;
            if(j < n)
                zeros.push_back(j - i - 1);
            i = j;
        }
    }
    void generatePresum(const vector<int> &zeros)
    {
        for(int i = 0; i < zeros.size(); i++)
            presum.push_back(presum.back() + zeros[i]);
    }
    inline int getRangeSum(int left, int right)
    {
        return presum[right + 1] - presum[left];
    }
    
    int minMoves(vector<int>& nums, int k) {
        generateZeros(nums);

        //初始化计算第一个窗口
        int cost = 0;
        int left = 0, right = k - 2;
        for(int i = left; i <= right; i++)
        {
            cost += zeros[i]*(min(i -left + 1, right - i + 1));
        }

        //滑动窗口
        int minCost = cost;
        generatePresum(zeros);
        int l = 1, r = l + k - 2;
        for(; r < zeros.size(); l++, r++)
        {
            int mid = (l + r) / 2;
            cost -= getRangeSum(l - 1, mid - 1);
            cost += getRangeSum(mid + k%2, r);
            minCost = min(minCost, cost);
        }
        return minCost;
    }
};
```
</details>

<br>

---
* ### 差分： 
存放 数组的前缀和数组 前后元素的差值  
对差分数组求前缀和即得到原数组  
```c++   
void add(vector<int> c, int l, int r)
{
    c[l] ++;
    c[r+1] --; // “标记 +1” 的操作到此为止
}
```
## 题目
--- 
### &emsp; 798. 得分最高的最小轮调 :rage: HARD
关键思路： 
- 对长度为n的数组 有n-1种轮调 即对任意元素，有n-1种可取的新下标i-k  (最终映射为`(i-k+n)%n`)
- 先求每个元素`nums[i]`可得分轮调k的上下界，再对<b>对应区间（可取的k值）进行标记操作</b>   
- 为转化取模运算带来的问题：设k可取负数 新下标`i-k`满足 `0<=i-k<=n-1` --> `i-(n-1) <= k <=i`  （ 基于i值对[0,n-1]的区间映射 ）  
- 为满足得分 `nums[i] <= i-k` -->  `k <= i-nums[i]`  
- 故对一个元素nums[i],可以得分的k取值下界为`i-(n-1)`，上界为`i-nums[i]`   
- 设`a = (i-(n-1)+n) % n`， `b = (i-nums[i]+n) % n`  &emsp; ( 此时将k的域映射回了[0,n-1] )  
- 若a <= b ，这是一个连续区间  
- 否则区间分为[0, b] 与 [a, n-1]两段 
- 对<b>属于这些区间的k取值进行标记+1</b>, 最终标记值最大的k值即为答案  
- <b>使用差分数组实现区间标记操作</b>  

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    // 使用差分实现标记操作
    void add(vector<int> &c, int l, int r)
    {
        count[l] ++;
        count[r + 1] --;
    }
    int bestRotation(vector<int>& nums) {
        int n = nums.size();
        vector<int> kC(n + 1, 0);// 差分数组
        for(int i = 0; i < n; i++)
        {
            int a = (i - (n - 1) + n) % n;
            int b = (i - nums[i] + n) % n;
            if(a <= b)
            {
                add(kC, a, b);
            }
            else
            {
                add(kC, 0, b);
                add(kC, a, n - 1);
            }
        }
        // 对差分数组求前缀和 还原区间
        for(int i = 1; i <= n; i++)
        {
            kC[i] += kC[i - 1];
        }
        int ans = 0, count = kC[0];
        for(int i = 1; i <= n; i++)
        {
            if(kC[i] > count)
            {
                count = kC[i];
                ans = i;
            }
        }
        return ans;
    }
};
```
</details>

<br>

---
* ### 单调队列/单调栈 ： 
常用于<b>区间最值问题</b>  
单调队列使用`std::deque` &emsp; 单调栈可使用`std::stack`

## 题目
--- 
### &emsp; 239. 滑动窗口最大值 :rage: HARD
关键思路：
- 使用<b>单调减队列</b> 存储对应下标值

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        deque<int> d; // 单调减队列，存储对应下标值
        vector<int> ans;
        int n = nums.size();
        int i = 0;
        for(i = 0; i < k; i++) // 初始化第一个区间
        {
            while(!d.empty() && nums[d.back()] < nums[i])
                d.pop_back();
            d.push_back(i);
        }
        ans.push_back(nums[d.front()]);
        for(int i = k; i < n; i++)
        {
            while(!d.empty() && d.front() < i - k + 1)
                d.pop_front();
            while(!d.empty() && nums[d.back()] < nums[i])
                d.pop_back();
            d.push_back(i);
            ans.push_back(nums[d.front()]);
        }
        return ans;
    }
};
```
</details>