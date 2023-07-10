### 概念
* 滑动窗口
* 前缀和
* 差分
* 单调队列 / 单调栈
* 其他

------
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

---
### &emsp; 1658. 将x减小到0的最小操作数 MID
关键思路：
- <b>逆向考虑问题</b>
- 原操作区间为数组的前面一段与后面一段
- 转换为求 数组中和为s-x最长连续子数组

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    int minOperations(vector<int>& nums, int x) {
        int ans = -1, cnt = 0, s = accumulate(nums.begin(), nums.end(), 0);
        int left = 0, right = 0;
        int n = nums.size();

        while(right < n)
        {
            cnt += nums[right];
            while(left <= right && cnt > s-x)
            {
                cnt -= nums[left];
                left++;
            }
            if(cnt == s-x)
                ans = max(ans, right - left + 1);
            right++;
        }
        return ans == -1? ans : n - ans;
    }
};
```
</details>
<br>

---
### &emsp; 2762. 不间断子数组 MID
关键思路：
- 滑动窗口 使用<b>平衡树or哈希表</b> 维护窗口内最大值与最小值
- 由于绝对值之差至多为2，至多维护3个数，添加和删除可以视为O（1）的
- 对每一个`right`记一次数；当前满足条件的`left`对应`[left, right]`的子数组个数

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    long long continuousSubarrays(vector<int>& nums) {
        long long ans = 0;
        multiset<int> s;
        int left = 0, n = nums.size();
        for(int right = left; right < n; right++)
        {
            s.insert(nums[right]);
            while(*s.rbegin() - *s.begin() > 2)
            {
                s.erase(s.find(nums[left++])); // 先find 只去掉一个数
            }
            ans += right - left + 1; // 以right作为右端点的[left, right]的子数组个数
        }
        return ans;
    }
};
```
</details>
<br>

---
### &emsp; 2763. 所有子数组中不平衡数字之和 :rage: HARD
关键思路：
- 遍历左端点，枚举右端点 维护当前不平衡度`cnt`
- 使用一个`vis[]`数组记录当前窗口中的数字
- 若`x=nums[i]` 之前出现过，那么排序后必然与另一个`x`相邻，`cnt`不变
- 否则看是否出现过`x-1`与`x+1`；都没出现过时，`cnt++`;只有一个时，`cnt`不变；都出现过时，`cnt--`
- 优化：可以用`int vis[n+1]`，值为当前遍历的左端点`i`，这样就无需每次都`memset`了

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    int sumImbalanceNumbers(vector<int> &nums) {
        int ans = 0, n = nums.size();
        bool vis[n + 2];
        for(int i = 0; i < n; i++)
        {
            memset(vis, 0, sizeof(vis));
            vis[nums[i]] = true;
            int cnt = 0;
            for(int j = i + 1; j < n; j++)
            {
                int x = nums[j];
                if(!vis[x])
                {
                    cnt += 1 - vis[x - 1] - vis[x + 1];
                    vis[x] = true;
                }
                ans += cnt;
            }
        }
        return ans;
    }
};
```
</details>  
<br>

关键思路2：
- <b>贡献法</b>：对一个元素，若其所在排序子数组的左侧元素与它差值大于1 则贡献1个不平衡数字
- *重复数字如何计算贡献？* 规定当有多个相同元素`x`时 由最右侧`x`提供贡献（假定最右侧的`x`排序后排到最前）
- 对一个`x=nums[i]` 计算左右边界：求右边最近的`x`或`x-1` 左边最近的`x-1`（左边可以有相同的）；否则`left[i] = -1`、 `right[i] = n`
- 最后减去`x`作为子数组最小值的情况 总和即为子数组个数

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    int sumImbalanceNumbers(vector<int>& nums) {
        int n = nums.size();
        int right[n]; // 满足条件的右端点
        int idx[n+1]; // x to id
        fill(idx, idx + n + 1, n);
        for(int i = n-1; i >= 0; i--) // 计算各元素右边界 从右向左遍历
        {
            int x = nums[i];
            right[i] = min(idx[x], idx[x-1]); // 右边最近的x或x-1作为右边界 不存在时为n
            idx[x] = i;
        }

        int ans = 0;
        memset(idx, -1, sizeof(idx)); // 维护左端点
        for(int i = 0; i < n; i++) // 统计能产生多少贡献
        {
            int x = nums[i];
            ans += (i - idx[x - 1]) * (right[i] - i); // 包含i的子数组个数 可选左端点数*可选右端点数
            idx[x] = i;
        }
        return ans - n*(n+1)/2;
    }
};
```
</details>
<br>

------
* ### 前缀和： 
可以快速求子数组的和（转换为两个前缀和的差）  
为方便计算，常用左闭右开区间 `[left,right)` 来表示子数组（`s[0] = 0`），此时子数组 `[left,right]` 的和为 `s[right+1]−s[left]`  
如果要计算的子数组恰好是一个前缀，则`s[right+1] - s[0]` 


## 题目
--- 
### &emsp; 1177. 构建回文串检测 MID
关键思路：
- 可以重新排列成回文意味着什么？ 偶数长度串时，各字母出现次数都是偶数；奇数长度串时，只有一个字母出现次数是奇数 
- 对于可修改次数`k`，如果有`m`个字母出现次数是奇数,修改其中 `⌊m/2⌋` 个字母满足`⌊m/2⌋ <= k` 时，可以构成回文串  
- 如何快速求出子串中每种字母的个数？
- 故使用 <b>前缀和（异或前缀和）&emsp;  统计并维护区间各字母出现次数的奇偶性</b>
- 压缩状态空间 一种字母对应一个bit
- 两个int异或，统计结果中1的个数，即区间出现奇数次的字母个数
  
<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    // 如何快速求出子串中每种字母的个数
    vector<bool> canMakePaliQueries(string s, vector<vector<int>>& queries) {
        int n = s.length(), nq = queries.size();
        int sum[n+1]; // 位运算 压缩存储26个字母对应的状态bit
        sum[0] = 0;
        for(int i = 0; i < n; i++)
        {
            int bit = 1 << (s[i] - 'a'); // 哪个bit
            sum[i+1] = sum[i] ^ bit; // 1、0对应某字母出现次数的奇偶性
        }

        vector<bool> ans(nq);
        for(int i = 0; i < nq; i++)
        {
            auto& query = queries[i];
            int left = query[0], right = query[1], k = query[2];
            int m = __builtin_popcount(sum[right + 1] ^ sum[left]); // 1的个数
            ans[i] = (m / 2 <= k);
        }
        return ans;
    }
};
```
</details>
<br>

--- 
### &emsp; 1590. 使数组和能被P整除 MID
关键思路：
- 设所有元素和modP为`countModP` 问题转化为：寻找最短的一段区间，其区间和与所有元素和modP同余
- “x y modP同余” 等价于 “x % p == y % p” （当x y 均非负数时）
- 若x取任意整数 等价于 （x % p + p）% p == y % p
- 遍历数组，存储 <b>当前遍历位置的前缀和 mod P</b>
- <b>使用哈希表维护能得到这个值的下标</b>（遍历过程使得这个下标是当前最右的）：`{s[i] % P, index}`
- 寻找 `s[r] - s[left] % P == count % P`，即寻找 `(s[r] - countModP) % P == s[left] mod P`
- 遍历时，查找哈希表对应元素是否存在
  
<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    // 前缀和 mod p 用哈希表缓存结果值对应的下标位置
    int minSubarray(vector<int>& nums, int p) {
        int n = nums.size();
        int countModP = accumulate(nums.begin(), nums.end(), 0LL) % p; 
        int ans = n , s = 0; // s维护当前计算到的modP前缀和
        unordered_map<int, int> last;
        last[0] = -1;
        for(int i = 0; i < n; i++)
        {
            s = (s + nums[i]) % p;
            last[s] = i;
            auto it = last.find((s - countModP + p) % p); // +p 保证结果不为负数
            if(it != last.end())
            {
                ans = min(ans, i - it->second);
            }
        }
        return ans == n? -1: ans;
    }
};
```
</details>
<br>

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

------
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

------
* ### 单调队列/单调栈 ： 
常用于<b>区间最值问题</b>   
<b>用于快速定位数组区间中具有某个最值性质的位置</b>   
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

--- 
### &emsp; 1438. 绝对差不超过限制的最长连续子数组 MID
关键思路：
- 两个单调队列
- 使用 <b>单调减队列</b> 队头维护窗口最大值
- 使用 <b>单调增队列</b> 队头维护窗口最小值

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    int longestSubarray(vector<int>& nums, int limit) {
        deque<int> queMax, queMin;
        //单调减队列队头维护窗口最大值 单调增队列队头维护窗口最小值
        int n = nums.size();
        int left = 0, right = 0;
        int ans = 0;
        while(right < n)
        {
            while(!queMax.empty() && nums[queMax.back()] < nums[right])
                queMax.pop_back();
            while(!queMin.empty() && nums[queMin.back()] > nums[right])
                queMin.pop_back();
            queMax.push_back(right);
            queMin.push_back(right);

            while(!queMax.empty() && !queMin.empty() && nums[queMax.front()] - nums[queMin.front()] > limit)
            {
                if(queMax.front() == left)
                    queMax.pop_front();
                if(queMin.front() == left)
                    queMin.pop_front();
                left++;
            }
            ans = max(ans, right - left + 1);
            right++;
        }
        return ans;
    }
};
```
</details>

<br>

------
* ### 其他： 

## 题目
--- 
### &emsp; 16. 最接近的三数之和 MID
关键思路：
- 先排序 枚举第一个数 然后双指针枚举第二个第三个数
- 注意优化：先计算枚举当前第一个数i时的最大和 最小和；并跳过重复数字（与之前枚举过的情况等价）

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    int threeSumClosest(vector<int>& nums, int target) {
        sort(nums.begin(), nums.end());
        int minDiff= 0x3f3f3f3f;
        int ans = 0;
        int n = nums.size();
        for(int i = 0; i < n - 2; i++)
        {
            if(i > 0 && nums[i] == nums[i - 1]) // 之前枚举过
                continue;

            int s = nums[i] + nums[i + 1] + nums[i + 2];
            if(s > target) // 最小的情况比target大
            {
                if(s - target < minDiff)
                {
                    ans = s;
                    break;
                }
            }

            s = nums[i] + nums[n - 2] + nums[n - 1];
            if (s < target) { // 最大的情况比target小
                if (target - s < minDiff)
                {
                    minDiff = target - s;
                    ans = s;
                }
                continue;
            }

            int j = i + 1, k = n - 1;
            while(j < k)
            {
                int sum = nums[i] + nums[j] + nums[k];
                if(abs(sum - target) < minDiff)
                {
                    minDiff = abs(sum - target);
                    ans = sum;
                }

                if(sum == target)
                    return sum;
                else if(sum < target)
                    j++;
                else
                    k--;
            }
        }
        return ans;
    }
};
```
</details>
<br>

--- 
### &emsp; 1163. 按字典序排在最后的子串 :rage: HARD
关键思路：
- 寻找字典序最大的后缀子串
- 使用<b>双指针</b> 从头开始 
- i=0 为当前最大子串的开始，j=1 为当前考虑子串的开始，k=0 记录当前比较位置

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    string lastSubstring(string s) {
        int i = 0, j = 1, k = 0;
        int n = s.size();
        while(j + k < n)
        {
            if(s[i+k] == s[j+k])
                k++;
            else if(s[i+k] < s[j+k])
            {
                i += k+1; //以[i, i+k]为开始的都比不过以相应[j, j+k]为起始的
                k = 0;
                if(i >= j) // i始终在j前
                    j = i + 1;
            }
            else
            {
                j += k+1;
                k = 0;
            }
        }
        return s.substr(i);
    }
};
```
</details>
<br>

--- 
### &emsp; 1498. 满足条件的子序列数目 MID
关键思路：
- 首先对数组排序
- 对于某一个左端点（最小值），存在一个满足条件的最大右端点
- 使用<b>双指针</b> 从首尾开始 
- 统计当前左右端点对应区间中 包含左端点（最小值）的子序列数目
- 左端点必选（1种状态），其他点选or不选（2种状态）
- pow数组预处理计算`2^i`，否则会爆long long

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    const int MOD = 1e9 + 7;
    int numSubseq(vector<int>& nums, int target) {
        sort(nums.begin(), nums.end());
        int n = nums.size();
        int pow[n]; // 预处理幂 否则会爆longlong
        pow[0] = 1;
        for(int i = 1; i < n; i++)
        {
            pow[i] = (pow[i-1]*2) % MOD;
        }
        int left = 0, right = nums.size() - 1;
        int cnt = 0;
        while(left <= right)
        {
            if(nums[left] + nums[right] > target)
            {
                right--;
                continue;
            }
            // 包含left的子数组数目 left必选 其它的选or不选
            cnt = (cnt + pow[right - left]) % MOD; // 2^(n-1)
            left++;
        }
        return cnt;
    }
};
```
</details>
<br>