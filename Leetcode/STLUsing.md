## Leetcode中利用STL中数据结构求解的题目
- 堆、栈、队列
- 哈希表、哈希集合
- map、set（平衡二叉树）？

------
## 堆、栈、队列

### 题目
---
### &emsp; 295. 数据流的中位数 :rage: HARD
关键思路：
- 设计可以快速获得中位数的数据流
- 利用两个堆维护左右两边的数据 left为大顶堆 right为小顶堆
- 定义并维护left 与 right的 size关系 &emsp; `left.size() = right.size() or right.size()+1`

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class MedianFinder {
public:
    priority_queue<int, vector<int>, less<int>> left; // 大顶堆
    priority_queue<int, vector<int>, greater<int>> right; // 小顶堆

    MedianFinder() {

    }
    
    void addNum(int num) {
        if(left.size() == right.size())
        {
            if(!right.empty() && num > right.top())
            {
                int t = right.top();
                right.pop();
                left.push(t);
                right.push(num);
            }
            else
                left.push(num);
        }
        else
        {
            if(!left.empty() && num < left.top())
            {
                int t = left.top();
                left.pop();
                right.push(t);
                left.push(num);
            }
            else
                right.push(num);
        }
    }
    
    double findMedian() {
        if(left.size() != right.size())
            return (double)left.top();
        return ((double)left.top() + right.top())/2;
    }
};
```
</details>
<br>

---
### &emsp; 313. 超级丑数 MID
关键思路：
- 丑数列`ans` 是由各prime序列合并后的子集 &emsp; 使用「已有丑数」乘上「给定质因数」primes[i] 得到该prime序列
- 虚拟的 “多路归并” 设计数据结构 `( 当前值(ans[index]*prime), 来自于哪个序列, 对应ans数组的index )`
- <b>使用优先队列</b>
- 优先队列中 维护各列中最小的元素
- 判断是否能填入丑数列
- 注意这一构造过程中体现的 “单调性”

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    // 基于优先队列的多路归并
    typedef struct Info{
        long val;
        int prime_from;
        int index_ans;
        Info(long a, int b, int c)
        {
            val = a; prime_from = b; index_ans =c;
        }
    }info;
    struct cmp {
        bool operator()(const info &a, const info &b) {
            return a.val > b.val;
        }
    };
    int nthSuperUglyNumber(int n, vector<int>& primes) {
        vector<int> ans;
        priority_queue<info, vector<info>, cmp> p_q;
        ans.push_back(1);
        for(int i = 0; i < primes.size(); i++)
        {
            // 由ans[0] 构造各prime路对应的下一元素
            // 各prime路的第i各元素 = ans[i]*prime
            p_q.emplace(primes[i], i, 0);
        }

        for(int i = 1; i < n; )
        {
            if(p_q.empty()) cout<<"error 1"<<endl;
            info t = p_q.top();
            p_q.pop();
            if(t.val > ans[i - 1])
            {
                i++;
                ans.push_back(t.val); // t.val == ans[i]
            }
            p_q.emplace((long)ans[t.index_ans + 1]*primes[t.prime_from], t.prime_from, t.index_ans + 1);
            // 利用已构造出的ans[] 继续构造此路后续的部分
        }
        return ans[n - 1];
    }
};
```
</details>
<br>

---
### &emsp; 373. 查找和最小的K对数字 MID
关键思路：
- 多路（`nums1.size()`）归并问题 取长度最小的为nums1 减小堆中数据量
- <b>使用优先队列</b>
- 堆中存放两个数组对应下标

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    vector<vector<int>> kSmallestPairs(vector<int>& nums1, vector<int>& nums2, int k) {
        vector<vector<int>> ans;
        int n = nums1.size();
        int m = nums2.size();
        bool flag =true;
        if(n > m) // 取长度较小的为nums1
        {
            swap(nums1, nums2);
            swap(n, m);
            flag = false;
        }

        auto cmp = [&](const auto& a, const auto& b){
            return nums1[a.first] + nums2[a.second] > nums1[b.first] + nums2[b.second];
        };
        priority_queue<pair<int, int>, vector<pair<int, int>>, decltype(cmp)> p_q(cmp);

        for(int i = 0; i < min(n,k); i++)
            p_q.emplace(i, 0);
        while(ans.size() < k && p_q.size())
        {
            auto [a, b] = p_q.top();
            p_q.pop();
            flag ? ans.push_back({nums1[a], nums2[b]}) : ans.push_back({nums2[b], nums1[a]});
            if(b + 1 < m)
                p_q.emplace(a, b + 1);
        }
        return ans;
    }
};
```
</details>
<br>

---
### &emsp; 659. 分割数组为连续子序列 MID
关键思路：
- 贪心
- <b>使用哈希表与优先队列 &emsp; 映射结尾数字 x 到可能的子序列上</b>
- 优先队列中 维护这些子序列的长度
- 基于优先队列 将 x 添加到较短的子序列

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    bool isPossible(vector<int>& nums) {
        unordered_map<int, priority_queue<int, vector<int>, greater<int>>> mp; 
        // 结尾数字 映射到 以其结尾的各串长度 
        // 小顶堆实现先填充短的序列
        for(const auto& x: nums)
        {
            if(mp.find(x) == mp.end()) // 先创建x对应的表项
            {
                mp[x] = priority_queue<int, vector<int>, greater<int>>();
            }

            if(mp.find(x-1) != mp.end()) // x可以加入某个序列
            {
                int len = mp[x-1].top() + 1;
                mp[x-1].pop();
                if(mp[x-1].empty()) // 如果没有序列以x-1结尾了 删除该哈希表项
                    mp.erase(x-1);
                mp[x].push(len);
            }
            else
                mp[x].push(1); // 开一个新序列
        }
        for(const auto& item: mp) // 遍历哈希表 即遍历所有子序列
        {
            if(item.second.top() < 3)
                return false;
        }
        return true;
    }
};
```
</details>
<br>

---
### &emsp; 767. 重构字符串 MID
关键思路：
- 贪心
- <b>优先队列 &emsp; 维护各字符优先级</b>
- 当有一个字符数量超过`(length + 1)/2`时 重排肯定会相邻

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    string reorganizeString(string s) {
        if(s.length() < 2)
            return s;
        vector<int> cnt(26, 0);
        int maxCount = 0;
        for(char c : s)
        {
            cnt[c - 'a']++;
            maxCount = max(maxCount, cnt[c - 'a']);
        }
        if(maxCount > (s.length() + 1) / 2)
            return "";
        
        auto cmp = [&](const char &c1, const char &c2){
            return cnt[c1 - 'a'] < cnt[c2 - 'a'];
        };
        priority_queue<char, vector<char>, decltype(cmp)> p_q{cmp};
        string res;
        for(char c = 'a'; c <= 'z'; c++)
        {
            if(cnt[c - 'a'] > 0)
                p_q.emplace(c);
        }
        while(p_q.size() > 1)
        {
            char c1 = p_q.top();
            p_q.pop();
            char c2 = p_q.top();
            p_q.pop();
            res += c1;
            res += c2;
            if(--cnt[c1 -'a'])
                p_q.emplace(c1);
            if(--cnt[c2 -'a'])
                p_q.emplace(c2);
        }
        if(!p_q.empty())
            res += p_q.top();
        return res;
    }
};
```
</details>
<br>

---
### &emsp; 857.雇佣K名工人的最低成本 :rage: HARD
关键思路：
- 对各个工人的 `wage[i]/quality[i]` 排序
- 从[0,k-1]开始 向右 <b>枚举</b> 各个可选的子区间 判断是否能更新 `ans= q_sum * wage[i]/quality[i]`
- <b>优先队列</b> &emsp; quality的大顶堆 计算当前可选的子区间最小的k个quality和

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    double mincostToHireWorkers(vector<int>& quality, vector<int>& wage, int k) {
        int nums = quality.size();
        vector<int> id(nums); // 记录下标的辅助数组 用于排序
        iota(id.begin(), id.end(), 0); // 范围赋值 从0开始递增
        sort(id.begin(), id.end(), [&](int i, int j){
            return wage[i] * quality[j] < quality[i] * wage[j];}); // 按 r=w/q 从小到大排序 注意int用乘法比较
        //总开销 q_sum*R 选定r[k-1]作为R

        priority_queue<int, vector<int>, less<int>> p_q; // 大顶堆
        int q_sum = 0; // 利用优先队列寻找最小化 q_sum
        for(int i = 0; i < k; i++)
        {
            p_q.emplace(quality[id[i]]);
            q_sum += quality[id[i]];
        }
        double ans = q_sum * (double)wage[id[k-1]] / quality[id[k-1]];
        // 排序保证了r的单调增，向右枚举不同的r，看是否有更小q_sum的k子区间，再判断能否更新ans
        for(int i = k; i < nums; i++)
        {
            int q = quality[id[i]];
            if(q < p_q.top())
            {
                q_sum -= (p_q.top() - q);
                p_q.pop();
                p_q.push(q);
                ans = min(ans, q_sum * (double)wage[id[i]] / quality[id[i]]); // 看是否能更新ans
            }
        }
        return ans;
    }
};
```
</details>
<br>

---
### &emsp; 1106. 解析布尔表达式 :rage: HARD

关键思路：
- 使用<b>两个栈ops、value分别存放操作符和值</b>
- 对于 ' ( '，在value栈中放一个占位符，标志出栈结束
- 运算函数 `bool val = cal(a, b, op)` 对初始情况的判定 

<details> 
<summary> <b>C++ Code</b> </summary>

``` c++
class Solution {
public:
    bool cal(bool a, char b, char op)
    {
        if(op == '!')
            return !a;
        if(op == '&')
        {
            if(b == ' ')
                return a;
            return a&b;
        }
        if(op == '|')
        {
            if(b == ' ')
                return a;
            return a|b;
        }
        return false;
    }

    bool parseBoolExpr(string expression) {
        stack<char> ops; // 栈 存放操作
        stack<char> values; // 栈 存放值
        bool res = true;

        for(const auto &c: expression)
        {
            switch(c)
            {
                case ',':
                    continue;
                case 't':
                    values.push(true);
                    break;
                case 'f':
                    values.push(false);
                    break;
                case '!':
                    ops.push(c);
                    break;
                case '&':
                    ops.push(c);
                    break;
                case '|':
                    ops.push(c);
                    break;
                case '(':
                    values.push('0'); // 占位 出栈标记括号结束
                    break;
                case ')':
                {
                    char temp = values.top();
                    char _val = ' ';
                    char op = ops.top();
                    ops.pop();
                    while(temp != '0')
                    {
                        values.pop();
                        _val = cal(temp, _val, op);
                        temp = values.top();
                    }
                    values.pop(); // pop '0'
                    values.push(_val);
                    break;
                }
                default:
                    return false;
            }
        }
        res = values.top();
        return res;
    }
};
```
</details>
<br>

---
### &emsp; 1642. 可以到达的最远建筑 MID
关键思路：
- 贪心，尽可能在高度差大的地方使用梯子 &emsp; <b>使用优先队列维护高度差</b>  
- 问题在于在无法抵达末尾时，`如何判断各高度差的 “先后次序”`  
- 解决方案：使用<b>size有限的优先队列（小顶堆）</b>即可，当大小超出size时，取出堆顶，将这个高度差改为使用砖块  

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    int furthestBuilding(vector<int>& heights, int bricks, int ladders) {
        priority_queue<int, vector<int>, greater<int>> up_heights; // 小顶堆
        int bricks_used = 0; // 当前使用的砖块数
        for(int i = 0; i < heights.size()-1; i++)
        {
            int delta_height = heights[i+1] - heights[i];
            if(delta_height > 0)
                up_heights.push(delta_height);
            if(up_heights.size() > ladders)
            {
                int use_bricks = up_heights.top(); // 取出最小值
                up_heights.pop();
                bricks_used += use_bricks;
                if(bricks_used > bricks)
                    return i;
            }
        }
        return heights.size()-1;
    }
};
```
</details>
<br>

---
### &emsp; 1705 吃苹果的最大数目 MID
关键思路：  
- <b>使用优先队列，贪心，优先吃过期日期早的苹果</b>  
- 不再产生新苹果后，看还能再吃多少天（这时候不用考虑“不吃苹果”的选项了）
- 需要注意的是，优先队列中存放 `pair<ddl，num>` 以减少堆操作数目，节约时间  

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    int eatenApples(vector<int>& apples, vector<int>& days) {
        int day = 0;
        int day_napple = 0;
        priority_queue<int, vector<int>, greater<int>> deadline; // 小顶堆 存放到期时间 
        for(day = 0; day < apples.size(); day++)
        {
            for(int i = 0; i < apples[day]; i++)
                deadline.push(day + days[day] - 1);
            while(true)
            {
                if(deadline.empty())
                {
                    day_napple++;
                    break;
                }
                int ddl = deadline.top();
                deadline.pop();
                if(ddl < day)
                    continue;
                else
                    break;
            }
        }
        while(!deadline.empty())
        {
            int ddl = deadline.top();
            deadline.pop();
            if(ddl < day)
                continue;
            else
                day++;
        }
        return day-day_napple;
    }
};
```
</details>
<br>

---
### &emsp; 2751 机器人碰撞 :rage: HARD
关键思路：  
- <b>使用栈</b>维护向右运动（等待被碰撞的机器人）  
- 遇到向左的机器人，考虑与栈中机器人依次碰撞
- 返回最终 `health[i]` 不为 0 的机器人

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    vector<int> survivedRobotsHealths(vector<int>& positions, vector<int>& healths, string directions) {
        int n = positions.size(), id[n];
        iota(id, id + n, 0);
        sort(id, id + n, [&](const int i, const int j) {
            return positions[i] < positions[j];
        }); // id根据position排序

        stack<int> st;
        for(int i : id)
        {
            if(directions[i] == 'R') 
            { // 向右，存入栈中
                st.push(i);
                continue;
            }

            // 向左，与栈中向右的机器人碰撞
            while(!st.empty())
            {
                int top = st.top();
                if(healths[top] > healths[i])
                {
                    healths[top]--;
                    healths[i] = 0;
                    break;
                }
                else if(healths[top] == healths[i])
                {
                    healths[top] = healths[i] = 0;
                    st.pop();
                    break;
                }
                else
                {
                    healths[top] = 0;
                    healths[i]--;
                    st.pop();
                }
            }
        }
        healths.erase(remove(healths.begin(), healths.end(), 0), healths.end()); // 删除0
        return healths;
    }
};
```
</details>
<br>

------
## 哈希表、哈希集合

### 题目
---
### &emsp; 1016. 子串能表示从1到N数字的二进制串 MID
关键思路1： O（m logn）
- 遍历所有子串 `j`右移扩展子串`[i, j]`： `x = (x << 1) | (s[j] - '0');`
- 使用哈希集合记录子串对应的数字
- 最终集合大小为n时，满足 “ [1,n] 内所有整数的二进制串都是 S 的子串 ”

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    bool queryString(string s, int n) {
        unordered_set<int> seen;
        int m = s.length();
        for(int i = 0; i < m; i++)
        {
            int x = s[i] - '0';
            if(x == 0)  // 从1开始
                continue;
            for(int j = i + 1; x <= n; j++) // 记录以s[i]开头的子串对应数字
            {
                seen.insert(x);
                if(j == m) // 要先记录上一次的
                    break;
                x = (x << 1) | (s[j] - '0'); // 子串 [i, j]
            }
        }
        return seen.size() == n;
    }
};
```
</details>

关键思路2： O（m）
- S 长度需满足的条件，不满足直接 return false：
  - 设 n 的二进制长度为 `k+1`；则区间 `[2^k, n] `中数字长度均为 `k+1`，共 `n - 2^k + 1`个数字；S长度应满足 `m > k + 1 + (n - 2^k + 1) = n - 2^k + k + 1`
  - 区间 `[2^(k-1), 2^k - 1]` 中数字长度均为 `k`，共 `2^(k-1)`个数字；S长度还应满足 `m > k + 2^(k-1) - 1`
- 注意到，对于区间`[2^(k-1), 2^k - 1]`中的数字，右移一位（子串的子串）可以得到更小值的区间`[2^(k-2), 2^(k-1) - 1]` 中的所有数字
- 故只需判断那两个区间中的数字 在S中是否都存在子串相对应 即可。
- <b>使用长度k 和 k+1的滑动窗口</b>

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    bool queryString(string s, int n) {
        if (n == 1)
            return s.find('1') != string::npos;

        int m = s.length();
        // __builtin_clz: 返回前导0个数
        int k = 31 - __builtin_clz(n); // n 的二进制长度减一
        if (m < max(n - (1 << k) + k + 1, (1 << (k - 1)) + k - 1))
            return false;

        // 对于长为 k 的在 [lower, upper] 内的二进制数，判断这些数 s 是否都有
        auto check = [&](int k, int lower, int upper) -> bool {
            if (lower > upper)
                return true;

            unordered_set<int> seen;
            int mask = (1 << (k - 1)) - 1; 
            int x = stoi(s.substr(0, k - 1), nullptr, 2);
            for (int i = k - 1; i < m; i++) 
            {
                // & mask 可以去掉长度k串的最高比特位，从而实现滑窗的「出」
                // << 1 | (s[i] - '0') 即为滑窗的「入」
                x = ((x & mask) << 1) | (s[i] - '0');
                if (lower <= x && x <= upper)
                    seen.insert(x);
            }
            return seen.size() == upper - lower + 1;
        };

        return check(k, n / 2 + 1, (1 << k) - 1) && check(k + 1, 1 << k, n);
    }
};
```
</details>
<br>

---
### &emsp; 1072. 按列翻转得到最大值等行数 MID
关键思路： 
- 矩阵中元素相同or互补的两行可以通过翻转各自变成相等行
- <b>这样的两行通过边相连接 答案就是最大连通块的大小</b>
- 无需建图。<b>用哈希表计数统计这些 “连通行”</b> 当以1开头时 翻转为以0开头后记入哈希表

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    int maxEqualRowsAfterFlips(vector<vector<int>>& matrix) {
        int ans = 0, n = matrix[0].size();
        unordered_map<vector<bool>, int> cnt;
        for(const auto &row : matrix)
        {
            vector<bool> t(row.begin(), row.end());
            if(t[0])
            {
                t.flip();
            }
            ans = max(ans, ++cnt[std::move(t)]);
            //std::move 将一个左值强制转化为右值引用
        }
        return ans;
    }
};
```
</details>
<br>

---
### &emsp; 2475. 数组中不等三元组的数目 EASY
关键思路： 
- 注意<b>数组元素的相对顺序不影响结果</b> 即在枚举过程中无需想哪个是i，哪个是j，哪个是k，只需考虑枚举出一个不同数字组成的三元组（n1，n2，n3）
- 用哈希表记录各数字的个数（个数 即意味着有多少个可选位序）
- 遍历哈希表（保证元素间不同） 枚举三元组的中间数字及其个数
- 这里遍历哈希表无需考虑大小顺序，只需枚举不同数字
- 迭代更新三元组可选左边数字 计算得到右边数字的个数

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    int unequalTriplets(vector<int>& nums) {
        unordered_map<int, int> cnt;
        for(int v:nums)
        {
            cnt[v]++;
        }
        int ans = 0, leftNum = 0, n = nums.size();
        for(auto& [_, midNum] : cnt) // 枚举中间数字及其个数 遍历哈希表保证元素间不同
        {
            int rightNum = n - leftNum - midNum;
            ans += leftNum * midNum * rightNum;
            leftNum += midNum; // 去当左侧元素
        }
        return ans;
    }
};
```
</details>
<br>