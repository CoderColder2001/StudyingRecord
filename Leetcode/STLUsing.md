## Leetcode中利用STL中数据结构求解的题目

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

---
### &emsp; 1642. 可以到达的最远建筑 MID
关键思路：
贪心，尽可能在高度差大的地方使用梯子 &emsp; <b>使用优先队列维护高度差</b>  
问题在于在无法抵达末尾时，`如何判断各高度差的 “先后次序”`  
解决方案：使用<b>size有限的优先队列（小顶堆）</b>即可，当大小超出size时，取出堆顶，将这个高度差改为使用砖块  

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

---
### &emsp; 1705 吃苹果的最大数目 MID
关键思路：  
<b>使用优先队列，贪心，优先吃过期日期早的苹果</b>  
不再产生新苹果后，看还能再吃多少天（这时候不用考虑“不吃苹果”的选项了）
需要注意的是，优先队列中存放 `pair<ddl，num>` 以减少堆操作数目，节约时间  

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