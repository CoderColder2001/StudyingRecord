## Leetcode中利用STL中数据结构求解的题目

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
### &emsp; 1106. 解析布尔表达式 HARD

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