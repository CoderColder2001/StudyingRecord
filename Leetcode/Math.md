## Content

------
### 题目

---
### &emsp; 1015. 可被K整除的最小整数 MID
关键思路：  
- （a + b）mod k == （（a mod k）+（b mod k））mod k
- （a * b）mod k == （（a mod k）*（b mod k））mod k
- 从小到大枚举，第一个 mod k == 0 的即为答案
- 根据mod运算规则，下一轮计算结果为 `x =（10 * x + 1）mod k`
- 用哈希表判断环，若出现一个已经出现过的结果，则存在环
- 也可以通过循环轮次判断，根据<b>抽屉原理</b>，最多存在k轮计算；这也说明了算法时间复杂度为O（k）

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    int smallestRepunitDivByK(int k) {
        unordered_set<int> seen;
        int x = 1 % k;
        while(x && !seen.count(x))
        {
            seen.insert(x);
            x = (x*10 + 1) % k;
        }
        return x ? -1 : seen.size() + 1;    
    }
};
```
</details> 

<br>