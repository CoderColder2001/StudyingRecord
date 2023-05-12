## Content

------
### 题目

---
### &emsp; 1015. 可被K整除的最小整数 MID
关键思路：  
- <b>模运算转化</b>
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

---

### &emsp; 1330. 翻转子数组得到的最大数组值 :rage:HARD
关键思路：  
- <b>绝对值运算转化</b>
- 只有翻转部分和未翻转部分的交界会改变
- i 为 0 或 j 为 n-1 时，O（n）枚举
- `|x| = max(x, -x)`
- `max(a,b) + max(c,d) = max{a+c, a+d, b+c, b+d}`
- <img src ="./pic/math_1.png" width = "80%">
- 把含j的提到前，含i的置于后，可以理解为将问题转化为求（四个）f1(i) + f2(j)的最大值：**在枚举j的过程中：维护f1(i)最大值即可**

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    static constexpr int INF = 0x3f3f3f3f;

    int maxValueAfterReverse(vector<int>& nums) {
        const int n = nums.size();
        const int a = nums[0], b = nums[n-1];
        int premax[2][2] = {-INF, -INF, -INF, -INF};
        int sum = 0, ans = 0;
        for(int i = 1; i < n; i++)
        {
            const int x = nums[i-1], y = nums[i], d = abs(x-y);
            sum += d;
            ans = max(ans, max({
                abs(x - b), // j为n-1的特殊情况 枚举子数组左端点
                abs(y - a), // i为0的特殊情况 枚举子数组右端点
                premax[0][0] - x - y,
                premax[0][1] - x + y,
                premax[1][0] + x - y,
                premax[1][1] + x + y
            }) - d);
            // 更新前缀最大值信息
            premax[0][0] = max(premax[0][0], x + y - d);
            premax[0][1] = max(premax[0][1], x - y - d);
            premax[1][0] = max(premax[1][0], -x + y - d);
            premax[1][1] = max(premax[1][1], -x - y - d);
        }
        return sum + ans;
    }
};
```
</details> 
<br>