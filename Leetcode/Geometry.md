# 计算几何
---
## Content
- 求交（判定）问题

<br>

------
## 求交（判定）问题
---
### 题目
---
### &emsp; 1401. 圆与矩形是否有重叠 MID
关键思路：  
- 对于一个点，若其到圆心距离 `sqrt((x-x0)^2 + (y-y0)^2)` 小于 `r`，则这个点在圆内
- 求矩形内使得`(x-x0)`与`(y-y0)`最小的点；问题转化为求 `x∈[x1,x2]`时，`a=∣x−x0∣` 的最小值，以及 `y∈[y1,y2]` 时 `b=∣y−y0∣` 的最小值
- 分情况讨论`x0`与`x1`、`x2`的大小关系，得到最小值；`y`同理

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    bool checkOverlap(int radius, int xCenter, int yCenter, int x1, int y1, int x2, int y2) {
        auto f = [](int i, int j, int k) -> int {
            if(i <= k && k <= j) {
                return 0;
            }
            return k < i ? i - k : k - j;
        };
        int a = f(x1, x2, xCenter);
        int b = f(y1, y2, yCenter);
        return a*a + b*b <= radius*radius;
    }
};
```
</details> 
<br>
