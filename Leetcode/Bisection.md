## 二分法
---
<b>最大化最小值</b> 或 <b>最小化最大值</b> => 二分答案  
利用数组的有序性和目标问题的单调性  

三种形式：
- 寻找一个数
- 寻找左侧边界
- 寻找右侧边界

<br>  

一些条件转化：
- 最基本：`>= x`
- `> x` => `>= x + 1`
- `< x` => `>= x` 的左边一个数
- `<= x` => `> x` 的左边一个数

关键：
- 确定搜索区间 `[left, right] 、 [left, right) 、 (left, right)`三种写法 &emsp;这决定了循环结束条件以及转移方程  
- 确定循环不变量（区间满足的性质，或者说L R满足的性质）

---
### 寻找一个数

```c++
int binarySearch(int[] nums, int target) {
    int left = 0; 
    int right = nums.length - 1; // 注意

    while(left <= right) { // [left, right] 闭区间写法
        int mid = left + (right - left) / 2;
        if(nums[mid] == target)
            return mid; 
        else if (nums[mid] < target)
            left = mid + 1; // 注意
        else if (nums[mid] > target)
            right = mid - 1; // 注意
    }
    return -1;
}
```

### 寻找左侧边界（第一个xxx）

循环不变量： <b>`left-1` 一定指向小于它的数</b> （半闭半开写法）  

```c++
int left_bound(int[] nums, int target) {
    if (nums.length == 0) return -1;
    int left = 0;
    int right = nums.length; // 注意
    
    while (left < right) { // [left, right)
        int mid = (left + right) / 2;
        if (nums[mid] == target) {
            right = mid; // 看看有没有可能更左
        } else if (nums[mid] < target) {
            left = mid + 1;
        } else if (nums[mid] > target) {
            right = mid; // 注意
        }
    }
    return left; // left == right
}
```

### 寻找右侧边界（最后一个xxx）

循环不变量： <b>`right` 一定指向大于它的数</b>  （半闭半开写法）  

```c++
int right_bound(int[] nums, int target) {
    if (nums.length == 0) return -1;
    int left = 0, right = nums.length;
    
    while (left < right) { // [left, right)
        int mid = (left + right) / 2;
        if (nums[mid] == target) {
            left = mid + 1; // 看看有没有可能更右
        } else if (nums[mid] < target) {
            left = mid + 1;
        } else if (nums[mid] > target) {
            right = mid;
        }
    }
    return left - 1; // left - 1 == right - 1 
    // "-1" 是状态转移决定的（也可以说是由选取的搜索空间决定的） 
    // 在结束时 *left > target或越界
}
```
## Leetcode中利用二分求解的题目

------
### &emsp; 215. 数组中第K大元素 MID
关键思路：
- 单侧降序快速排序模板题
- 判断partition返回的划分位置与K的关系 更新区间

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    inline void swap(vector<int>& nums, int l, int r)
    {
        int t = nums[l];
        nums[l] = nums[r];
        nums[r] = t;
    }
    int partition(vector<int>& nums, int left, int right)
    {
        int pivot = nums[left]; // 取最左为基准
        int l = left + 1;
        int r = right;
        while(true)
        {
            // 降序排序对应的快速选择
            // 找左边第一个比pivot小的与右边第一个比pivot大的
            while(l <= r && nums[l] > pivot)
                l++;
            while(r >= l && nums[r] < pivot)
                r--;
            if(l >= r)
                break;
            swap(nums, l, r);
            l++;
            r--;
        }
        swap(nums, left, r);
        return r; // 返回划分位置
    }
    int findKthLargest(vector<int>& nums, int k) {
        int left = 0, right = nums.size() - 1;
        while(true)
        {
            int pivot_i = partition(nums, left, right);
            if(pivot_i == k - 1)
                return nums[k - 1];
            else if(pivot_i < k - 1)
                left = pivot_i + 1;
            else if(pivot_i > k - 1)
                right = pivot_i - 1;
        }
        return -1;
    }
};
```
</details>
<br>

---
### &emsp; 2517. 礼盒的最大甜蜜度 MID
关键思路：
- 如果一个甜蜜度为 `x` 的礼盒是可行的，那么甜蜜度小于 `x` 的礼盒也是可行的；<b>问题存在着单调性</b>
- 因此可以使用二分查找答案的方法，找到最大的可行甜蜜度
- 二分寻找满足条件的最大值 即右侧边界

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    int maximumTastiness(vector<int>& price, int k) {
        sort(price.begin(), price.end());
        int n = price.size();

        int l = 0, r = price.back() - price[0];
        auto check = [&price, k](int x) -> bool {
            int cnt = 0;
            int pre = -x; // 上一个选取的糖果价格 初始使 cur - pre >= x 恒成立
            for(int cur : price)
            {
                if(cur - pre >= x)
                {
                    pre = cur;
                    if(++cnt >= k)
                        return true;
                }
            }
            return false;
        };
        // 二分寻找满足条件的最大值 右侧边界
        while(l <= r)
        {
            int mid = (l + r) >> 1;
            if(check(mid)) // [, l-1]满足条件
                l = mid + 1;
            else //[mid, old_r]不满足条件
                r = mid - 1;
        }
        return l - 1;
    }
};
```
</details>
<br>

---
### &emsp; 2560. 打家劫舍IV MID
关键思路：
- 二分 + DP &emsp; 取二分中点为`mx`，check函数通过dp实现
- `f[i]` 表示从下标`[0， i]` 偷金额不超过 `mx` 的房屋，至多能偷几间
- `f[i]`仅仅依赖于`f[i-1]`和`f[i-2]`，可以空间优化

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
    bool check(vector<int>& nums, int k, int mx)
    {
        int f0 = 0, f1 = 0; // 两个状态
        for(int x : nums)
        {
            if(x > mx)
                f0 = f1;
            else
            {
                int tmp = f1;
                f1 = max(f1, f0 + 1); // f[i] = max(f[i-1], f[i-2] + 1)
                f0 = tmp;
            }
        }
        return f1 >= k;
    }
public:
    int minCapability(vector<int>& nums, int k) {
        int left = 0, right = *max_element(nums.begin(), nums.end());
        while(left <= right) // [l, r]
        {
            int mid = left + (right - left) / 2;
            if(check(nums, k, mid))
                right = mid - 1;
            else
                left = mid + 1;
        }
        return right + 1;
    }
};
```
</details>
<br>

