## 二分法
---
三种形式：
- 寻找一个数
- 寻找左侧边界
- 寻找右侧边界

关键：
- 确定搜索区间 `[left, right] 、 [left, right) 、 (left, right)` &emsp;这决定了循环结束条件以及转移方程  
- 确定循环不变量（区间满足的性质，或者说L R满足的性质）

---
### 寻找一个数

```c++
int binarySearch(int[] nums, int target) {
    int left = 0; 
    int right = nums.length - 1; // 注意

    while(left <= right) {
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

### 寻找左侧边界

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

### 寻找右侧边界

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
