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