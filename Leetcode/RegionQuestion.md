### 概念
* 差分
* 单调队列 / 单调栈
---
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
### &emsp; 798. 得分最高的最小轮调
先求每个元素 nums[i] 可得分轮调k的上下界，再对对应区间（可取的k值）进行标记操作  
  
对长度为n的数组 有n-1种轮调 即对任意元素，有n-1种可取的新下标i-k  (最终映射为(i-k+n)%n)  
为转化取模的问题 设k可取负数 新下标i-k满足 0<=i-k<=n-1 --> i-(n-1) <= k <=i  (基于i值对[0,n-1]的区间映射)  

为满足得分 nums[i] <= i-k -->  k <= i-nums[i]  
故对一个元素nums[i],可以得分的k取值下界为i-(n-1)，上界为i-nums[i]   
设a = (i - (n-1) + n) % n, b =(i - nums[i] + n) % n  &emsp; ( 此时将k的域映射回了[0,n-1] )  
若a <= b ，这是一个连续区间  
否则区间分为[0, b] 与 [a, n-1]两段 
对属于这些区间的k取值进行标记+1, 最终标记值最大的k值即为答案  
使用差分数组实现区间标记操作  
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


---
* ### 单调队列/单调栈 ： 
常用于区间最值问题  
单调队列使用std::deque &emsp; 单调栈可使用std::stack

## 题目
--- 
### &emsp; 239. 滑动窗口最大值

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