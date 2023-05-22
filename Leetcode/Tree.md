## Content
- DFS

<br>

------
## DFS
### 概念
---
---
### 题目
---
### &emsp; 1080. 根到叶路径上的不足节点 MID
关键思路：
- <b>DFS</b> 递归调用自身 每次调用把limit减去当前节点值
- 对叶节点 判断是否要被删除 <b>（并通过return的指针向上传递）</b>
- 递归完左右子节点后 若左右子节点都被删除了，等价于所有经过该节点的路径值都小于`limit`，该节点也需要被删除

<details> 
<summary> <b>C++ Code</b> </summary>

``` c++
class Solution {
public:
    TreeNode* sufficientSubset(TreeNode* root, int limit) {
        limit -= root->val;
        if(root->left == root->right) // nullptr
            return limit > 0 ? nullptr : root;
        if(root->left)
            root->left = sufficientSubset(root->left, limit);
        if(root->right)
            root->right = sufficientSubset(root->right, limit);

        return (root->left || root->right)? root : nullptr;
    }
};
```
</details>
<br>

