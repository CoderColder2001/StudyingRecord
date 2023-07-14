## Content
- DFS

不要一开始就陷入细节；先思考 <b>整棵树与其（左右）子树的关系</b>（原问题与子问题）
<br>

------
## DFS
### 概念
---
---
### 题目
---
### &emsp; 979. 在二叉树中分配硬币 MID
关键思路：
- 每枚硬币移动的路径长度并不好计算，但是若把这些路径叠起来，**转换成 每条边经过了多少枚硬币** ，就容易计算了
- 所有路径长度之和，等同于把「每条边出现在多少条路径中」相加
- <b>DFS + 贡献法</b> DFS计算节点贡献
- 硬币在节点上移动 思考对于一个非根节点（子树而言）要向其父节点移入or移出多少硬币
- 注意对根节点有`abs(coins - nodes) == 0` 不影响ans，无需特判

<details> 
<summary> <b>C++ Code</b> </summary>

``` c++
class Solution {
public:
    int ans = 0;
    pair<int, int> dfs(TreeNode* node) // 返回硬币数和节点数
    {
        if(node == nullptr)
            return {0, 0};
        auto [coins_l, nodes_l] = dfs(node->left);
        auto [coins_r, nodes_r] = dfs(node->right);
        int coins = coins_l + coins_r + node->val;
        int nodes = nodes_l + nodes_r + 1;
        ans += abs(coins - nodes); // 产生贡献
        return {coins, nodes};
    }

    int distributeCoins(TreeNode* root) {
        dfs(root);
        return ans;
    }
};
```
</details>
<br>

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

---
### &emsp; 1110. 删点成林 MID
关键思路：
- 使用哈希集合记录需要删除的节点以便快速查询
- <b>DFS</b> 遍历树；<b>使用返回值表示当前子树根节点是否被删除了</b>
- 当前节点被删除时，根据左右子树DFS递归返回的结果（左右子节点是否被删除），将其加入答案
- 注意DFS递归调用的位置

<details> 
<summary> <b>C++ Code</b> </summary>

``` c++
class Solution {
    vector<TreeNode*> ans;
    unordered_set<int> s; // 快速判断是否在delete中

    TreeNode* dfs(TreeNode* node)
    {
        if(node == nullptr)
            return nullptr;
        node->left = dfs(node->left); // 递归判断左子树删不删
        node->right = dfs(node->right); // 递归判断右子树删不删
        if(!s.count(node->val)) // 当前结点不删
            return node;
        
        if(node->left)
            ans.push_back(node->left);
        if(node->right)
            ans.push_back(node->right);
        return nullptr;
    }
public:
    vector<TreeNode*> delNodes(TreeNode* root, vector<int>& to_delete) {
        for(int x : to_delete)
            s.insert(x);
        if(dfs(root))
            ans.push_back(root);
        return ans;
    }
};
```
</details>
<br>
