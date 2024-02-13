[TOC]

## Content
- basic
- DFS

不要一开始就陷入细节；先思考 <b>整棵树与其（左右）子树的关系</b>（原问题与子问题）
<br>

------
## basic
### 二叉树前序遍历

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    vector<int> preorderTraversal(TreeNode* root) {
        vector<int> ans;
        stack<TreeNode*> s;
        if(root == nullptr)
            return ans;
        s.push(root);
        ans.push_back(root->val);
        TreeNode* p = root->left;
        while(!s.empty())
        {
            while(p)
            {
                s.push(p);
                ans.push_back(p->val);
                p = p->left;
            }
            TreeNode* t = s.top();
            s.pop();
            if(t->right)
            {
                s.push(t->right);
                ans.push_back(t->right->val);
                p = t->right->left;
            }
        }
        return ans;
    }
};
```
</details>
<br>

### 二叉树中序遍历

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
        vector<int> ans;
        stack<TreeNode*> s;
        if(root == nullptr)
            return ans;

        s.push(root);
        TreeNode* p = root->left; // 指示下一个进栈
        while(!s.empty())
        {
            while(p != nullptr) // 左子树向左遍历
            {
                s.push(p);
                p = p->left;
            }

            TreeNode* t = s.top();
            ans.push_back(t->val);
            s.pop(); // 出栈 处理右子树
            if(t->right)
            {
                s.push(t->right);
                p = t->right->left;
            }
        }
        return ans;
    }
};
```
</details>
<br>

### 二叉树后序遍历

先“根右左”遍历，然后reverse
<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    vector<int> postorderTraversal(TreeNode* root) {
        vector<int> ans;
        if(root == nullptr)
            return ans;
        stack<TreeNode*> s;
        s.push(root);
        while(!s.empty())
        {
            TreeNode* node = s.top();
            s.pop();
            ans.push_back(node->val);
            if(node->left)
                s.push(node->left);
            if(node->right)
                s.push(node->right);
        }
        reverse(ans.begin(), ans.end());
        return ans;
    }
};
```
</details>
<br>

---
### 二叉树morris遍历（以中序为例）
主要思想：
- 对于当前节点`x`，准备处理左子树时，先找到其左子树的最右节点（中序遍历输出的前一个节点）`predecessor`，将其右指针指向自己，然后正常向左遍历  
- 这样在左子树遍历完成后可以通过这个指针走回`x`，且能通过这个指针知晓我们已经遍历完成了左子树    

每个节点会被遍历两次，O(2n)；空间复杂度为 O(1)   
<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
        vector<int> res;
        TreeNode *predecessor = nullptr;

        while (root != nullptr) {
            if (root->left != nullptr) {
                // predecessor 节点就是当前 root 节点向左走一步，然后一直向右走至无法走为止
                predecessor = root->left;
                while (predecessor->right != nullptr && predecessor->right != root) {
                    predecessor = predecessor->right;
                }
                
                // 让 predecessor 的右指针指向 root，继续遍历左子树
                if (predecessor->right == nullptr) {
                    predecessor->right = root;
                    root = root->left;
                }
                // 说明左子树已经访问完了，我们需要断开链接
                else {
                    res.push_back(root->val);
                    predecessor->right = nullptr;
                    root = root->right;
                }
            }
            // 如果没有左孩子，则直接访问右孩子
            else {
                res.push_back(root->val);
                root = root->right;
            }
        }
        return res;
    }
};
```
</details>
<br>

------
## DFS
### 概念
---
---
### 题目
---
### &emsp; 236. 二叉树的最近公共祖先 MID
关键思路：
- 分析性质，最近公共祖先为要么是p、q中的一个，要么使p、q分布在两侧
- <b>DFS递归寻找子树中的p或q</b>
- 递归结果`rtn`不为`nullptr`时，意味着在`rtn`子树下找到了p或q（递归保证是“最深”的）

<details> 
<summary> <b>C++ Code</b> </summary>

``` c++
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if(root == nullptr || root == p || root == q) // 遇到叶节点、p或q时从底至顶回溯
            return root;
        TreeNode* left = lowestCommonAncestor(root->left, p, q);
        TreeNode* right = lowestCommonAncestor(root->right, p, q);

        // 单侧为nullptr 返回另一侧结果
        if(left == nullptr) return right; 
        if(right == nullptr) return left;
        
        return root; // p q 分布在异侧
    }
};
```
</details>
<br>

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
### &emsp; 987. 二叉树的垂序遍历 :rage: HARD
关键思路：
- 需要知道每个节点的行号row、列号col以及节点值val
- 按照col分组 每组的val根据row从小到大排序
- <b>使用DFS获取每个节点信息，用有序map记录</b>
- 也可以在DFS的同时记录col最小值，从而可以使用unordered_map

<details> 
<summary> <b>C++ Code</b> </summary>

``` c++
class Solution {
    map<int, vector<pair<int, int>>> groups;

    void dfs(TreeNode* node, int row, int col)
    {
        if(node == nullptr)
            return;
        groups[col].emplace_back(row, node->val);
        dfs(node->left, row + 1, col - 1);
        dfs(node->right, row + 1, col + 1);
    }
public:
    vector<vector<int>> verticalTraversal(TreeNode* root) {
        dfs(root, 0, 0);
        vector<vector<int>> ans;
        for(auto &[_, g] : groups)
        {
            ranges::sort(g);
            vector<int> vals;
            for(auto &[_, val] : g)
                vals.push_back(val);
            
            ans.push_back(vals);
        }
        return ans;
    }
};
```
</details>
<br>

- 也可以把所有 `(col,row,val)` 全部丢到同一个列表中，排序后按照 `col` 分组
<details> 
<summary> <b>C++ Code</b> </summary>

``` c++
lass Solution {
public:
    vector<vector<int>> verticalTraversal(TreeNode* root) {
        vector<tuple<int, int, int>> data;
        function<void(TreeNode*, int, int)> dfs = [&](TreeNode* node, int row, int col) {
            if(node == nullptr)
                return;
            data.emplace_back(col, row, node->val);
            dfs(node->left, row + 1, col - 1);
            dfs(node->right, row + 1, col + 1);
        };
        dfs(root, 0, 0);

        vector<vector<int>> ans;
        ranges::sort(data);
        int last_col = INT_MIN;
        for(auto &[col, _, val] : data)
        {
            if(col != last_col)
            {
                last_col = col;
                ans.push_back({}); // 一个新的组
            }
            ans.back().push_back(val);
        }
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

---
### &emsp; 1123. 最深叶节点的最近公共祖先 MID
关键思路：
- 思考问题转化 *要找的这个节点是在当前节点的左子树，还是右子树，还是就是当前节点*
- 递归这棵树，同时维护一个全局最大深度
- 在“递”的过程中，向下传递深度`depth`
- 在“归”的过程中，上传当前子树最深叶节点的深度
- 设左子树最深叶节点的深度为 `leftMaxDepth`，右子树最深叶节点的深度为 `rightMaxDepth`。如果 `leftMaxDepth == rightMaxDepth == maxDepth`，那么更新答案为当前节点。注意这并不代表我们找到了答案，如果后面发现了更深的叶节点，那么答案还会更新

<details> 
<summary> <b>C++ Code</b> </summary>

``` c++
class Solution {
public:
    TreeNode* lcaDeepestLeaves(TreeNode* root) {
        TreeNode *ans = nullptr;
        int max_depth = -1; // 全局最大深度 
        function<int(TreeNode*, int)> dfs = [&](TreeNode *node, int depth) {
            if(node == nullptr)
            {
                max_depth = max(max_depth, depth);
                return depth;
            }
            int left_max_depth = dfs(node->left, depth + 1);
            int right_max_depth = dfs(node->right, depth + 1);
            if(left_max_depth == right_max_depth && left_max_depth == max_depth)
                ans = node;

            return max(left_max_depth, right_max_depth);
        };
        dfs(root, 0);
        return ans;
    }
};
```
</details>
<br>

---
### &emsp; 2003. 每棵子树内缺失的最小基因值 :rage: HARD
关键思路：
- 思考问题转化： 以每个节点为根的子树中，权重集合在 [1,n+1] 范围内缺失的最小数
- <b>从节点 1 开始往根找（从深到浅）</b> 记录并更新<b>以节点 i 为根的子树包含的所有基因值</b>； 这条路径以外的点 ans 都为 1
- 可以用数组代替哈希表标记 即使nums包括了1到n所有数，ans[i]也不会超过n+1（即对于超过n+1的nums[i]来说都可以视为n+2）
- 进一步地，也可以视作n+1，因为如果有nums[i]超过n，那么ans[i]最大不会超过n（因为前面一定有一个数不是任何一个nums[i]）

<details> 
<summary> <b>C++ Code</b> </summary>

``` c++
class Solution {
public:
    vector<int> smallestMissingValueSubtree(vector<int>& parents, vector<int>& nums) {
        int n = parents.size();
        vector<int> ans(n, 1);
        auto it = find(nums.begin(), nums.end(), 1);
        if(it == nums.end()) // 不存在基因值为 1 的点
            return ans;
        
        vector<vector<int>> g(n);
        for(int i = 1; i < n; i++)
            g[parents[i]].push_back(i);

        vector<int> vis(n + 2);
        stack<int> nodes; // dfs 子树的节点
        int mex = 2; // 缺失的最小基因值
        int pre = -1;
        int node = it - nums.begin(); // 节点 1
        while(node >= 0)
        {
            vis[min(nums[node], n + 1)] = true; // 标记基因值
            for(int son : g[node])
            {
                if(son != pre) // pre的子树这一边就无需遍历了
                    nodes.push(son); // 接下来需要遍历的点
            }
            while(!nodes.empty()) // dfs每个son
            {
                int x = nodes.top();
                nodes.pop();
                vis[min(nums[x], n + 1)] = true;
                for(int son : g[x])
                    nodes.push(son);
            }
            while(vis[mex])
                mex++;
            ans[node] = mex;
            pre = node;
            node = parents[node]; // 往上走
        }
        return ans;
    }
};
```
</details>
<br>