[TOC]

## Content
- basic
- BST二叉搜索树
- DFS
- other

不要一开始就陷入细节；先思考 <b>整棵树与其（左右）子树的关系</b>（原问题与子问题）
<br>

------
## basic
### 二叉树前序遍历
迭代栈模拟中比较难处理的在于从当前节点 $u$ 的子节点 $v_1$返回时，需要记录当前已经遍历完成哪些子节点，才能找到下一个需要遍历的节点   
（N叉树可以使用哈希表）  
（或者子节点从右向左逆序入栈）

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
        TreeNode* p = root->left; // 下一个进栈（遍历）的节点
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

N叉树：
<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    vector<int> preorder(Node* root) {
        vector<int> res;
        if(root == nullptr)
            return res;

        stack<Node* > st;
        st.emplace(root);
        while(!st.empty())
        {
            Node* node = st.top();
            st.pop();
            res.emplace_back(node->val);
            for(auto it = node->children.rbegin(); it != node->children.rend(); it++) // 倒序入栈
                st.emplace(*it);
        }
        return res;
    }
};
```
</details>
<br>

---
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

---
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

N叉树：  
等待子节点全部出栈  
<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    vector<int> postorder(Node* root) {
        vector<int> res;
        if(root == nullptr)
            return res;
        
        stack<Node*> st;
        unordered_set<Node*> visited;
        st.emplace(root);
        while(!st.empty())
        {
            Node* node = st.top();
            if(node->children.size() == 0 || visited.count(node)) // 叶子节点或子节点已经全部访问（子节点已全部出栈）
            {
                res.push_back(node->val);
                st.pop();
                continue;
            }
            for(auto it = node->children.rbegin(); it != node->children.rend(); it++)  // 倒序入栈
                st.emplace(*it);
            visited.emplace(node);
        }
        return res;
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

---
### 根据前序序列和后序序列构建二叉树
主要思想：
- <b>找到左右子树对应的序列在前序序列及后序序列中的位置</b>
- 用一个数组预处理 postorder 每个元素的下标，从而可以 O(1) 查找 `preorder[1]` 在 postorder 的位置，从而 O(1)知道左子树的大小
- 递归的终点：空节点 or 叶节点


<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    TreeNode* constructFromPrePost(vector<int>& preorder, vector<int>& postorder) {
        int n = postorder.size();
        vector<int> index(n + 1);
        for(int i = 0; i < n; i++)
            index[postorder[i]] = i; // value to id

        function<TreeNode* (int, int, int, int)> dfs = [&](int pre_l, int pre_r, int post_l, int post_r) -> TreeNode* {
            if(pre_l == pre_r) //空节点
                return nullptr;
            if(pre_l + 1 == pre_r) // 叶节点
                return new TreeNode(preorder[pre_l]);

            int left_size = index[preorder[pre_l + 1]] - post_l + 1; // 左子树大小
            TreeNode* left = dfs(pre_l + 1, pre_l + 1 + left_size, post_l, post_l + left_size);
            TreeNode* right = dfs(pre_l + 1 + left_size, pre_r, post_l + left_size, post_r - 1);
            return new TreeNode(preorder[pre_l], left, right);
        };
        return dfs(0, n, 0, n); // 左闭右开
    }
};
```
</details>
<br>

------
## BST二叉搜索树
### 概念
---
BST性质：
- 左子树的节点值都小于根节点的值
- 右子树的节点值都大于根节点的值
- 任意节点的左子树和右子树都是二叉搜索树
- **中序遍历BST 可以得到有序数组**

---
### 题目
---
### &emsp; 938. 二叉搜索树的范围和 EASY
关键思路：
- DFS，利用BST的性质确定递归下去的方向

<details> 
<summary> <b>C++ Code</b> </summary>

``` c++
class Solution {
public:
    int rangeSumBST(TreeNode* root, int low, int high) {
        if(root == nullptr)
            return 0;
        int x = root->val;
        if(x > high)
            return rangeSumBST(root->left, low, high); // 右子树没有节点在范围内，只需递归左子树
        if(x < low)
            return rangeSumBST(root->right, low, high); // 左子树没有节点在范围内，只需递归右子树
        return x + rangeSumBST(root->left, low, high) + rangeSumBST(root->right, low, high);
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
- 等价于 <b>找最深的包含p、q的子树</b>
- 分析性质，最近公共祖先为要么是p、q中的一个，要么使p、q分布在两侧
- <b>DFS 递归寻找子树中的 p 或 q</b>
- 递归结果`rtn`不为`nullptr`时，意味着在`rtn`子树下找到了 p 或 q （递归保证是“最深”的） 一直向上传递

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

        // 单侧为nullptr 说明这侧子树没有p或q 返回另一侧结果
        if(left == nullptr) return right; 
        if(right == nullptr) return left;
        
        return root; // 左右都不为nullptr，说明p q 分布在异侧
        // 在这里实现了“汇聚”
    }
};
```
</details>
<br>

---
### &emsp; 572. 另一棵树的子树 EASY
关键思路：
- 子问题：判断相同的树（注意两个树的前序序列相同不一定代表树相同）
- 暴力遍历的优化：只判断深度与`subRoot`相同的节点（递归计算深度时，若当前节点深度与`subRoot`相同，调用isSameTree进行判断）

<details> 
<summary> <b>C++ Code</b> </summary>

``` c++
class Solution {
public:
    int getHeight(TreeNode* root) // 二叉树的最大深度
    {
        if(root == nullptr)
            return 0;
        int left_h = getHeight(root->left);
        int right_h = getHeight(root->right);
        return max(left_h, right_h) + 1;
    }
    bool isSameTree(TreeNode* p, TreeNode* q)
    {
        if(p == nullptr || q == nullptr)
            return p == q;
        return p->val == q->val &&
                isSameTree(p->left, q->left) &&
                isSameTree(p->right, q->right);
    }
    bool isSubtree(TreeNode* root, TreeNode* subRoot) {
        int hs = getHeight(subRoot);

        // 返回root的高度以及是否找到了subRoot
        function<pair<int, bool>(TreeNode*)> dfs = [&](TreeNode* node) -> pair<int, bool> {
            if(node == nullptr)
                return {0, false};

            auto [left_h, left_found] = dfs(node->left);
            auto [right_h, right_found] = dfs(node->right);
            if(left_found || right_found)
                return {0, true};

            int node_h = max(left_h, right_h) + 1;
            return {node_h, node_h == hs && isSameTree(node, subRoot)};
        };
        return dfs(root).second;
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
### &emsp; 1766. 互质树 :rage: HARD
关键思路：
- 对一个节点`x` <b>枚举</b> `[1,50]` 中与 `nums[x]` 互质的数
- <b>通过一个映射关系找到深度最大的某一值对应的节点</b>
- 存储节点值等于 `i` 的最近祖先的深度和节点编号（递归到这一层时的“信息”）
- <b>DFS的过程中更新各节点的信息与当前“全局信息”</b>
- 回溯时 恢复现场

<details> 
<summary> <b>C++ Code</b> </summary>

``` c++
const int MX = 51;
vector<int> coprime[MX];
// 预处理 在coprime[i]保存[1, MX)中所有与i互质的数
auto init = [] {
    for(int i = 1; i < MX; i++) 
    {
        for(int j = 1; j < MX; j++)
            if(gcd(i, j) == 1)
                coprime[i].push_back(j);
    } 
    return 0;
}();

class Solution {
    vector<vector<int>> g;
    vector<int> ans;
    pair<int, int> val_depth_id[MX]; // 节点值等于i的最近祖先（最深）的深度和节点编号

    void dfs(int x, int fa, int depth, vector<int> &nums)
    {
        int val = nums[x]; // x 的节点值
        // 计算与 val 互质的数中，深度最大的节点编号
        int max_depth = 0;
        for(int j : coprime[val]) // 枚举 遍历互质的val
        {
            auto [depth, id] = val_depth_id[j];
            if(depth > max_depth) 
            {
                max_depth = depth;
                ans[x] = id;
            }
        }

        auto tmp = val_depth_id[val]; // 用于恢复现场
        val_depth_id[val] = {depth, x}; // 更新递归到此时 val 对应的节点深度和节点编号
        for(int y : g[x]) 
        {
            if(y != fa)
                dfs(y, x, depth + 1, nums);
        }
        val_depth_id[val] = tmp; // 恢复现场
    }
public:
    vector<int> getCoprimes(vector<int>& nums, vector<vector<int>>& edges) {
        int n = nums.size();
        g.resize(n);
        for(auto &e : edges)
        {
            int x = e[0], y = e[1];
            g[x].push_back(y);
            g[y].push_back(x);
        }

        ans.resize(n, -1);
        dfs(0, -1, 1, nums);
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

------
## other
---
### 题目
---
### &emsp; 331. 验证二叉树的前序序列化 MID
关键思路：
- 遍历序列的过程其实等价于遍历这颗树
- <b>统计树的“出度-入度”值`diff`</b>
- 根节点提供2出度，非空节点提供1入度2出度，空节点提供1入度
- 法二：也可以利用 <b>栈</b>，每次遍历到两个空节点时意味着遇到了一个叶子节点，顶部三个元素出栈，替换成一个`"#"`

<details> 
<summary> <b>C++ Code</b> </summary>

``` c++
class Solution {
public:
    bool isValidSerialization(string preorder) {
        stringstream ss(preorder);
        string s;
        int diff = 1;
        while(getline(ss, s, ','))
        {
            diff--; // 入度+1
            if(diff < 0 ) // 还没有遍历它的子节点 必须满足出度大于入度
                return false;

            if(s != "#")
                diff += 2;
        }
        return diff == 0;
    }
};
```
</details>
<br>

---
### &emsp; 1261. 在受污染的二叉树中查找元素 MID
关键思路：
- 二进制数位即代表了向左/向右的移动路径

<details> 
<summary> <b>C++ Code</b> </summary>

``` c++
class FindElements {
    TreeNode *root;
public:
    FindElements(TreeNode* root) : root(root){

    }
    
    bool find(int target) {
        target++;
        auto cur = root;
        for(int i = 30 - __builtin_clz(target); i >= 0; i--) // 从次高位开始枚举
        {
            int bit = target >> i & 1;
            cur = bit ? cur->right : cur->left;
            if(cur == nullptr)
                return false; // 走到空节点
        }
        return true;
    }
};
```
</details>
<br>

---
### &emsp; 2673. 使二叉树所有路径值相等的最小代价 MID
关键思路：
- 影响一个节点值的同时，也会影响它的子孙节点的路径值
- <b>从最后一个非叶节点开始自底向上计算</b>
- 再递归考虑子节点对父节点的路径和的影响

<details> 
<summary> <b>C++ Code</b> </summary>

``` c++
class Solution {
public:
    int minIncrements(int n, vector<int>& cost) {
        int ans = 0;
        for(int i = n / 2; i > 0; i--) // 从最后一个非叶节点开始
        {
            ans += abs(cost[i * 2 -1] - cost[i * 2]); // 两个子节点变成一样的
            cost[i - 1] += max(cost[i * 2 - 1], cost[i * 2]); // 更新父节点的路径和 
        }
        return ans;
    }
};
```
</details>
<br>