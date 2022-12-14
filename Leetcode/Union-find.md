## 并查集
---
`核心思想`：每一个集合选择一个 **代表元**，集合的其他元素指向这个代表元  
通过代表元维护一个集合的共有性质  

路径压缩：每个节点直接指向代表元  
按秩合并：把简单的集合（树）往复杂的集合（树）上合并  

可以通过一个节点的邻居查询相应连通块的性质

### **模板**
<details>
<summary> <b> C++ Code</b> </summary>

``` c++
class UnionFind {
public:
    vector<int> father; // 代表元
    vector<int> size; // 秩
    int n; // 节点树
    int comp_cnt; // 集合数

    UnionFind(int _n) : n(_n), comp_cnt(_n), father(_n), size(_n, 1){
        iota(father.begin(), father.end(), 0);
    }
    int find(int x) // 更新并返回father[x]
    {
        if(x != father[x])
            father[x] = find(father[x]);
        return father[x];
    }
    bool unite(int x, int y)
    {
        x = find(x);
        y = find(y);
        cout<<"unite "<<x<<" "<<y<<endl;
        if(x == y)
            return false;
        if(size[x] < size[y])
            swap(x, y);
        father[y] = x;
        size[x] += size[y];
        comp_cnt--;
        return true;
    }
};
```
</details>

---
## Leetcode中利用并查集的题目

---
### &emsp; 1697. 检查边长度限制的路径是否存在 :rage: HARD
关键思路：
- queries数组 离线查询
- 根据对边长度的限制 从小到大排序queries_id
- 对边集根据边长排序 随着当前遍历到的查询（限制长度满足单调增）依次添加边
- <b>使用并查集维护图的连通性</b> &emsp; 在并查集中添加`judgeConnection`方法

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class UnionFind {
public:
    vector<int> father; // 代表元
    vector<int> size; // 秩
    int n; // 节点树
    int comp_cnt; // 集合数

    UnionFind(int _n) : n(_n), comp_cnt(_n), father(_n), size(_n, 1){
        iota(father.begin(), father.end(), 0);
    }
    int find(int x) // 更新并返回father[x]
    {
        if(x != father[x])
            father[x] = find(father[x]);
        return father[x];
    }
    bool unite(int x, int y)
    {
        x = find(x);
        y = find(y);
        cout<<"unite "<<x<<" "<<y<<endl;
        if(x == y)
            return false;
        if(size[x] < size[y])
            swap(x, y);
        father[y] = x;
        size[x] += size[y];
        comp_cnt--;
        return true;
    }
    bool judgeConnection(int x, int y) // 使用并查集判断连通性
    {
        x = find(x);
        y = find(y);
        return x == y;
    }
};
class Solution {
public:
    vector<bool> distanceLimitedPathsExist(int n, vector<vector<int>>& edgeList, vector<vector<int>>& queries) {
        vector<int> q_id(queries.size());
        iota(q_id.begin(), q_id.end(), 0);
        // 对queries的id依limit增序排序
        sort(q_id.begin(), q_id.end(), [&](int i, int j){
            return queries[i][2] < queries[j][2];
        });

        // 将edgeList依dis增序排序
        sort(edgeList.begin(), edgeList.end(), [](const auto& e1, const auto& e2){
            return e1[2] < e2[2];
        });

        UnionFind uf(n);
        int i = 0; // edgeList的指针
        vector<bool> ans(queries.size());
        for(int query : q_id)
        {
            while(i < edgeList.size() && edgeList[i][2] < queries[query][2]) // 根据当前query 添加满足条件的边 连接图
            {
                uf.unite(edgeList[i][0], edgeList[i][1]);
                i++;
            }
            ans[query] = uf.judgeConnection(queries[query][0], queries[query][1]);
        }
        return ans;
    }
};
```
</details>
