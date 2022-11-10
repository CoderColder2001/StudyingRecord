## Content
- 状态压缩 + BFS/DFS

<br>

------
## 状态压缩 + BFS/DFS
### 概念
---
关键在于考虑 <b>抽象状态的定义与转移</b>

<br>

---
### 题目
---
### &emsp; 864. 获取所有钥匙的最短路径 HARD
`int state` 存储二进制状态 表示当前是否持有某个钥匙  
最终状态定义 <b>(x,y,keys)</b> &emsp; 即当前坐标与当前所持有的钥匙  

状态转移：从队列头取出一个点，可获得此处的位置和携带钥匙，向四个方向查看是否能走。  
“ 能走 ” 等价于： 下个状态不越界 && 没访问过下个状态 && （下个状态是通路 ||（下个状态是大写字母 && 当前我有下个状态需要的钥匙））

<details> 
<summary> <b>C++ Code</b> </summary>

``` c++
class Solution {
public:
    const static int N = 35, K = 7, INF = 0x3f3f3f3f; 
    bool states[N][N][1<<K];
    int dx[4] = {0, 1, 0, -1};
    int dy[4] = {-1, 0, 1, 0};

    int shortestPathAllKeys(vector<string>& grid) {
        int m = grid.size();
        int n = grid[0].size();
        int k = 0; // 钥匙&锁数量
        int start_x = -1, start_y = -1;

        // init
        for(int i = 0; i < grid.size(); i++)
        {
            for(int j = 0; j < grid[i].size(); j++)
            {
                char c = grid[i][j];
                if(c == '@')
                {
                    start_x = i;
                    start_y = j;
                }
                else if(c <= 'z' && c >= 'a')
                    k++;
            }
        }

        //BFS
        int step = 0;
        queue<tuple<int, int, int>> q;
        q.emplace(start_x, start_y, 0);
        while(!q.empty())
        {
            int qSize = q.size(); // 注意size()不能写在循环判断内
            for(int i = 0; i < qSize; i++)
            {
                auto [x, y, keyState] = q.front();
                q.pop();
                if(keyState == (1 << k) - 1) // 找到了所有钥匙
                    return step;
                    
                for(int move = 0; move < 4; move++)
                {
                    int _x = x + dx[move];
                    int _y = y + dy[move];
                    if(_x >= 0 && _x < m && _y >= 0 && _y < n) // 边界范围内移动
                    {
                        char c = grid[_x][_y];
                        int _keyState = keyState;
                        if(c == '#') // 墙
                            continue;
                        else if(c <= 'Z' && c >= 'A' && !(_keyState>>(c-'A') & 1)) // 还没有钥匙
                            continue;
                        else if(c <= 'z' && c >= 'a') // 得到钥匙
                            _keyState = _keyState | 1<<(c-'a');

                        if(!states[_x][_y][_keyState]) // 未访问过这个状态
                        {
                            states[_x][_y][_keyState] = true;
                            q.emplace(_x, _y, _keyState);
                        }
                    }
                }
            }
            step++;
        }

        return -1;
    }
};
```
</details>