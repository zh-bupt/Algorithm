### [LeetCode #42: Trapping Rain Water](https://leetcode.com/problems/trapping-rain-water/)
#### 题目描述:
> 给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。
> ![image](https://i.loli.net/2020/05/12/fmTMv9sN43Ww1kj.png)
> 上面是由数组 [0,1,0,2,1,0,1,3,2,1,2,1] 表示的高度图，在这种情况下，可以接 6 个单位的雨水（蓝色部分表示雨水）。
> 示例:
> 输入: [0,1,0,2,1,0,1,3,2,1,2,1]
> 输出: 6
#### 解法一:
> 可以观察到要形成一个水洼需要两边的方块比中间的高，而且一个形状不规则的水洼可以被分解成几个形状规则的水洼，如图：
> ![不规则的水洼的分解](https://i.loli.net/2020/05/12/ouX3vC1zsipYPjT.jpg)
> 因此我们在遍历数组时维护一个栈。如果当前的条形块小于或等于栈顶的条形块，我们将条形块的索引和高度入栈，这样栈里面存放的就是一些下标递增但是高度非递增的块。如果我们发现一个条形块长于栈顶，我们可以确定栈顶的条形块被当前条形块和栈的前一个条形块形成一个水洼（如果站定几个条形块高度一样则可认为形成了一个高度为零的水洼），而且我们可以根据高度和宽度算出水洼的面积。这样只需要一次遍历就可以计算出结果，代码如下:

**C++代码:**
```c++
typedef struct node {
    int h, x;
    node(int hh, int xx):h(hh),x(xx) {}
} node;

class Solution {
public:
    int trap(vector<int>& height) {
        stack<node> s;
        int result = 0;
        for (int i = 0; i < height.size(); ++i) {
            while(!s.empty() && s.top().h < height[i]) {
                // 记录底部的高度, 因为有可能是一个不规则的水洼
                int t = s.top().h;
                // 将底部从堆栈中弹出
                s.pop();
                // 保证水洼的左边有方块
                if (!s.empty()){
                     // 水洼面积等于宽度乘上两边方块高度的最小值(减去底部高度)
                    result += (i - s.top().x - 1) * (min(height[i], s.top().h) - t);
                }
            }
            s.push(node(height[i], i));
        }
        return result;
    }
};
```