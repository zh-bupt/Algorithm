### [百练 OJ #2787：算 24](http://bailian.openjudge.cn/practice/2787/)
#### 题目描述：
> 给出4个小于10个正整数，你可以使用加减乘除4种运算以及括号把这4个数连接起来得到一个表达式。现在的问题是，是否存在一种方式使得得到的表达式的结果等于24。
> 这里加减乘除以及括号的运算结果和运算的优先级跟我们平常的定义一致（这里的除法定义是实数除法）。
> 比如，对于5，5，5，1，我们知道5 * (5 – 1 / 5) = 24，因此可以得到24。又比如，对于1，1，4，2，我们怎么都不能得到24。

#### 解法一:
> 使用深度优先的搜索方法，首先从中选择两个数进行加减乘除，和剩下的数放进一个数组。后面重复这个过程，直到数组中只有一个数的时候判断这个数是不是 24，如果是 24 则表示能能通过这几个数算出 24，否则，则不能算出 24。
>
>例如 (1, 2, 3, 4) 一次成功的搜索:
>(1, 2, 3, 4) -> (1 * 2, 3, 4) -> ((1 * 2) * 3, 4) -> (((1 * 2) * 3) * 4)

**C++ 代码:**
```c++
#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

const float epsilon = 0.00001;

// 判断浮点数相等不能使用等号, 会有误差
// 因此要使用这个 equal 函数判断两个浮点数是否相等
bool equal(float a, float b) {
    return fabs(a - b) < epsilon;
}

bool calculate24(vector<float> v) {
    // 如果数组中只有一个数了, 则判断这个数是不是 24
    // 如果是 24, 则返回 true, 否则返回 false
    if (v.size() == 1) return equal(v[0], 24);
    // 将数组中的数两两组合进行加减乘除(减法和除法有顺序)
    for (int i = 0; i < v.size() - 1; ++i) {
        for (int j = i + 1; j < v.size(); ++j) {
            vector<float> temp(v.size() - 1);
            int index = 0;
            for (int k = 0; k < v.size(); ++k) {
                if (k != i && k != j) {
                    temp[index++] = v[k];
                }
            }
            temp[index] = v[i] + v[j];
            if (calculate24(temp)) return true;
            temp[index] = v[i] - v[j];
            if (calculate24(temp)) return true;
            temp[index] = v[j] - v[i];
            if (calculate24(temp)) return true;
            temp[index] = v[i] * v[j];
            if (calculate24(temp)) return true;
            // 除法需要保证除数不为 0
            if (!equal(v[j], 0)) {
                temp[index] = v[i] / v[j];
                if (calculate24(temp)) return true;
            }
            if (!equal(v[i], 0)) {
                temp[index] = v[j] / v[i];
                if (calculate24(temp)) return true;
            }
        }
    }
    return false;
}

int main() {
    while (true) {
        vector<float> v(4);
        scanf("%f %f %f %f", &v[0], &v[1], &v[2], &v[3]);
        if (v[0] == 0 && v[1] == 0 && v[2] == 0 && v[3] == 0) {
            break;
        }
        if (calculate24(v)) {
            printf("YES\n");
        } else {
            printf("NO\n");
        }
    }
    return 0;
}
```