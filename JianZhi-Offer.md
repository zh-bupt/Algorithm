### [!]1. 二维数组查找

#### 题目描述

> 在一个二维数组中 (每个一维数组的长度相同), 每一行都按照从左到右递增的顺序排序, 每一列都按照从上到下递增的顺序排序. 请完成一个函数, 输入这样的一个二维数组和一个整数, 判断数组中是否含有该整数.

#### 解法一: 二分查找

> 针对每一列进行二分查找, 如果找到则返回 ```true```, 否则返回 ```false```. 对于 $m \times n$ 的二维数组, 时间复杂度为 $O(m \log n)$

```c++
class Solution {
public:
    bool Find(int target, vector<vector<int> > array) {
        int row = array.size();
        if (row == 0) return false;
        int col = array[0].size();
        if (col == 0) return false;
        for (auto a:array) {
            int l = 0, r = col - 1;
            while (l <= r) {
                int mid = (l + r) / 2;
                if (a[mid] == target) return true;
                if (a[mid] > target) r = mid - 1;
                else l = mid + 1;
            }
        }
        return false;
    }
};
```

#### 解法二:

> 从左下角开始遍历, 如果遇到比目标数大的值则向上移动, 否则向右移动. 如果找到了则返回 ```true```; 如果到了右上角还没有找到, 则返回 ```false```. 时间复杂度为 $O(m+n)$.

```c++
class Solution {
public:
    bool Find(int target, vector<vector<int> > array) {
        int row = array.size();
        if (row == 0) return false;
        int col = array[0].size();
        if (col == 0) return false;
        int i = row - 1, j = 0;
        while (i >= 0 && j < col) {
            if (array[i][j] == target) return true;
            if (array[i][j] > target) {
                i--;
            } else {
                j++;
            }
        }
        return false;
    }
};
```

### 3. 从尾到头打印链表

#### 题目描述

>  输入一个链表，按链表从尾到头的顺序返回一个ArrayList。

```c++
/**
*  struct ListNode {
*        int val;
*        struct ListNode *next;
*        ListNode(int x) :
*              val(x), next(NULL) {
*        }
*  };
*/
class Solution {
public:
    vector<int> printListFromTailToHead(ListNode* head) {
        ListNode *cur = head, *pre = NULL, *temp;
        int len = 0;
        while (cur != NULL) {
            temp = cur->next;
            cur->next = pre;
            pre = cur;
            cur = temp;
            len++;
        }
        cur = pre;
        int index = 0;
        vector<int> result(len);
        while (cur != NULL) {
            result[index++] = cur->val;
            cur = cur->next;
        }
        return result;
    }
};
```

### 4. 重建二叉树

#### 题目描述

> 输入某二叉树的前序遍历和中序遍历的结果, 请重建出该二叉树. 假设输入的前序遍历和中序遍历的结果中都不含重复的数字. 例如输入前序遍历序列 ```{1,2,4,7,3,5,6,8}``` 和中序遍历序列 ```{4,7,2,1,5,3,8,6}```, 则重建二叉树并返回.
#### 解法一: 递归

> TODO

```c++
/**
 * Definition for binary tree
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    vector<int> pre, vin;
    TreeNode* help(pair<int, int> pre_p, pair<int, int> vin_p) {
        if (pre_p.first > pre_p.second) return NULL;
        if (pre_p.first == pre_p.second) return new TreeNode(pre[pre_p.first]);
        int cur = pre[pre_p.first];
        TreeNode* node = new TreeNode(cur);
        pair<int, int> pre_pl, pre_pr, vin_pl, vin_pr;
        for (int i = vin_p.first; i <= vin_p.second; ++i) {
            if (vin[i] == cur){
                int len1 = i - vin_p.first, len2 =  vin_p.second - i; 
                vin_pl = make_pair(vin_p.first, i - 1);
                vin_pr = make_pair(i + 1, vin_p.second);
                pre_pl = make_pair(pre_p.first + 1, pre_p.first + len1);
                pre_pr = make_pair(pre_p.first + 1 + len1, pre_p.second);
                break;
            }
        }
        node->left = help(pre_pl, vin_pl);
        node->right = help(pre_pr, vin_pr);
        return node;
    }
    
    TreeNode* reConstructBinaryTree(vector<int> pre,vector<int> vin) {
        this->pre = pre;
        this->vin = vin;
        int len = pre.size();
        return help(make_pair(0, len - 1), make_pair(0, len - 1));
    }
};
```

### 5. 用两个栈实现队列

#### 题目描述

> 用两个栈来实现一个队列, 完成队列的 Push 和 Pop 操作. 队列中的元素为 ```int``` 类型.
#### 解法一:

> 使用两个栈, 一个(```stack1```)用来压进队列, 一个(```stack2```)用来出队列. 压进队列的时候只需要向第一个栈里面压即可, 出队列的时候检查第二个队列是否为空, 不为空的话栈顶即为队列的第一个元素; 如果为空的话, 则将第一个栈的所用元素出栈依次压入第二个栈, 然后栈顶元素即为队列的第一个元素(若两个栈都为空的时候从队列取元素则抛异常).

```c++
class Solution
{
public:
    void push(int node) {
        stack1.push(node);
    }

    int pop() {
        if (stack2.empty()) {
            while (!stack1.empty()) {
                stack2.push(stack1.top());
                stack1.pop();
            }
        }
        int node = stack2.top();
        stack2.pop();
        return node;
    }

private:
    stack<int> stack1;
    stack<int> stack2;
};
```

### 6. 旋转数组的最小数字

#### 题目描述

> 把一个数组最开始的若干个元素搬到数组的末尾, 我们称之为数组的旋转. 输入一个非递减排序的数组的一个旋转, 输出旋转数组的最小元素. 例如数组 ```{3,4,5,1,2}``` 为 ```{1,2,3,4,5} ```的一个旋转, 该数组的最小值为1.
> NOTE: 给出的所有元素都大于0, 若数组大小为0, 请返回0.
#### 解法一:

> TODO

```c++
class Solution {
public:
    int minNumberInRotateArray(vector<int> rotateArray) {
        if (rotateArray.size() == 0) return 0;
        int l = 0, r = rotateArray.size() - 1;
        if (rotateArray[l] < rotateArray[r]) return rotateArray[0];
        while (l <= r) {
            int mid = (l + r) / 2;
            if (rotateArray[mid] > rotateArray[mid + 1]) return rotateArray[mid + 1];
            if (rotateArray[mid] >= rotateArray[l]) {
                l = mid + 1;
            } else {
                r = mid - 1;
            }
        }
    }
};
```

### [!]9. 变态跳台阶

#### 题目描述

> 一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法。
#### 解法一:
> 用 $f(n)$ 表示跳上第 $n$ 级台阶有多少种走法, 则:
>
> $
> f(n)=f(n-1) + f(n-2) +\cdots+f(1)
> $


```c++
class Solution {
public:
    int jumpFloorII(int number) {
        vector<int> dp(number + 1, 0);
        dp[0] = 1;
        for (int i = 1; i <= number; ++i) {
            for (int j = 0; j < i; ++j) {
                dp[i] += dp[j];
            }
        }
        return dp[number];
    }
};
```

#### 解法二:

> 用 $f(n)$ 表示跳上第 $n$ 级台阶有多少种走法, 则:
>
> $
> f(n)=f(n-1) + f(n-2) +\cdots+f(1)
> $
> $
> f(n - 1)=f(n-2) + f(n-2) +\cdots+f(1)
> $
>
> 两式相减得: $f(n) = 2f(n-1)$

```c++
class Solution {
public:
    int jumpFloorII(int number) {
        if (number == 0) return 1;
        if (number == 1) return 1;
        int result = 1;
        for (int i = 0; i < number - 1; ++i) {
            result <<= 1;
        }
        return result;
    }
};
```

### [!]10. 矩形覆盖

#### 题目描述

> 我们可以用 $2*1$ 的小矩形横着或者竖着去覆盖更大的矩形. 请问用 n 个 $2*1$ 的小矩形无重叠地覆盖一个 $2*n$ 的大矩形, 总共有多少种方法?
>
> 比如 n = 3 时, $2*3$ 的矩形块有 3 种覆盖方法:
>
> ![](https://i.loli.net/2020/04/29/1f8tNGjmSJMvlFh.png)

#### 解法一:

> 只需考虑第一个矩形的放置方法即可, 如果用 $f(n)$ 表示拼成 $2*n$ 的矩形的方式, 则由下图可知:
>
> ![](https://i.loli.net/2020/05/02/CdDaRKrm4pkoqEV.jpg)
>
> $f(n) = f(n-1)+f(n-2), n > 2$

```c++
class Solution {
public:
    int rectCover(int number) {
        if (number == 0) return 0;
        if (number == 1) return 1;
        if (number == 2) return 2;
        int i1 = 1, i2 = 2, temp;
        for (int i = 0; i < number - 2; ++i) {
            temp = i2;
            i2 = i1 + i2;
            i1 = temp;
        }
        return i2;
    }
};
```

### 13. 调整数组顺序使奇数位于偶数前面

#### 题目描述

> 输入一个整数数组, 实现一个函数来调整该数组中数字的顺序, 使得所有的奇数位于数组的前半部分, 所有的偶数位于数组的后半部分, 并保证奇数和奇数, 偶数和偶数之间的相对位置不变.

#### 解法一: 冒泡排序

```c++
class Solution {
public:
    void reOrderArray(vector<int> &array) {
        int len = array.size();
        for (int i = 0; i < len; ++i) {
            for (int j = 0; j < len - i - 1; ++j) {
                if (array[j] % 2 == 0 && array[j+1] % 2 != 0) {
                    int temp = array[j];
                    array[j] = array[j+1];
                    array[j+1] = temp;
                }
            }
        }
    }
};
```

#### 解法二: 队列

```c++
class Solution {
public:
    void reOrderArray(vector<int> &array) {
        queue<int> even_q, odd_q;
        for (int i:array) {
            if (i % 2 == 0) {
                even_q.push(i);
            } else {
                odd_q.push(i);
            }
        }
        int index = 0;
        while(!odd_q.empty()) {
            array[index++] = odd_q.front();
            odd_q.pop();
        }
        while(!even_q.empty()) {
            array[index++] = even_q.front();
            even_q.pop();
        }
    }
};
```

### 14. 链表中的导数第 K 个节点

#### 题目描述

>  输入一个链表, 输出该链表中倒数第 k 个结点.

#### 解法一: 双指针

> 设置两个指针, 一个指针超前另一个指针 k 个结点, 然后一起向后移动, 当超前的指针为空时, 后面的指针所指的结点就是倒数第 k 个结点了.

```c++
/*
struct ListNode {
	int val;
	struct ListNode *next;
	ListNode(int x) :
			val(x), next(NULL) {
	}
};*/
class Solution {
public:
    ListNode* FindKthToTail(ListNode* pListHead, unsigned int k) {
        ListNode *p1 = pListHead, *p2 = pListHead;
        for (int i = 0; i < k; ++i) {
            if (p2 != NULL) {
                p2 = p2->next;
            } else {
                return NULL;
            }
        }
        while (p2 != NULL) {
            p1 = p1->next;
            p2 = p2->next;
        }
        return p1;
    }
};
```

### 15. 反转链表

#### 题目描述

>  输入一个链表，反转链表后，输出新链表的表头。

#### 解法一:

```c++
/*
struct ListNode {
	int val;
	struct ListNode *next;
	ListNode(int x) :
			val(x), next(NULL) {
	}
};*/
class Solution {
public:
    ListNode* ReverseList(ListNode* pHead) {
        ListNode *cur = pHead, *pre = NULL, *temp;
        while (cur != NULL) {
            temp = cur->next;
            cur->next = pre;
            pre = cur;
            cur = temp;
        }
        return pre;
    }
};
```

### 16. 合并两个有序链表

#### 题目描述

>  输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。

#### 解法一:

> 类似于归并排序

```c++
/*
struct ListNode {
	int val;
	struct ListNode *next;
	ListNode(int x) :
			val(x), next(NULL) {
	}
};*/
class Solution {
public:
    ListNode* Merge(ListNode* pHead1, ListNode* pHead2){
        ListNode *pHead = new ListNode(0), *cur = pHead;
        while (pHead1 != NULL && pHead2 != NULL) {
            if (pHead1->val < pHead2->val) {
                cur->next = pHead1;
                pHead1 = pHead1->next;
                cur = cur->next;
            } else {
                cur->next = pHead2;
                pHead2 = pHead2->next;
                cur = cur->next;
            }
        }
        while (pHead1 != NULL) {
            cur->next = pHead1;
            pHead1 = pHead1->next;
            cur = cur->next;
        }
        while (pHead2 != NULL) {
            cur->next = pHead2;
            pHead2 = pHead2->next;
            cur = cur->next;
        }
        return pHead->next;
    }
};
```

### [!]17. 树的子结构

#### 题目描述

> 输入两棵二叉树A, B, 判断 B 是不是 A 的子结构. (ps: 我们约定空树不是任意一个树的子结构.)

#### 解法一: 递归

> **注意:** 刚开始的时候对子结构的理解出现了偏差, 实际上 A 树中任意一个部分都是 A 树的子结构, 刚开始以为只有从当前节点一直到叶结点才是一个子结构.
>
> TODO

```c++
/*
struct TreeNode {
	int val;
	struct TreeNode *left;
	struct TreeNode *right;
	TreeNode(int x) :
			val(x), left(NULL), right(NULL) {
	}
};*/
class Solution {
public:
    bool IsSubtree(TreeNode* root1, TreeNode* root2) {
        if (root2 == NULL) return true;
        if (root1 == NULL) return false;
        return (root1->val == root2->val) &&
            IsSubtree(root1->left, root2->left) &&
            IsSubtree(root1->right, root2->right);
    }
    
    bool HasSubtree(TreeNode* pRoot1, TreeNode* pRoot2) {
        if (pRoot2 == NULL || pRoot1 == NULL) return false;
        if (IsSubtree(pRoot1, pRoot2)) return true;
        return HasSubtree(pRoot1->left, pRoot2) || HasSubtree(pRoot1->right, pRoot2);
    }
};
```

### 18. 二叉树镜像

#### 题目描述

> 操作给定的二叉树，将其变换为源二叉树的镜像。

```txt
输入描述:
二叉树的镜像定义：
源二叉树:
    	    8
    	   /  \
    	  6   10
    	 / \  / \
    	5  7 9 11
镜像二叉树:
    	    8
    	   /  \
    	  10   6
    	 / \  / \
    	11 9 7  5
```

#### 解法一: 递归

```c++
/*
struct TreeNode {
	int val;
	struct TreeNode *left;
	struct TreeNode *right;
	TreeNode(int x) :
			val(x), left(NULL), right(NULL) {
	}
};*/
class Solution {
public:
    void Mirror(TreeNode *pRoot) {
        if (pRoot == NULL) return;
        TreeNode *temp = pRoot->left;
        pRoot->left = pRoot->right;
        pRoot->right = temp;
        Mirror(pRoot->left);
        Mirror(pRoot->right);
    }
};
```

### 19. 顺时针打印矩阵

#### 题目描述

>  输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字.
>
> 例如，如果输入如下4 X 4矩阵：
> ```
> 1 2 3 4 
> 5 6 7 8
> 9 10 11 12
> 13 14 15 16 
> ```
> 则依次打印出数字1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10.

#### 解法一:

```c++
class Solution {
public:
    vector<int> printMatrix(vector<vector<int> > matrix) {
        int len1 = matrix.size();
        if (len1 == 0) return {};
        int len2 = matrix[0].size();
        if (len2 == 0) return {};
        int len = len1 * len2;
        vector<int> result(len);
        int dir_arr[4][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
        int up = 0, down = len1 - 1, left = 0, right = len2 - 1;
        int index = 0, i = 0, j = 0, dir = 0;
        while (index < len) {
            result[index++] = matrix[i][j];
            if (dir == 0 && j == right) {
                up += 1;
                dir = (dir + 1) % 4;
            }
            if (dir == 1 && i == down) {
                right -= 1;
                dir = (dir + 1) % 4;
            }
            if (dir == 2 && j == left) {
                down -= 1;
                dir = (dir + 1) % 4;
            }
            if (dir == 3 && i == up) {
                left += 1;
                dir = (dir + 1) % 4;
            }
            i += dir_arr[dir][0];
            j += dir_arr[dir][1];
        }
        return result;
    }
};
```

### [!]20. 包含 min 函数的栈

#### 题目描述

> 定义栈的数据结构, 请在该类型中实现一个能够得到栈中所含最小元素的 ```min``` 函数 (时间复杂度应为 $O(1)$)
>
> 注意: 保证测试中不会当栈为空的时候, 对栈调用 ```pop()``` 或者 ```min()``` 或者 ```top()``` 方法.

#### 解法一:

> TODO

```c++
class Solution {
private:
    stack<int> value_s;
    stack<int> min_s;
public:
    void push(int value) {
        value_s.push(value);
        if (min_s.empty() || min_s.top() >= value) {
            min_s.push(value);
        }
    }
    void pop() {
        if (value_s.top() == min_s.top()) {
            min_s.pop();
        }
        value_s.pop();
    }
    int top() {
        return value_s.top();
    }
    int min() {
        return min_s.top();
    }
};
```

### [!]21. 栈的压入弹出序列

#### 题目描述

> 输入两个整数序列, 第一个序列表示栈的压入顺序, 请判断第二个序列是否可能为该栈的弹出顺序. 假设压入栈的所有数字均不相等. 例如序列 1,2,3,4,5 是某栈的压入顺序, 序列 4,5,3,2,1 是该压栈序列对应的一个弹出序列, 但 4,3,5,1,2 就不可能是该压栈序列的弹出序列. (注意: 这两个序列的长度是相等的)

#### 解法一:

> TODO

```c++
class Solution {
public:
    bool IsPopOrder(vector<int> pushV,vector<int> popV) {
        if (pushV.size() == 0) return false;
        stack<int> s;
        for (int i = 0, j = 0; i < pushV.size(); ++i) {
            s.push(pushV[i]);
            while (j < popV.size() && !s.empty() && s.top() == popV[j]) {
                s.pop();
                j++;
            }
        }
        return s.empty();
    }
};
```

### 22. 从上往下打印二叉树

#### 题目描述

> 从上往下打印出二叉树的每个节点，同层节点从左至右打印。

#### 解法一: 使用队列

```c++
/*
struct TreeNode {
	int val;
	struct TreeNode *left;
	struct TreeNode *right;
	TreeNode(int x) :
			val(x), left(NULL), right(NULL) {
	}
};*/
class Solution {
public:
    vector<int> PrintFromTopToBottom(TreeNode* root) {
        queue<TreeNode*> q;
        q.push(root);
        vector<int> v;
        while (!q.empty()) {
            TreeNode* node = q.front();
            q.pop();
            if (node != NULL) {
                v.push_back(node->val);
                q.push(node->left);
                q.push(node->right);
            }
        }
        return v;
    }
};
```

### [!]23. 二叉搜索树的后序遍历序列

#### 题目描述

> 输入一个非空整数数组, 判断该数组是不是某二叉搜索树的后序遍历的结果. 如果是则输出 Yes, 否则输出 No. 假设输入的数组的任意两个数字都互不相同.

#### 解法一: 递归

> 二叉搜索树满足其左子树的所有值都小于根结点的值, 右子树的所有值都大于根结点的值, 而且它的子树也满足这样的性质. 由这个性质我们就可以递归的判断一个二叉树是不是二叉搜索树了. 而且后序遍历序列最后一个数一定是根结点, 前面的序列的为左子树的后序遍历序列和右子树的后序遍历序列. 由此就可以写出下面的递归的程序:

```c++
class Solution {
public:
    bool help(vector<int> v, int start, int end) {
      	// 递归结束条件, 当前的子树根结点是一个值或者为空
        if (start >= end) return true;
        int root = v[end], i = end - 1;
      	// 从后往前遍历, 找到左右子树的分界点, 
      	// 此时同时保证了右子树的所有值都大于根结点的值
        for (; i >= start; --i) {
            if (v[i] < root) break;
        }
      	// 判断左子树的所有值是否都小于根结点的值
        for (int j = start; j <= i; ++j) {
            if (v[j] > root) return false;
        }
      	// 递归判断其左子树和右子树是不是二叉搜索树
        return help(v, start, i) && help(v, i + 1, end - 1);
    }
    
    bool VerifySquenceOfBST(vector<int> sequence) {
        if (sequence.size() == 0) return false;
        return help(sequence, 0, sequence.size() - 1);
    }
};
```

#### 解法二: 非递归

> 上面的程序可以用栈来消除递归.

```c++
class Solution {
public:
    bool VerifySquenceOfBST(vector<int> sequence) {
        if (sequence.size() == 0) return false;
        if (sequence.size() == 1) return true;
        stack<pair<int, int> > s;
        s.push(make_pair(0, sequence.size() - 1));
        while (!s.empty()) {
            int start = s.top().first, end = s.top().second;
            s.pop();
            if (start >= end) continue;
            int i = start;
          	// 找到左子树的最后一个数
            while (sequence[i] < sequence[end]) {
                ++i;
            }
          	// 将左右子树放入堆栈中
            s.push(make_pair(start, i - 1));
            s.push(make_pair(i, end - 1));
          	// 判断右子树是不是满足右子树的所有值都大于根结点
            while (sequence[i] > sequence[end] && i < end) {
                ++i;
            }
          	// 不是则返回 false
            if (i < end) return false;
        }
        return true;
    }
};
```

### 24. 二叉树中和为某一值的路径

#### 题目描述

> 输入一颗二叉树的根节点和一个整数, 打印出二叉树中结点值的和为输入整数的所有路径. 路径定义为从树的根结点开始往下一直到叶结点所经过的结点形成一条路径.

#### 解法一: 深度优先遍历

```c++
/*
struct TreeNode {
	int val;
	struct TreeNode *left;
	struct TreeNode *right;
	TreeNode(int x) :
			val(x), left(NULL), right(NULL) {
	}
};*/
class Solution {
private:
    vector<vector<int> > result;
public:
    void help(TreeNode* root, int number, vector<int> path) {
        path.push_back(root->val);
        number -= root->val;
        if (root->left == NULL && root->right == NULL){
            if (number == 0) {
                result.push_back(path);
            }
            return;
        }
        if (root->left != NULL){
            help(root->left, number, path);
        }
        if (root->right != NULL) {
            help(root->right, number, path);
        }
    }
    
    vector<vector<int> > FindPath(TreeNode* root,int expectNumber) {
        if (root == NULL) return {};
        help(root, expectNumber, vector<int>());
        return result;
    }
};
```

### [!]25. 复杂链表的深拷贝

#### 题目描述

> 输入一个复杂链表 (每个节点中有节点值, 以及两个指针, 一个指向下一个节点, 另一个特殊指针 ```random``` 指向一个随机节点), 请对此链表进行深拷贝, 并返回拷贝后的头结点. (注意, 输出结果中请不要返回参数中的节点引用, 否则判题程序会直接返回空.)

#### 解法一:

> 复杂链表复制步骤:
>
> ![](https://i.loli.net/2020/05/11/yIjqTouZUG8aS6M.jpg)

```c++
/*
struct RandomListNode {
    int label;
    struct RandomListNode *next, *random;
    RandomListNode(int x) :
            label(x), next(NULL), random(NULL) {
    }
};
*/
class Solution {
public:
    RandomListNode* Clone(RandomListNode* pHead)
    {
        if (pHead == NULL) return NULL;
        RandomListNode* cur = pHead;
        // 复制结点
        while (cur != NULL) {
            RandomListNode* temp = new RandomListNode(cur->label);
            temp->next = cur->next;
            cur->next = temp;
            cur = temp->next;
        }
        cur = pHead;
      	// 复制 Random 指针的关系
        while (cur != NULL) {
            if (cur->random != NULL) {
                cur->next->random = cur->random->next;
            }
            cur = cur->next->next;
        }
      	// 拆分链表
        RandomListNode *cur1 = pHead, *head2 = pHead->next, *cur2 = head2;
        while (cur2->next != NULL){
            cur1->next = cur1->next->next;
            cur2->next = cur2->next->next;
            cur1 = cur1->next;
            cur2 = cur2->next;
        }
        cur1->next = NULL;
        return head2;
    }
};
```

### [!]26. 二叉搜索树与双向链表

#### 题目描述

> 输入一棵二叉搜索树, 将该二叉搜索树转换成一个排序的双向链表. 要求不能创建任何新的结点, 只能调整树中结点指针的指向.

#### 解法一: 堆栈

> TODO

```c++
/*
struct TreeNode {
	int val;
	struct TreeNode *left;
	struct TreeNode *right;
	TreeNode(int x) :
			val(x), left(NULL), right(NULL) {
	}
};*/
class Solution {
public:
    TreeNode* Convert(TreeNode* pRootOfTree){
        if (pRootOfTree == NULL) return NULL;
        stack<TreeNode*> s;
        TreeNode *head = NULL, *cur, *temp;
        s.push(pRootOfTree);
        while(!s.empty()) {
            while (s.top()->left != NULL) {
                temp = s.top()->left;
                s.top()->left = NULL;
                s.push(temp);
            }
            if (head == NULL) {
                head = s.top();
                cur = s.top();
            } else {
                cur->right = s.top();
                s.top()->left = cur;
                cur = cur->right;
            }
            s.pop();
            if (cur->right != NULL) {
                s.push(cur->right);
            }
        }
        return head;
    }
};
```

### [!]26. 字符串的排列

#### 题目描述

> 输入一个字符串, 按字典序打印出该字符串中字符的所有排列. 例如输入字符串 abc, 则打印出由字符 a, b, c 所能排列出来的所有字符串 abc, acb, bac, bca, cab 和 cba.
> 
> **输入描述**
> 
> 输入一个字符串, 长度不超过9(可能有字符重复), 字符只包括大小写字母。

#### 解法一:  非递归方法

> 一般而言, 设 $P$ 是 $[1,n]$ 的一个全排列, 下面是求字典序下一个排列的算法:
> $$P=P_1P_2…P_n=P_1P_2…P_{j-1}P_jP_{j+1}…P_{k-1}P_kP_{k+1}…P_n$$

```txt
Find:
j=max{i|P[i] < P[i+1]}
k=max{i|P[i] > P[j]}
1, 对换P[j]，P[k]，
2, 将 P[j+1], …, P[k-1], P[j], P[k+1], …, P[n] 翻转
```
> $P’= P_1P_2…P_{j-1}P_kP_n…P_{k+1}P_jP_{k-1}…P_{j+1}$ 即 $P$ 的下一个
>
> 代码如下:

```c++
class Solution {
public:
    bool NextPermutation(string &str) {
        int len = str.length(), i = len - 1;
      	// 找到满足 str[i] < str[i+1] 的最大的 i
      	// 记为 index1
        for (; i > 0; --i) {
            if (str[i - 1] < str[i]) break;
        }
      	// 如果应已经是降序排列了, 则这已经是最后一个排列了
        if (i == 0) return false;
        int index1 = i - 1;
      	// 寻找 index1 之后满足 str[i] <= str[index1] 最大的 i, 记为 index2
        for (; i < len; ++i) {
            if (str[i] <= str[index1]) break;
        }
        int index2 = i - 1;
      	// 交换 str[index1] 和 str[index2]
        char temp = str[index1];
        str[index1] = str[index2];
        str[index2] = temp;
      	// 倒序 index1 之后串
        for (int i = index1 + 1, j = len - 1; i < j; ++i, --j) {
            char temp = str[i];
            str[i] = str[j];
            str[j] = temp;
        }
        return true;
    }
    
    vector<string> Permutation(string str) {
        if (str.length() == 0) return {};
        vector<string> v;
        sort(str.begin(), str.end());
        v.push_back(str);
        while (NextPermutation(str)) {
            v.push_back(str);
        }
        return v;
    }
};
```

### [!]28. 数组中出现超过一半的数字

#### 题目描述

>数组中有一个数字出现的次数超过数组长度的一半, 请找出这个数字. 例如输入一个长度为 9 的数组{1,2,3,2,2,2,5,4,2}. 由于数字 2 在数组中出现了 5 次, 超过数组长度的一半, 因此输出2. 如果不存在则输出0.

#### 解法一: 排序

> 如果该数组中有数字出现超过其长度的一半, 那么排序后数组的中间那个数(length / 2 处)一定是出现次数超过一半的数字. 所以我们可以将数组排序, 然后再去验证中间那个数字的次数是不是超过了一半. 由于要排序, 所以时间复杂度为 $O(nlogn)$

```c++
class Solution {
public:
    int MoreThanHalfNum_Solution(vector<int> numbers) {
        int len = numbers.size();
        if (len == 0) return 0;
        if (len == 1) return numbers[0];
        sort(numbers.begin(), numbers.end());
        int temp = numbers[len / 2], count = 0;
        for (int i:numbers) {
            if (i == temp) count++;
        }
        if (count > len / 2) return temp;
        return 0;
    }
};
```

#### 解法二:

> 如果数组中有出现次数超过一半的数字, 那我们将两个不同的数字两两消除, 最后剩余所有的数字都是一样的, 剩余的数一定是那个数字. 所以可以从数组开始遍历, 设置一个计数器, 初始值为0. 如果计数器为零则将当前数字记录下来, 计数器加一; 如果计数器不是零, 则看记录的数字和当前数字是否一样,如果一样, 则计数器加一; 如果不一样, 则将计数器减一. 最后同样再验证记录的那个数字的次数是不是超过了一半. 时间复杂度为 $O(n)$.

```c++
class Solution {
public:
    int MoreThanHalfNum_Solution(vector<int> numbers) {
        int len = numbers.size();
        if (len == 0) return 0;
        if (len == 1) return numbers[0];
        int count = 0, rec = 0;
        for (int i:numbers) {
            if (count == 0) {
                rec = i;
                count ++;
            } else {
                if (i == rec) {
                    count++;
                } else {
                    count--;
                }
            }
        }
        count = 0;
        for (int i:numbers) {
            if (i == rec) count++;
        }
        if (count > len / 2) return rec;
        return 0;
    }
};
```

### [!]29. 最小的 k 个数

#### 题目描述

> 输入n个整数, 找出其中最小的K个数. 例如输入 4,5,1,6,2,7,3,8 这 8 个数字, 则最小的 4 个数字是 1,2,3,4.

#### 解法一: 选择排序或冒泡排序

> 利用冒泡或者选择排序, 排出前 k 项即可. 
>
> 复杂度为 $O(k \times (n + (n-1)+ \cdots +(n-k+1)))=O(nk^2)$

```c++
class Solution {
public:
    vector<int> GetLeastNumbers_Solution(vector<int> input, int k) {
        if (k > input.size()) return {};
        vector<int> result;
        for (int i = 0; i < k; ++i) {
            int temp = input[i], index = i;
            for (int j = i + 1; j < input.size(); ++j) {
                if (input[j] < temp) {
                    temp = input[j];
                    index = j;
                }
            }
            input[index] = input[i];
            input[i] = temp;
            result.push_back(temp);
        }
        return result;
    }
};
```

#### 解法二:

> 排序, 然后返回最小的 k 个数. 复杂度 $O(n \log n)$

```c++
class Solution {
public:
    vector<int> GetLeastNumbers_Solution(vector<int> input, int k) {
        if (k > input.size()) return {};
        if (k == 0 || input.size() == 0) return {};
        vector<int> result;
        sort(input.begin(), input.end());
        for (int i = 0; i < k; ++i) {
            result.push_back(input[i]);
        }
        return result;
    }
};
```

#### 解法三: 利用大顶堆

> 利用前 k 个数构造大顶堆, 然后遍历后面的数, 如果小于堆顶则弹出堆顶, 插入这个数. 结束后, 大顶堆里的 k 个数即为最小的 k 个数了, 而且当数据很多无法一次加载进内存的时候也可以使用这个方法. 复杂度为 $O(n \log k)$

```c++
class Solution {
public:
    vector<int> GetLeastNumbers_Solution(vector<int> input, int k) {
        if (k > input.size()) return {};
        if (k == 0 || input.size() == 0) return {};
        vector<int> result;
        priority_queue<int, vector<int> > pq;
        for (int i = 0; i < k; ++i) {
             pq.push(input[i]);
        }
        for (int i = k; i < input.size(); ++i) {
            if (input[i] < pq.top()) {
                pq.pop();
                pq.push(input[i]);
            }
        }
        while (!pq.empty()) {
            result.push_back(pq.top());
            pq.pop();
        }
        return result;
    }
};
```

### 30. 连续子数列最大和

#### 题目描述

> HZ 偶尔会拿些专业问题来忽悠那些非计算机专业的同学. 今天测试组开完会后, 他又发话了: 在古老的一维模式识别中, 常常需要计算连续子向量的最大和, 当向量全为正数的时候, 问题很好解决. 但是, 如果向量中包含负数, 是否应该包含某个负数, 并期望旁边的正数会弥补它呢? 例如: {6,-3,-2,7,-15,1,2,2}, 连续子向量的最大和为 8 (从第 0 个开始, 到第 3 个为止). 给一个数组, 返回它的最大连续子序列的和, 你会不会被他忽悠住? (子向量的长度至少是 1)

#### 解法一: 动态规划

> 如果我们用 $dp[i]$ 表示包含数组中第 $i$ 个数的最大连续子序列的和, 那么我们可以得到:
> $$
> dp[i+1]=\left\{
> \begin{array}\\
> dp[i]+array[i+1], dp[i] \ge 0\\
> array[i+1], dp[i] \lt 0
> \end{array}
> \right.
> $$
> 在写代码的时候并不需要一个 dp 数组, 我们只需要记录上一的值就行了.

```c++
class Solution {
public:
    int FindGreatestSumOfSubArray(vector<int> array) {
        if (array.size() == 0) return 0;
        int dp = array[0], result = array[0];
        for (int i = 1; i < array.size(); ++i) {
            if (dp >= 0) {
                dp = dp + array[i];
            } else {
                dp = array[i];
            }
            result = max(dp, result);
        }
        return result;
    }
};
```



### 31. 从 1 到 n 中 1 出现的次数

#### 题目描述

> ACMer 想求出 1\~13 的整数中 1 出现的次数, 并算出 100\~1300 的整数中 1 出现的次数. 为此他特别数了一下1\~13 中包含1的数字有 1、10、11、12、13 因此共出现 6 次, 但是对于后面问题他就没辙了. ACMer 希望你们帮帮他, 并把问题更加普遍化, 可以很快的求出任意非负整数区间中 1 出现的次数 ( 从 1 到 n 中 1 出现的次数).

#### 解法一:

```c++
class Solution {
public:
    int CountOne(int n) {
        int count = 0;
        while (n > 0) {
            if (n % 10 == 1) count ++;
            n = n / 10;
        }
        return count;
    }
    
    int NumberOf1Between1AndN_Solution(int n){
        int count = 0;
        for (int i = 1; i <=n; ++i) {
            count += CountOne(i);
        }
        return count;
    }
};
```

### 32. 把数组排成最小的数

#### 题目描述:

> 输入一个正整数数组, 把数组里所有数字拼接起来排成一个数, 打印能拼接出的所有数字中最小的一个. 例如输入数组 {3, 32, 321}, 则打印出这三个数字能排成的最小数字为 321323.

#### 解法一:

> 其实就是将原数组排序后拼接起来, 比较大小的规则就是两个数 a, b, 如果 ab < ba 则表示 a 要排在前面, 反之则相反.

```c++
class Solution {
public:
    static bool compare(int a, int b) {
        string s1 = to_string(a) + to_string(b), s2 = to_string(b) + to_string(a);
        return s1 < s2;
    }

    string PrintMinNumber(vector<int> numbers) {
        sort(numbers.begin(), numbers.end(), compare);
        string result = "";
        for (int i:numbers) {
            result = result + to_string(i);
        }
        return result;
    }
};
```

### [!]33. 丑数

#### 题目描述

> 把只包含质因子 2, 3 和 5 的数称作丑数 (Ugly Number). 例如 6, 8 都是丑数, 但 14 不是, 因为它包含质因子 7. 习惯上我们把 1 当做是第一个丑数, 求按从小到大的顺序的第 N 个丑数.

#### 解法一:

> 首先从丑数的定义我们知道，一个丑数的因子只有2,3,5，那么丑数$p = 2 ^ x * 3 ^ y * 5 ^ z$，换句话说一个丑数一定由另一个丑数乘以2或者乘以3或者乘以5得到，那么我们从1开始乘以2,3,5，就得到2,3,5三个丑数，在从这三个丑数出发乘以2,3,5就得到4，6,10,6，9,15,10,15,25九个丑数，我们发现这种方法得到重复的丑数，而且我们题目要求第N个丑数，这样的方法得到的丑数也是无序的。那么我们可以维护三个队列: 
>
>    **（1）丑数数组： 1**  
>
>    **乘以2的队列：2**  
>
>    **乘以3的队列：3**  
>
>    **乘以5的队列：5**  
>
>    **选择三个队列头最小的数2加入丑数数组，同时将该最小的数乘以****2,3,5****放入三个队列；**  
>
>    **（2）丑数数组：1,2**  
>
> ​    **乘以2的队列：4**   
>
> ​    **乘以3的队列：3，6**   
>
> ​    **乘以5的队列：5，10**   
>
> ​    **选择三个队列头最小的数3加入丑数数组，同时将该最小的数乘以****2,3,5****放入三个队列；**   
>
> ​    **（3）丑数数组：1,2,3**   
>
> ​    **乘以2的队列：4,6**   
>
> ​    **乘以3的队列：6,9**   
>
> ​    **乘以5的队列：5,10,15**   
>
> ​    **选择三个队列头里最小的数4加入丑数数组，同时将该最小的数乘以****2,3,5****放入三个队列；**   
>
> ​    **（4）丑数数组：1,2,3,4**   
>
> ​    **乘以2的队列：6，8**   
>
> ​    **乘以3的队列：6,9,12**   
>
> ​    **乘以5的队列：5,10,15,20**   
>
> ​    **选择三个队列头里最小的数5加入丑数数组，同时将该最小的数乘以****2,3,5****放入三个队列；**   
>
> ​    **（5）丑数数组：1,2,3,4,5**   
>
> ​    **乘以2的队列：6,8,10，**   
>
> ​    **乘以3的队列：6,9,12,15**   
>
> ​    **乘以5的队列：10,15,20,25**   
>
> ​    **选择三个队列头里最小的数6加入丑数数组，但我们发现，有两个队列头都为6，所以我们弹出两个队列头，同时将12,18,30放入三个队列；**   
>
> ​    **……………………**   
>
> ​    **疑问：**   
>
> ​    **1.为什么分三个队列？**   
>
> ​    **丑数数组里的数一定是有序的，因为我们是从丑数数组里的数乘以2,3,5选出的最小数，一定比以前未乘以2,3,5大，同时对于三个队列内部，按先后顺序乘以2,3,5分别放入，所以同一个队列内部也是有序的；**   
>
> ​    **2.为什么比较三个队列头部最小的数放入丑数数组？**   
>
> ​    **因为三个队列是有序的，所以取出三个头中最小的，等同于找到了三个队列所有数中最小的。**   
>
> ​    **实现思路：**   
>
> ​    **我们没有必要维护三个队列，只需要记录三个指针显示到达哪一步；“|”表示指针,arr表示丑数数组；**   
>
> ​    **（1）1**   
>
> ​    **|2**   
>
> ​    **|3**   
>
> ​    **|5**   
>
> ​    **目前指针指向0,0,0，队列头arr[0] \* 2 = 2, arr[0] \* 3 = 3, arr[0] \* 5 = 5**   
>
> ​    **（2）1 2**   
>
> ​    **2 |4**   
>
> ​    **|3 6**   
>
> ​    **|5 10**   
>
> ​    **目前指针指向1,0,0，队列头arr[1] \* 2 = 4, arr[0] \* 3 = 3, arr[0] \* 5 = 5**   
>
> ​    **（3）1 2 3**   
>
> ​    **2| 4 6**   
>
> ​    **3 |6 9**    
>
> ​    **|5 10 15**   
>
>    **目前指针指向1,1,0，队列头arr[1] \* 2 = 4, arr[1] \* 3 = 6, arr[0] \* 5 = 5**  
>
>    **………………**

```c++
class Solution {
public:
    int GetUglyNumber_Solution(int index) {
        if (index < 7) return index;
        int p2 = 0, p3 = 0, p5 = 0, result = 1;
        vector<int> v;
        v.push_back(result);
        while (v.size() < index) {
            result = min(v[p2] * 2, min(v[p3] * 3, v[p5] * 5));
            if (result == v[p2] * 2) ++p2;
            if (result == v[p3] * 3) ++p3;
            if (result == v[p5] * 5) ++p5;
            v.push_back(result);
        }
        return result;
    }
};
```

### 34. 第一个只出现一次的字符

#### 题目描述

> 在一个字符串 (0 <= 字符串长度 <= 10000, 全部由字母组成) 中找到第一个只出现一次的字符, 并返回它的位置, 如果没有则返回 -1(需要区分大小写). (从 0 开始计数).

#### 解法一: HashMap

> 用 HashMap 统计每个字符出现的次数以及第一次出现的位置, 然后遍历 HashMap 找出第一个只出现一次的字符.

```c++
class Solution {
public:
    int FirstNotRepeatingChar(string str) {
        if (str.length() == 0) return -1;
        unordered_map<char, int> m1, m2;
        for (int i = 0; i < str.size(); ++i) {
            m1[str[i]]++;
            if (m1[str[i]] == 1) m2[str[i]] = i;
        }
        int result = 10001;
        for (auto p:m1) {
            if (p.second == 1) {
                result = min(result, m2[p.first]);
            }
        }
        return result == 10001 ? -1:result;
    }
};
```

### [!]35. 数组中的逆序对

#### 题目描述

> 在数组中的两个数字, 如果前面一个数字大于后面的数字, 则这两个数字组成一个逆序对. 输入一个数组, 求出这个数组中的逆序对的总数P. 并将 P 对 1000000007 取模的结果输出, 即输出 P%1000000007.
>
> **输入描述:**
>
> > 题目保证输入的数组中没有的相同的数字
> >
> > 数据范围:
> >
> > ​	对于 50% 的数据, size <= 10^4
> >
> > ​	对于 75% 的数据, size <= 10^5
> >
> > ​	对于 100% 的数据, size <= 2 * 10^5
>
> **示例1**
>
> > 输入:
> >
> > ```1, 2, 3, 4, 5, 6, 7, 0```
> >
> > 输出:
> >
> > ```7```

#### 解法一: 利用归并排序统计逆序对的数量

```c++
class Solution {
public:
    typedef long long ll;
    ll mod = 1000000007;
    int InversePairs(vector<int> data) {
        ll len = data.size();
        int result = 0;
        for (ll step = 1; step < len;) {
            ll group = ceil((float)len / (step * 2));
            for (ll j = 0; j < group; ++j) {
                ll s = j * (step * 2), s1 = s, mid = min(s + step, len), s2 = mid, end = min(mid + step, len);
                vector<int> temp(end - s);
                ll index = 0;
                while (s1 < mid && s2 < end) {
                    if (data[s1] > data[s2]) {
                        temp[index++] = data[s2++];
                        result = (result + (mid - s1)) % mod;
                    } else {
                        temp[index++] = data[s1++];
                    }
                }
                while(s1 < mid) {
                    temp[index++] = data[s1++];
                }
                while(s2 < end) {
                    temp[index++] = data[s2++];
                }
                for (int i : temp) {
                    data[s++] = i;
                }
            }
            step <<= 1;
        }
        return result;
    }
};
```

### [!]36. 两个链表的第一个公共结点

#### 题目描述

> 输入两个链表, 找出它们的第一个公共结点. (注意因为传入数据是链表, 所以错误测试数据的提示是用其他方式显示的, 保证传入数据是正确的)

#### 解法一: 双指针

> 首先要清楚的一点是两个单链表如果有公共结点, 那么公共结点之后的部分两个链表一定是共用的. 那么我们只需要先求出两个链表的长度差, 然后设置两个指针分别指向两个链表的头, 让长链表的指针先走长度差个结点, 然后两个指针一起向前走, 直到两个指针指向同一个结点, 该结点就是两个链表的公共结点.

```c++
/*
struct ListNode {
	int val;
	struct ListNode *next;
	ListNode(int x) :
			val(x), next(NULL) {
	}
};*/
class Solution {
public:
    ListNode* FindFirstCommonNode( ListNode* pHead1, ListNode* pHead2) {
        int l1 = 0, l2 = 0;
        ListNode *cur = pHead1;
        while (cur != NULL) {
            cur = cur->next;
            l1++;
        }
        cur = pHead2;
        while (cur != NULL) {
            cur = cur->next;
            l2++;
        }
        if (l1 == 0 || l2 == 0) return NULL;
        ListNode *node1, *node2;
        int diff;
        if (l1 > l2) {
            diff = l1 - l2;
            node1 = pHead1;
            node2 = pHead2;
        } else {
            diff = l2 - l1;
            node1 = pHead2;
            node2 = pHead1;
        }
        for (int i = 0; i < diff; ++i) {
            node1 = node1->next;
        }
        while (node1 != NULL && node2!= NULL && node1 != node2) {
            node1 = node1->next;
            node2 = node2->next;
        }
        return node1;
    }
};
```

### 解法二: 双指针

> TODO

```c++
/*
struct ListNode {
	int val;
	struct ListNode *next;
	ListNode(int x) :
			val(x), next(NULL) {
	}
};*/
class Solution {
public:
    ListNode* FindFirstCommonNode( ListNode* pHead1, ListNode* pHead2) {
        ListNode *p1 = pHead1, *p2 = pHead2;
        while (p1 != p2) {
            p1 = p1 == NULL ? pHead2:p1->next;
            p2 = p2 == NULL ? pHead1:p2->next;
        }
        return p1;
    }
};
```

### 37. 数字在排序数组中出现的次数

#### 题目描述

>  统计一个数字在排序数组中出现的次数.

#### 解法一: 遍历

```c++
class Solution {
public:
    int GetNumberOfK(vector<int> data ,int k) {
        int count = 0;
        for (int i = 0; i < data.size(); ++i) {
            if (data[i] == k) count++;
        }
        return count;
    }
};
```

#### 解法二: 二分查找

```c++
class Solution {
public:
    int binary_search(vector<int> data, int target) {
        int l = 0, r = data.size() - 1;
        while (l <= r) {
            int mid = (l + r) / 2;
            if (data[mid] == target) return mid;
            if (data[mid] > target) {
                r = mid - 1;
            } else {
                l = mid + 1;
            }
        }
        return -1;
    }
    int GetNumberOfK(vector<int> data ,int k) {
        int count = 0, index = binary_search(data, k);
        if (index == -1) return count;
        for (int i = index; i < data.size();++i) {
            if (data[i] == k) count++;
            else break;
        }
        for (int i = index - 1; i >= 0; --i) {
            if (data[i] == k) count++;
            else break;
        }
        return count;
    }
};
```

#### 解法三: 二分查找两个边界

```c++
class Solution {
public:
    int binary_search(vector<int> data, float target) {
        int l = 0, r = data.size() - 1;
        while (l <= r) {
            int mid = (l + r) / 2;
            if (data[mid] > target) {
                r = mid - 1;
            } else {
                l = mid + 1;
            }
        }
        return l;
    }
    int GetNumberOfK(vector<int> data ,int k) {
        int l = binary_search(data, k - 0.5), r = binary_search(data, k + 0.5);
        return r - l;
    }
};
```

### 38. 二叉树的深度

#### 题目描述

> 输入一棵二叉树, 求该树的深度. 从根结点到叶结点依次经过的结点(含根, 叶结点)形成树的一条路径, 最长路径的长度为树的深度.

#### 解法一: 递归

```c++
/*
struct TreeNode {
	int val;
	struct TreeNode *left;
	struct TreeNode *right;
	TreeNode(int x) :
			val(x), left(NULL), right(NULL) {
	}
};*/
class Solution {
public:
    int TreeDepth(TreeNode* pRoot){
        if (pRoot == NULL) return 0;
        return max(TreeDepth(pRoot->left), TreeDepth(pRoot->right)) + 1;;
    }
};
```

### [!]39. 平衡二叉树

#### 题目描述

> 输入一棵二叉树, 判断该二叉树是否是平衡二叉树. 在这里, 我们只需要考虑其平衡性, 不需要考虑其是不是排序二叉树.

#### 解法一:

```c++
class Solution {
public:
    bool help(TreeNode* pRoot, int &depth) {
        if (pRoot == NULL) return true;
        int depthLeft = depth + 1;
        if (!help(pRoot->left, depthLeft)) {
            return false;
        }
        int depthRight = depth + 1;
        if (!help(pRoot->right, depthRight)) {
            return false;
        }
        if (abs(depthLeft - depthRight) > 1) {
            return false;
        }
        depth = max(depthLeft, depthRight);
        return true;
    }
    
    bool IsBalanced_Solution(TreeNode* pRoot) {
        int depth = 1;
        return help(pRoot, depth);
    }
};
```

### [!]40. 数组中只出现一次的数字

#### 题目描述

> 一个整型数组里除了两个数字之外, 其他的数字都出现了两次. 请写程序找出这两个只出现一次的数字.

#### 解法一:

> 遍历数组, 维护一个集合, 当集合中没有当前的数字的时候则添加进去, 如果已经有了当前的数字, 则移除里面的这个数字, 最后集合里面剩下的就是要找的两个数字了, 这种方法 时间复杂度为 $O(n)$, 空间复杂度也为 $O(n)$. 这种方法代码就不写了.

#### 解法二:

> 我们知道位运算中异或的性质是两个相同数字异或为0, 一个数和 0 异或还是它本身. 所以当只有一个数出现一次时, 其他数都出现两次, 我们把数组中所有的数依次异或运算, 最后剩下的就是落单的那个数. 因为成对儿出现的都抵消了. 但是现在有两个都只出现了一次. 我们首先还是先异或, 剩下的数字肯定是 A, B 异或的结果, 这个结果的二进制中的 1，表现的是 A 和 B 的不同的位. 我们就取(从低位到高位)第一个 1 所在的位数记为 i, 接着把原数组分成两组, 分组标准是第 i 位是否为 1. 这样, 相同的数肯定在一个组, 因为相同数字所有位都相同, 第 i 位肯定也一样. 而不同的数, 肯定不在一组, 即 A, B 被分在了不同的组, 然后把这两个组按照最开始的思路, 依次异或, 剩余的两个结果就是这两个只出现一次的数字.
>
> 这个方法的时间复杂度为 $O(n)$, 空间复杂度为 $O(1)$.

```c++
class Solution {
public:
    void FindNumsAppearOnce(vector<int> data,int* num1,int *num2) {
        int xor0 = 0, index = 0;
        for (int i:data) xor0 ^= i;
        while (!((xor0 >> index) & 1)) {
            index++;
        }
        int xor1 = 0, xor2 = 0;
        for (int i:data) {
            if ((i >> index) & 1) {
                xor1 ^= i;
            } else {
                xor2 ^= i;
            }
        }
        num1[0] = xor1;
        num2[0] = xor2;
    }
};
```

### [!]41. 和为 S 的连续正数序列

#### 题目描述

> 小明很喜欢数学, 有一天他在做数学作业时, 要求计算出 9~16 的和, 他马上就写出了正确答案是100. 但是他并不满足于此, 他在想究竟有多少种连续的正数序列的和为 100 (至少包括两个数). 没多久, 他就得到另一组连续正数和为 100 的序列: 18, 19, 20, 21, 22. 现在把问题交给你, 你能不能也很快的找出所有和为 S 的连续正数序列? Good Luck!
>
> **输出描述:**
>
> > 输出所有和为 S 的连续正数序列. 序列内按照从小至大的顺序, 序列间按照开始数字从小到大的顺序.

#### 解法一:

```c++
class Solution {
public:
    vector<vector<int> > FindContinuousSequence(int sum) {
        int i = 1, j = 1, temp = 0;
        vector<vector<int> > result;
        while (j < sum) {
            while (temp + j > sum) {
                temp = temp - i;
                i++;
            }
            temp = temp + j;
            j++;
            if (temp == sum) {
                vector<int> v;
                for (int k = i; k < j; k++) {
                    v.push_back(k);
                }
                result.push_back(v);
            }
        }
        return result;
    }
};
```

### 42. 和为 S 的两个数字

#### 题目描述

> 输入一个递增排序的数组和一个数字 S, 在数组中查找两个数, 使得他们的和正好是 S, 如果有多对数字的和等于 S, 输出两个数的乘积最小的.
>
> 输出描述:
>
> > 对应每个测试案例, 输出两个数, 小的先输出.

#### 解法一: 二分查找

```c++
class Solution {
public:
    int binary_search(vector<int> array, int target) {
        int l = 0, r = array.size() -1;
        while (l <= r) {
            int mid = (l + r) / 2;
            if (array[mid] == target) return mid;
            if (array[mid] > target) {
                r = mid - 1;
            } else {
                l = mid + 1;
            }
        }
        return -1;
    }
    
    vector<int> FindNumbersWithSum(vector<int> array,int sum) {
        vector<int> result;
        for (int i = 0; i < array.size() && array[i] <= sum - array[i]; ++i) {
            if (binary_search(array, sum - array[i]) != -1) {
                result.push_back(array[i]);
                result.push_back(sum - array[i]);
                break;
            }
        }
        return result;
    }
};
```

#### [!]解法二: 双指针遍历

```c++
class Solution {
public:    
    vector<int> FindNumbersWithSum(vector<int> array,int sum) {
        vector<int> result;
        int i = 0, j = array.size() - 1;
        while (i <= j) {
            if (array[i] + array[j] == sum) {
                result.push_back(array[i]);
                result.push_back(array[j]);
                break;
            }
            if (array[i] + array[j] > sum) {
                j--;
            } else {
                i++;
            }
        }
        return result;
    }
};
```

### 43. 左旋转字符串

#### 题目描述

> 汇编语言中有一种移位指令叫做循环左移(ROL), 现在有个简单的任务, 就是用字符串模拟这个指令的运算结果. 对于一个给定的字符序列 S, 请你把其循环左移K位后的序列输出. 例如, 字符序列 S=”abcXYZdef”, 要求输出循环左移3位后的结果, 即 “XYZdefabc”. 是不是很简单? OK, 搞定它!

#### 解法一:

```c++
class Solution {
public:
    string LeftRotateString(string str, int n) {
        int len = str.length();
        if (len == 0) return str;
        n = n % len;
        if (n == 0) return str;
        return str.substr(n, len - n) + str.substr(0, n);
    }
};
```

#### [!]解法二: 三次翻转字符串

> $XY=(Y^TX^T)^T$

```c++
class Solution {
public:
    string LeftRotateString(string str, int n) {
        if (str.length() == 0) return str;
        n = n % str.length();
        reverse(str.begin(), str.begin() + n);
        reverse(str.begin() + n, str.end());
        reverse(str.begin(), str.end());
        return str;
    }
};
```

### 44. 翻转单词顺序列

#### 题目描述

> 牛客最近来了一个新员工Fish, 每天早晨总是会拿着一本英文杂志, 写些句子在本子上. 同事 Cat 对 Fish 写的内容颇感兴趣, 有一天他向 Fish 借来翻看, 但却读不懂它的意思. 例如, "student. a am I". 后来才意识到, 这家伙原来把句子单词的顺序翻转了, 正确的句子应该是 "I am a student.". Cat 对一一的翻转这些单词顺序可不在行, 你能帮助他么?

#### 解法一: 先拆分再组装

```c++
class Solution {
public:
    string ReverseSentence(string str) {
        vector<string> v;
        int i = 0, j = 0;
        for (; i < str.length(); ++i) {
            if (str[i] == ' ') {
                v.push_back(str.substr(j, i - j));
                j = i + 1;
            }
        }
        v.push_back(str.substr(j, i - j));
        string result = "";
        if (v.size() == 0) return result;
        for (int i = v.size() - 1; i > 0; --i) {
            result = result + v[i] + " ";
        }
        result = result + v[0];
        return result;
    }
};
```

#### [!]解法二: 翻转再翻转

> 遍历整个字符串, 先将里面的每个单词翻转, 然后将整个字符串翻转.

```c++
class Solution {
public:
    string ReverseSentence(string str) {
        vector<string> v;
        int i = 0, j = 0;
        for (; i < str.length(); ++i) {
            if (str[i] == ' ') {
                reverse(str.begin() + j, str.begin() + i);
                j = i + 1;
            }
        }
        reverse(str.begin() + j, str.begin() + i);
        reverse(str.begin(), str.end());
        return str;
    }
};
```

### 45. 扑克牌顺子

#### 题目描述

> LL 今天心情特别好, 因为他去买了一副扑克牌, 发现里面居然有 2 个大王, 2 个小王(一副牌原本是 54 张) ... 他随机从中抽出了 5 张牌, 想测测自己的手气, 看看能不能抽到顺子, 如果抽到的话,他决定去买体育彩票, 嘿嘿!! "红心 A, 黑桃 3, 小王, 大王, 方片 5", "Oh My God!" 不是顺子 ..... LL 不高兴了, 他想了想, 决定大/小王可以看成任何数字, 并且 A 看作 1, J 为 11, Q 为 12, K 为 13. 上面的 5 张牌就可以变成 "1, 2, 3, 4, 5"(大小王分别看作 2 和 4), "So Lucky!". LL 决定去买体育彩票啦. 现在, 要求你使用这幅牌模拟上面的过程, 然后告诉我们 LL 的运气如何, 如果牌能组成顺子就输出 true, 否则就输出 false. 为了方便起见, 你可以认为大小王是0.

#### 解法一:

> 排序, 然后从不是 0 的地方遍历, 如果是连续的就往后遍历, 如果不连续则使用王代替, 最终看是不是能够组成顺子.

```c++
class Solution {
public:
    bool IsContinuous( vector<int> numbers ) {
        if (numbers.size() < 5) return false;
        sort(numbers.begin(), numbers.end());
        int s, joker = 0, index = 0;
        for (; index < numbers.size(); ++index) {
            if (numbers[index] == 0) joker++;
            if (numbers[index] != 0) {
                s = numbers[index];
                break;
            }
        }
        for (int i = index + 1; i < numbers.size();) {
            if (numbers[i] == s + 1) {
                ++i;
                s++;
            }
            else {
                if (joker > 0) {
                    joker--;
                    s++;
                } else {
                    return false;
                }
            }
        }
        return true;
    }
};
```

#### 解法二:

> 满足条件:
>
> 1. max - min < 5
> 2. 除0外没有重复的数字(牌)
> 3. 数组长度为5

```c++
class Solution {
public:
    bool IsContinuous( vector<int> numbers ) {
        if (numbers.size() < 5) return false;
        int low = 14, high = 0;
        vector<int> memo(14, 0);
        for (int i:numbers) {
            if (i == 0) continue;
            memo[i]++;
            if (memo[i] > 1) return false;
            low = min(low, i);
            high = max(high, i);
        }
        return high - low < 5;
    }
};
```

### 46. 孩子们的游戏(圆圈中最后剩下的数)

#### 题目描述

> 每年六一儿童节,牛客都会准备一些小礼物去看望孤儿院的小朋友,今年亦是如此。HF作为牛客的资深元老,自然也准备了一些小游戏。其中,有个游戏是这样的:首先,让小朋友们围成一个大圈。然后,他随机指定一个数m,让编号为0的小朋友开始报数。每次喊到m-1的那个小朋友要出列唱首歌,然后可以在礼品箱中任意的挑选礼物,并且不再回到圈中,从他的下一个小朋友开始,继续0...m-1报数....这样下去....直到剩下最后一个小朋友,可以不用表演,并且拿到牛客名贵的“名侦探柯南”典藏版(名额有限哦!!^_^)。请你试着想下,哪个小朋友会得到这份礼品呢？(注：小朋友的编号是从0到n-1)
> 如果没有小朋友，请返回-1

#### 解法一:

> 使用一个循环链表模拟上面这个过程, 直到链表里只有一个结点就结束.

```c++
typedef struct Node{
    int val;
    struct Node* next;
    Node(int x):val(x), next(NULL) {}
} Node;

class Solution {
public:
    int LastRemaining_Solution(int n, int m){
        if (n == 0 || m == 0) return -1;
        Node *head = new Node(0), *cur = head, *pre;
        for (int i = 0; i < n; ++i) {
            cur->next = new Node(i);
            cur = cur->next;
        }
        head = head->next;
        cur->next = head;
        pre = cur;
        cur = cur->next;
        int index = 0;
        while (cur != cur->next) {
            if (index == m - 1) {
                pre->next = cur->next;
                cur = cur->next;
                index = 0;
            } else {
                index++;
                pre = pre->next;
                cur = cur->next;
            }
            
        }
        return cur->val;
    }
};
```

#### 解法二: 数学大法好

> 第一个删除的人的编号是 $k=(m-1)\mod n$, 那么剩余的编号就是:
>
> $\{1, 2, 3, \cdots ,k-1, k+1, \cdots , n-1\}$
>
> 如果我们用 $f(n,m)$ 表示最终结果, $q(n-1, m)$ 表示序列 $\{k+1, k+2, \cdots ,n-1, 0, 1, \cdots , k - 1\}$ 的最终结果, 则 $f(n,m)=q(n-1, m)$.
>
> 如果我们将序列 $\{k+1, k+2, \cdots ,n-1, 0, 1, \cdots , k - 1\}$ 转换成序列 $\{0, 1, 2, \cdots ,n-2\}$ 那么将得到的序号转换回去就能知道下一个人的标号是多少了. 转换方式如下:
> $$
> k+1 \rightarrow 0 \\
> k+2 \rightarrow 1 \\
> ... \\
> k-1 \rightarrow n-2 \\
> $$
> 可以得出转换函数为 $p(x)=(x-k-1) \mod n$, 那么其反函数 $p^{-1}(y)=(y+k+1) \mod n$ 就能将得到的序号转换回去. 因此:
> $$
> \begin{aligned} \\
> f(n,m) &= q(n-1,m) \\
> &=p^{-1}(f(n-1,m)) \\
> &=(f(n-1, m)+k+1) \mod n \\
> \end{aligned}
> $$
> 又 $k=(m-1) \mod n$, 故 $(k+1) \mod n = m \mod n$
> $$
> f(n,m)=(f(n-1, m)+m) \mod n
> $$
> 所以最终的递推式为:
> $$
> \left\{
> \begin{array}\\
> f(1,m)=0 \\
> f(n,m)=(f(n-1,m)+m) \mod n, n \ge 2
> \end{array}
> \right.
> $$
> 

```c++
class Solution {
public:
    int LastRemaining_Solution(int n, int m){
        if (n == 0 || m == 0) return -1;
        int temp = 0;
        for (int i = 2; i <= n; ++i) {
            temp = (temp + m) % i;
        }
        return temp;
    }
};
```

### [!]47. 求 1+2+3+...+n

#### 题目描述

> 求1+2+3+...+n, 要求不能使用乘除法、for、while、if、else、switch、case 等关键字及条件判断语句(A?B:C)

#### 解法一:

> 利用短路特性以及递归

```c++
class Solution {
public:
    int Sum_Solution(int n) {
        if (n == 0) return 0;
        int ans = n;
        n && (ans += Sum_Solution(n - 1));
        return ans;
    }
};
```

### 48. 不用加减乘除做加法

### 49. 把字符串转换为整数

### 50. 数组中的重复数字

#### 题目描述

> 在一个长度为n的数组里的所有数字都在0到n-1的范围内。 数组中某些数字是重复的，但不知道有几个数字是重复的。也不知道每个数字重复几次。请找出数组中任意一个重复的数字。 例如，如果输入长度为7的数组{2,3,1,0,2,5,3}，那么对应的输出是第一个重复的数字2。

#### 解法一:

> 遍历, 同时记录这个数是否遇到过, 如果出现遇到过得数字的则返回 true 并记录. 空间复杂度 $O(n)$

```c++
class Solution {
public:
    // Parameters:
    //        numbers:     an array of integers
    //        length:      the length of array numbers
    //        duplication: (Output) the duplicated number in the array number
    // Return value:       true if the input is valid, and there are some duplications in the array number
    //                     otherwise false
    bool duplicate(int numbers[], int length, int* duplication) {
        vector<bool> visited(length, false);
        for (int i = 0; i < length; ++i) {
            if (visited[numbers[i]]) {
                duplication[0] = numbers[i];
                return true;
            }
            visited[numbers[i]] = true;
        }
        return false;
    }
};
```

#### 解法二:

> 由于数组里面的数字都是 0-n 之间的数字, 所以当我们遍历到一个数时, 可以将以这个数为下标的数加上 n, 如果在遍历时, 遇到``` numbers[numbers[i] % length] >= length``` 则说明之前已经有一个数和当前得数重复了. 空间复杂度为 $O(1)$.

```c++
class Solution {
public:
    // Parameters:
    //        numbers:     an array of integers
    //        length:      the length of array numbers
    //        duplication: (Output) the duplicated number in the array number
    // Return value:       true if the input is valid, and there are some duplications in the array number
    //                     otherwise false
    bool duplicate(int numbers[], int length, int* duplication) {
        for (int i = 0; i < length; ++i) {
            int index = numbers[i] % length;
            if (numbers[index] >= length) {
                duplication[0] = numbers[i] - length;
                return true;
            }
            numbers[index] += length;
        }
        return false;
    }
};
```

### 51. 构建乘积数组

#### 题目描述

> 给定一个数组A[0,1,...,n-1],请构建一个数组B[0,1,...,n-1],其中B中的元素B[i]=A[0]*A[1]*...*A[i-1]*A[i+1]*...*A[n-1]。不能使用除法。（注意：规定B[0] = A[1] * A[2] * ... * A[n-1]，B[n-1] = A[0] * A[1] * ... * A[n-2];）

#### 解法一:

> 从第 i 个位置将乘法分为两个部分

```c++
class Solution {
public:
    vector<int> multiply(const vector<int>& A) {
        int len = A.size();
        vector<int> B1(len, 1), B2(len, 1), B(len);
        for (int i = 1; i < len; ++i) {
            B1[i] = B1[i - 1] * A[i - 1];
        }
        for (int i = len - 2; i >= 0; --i) {
            B2[i] = B2[i + 1] * A[i + 1];
        }
        for (int i = 0; i < len; ++i) {
            B[i] = B1[i] * B2[i];
        }
        return B;
    }
};
```

### 52. 正则表达式匹配

#### 题目描述

> 请实现一个函数用来匹配包括'.'和'*'的正则表达式。模式中的字符'.'表示任意一个字符，而'*'表示它前面的字符可以出现任意次（包含0次）。 在本题中，匹配是指字符串的所有字符匹配整个模式。例如，字符串"aaa"与模式"a.a"和"ab*ac*a"匹配，但是与"aa.a"和"ab*a"均不匹配

#### 解法一:

```c++
class Solution {
public:
    bool match(char* str, char* pattern)
    {
        if (*str == '\0' && *pattern == '\0') return true;
        if (*str != '\0' && *pattern == '\0') return false;
        bool first_match = *str == *pattern || (*str != '\0' && *pattern == '.');
        if (*(pattern + 1) == '*') {
            return match(str, pattern + 2) || 
                (first_match && match(str + 1, pattern));
        } else {
            return first_match && match(str + 1, pattern + 1);
        }
    }
};
```

### 53. 表示数值的字符串

#### 题目描述

> 请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。例如，字符串"+100","5e2","-123","3.1416"和"-1E-16"都表示数值。 但是"12e","1a3.14","1.2.3","+-5"和"12e+4.3"都不是。

#### 解法一: 有限状态自动机

> 可以构造一个识别字符串是不是一个合法的数值得自动机, 自动机如下:
>
> ![](https://i.loli.net/2020/05/08/sG9QrMa2EV5cn8J.jpg)
>
> 代码如下:

```c++
class Solution {
public:
    typedef enum state{s0, s1, s2, s3, s4} state;
    bool isNumeric(char* string)
    {
        state s = s0;
        while(true) {
            switch (s) {
                case s0:
                    if (*string == '+' 
                        || *string == '-' 
                        || (*string >= '0' && *string <= '9')) {
                        s = s1;
                        string++;
                    } else {
                        return false;
                    }
                    break;
                case s1:
                    if (*string >= '0' && *string <= '9') {
                        s = s1;
                    } else if (*string == '.') {
                        s = s2;
                    } else if (*string == '\0') {
                        return true;
                    } else if (*string == 'E' || *string == 'e') {
                        s = s3;
                    } else {
                        return false;
                    }
                    string++;
                    break;
                case s2:
                    if (*string >= '0' && *string <= '9') {
                        s = s2;
                    } else if (*string == 'E' || *string == 'e') {
                        s = s3;
                    } else if (*string == '\0'){
                        return true;
                    } else {
                        return false;
                    }
                    string++;
                    break;
                case s3:
                    if (*string == '+' 
                        || *string == '-' 
                        || (*string >= '0' && *string <= '9')){
                        s = s4;
                        string++;
                    } else {
                        return false;
                    }
                    break;
                case s4:
                    if (*string >= '0' && *string <= '9') {
                        s = s4;
                    } else if (*string == '\0') {
                        return true;
                    } else {
                        return false;
                    }
                    string++;
                    break;
            }
        }
    }
};
```

### 54. 字符流中第一个不重复的字符

#### 题目描述

> 请实现一个函数用来找出字符流中第一个只出现一次的字符。例如，当从字符流中只读出前两个字符" go" 时，第一个只出现一次的字符是"g"。当从该字符流中读出前六个字符“google"时，第一个只出现一次的字符是"l". 
>
> **输出描述:**
>
> > 如果当前字符流没有存在出现一次的字符，返回#字符。

#### 解法一: Hash

```c++
class Solution
{
    int hash[256][2] = {{0, -1}};
    int index = 0;
public:
  //Insert one char from stringstream
    void Insert(char ch)
    {
        hash[ch][0]++;
        if (hash[ch][0] == 1) {
            hash[ch][1] = index;;
        }
        index++;
    }
  //return the first appearence once char in current stringstream
    char FirstAppearingOnce()
    {
        int result, resultIndex = index + 1;
        for (int i = 0; i < 256; ++i) {
            if (hash[i][0] == 1) {
                if (resultIndex > hash[i][1]) {
                    resultIndex = hash[i][1];
                    result = i;
                }
            }
        }
        if (resultIndex != index + 1) return result;
        return '#';
    }

};
```

### [!]55. 链表中环的入口结点

#### 题目描述

> 给一个链表, 若其中包含环, 请找出该链表的环的入口结点, 否则, 输出 ```null```.

#### 解法一: 快慢指针

> 这个问题可以分成两个部分, 一个是判断是否有环, 一个是找到入口结点.
>
> **判断是否有环**
>
> 可以设置两个指针, 一个快指针 ```fast``` (一次走两步)和一个慢指针 ```slow``` (一次走一步), 两个指针同时从头结点出发, 如果有环的话, 两个指针一定会相遇. 因为慢指针一旦进入环, 就可以看做快指针在追赶慢指针, 每次都更近一步, 最后一定会追上, 即两个指针相等.
>
> **找到入口结点**
>
> 如果链表有环的话, 设置两个指针, 一个从相遇的结点出发, 一个从头结点出发, 两个指针每次都走一步, 最后一定会在入口处相遇. 如图, 
>
> ![](https://i.loli.net/2020/05/11/GH941SmRrbc5wLu.jpg)
>
> 假设链表头到入口结点的长度为 $a$, 入口结点和快慢指针相遇的节点之间的长度为 $b$, 相遇结点到入口结点之间的距离为 $c$, 则有 
> $$
> 快指针路程=a+(b+c)*k+b, k \ge 1 \\
> 慢指针路程慢指针路程=a+b \\
> 2*慢指针路程=快指针路程
> $$
> 所以得到:
> $$
> a=(k-1)*(b+c)+c, k \ge 1
> $$
> 这个式子表示**链表头到环入口的距离=相遇点到环入口的距离+(k-1)圈环长度**, 所以两个指针分别从链表头和相遇点出发, 最后一定相遇于环入口.

```c++
/*
struct ListNode {
    int val;
    struct ListNode *next;
    ListNode(int x) :
        val(x), next(NULL) {
    }
};
*/
class Solution {
public:
    ListNode* EntryNodeOfLoop(ListNode* pHead){
        ListNode *fast = pHead, *slow = pHead;
        while (fast != NULL && fast->next != NULL) {
            fast = fast->next->next;
            slow = slow->next;
            if (fast == slow) {
                break;
            }
        }
        if (fast == NULL || fast->next == NULL) {
            return NULL;
        }
        slow = pHead;
        while (slow != fast) {
            slow = slow->next;
            fast = fast->next;
        }
        return slow;
    }
};
```

### 56. 删除链表中的重复结点

#### 题目描述

> 在一个排序的链表中，存在重复的结点，请删除该链表中重复的结点，重复的结点不保留，返回链表头指针。 例如，链表1->2->3->3->4->4->5 处理后为 1->2->5

#### 解法一: 双指针

```c++
/*
struct ListNode {
    int val;
    struct ListNode *next;
    ListNode(int x) :
        val(x), next(NULL) {
    }
};
*/
class Solution {
public:
    ListNode* deleteDuplication(ListNode* pHead){
        if (pHead == NULL) return NULL;
        ListNode *head = new ListNode(0);
        head->next = pHead;
        ListNode *p1 = head, *p2 = head->next;
        while (p2) {
            while (p2 && p1->next->val == p2->val) {
                p2 = p2->next;
            }
            if (p1->next->next == p2) {
                p1 = p1->next;
            } else {
                p1->next = p2;
            }
        }
        return head->next;
    }
};
```

### 57. 二叉树的下一个结点

#### 题目描述

> 给定一个二叉树和其中的一个结点，请找出中序遍历顺序的下一个结点并且返回。注意，树中的结点不仅包含左右子结点，同时包含指向父结点的指针。

#### 解法一:

```c++
/*
struct TreeLinkNode {
    int val;
    struct TreeLinkNode *left;
    struct TreeLinkNode *right;
    struct TreeLinkNode *next;
    TreeLinkNode(int x) :val(x), left(NULL), right(NULL), next(NULL) {
        
    }
};
*/
class Solution {
public:
    TreeLinkNode* isRightChild(TreeLinkNode* node) {
        while (node->next != NULL && node->next->left != node) {
            node = node->next;
        }
        return node->next;
    }
    
    TreeLinkNode* isRoot(TreeLinkNode* node) {
        node = node->right;
        if (node == NULL) return NULL;
        while (node->left != NULL) {
            node = node->left;
        }
        return node;
    }
    
    TreeLinkNode* isLeftChild(TreeLinkNode* node) {
        return node->next;
    }
    
    TreeLinkNode* GetNext(TreeLinkNode* pNode){
        if (pNode == NULL) return NULL;
        if (pNode->right != NULL || pNode->next == NULL) return isRoot(pNode);
        if (pNode->next->left == pNode) return isLeftChild(pNode);
        if (pNode->next->right == pNode) return isRightChild(pNode);
    }
};
```

### [!]58. 对称的二叉树

#### 题目描述

> 请实现一个函数，用来判断一颗二叉树是不是对称的。注意，如果一个二叉树同此二叉树的镜像是同样的，定义其为对称的。

#### 解法一: 递归

> 首先根节点以及其左右子树, 左子树的左子树和右子树的右子树相同, 左子树的右子树和右子树的左子树相同即可, 采用递归.

```c++
/*
struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
    TreeNode(int x) :
            val(x), left(NULL), right(NULL) {
    }
};
*/
class Solution {
public:
    bool help(TreeNode* root1, TreeNode* root2) {
        if (root1 == NULL) return root2 == NULL;
        if (root2 == NULL) return false;
        return (root1->val == root2->val) &&
            help(root1->left, root2->right) &&
            help(root1->right, root2->left);
    }
    
    bool isSymmetrical(TreeNode* pRoot){
        if (pRoot == NULL) return true;
        return help(pRoot->left, pRoot->right);
    }

};
```

TODO: 其他方法

### 59. 按之字形顺序打印二叉树

#### 题目描述

> 请实现一个函数按照之字形打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右至左的顺序打印，第三行按照从左到右的顺序打印，其他行以此类推。

#### 解法一: 堆栈

```c++
/*
struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
    TreeNode(int x) :
            val(x), left(NULL), right(NULL) {
    }
};
*/
class Solution {
public:
    vector<vector<int> > Print(TreeNode* pRoot) {
        vector<vector<int> > ans;
        if (pRoot == NULL) return ans;
      	// 表示当前层的方向
        bool l2r = true;
        stack<TreeNode*> s;
        s.push(pRoot);
        while (!s.empty()) {
            vector<int> temp;
            stack<TreeNode*> s2;
            while (!s.empty()) {
                TreeNode *node = s.top();
                s.pop();
                temp.push_back(node->val);
                if (l2r) {
                    if (node->left != NULL) {
                        s2.push(node->left);
                    }
                    if (node->right != NULL) {
                        s2.push(node->right);
                    }
                    
                } else {
                    if (node->right != NULL) {
                        s2.push(node->right);
                    }
                    if (node->left != NULL) {
                        s2.push(node->left);
                    }
                }
            }
            ans.push_back(temp);
            l2r = !l2r;
            s = s2;
        }
        return ans;
    }
};
```

### 60. 把二叉树打印成多行

#### 题目描述

> 从上到下按层打印二叉树，同一层结点从左至右输出。每一层输出一行。

#### 解法一: 队列

```c++
/*
struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
    TreeNode(int x) :
            val(x), left(NULL), right(NULL) {
    }
};
*/
class Solution {
public:
        vector<vector<int> > Print(TreeNode* pRoot) {
            vector<vector<int> > ans;
            queue<TreeNode*> q;
            q.push(pRoot);
          	// 设置空指针为每一行的结束
            q.push(NULL);
            while (q.front()) {
                vector<int> temp;
                while (q.front()) {
                    TreeNode* node = q.front();
                    q.pop();
                    temp.push_back(node->val);
                    if (node->left) {
                        q.push(node->left);
                    }
                    if (node->right) {
                        q.push(node->right);
                    }
                }
              	// 设置空指针为每一行的结束
                q.push(NULL);
                ans.push_back(temp);
                q.pop();
            }
            return ans;
        }
};
```

### [!]61. 序列化二叉树

#### 题目描述

>  请实现两个函数, 分别用来序列化和反序列化二叉树.
>
>  二叉树的序列化是指: 把一棵二叉树按照某种遍历方式的结果以某种格式保存为字符串, 从而使得内存中建立起来的二叉树可以持久保存. 序列化可以基于先序, 中序, 后序, 层序的二叉树遍历方式来进行修改, 序列化的结果是一个字符串, 序列化时通过 某种符号表示空节点(#), 以 ! 表示一个结点值的结束 (value!).
>
>  二叉树的反序列化是指: 根据某种遍历顺序得到的序列化字符串结果str, 重构二叉树.
>
>  例如, 我们可以把一个只有根节点为1的二叉树序列化为 "1", 然后通过自己的函数来解析回这个二叉树.

#### 解法一: 利用层序遍历序列化

> TODO

```c++
import java.util.Queue;
import java.util.LinkedList;
/*
public class TreeNode {
    int val = 0;
    TreeNode left = null;
    TreeNode right = null;

    public TreeNode(int val) {
        this.val = val;

    }

}
*/
public class Solution {
    String Serialize(TreeNode root) {
        if (root == null) return null;
        String s = "";
        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);
        while (!q.isEmpty()) {
            TreeNode node = q.poll();
            if (node == null) {
                s = s + "#";
            } else {
                s = s + String.valueOf(node.val) + "!";
                q.offer(node.left);
                q.offer(node.right);
            }
        }
        return s;
  }
    TreeNode Deserialize(String str) {
        if (str == null) return null;
        Queue<TreeNode> q = new LinkedList<>();
        TreeNode head = null;
        int pre = 0, cur = 0;
        for (; cur < str.length(); ++cur) {
            if (str.charAt(cur) == '!') {
                head = new TreeNode(Integer.parseInt(str.substring(pre, cur)));
                q.offer(head);
                break;
            }
        }
        cur ++;
        pre = cur;
        boolean flag = true;
        TreeNode left = null, right = null;
        while (cur < str.length()) {
            char c = str.charAt(cur);
            if (c == '!') {
                int val = Integer.parseInt(str.substring(pre, cur));
                if (flag) {
                    left = new TreeNode(val);
                    q.offer(left);
                    flag = !flag;
                } else {
                    right = new TreeNode(val);
                    q.offer(right);
                    TreeNode node = q.poll();
                    node.left = left;
                    node.right = right;
                    flag = !flag;
                }
                cur ++;
                pre = cur;
            } else if (c == '#') {
                if (flag) {
                    left = null;
                    flag = !flag;
                } else {
                    right = null;
                    TreeNode node = q.poll();
                    node.left = left;
                    node.right = right;
                    flag = !flag;
                }
                cur++;
                pre=cur;
            } else {
                cur ++;
            }
        }
        return head;
  }
}
```

### [!]62. 二叉搜索树的第 K 个节点

#### 题目描述

> 给定一棵二叉搜索树, 请找出其中的第k小的结点。例如, (5, 3, 7, 2, 4, 6, 8) 中, 按结点数值大小顺序第三小结点的值为4.

#### 解法一: 非递归中序遍历

```c++
/*
struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
    TreeNode(int x) :
            val(x), left(NULL), right(NULL) {
    }
};
*/
class Solution {
public:
    TreeNode* KthNode(TreeNode* pRoot, int k) {
        if (pRoot == NULL || k == 0) return NULL;
        int index = 1;
        stack<TreeNode*> s;
        TreeNode* node = pRoot;
        while (node != NULL || !s.empty()) {
            while (node != NULL) {
                s.push(node);
                node = node->left;
            }
            node = s.top();
            s.pop();
            if (index == k) {
                return node;
            }
            node = node->right;
            index++;
        }
        return NULL;
    }
};
```

### [!]63. 数据流中的中位数

#### 题目描述

> 如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。我们使用Insert()方法读取数据流，使用GetMedian()方法获取当前读取数据的中位数

#### 解法一: 大顶堆小顶堆

> 因为要求的是中位数, 那么这两个堆, 大顶堆用来存较小的数, 从大到小排列; 小顶堆存较大的数, 从小到大的顺序排序, 并且保证两个堆的大小相差不超过 1, 这样当当前数字合数为为偶数时中位数就是大顶堆的根节点与小顶堆的根节点和的平均数, 为奇数时就是小顶堆的堆顶. 这样需要保证小顶堆中的元素都大于等于大顶堆中的元素, 所以每次塞值, 并不是直接塞进去, 而是将一个堆的堆顶元素放进另一个堆. 具体做法为:
>
> - 当数目为偶数的时候, 将这个值插入大顶堆中, 再将大顶堆的堆顶插入到小顶堆中, 同时将大顶堆堆顶弹出;
> - 当数目为奇数的时候, 将这个值插入小顶堆中, 再将小顶堆的堆顶插入到大顶堆中, 同时将小顶堆堆顶弹出;
> - 取中位数的时候, 如果当前个数为偶数, 显然是取小顶堆和大顶堆堆顶值的平均值; 如果当前个数为奇数, 显然是取小顶堆的堆顶值
>
> 下面举个例子说明一下:
>
> 例如, 传入的数据为: [5, 2, 3, 4, 1, 6, 7, 0, 8], 那么按照要求, 输出是 "5.00 3.50 3.00 3.50 3.00 3.50 4.00 3.50 4.00", 那么按照上面的步骤(用 min 表示小顶堆, max 表示大顶堆):
>
> - 5 先进入大顶堆, 然后将大顶堆中最大值放入小顶堆中, 此时 min=[5], max=[无], avg=[5.00]
> - 2 先进入小顶堆, 然后将小顶堆中最小值放入大顶堆中, 此时 min=[5], max=[2], avg=[(5+2)/2]=[3.50]
> - 3 先进入大顶堆, 然后将大顶堆中最大值放入小顶堆中, 此时 min=[3,5], max=[2], avg=[3.00]
> - 4 先进入小顶堆, 然后将小顶堆中最小值放入大顶堆中, 此时 min=[4,5], max=[3,2], avg=[(4+3)/2]=[3.50]
> - 1 先进入大顶堆, 然后将大顶堆中最大值放入小顶堆中, 此时 min=[3,4,5], max=[2,1], avg=[3/00]
> - 6 先进入小顶堆, 然后将小顶堆中最小值放入大顶堆中, 此时 min=[4,5,6], max=[3,2,1], avg=[(4+3)/2]=[3.50]
> - 7 先进入大顶堆, 然后将大顶堆中最大值放入小顶堆中, 此时 min=[4,5,6,7], max=[3,2,1], avg=[4]=[4.00]
> - 0 先进入小顶堆, 然后将小顶堆中最大值放入小顶堆中, 此时 min=[4,5,6,7], max=[3,2,1,0], avg=[(4+3)/2]=[3.50]
> - 8 先进入大顶堆, 然后将大顶堆中最小值放入小顶堆中, 此时 min=[4,5,6,7,8], max=[3,2,1,0], avg=[4.00]

```c++
class Solution {
private:
    priority_queue<int> maxHeap;
    priority_queue<int, vector<int>, greater<int> > minHeap;
    bool flag = true;
public:
    void Insert(int num)
    {
        if (flag) {
            maxHeap.push(num);
            minHeap.push(maxHeap.top());
            maxHeap.pop();
        } else {
            minHeap.push(num);
            maxHeap.push(minHeap.top());
            minHeap.pop();
        }
        flag = !flag;
    }

    double GetMedian()
    { 
        if (minHeap.empty() && maxHeap.empty()) return 0.0;
        if (flag) {
            return (minHeap.top() + maxHeap.top()) / 2.0;
        }
        return minHeap.top();
    }

};
```

### 64. 滑动窗口最大值

### 65. 矩阵中的路径

#### 题目描述

> 请设计一个函数, 用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径. 路径可以从矩阵中的任意一个格子开始, 每一步可以在矩阵中向左, 向右, 向上, 向下移动一个格子. 如果一条路径经过了矩阵中的某一个格子, 则该路径不能再进入该格子. 例如 $\begin{bmatrix} a & b & c &e \\ s & f & c & s \\ a & d & e& e\\ \end{bmatrix}$ 矩阵中包含一条字符串 "bcced" 的路径, 但是矩阵中不包含 "abcb" 路径, 因为字符串的第一个字符 b 占据了矩阵中的第一行第二个格子之后, 路径不能再次进入该格子.

#### 解法一: DFS

```c++
class Solution {
private:
    char* matrix;
    int rows;
    int cols;
public:
    bool search(int i, int j, char* c, vector<vector<bool> > visited) {
        if (*c == '\0') return true;
        if (i < 0 || i >= rows || j < 0 || j >= cols) return false;
        if (visited[i][j] || *c != matrix[i * cols + j]) return false;
        visited[i][j] = true;
        return search(i - 1, j, c + 1, visited) ||
            search(i + 1, j, c + 1,visited) ||
            search(i, j + 1, c + 1, visited) ||
            search(i, j - 1, c + 1, visited);
    }
    
    bool hasPath(char* matrix, int rows, int cols, char* str) {
        this->rows = rows;
        this->cols = cols;
        this->matrix = matrix;
        vector<vector<bool> > visited(rows, vector<bool>(cols, false));
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                if (search(i, j, str, visited)) {
                    return true;
                }
            }
        }
        return false;
    }
};
```

### 66. 机器人的运动范围

#### 题目描述

> 地上有一个 m 行和 n 列的方格. 一个机器人从坐标 (0, 0) 的格子开始移动, 每一次只能向左, 右, 上, 下四个方向移动一格, 但是不能进入行坐标和列坐标的数位之和大于 k 的格子. 例如, 当 k 为 18 时, 机器人能够进入方格(35, 37), 因为 3 + 5 + 3 + 7 = 18. 但是, 它不能进入方格 (35, 38), 因为 3 + 5 + 3 + 8 = 19. 请问该机器人能够达到多少个格子?

#### 解法一: DFS (递归)

```c++
class Solution {
private:
    int rows;
    int cols;
    int threshold;
public:
    bool judge(int i, int j) {
        int sum = 0;
        while (i > 0) {
            sum = sum + i % 10;
            i = i / 10;
        }
        while (j > 0) {
            sum = sum + j % 10;
            j = j / 10;
        }
        return sum <= threshold;
    }
    
    void search(int i, int j, int &count, vector<vector<bool> > &visited) {
        if (i < 0 || i >= rows || j < 0 || j >= cols) return;
        if (visited[i][j]) return;
        if (judge(i, j)) {
            count++;
            visited[i][j] = true;
            search(i, j + 1, count, visited);
            search(i, j - 1, count, visited);
            search(i + 1, j, count, visited);
            search(i - 1, j, count, visited);
        }
    }
    
    int movingCount(int threshold, int rows, int cols) {
        this->rows = rows;
        this->cols = cols;
        this->threshold = threshold;
        int count = 0;
        vector<vector<bool> > visited(rows, vector<bool>(cols, false));
        search(0, 0, count, visited);
        return count;
    }
};
```

#### 解法二: BFS (非递归)

```c++
class Solution {
public:
    bool judge(int i, int j, int threshold) {
        int sum = 0;
        while (i > 0) {
            sum = sum + i % 10;
            i = i / 10;
        }
        while (j > 0) {
            sum = sum + j % 10;
            j = j / 10;
        }
        return sum <= threshold;
    }

    int movingCount(int threshold, int rows, int cols) {
        int count = 0;
        vector<vector<bool> > visited(rows, vector<bool>(cols, false));
        queue<pair<int, int> > q;
        q.push(make_pair(0, 0));
        while (!q.empty()) {
            auto t = q.front();
            q.pop();
            if (t.first < 0 || t.first >= rows || t.second < 0 || t.second >= cols) continue;
            if (visited[t.first][t.second]) continue;
            if (judge(t.first, t.second, threshold)) {
                count++;
                visited[t.first][t.second] = true;
                q.push(make_pair(t.first, t.second + 1));
                q.push(make_pair(t.first, t.second - 1));
                q.push(make_pair(t.first + 1, t.second));
                q.push(make_pair(t.first - 1, t.second));
            }
        }
        return count;
    }
};
```

### 67. 剪绳子

#### 题目描述

> 给你一根长度为n的绳子，请把绳子剪成整数长的m段（m、n都是整数，n>1并且m>1），每段绳子的长度记为k[0],k[1],...,k[m]。请问k[0]xk[1]x...xk[m]可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。
>
> **输入描述**
>
> ```
> 输入一个数n, 意义见题面. (2 <= n <= 60)
> ```
>
> **输出描述:**
>
> ```
> 输出答案
> ```
>
> **示例1**
>
> **输入**
>
> ```
> 8
> ```
>
> **输出**
>
> ```
> 18
> ```

#### 解法一:

> 结论是当 $k[0]=k[1]=...=k[m]$ 时乘积最大, 设 $k[i]=x$, 那么 $n=x \times m$, 乘积可以用下式表示
> $$
> f(x)=x^{\frac{n}{x}}
> $$
> 下面是 $f(x)$ 的导数:
> $$
> f'(x)=\frac{n}{x^2}(1-\ln x)x^{\frac{n}{x}}
> $$
> 通过 $f'(x)=0$ 可以得到当 $x = e$ 的时候 $f(x)$ 最大. 又因为 $x$ 的取值只能为整数, 且 $f(3)>f(2)$, 所以, 当 $n＞3$ 时, 将 n 尽可能地分割为 3 的和时, 乘积最大. 当 $n＞3$ 时, 有三种情况, 即 $n\mod 3=0$, $n \mod 3=1$, $n \mod 3 =2$, 故:
> $$
> f(x)=
> \left\{
> \begin{array} \\
> 3^{\frac{n}{3}}, n \mod 3 = 0 \\
> 4 \times 3^{\frac{n}{3}-1}, n \mod 3 = 1 \\
> 2 \times 3^{\frac{n}{3}}, n \mod 3 = 2
> \end{array}
> \right.
> $$
> 

```c++
class Solution {
public:
    int cutRope(int number) {
        if (number < 4) return number - 1;
        if (number % 3 == 0) {
            return pow(3, number / 3);
        }
        if (number % 3 == 1) {
            return 4 * pow(3, number / 3 - 1);
        }
        if (number % 3 == 2) {
            return 2 * pow(3, number / 3);
        }
    }
};
```

