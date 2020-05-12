### [LeetCode #93: Restore IP Addresses (复原IP地址)](https://leetcode.com/problems/restore-ip-addresses/)
#### 题目描述:
> 给定一个只包含数字的字符串, 复原它并返回所有可能的 IP 地址格式.
> 示例:
> 输入: "25525511135"
> 输出: ["255.255.11.135", "255.255.111.35"]

#### 解法一: DFS
> 使用深度优先的搜索方法, 在原始的数字串中一次加 '.' 将数字串分割, 直到添加了三个 '.' 将数字串分为四个部分(每个部分都能满足除零外不以零开头, 而且小于 256)


**C++代码:**
```c++
class Solution {
public:
    /*
    这个函数就是找从 index 开始还要添加 n 个 . 的ip 地址
    */
    void help(string s, int index, int n, vector<string> &result) {
        /*
        当 n 为 0 时表示不需要添加 . 了，但是需要判断最后一段是不满足条件
        */
        if (n == 0) {
            // 已经没有剩余的数字了, 不满足条件
            if (index == s.length()) return;
            // 最后一段是0开头, 但是剩余不止一个数字也不满足条件
            if (s[index] == '0' && index != s.length() - 1) return;
            // 最后一段满足条件，添加到结果数组里面
            if (stoi(s.substr(index)) < 256) result.push_back(s);
            return;
        }
        // 当 n 比 0 大，还需要继续在后面的数字串里面添加 .
        // 如果当前数字为 0 , 直接在后面添加 . , 递归调用
        if (s[index] == '0') {
            s.insert(index + 1, 1, '.');
            help(s, index + 2, n - 1, result);
            return;
        }
        // 用一个临时的 string  temp 拷贝当前的 string , 保证回溯的时候能够还原 string
        string temp;
        int len = s.length();
        // 当 index 处的数字不是 0 时, 可以在 index 所指的数字后零个一个两个数字后添加 .
        for (int i = index; i < len - 1 && (i - index + 1) <= 3; i++) {
            temp = s;
            // 大于 256 就不满足 ip 地址每一段的条件了
            if (stoi(temp.substr(index, i-index+1)) > 256) continue;
            // 满足条件, 添加 .
            temp.insert(i+1, 1, '.');
            // 继续添加后面的 .
            help(temp, i+2, n-1, result);
        }
    }
    
    vector<string> restoreIpAddresses(string s) {
        vector<string> result;
        if (s.length() > 12 || s.length() < 4) return result;
        help(s, 0, 3, result);
        return result;
    }
};
```