### [LeetCode #10: Regular Expression Matching (正则表达式匹配)](https://leetcode.com/problems/regular-expression-matching/)
#### 题目描述:
> 给你一个字符串 s 和一个字符规律 p, 请你来实现一个支持 '.' 和 '\*' 的正则表达式匹配。
> 
>> '.' 匹配任意单个字符
>> '\*' 匹配零个或多个前面的那一个元素
>
> 所谓匹配，是要涵盖 整个 字符串 s的，而不是部分字符串。
> 说明:
>> s 可能为空，且只包含从 a-z 的小写字母。
>> p 可能为空，且只包含从 a-z 的小写字母，以及字符 . 和 \*。
> 
> 示例 1:
>> 输入:
>> s = "aa"
>> p = "a"
>> 输出: false
>> 解释: "a" 无法匹配 "aa" 整个字符串。
>
>示例 2:
>> 输入:
>> s = "aa"
>> p = "a*"
>> 输出: true
>> 解释: 因为 '\*' 代表可以匹配零个或多个前面的那一个元素, 在这里前面的元素就是 'a'。因此，字符串 "aa" 可被视为 'a' 重复了一次。
>
>示例 3:
>> 输入:
>> s = "ab"
>> p = ".\*"
>> 输出: true
>> 解释: ".*" 表示可匹配零个或多个（'*'）任意字符（'.'）。
>
>示例 4:
>> 输入:
>> s = "aab"
>> p = "c\*a\*b"
>> 输出: true
>>解释: 因为 '\*' 表示零个或多个，这里 'c' 为 0 个, 'a' 被重复一次。因此可以匹配字符串 "aab"。
>
>示例 5:
>> 输入:
>> s = "mississippi"
>> p = "mis\*is\*p\*."
>> 输出: false

#### 解法一(回溯法):
> 从左向右检查匹配串能否匹配模式串, 由于模式串里面有 '\*', 当下一个字符是 '\*' 时我们可以匹配 0 个或者多个当前字符, 当匹配 0 个的时候模式串直接跳到 '\*' 后即可, 匹配串不变; 匹配多个时匹配串跳到下一个字符, 模式串仍在当前位置. 代码如下:

**C++代码**
```c++
class Solution {
public:
    bool isMatch(string s, string p) {
        if (p.length() == 0) return s.length() == 0;
        // 先判断当前字符是否匹配
        bool first_match = s.length() > 0 
            && (s[0] == p[0] || p[0] == '.');
        // 判断模式串下一个字符是否是 ’*‘
        if (p.length() > 1 && p[1] == '*') {
            // 如果是则考虑匹配零个和多个
            return (first_match && isMatch(s.substr(1), p))
                || isMatch(s, p.substr(2));
        } else {
            // 如果不是则匹配当前的字符
            return first_match && isMatch(s.substr(1), p.substr(1));
        }
    }
};
```

#### 解法二(动态规划)
> 从上面的解法可以看出这个问题具有最优子结构的性质, 如果我们用一个二维数组 ```dp``` 表示子串的匹配结果, 即 ```dp[i][j]``` 表示匹配串 ```s[i:]``` 和模式串 ```p[j:]``` 是否匹配, 最终结果即 ```dp[0][0]```.
> 当模式串下一个字符不是 '\*' 时, 我们有:
> ```dp[i][j] = match(s[i], p[j]) && dp[i+1][j+1]```
> 
> 当模式串下一个字符是 '\*' 时, 我们有:
> ```dp[i][j] = dp[i][j + 2] || (match(s[i], p[j]) && dp[i + 1][j])```
> 代码如下:

**C++代码**
```c++
class Solution {
public:
    bool isMatch(string s, string p) {
        vector<vector<bool>> dp(s.length() + 1, vector<bool>(p.length() + 1));
        dp[s.length()][p.length()] = true;
        for (int i = s.length(); i >= 0; --i) {
            for (int j = p.length() - 1; j >= 0; --j) {
                bool first_match = (i < s.length() && (s[i] == p[j] || p[j] == '.'));
                if (j + 1 < p.length() && p[j + 1] == '*') {
                    dp[i][j] = dp[i][j + 2]
                        || (first_match && dp[i + 1][j]);
                } else {
                    dp[i][j] = first_match && dp[i + 1][j + 1];
                }
           }
        }
        return dp[0][0];
    }
};
```
另一种写法:
```c++
class Solution {
public:
    typedef enum flag {UNSET, TRUE, FALSE};
    vector<vector<flag>> memo;
    
    bool isMatch(string s, string p) {
        memo = vector(s.length() + 1, vector<flag>(p.length() + 1, flag::UNSET));
        return dp(0, 0, s, p);
    }
    
    bool dp(int i, int j, string s, string p) {
        if (memo[i][j] != flag::UNSET) {
            return memo[i][j] == flag::TRUE;
        }
        bool result;
        if (j == p.length()) {
            result = i == s.length();
        } else {
            bool first_match = (i < s.length() && (s[i] == p[j] || p[j] == '.'));
            if (j + 1 < p.length() && p[j + 1] == '*') {
                result = dp(i, j + 2, s, p)
                    || (first_match && dp(i + 1, j, s, p));
            } else {
                result = first_match && dp(i + 1, j + 1, s, p);
            } 
        }
        memo[i][j] = result ? flag::TRUE : flag::FALSE;
        return result;
    }
};
```