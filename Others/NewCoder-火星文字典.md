### [牛客网: 火星文字典](https://www.nowcoder.com/questionTerminal/29d1622d47514670a85e98a1f47b8e2d)
#### 题目描述:
> 已知一种新的火星文的单词由英文字母（仅小写字母）组成，但是此火星文中的字母先后顺序未知。给出一组非空的火星文单词，且此组单词已经按火星文字典序进行好了排序（从小到大），请推断出此火星文中的字母先后顺序。
> 输入描述:
>> 一行文本，为一组按火星文字典序排序好的单词(单词两端无引号)，单词之间通过空格隔开
>
> 输出描述:
>> 按火星文字母顺序输出出现过的字母，字母之间无其他字符，如果无法确定顺序或者无合理的字母排序可能，请输出"invalid"(无需引号)
>
>示例1
>> 输入
>> z x
>> 输出
>> zx
>
> 示例2
>> 输入
>> wrt wrf er ett rftt
>> 输出
>> wertf
>
>示例3
>> 输入
>> z x z
>> 输出
>> invalid

#### 解法一
> 这个题目其实就是一道拓扑排序的题目. 解决思路如下:
> 用两个 Hash 表, 一个来存每个出现过的字符的入度, 另一个来存一个字符指向的字符集合, 即一个字符应该在另外字符的前面. 然后每次查找入度为零的字符, 并且将这个字符所指向的所有字符的入度减一, 如果有多个入度为零的字符则说明不止一种排序, 输出 "invalid", 如果没有入度为零的字符了, 则说明不能排序, 输出 "invalid". 代码如下:

```c++
#include <iostream>
#include <unordered_map>
#include <set>
#include <vector>
#include <cmath>
using namespace std;

string getOrder(vector<string> words){
    unordered_map<char, int> inDegree;
    unordered_map<char, multiset<char> > hashTable;
    for (auto word:words) {
        for (char c:word) {
            inDegree[c] = 0;
            hashTable[c] = multiset<char>();
        }
    }
    for(int i = 1; i < words.size(); ++i) {
        int j = 0, l1 = words[i].length(), l2 = words[i - 1].length();
        while (j < min(l1, l2) && words[i][j] == words[i-1][j]) {
            j ++;
        }
        if (j == min(l1, l2)) continue;
        // c1 排在 c2 前面
        // 因为相同的两个字符可能出现多次, 每次出现都贡献了一个入度
        // 所以要用 multiset 存指向的字符集
        char c1 = words[i-1][j], c2 = words[i][j];
        inDegree[c2] ++;
        hashTable[c1].insert(c2);
    }

    string ans = "";
    for (int i = 0; i < inDegree.size(); ++i) {
        int count = 0;
        char c;
        for (auto p:inDegree) {
            if (p.second == 0) {
                count ++;
                c = p.first;
            }
        }
        if (count == 1) {
            ans += c;
            inDegree[c] --;
            for (char temp:hashTable[c]) {
                inDegree[temp] --;
            }
        } else {
            ans = "invalid";
            break;
        }
    }
    return ans;
}

int main() {
    vector<string> v;
    string s;
    while(cin >> s && getchar() != '\n') {
        v.push_back(s);
    }
    v.push_back(s);
    cout << getOrder(v) << "\n";
}
```