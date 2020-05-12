### [LeetCode #33: Search in Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array/)
#### 题目描述:
> 假设按照升序排序的数组在预先未知的某个点上进行了旋转。
> (例如，数组 [0,1,2,4,5,6,7] 可能变为 [4,5,6,7,0,1,2])。
> 搜索一个给定的目标值，如果数组中存在这个目标值，则返回它的索引，否则返回 -1 。
> 你可以假设数组中不存在重复的元素。
> 你的算法时间复杂度必须是 O(log n) 级别。
#### 解法一:
> 根据题目的要求可以知道需要使用二分搜索的方式才能满足 O(log n) 的算法复杂度，但是这样的数组并不能直接使用二分查找。一次我们需要找到数组中最小元素的位置。
> 
> 查找最小元素的位置需要使用类似二分查找的方式:
> ![](https://i.loli.net/2020/03/12/9jBqVI1XQHE25sv.jpg)
> 可以看到数组旋转之后会变成上图的样子。二分查找最小元素位置示意如下:
> ![](https://i.loli.net/2020/03/12/2ncAuE5Zt1Yqvsd.jpg)
>
> 找到最小元素的位置后我们就可以分别对左右两边分别进行二分查找了，完整的代码如下:

**C++代码:**
```c++
class Solution {
public:
    int find_pivot(vector<int> nums) {
        int left = 0, right = nums.size() - 1, mid;
        if (nums[left] < nums[right]) return 0;
        while (left <= right) {
            mid = (left + right) / 2;
            if (nums[mid] > nums[mid + 1]) return mid + 1;
            if (nums[mid] < nums[left]) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return 0;
    }
    
    int binary_search(vector<int> nums, int target, int l, int r) {
        while(l <= r) {
            int mid = (l + r) / 2;
            if (nums[mid] == target) return mid;
            if (target < nums[mid]) {
                r = mid - 1;
            } else {
                l = mid + 1;
            }
        }
        return -1;
    }
    
    int search(vector<int>& nums, int target) {
        if (nums.size() == 0) return -1;
        if (nums.size() == 1) return nums[0] == target ? 0 : -1;
        int pivot = find_pivot(nums);
        if (pivot == 0) return binary_search(nums, target, 0, nums.size() - 1);
        if (target < nums[0]) return binary_search(nums, target, pivot, nums.size() - 1);
        return binary_search(nums, target, 0, pivot - 1);
    }
};
```