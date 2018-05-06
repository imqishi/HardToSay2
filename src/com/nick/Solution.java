package com.nick;

import java.math.BigInteger;
import java.util.*;

import static java.lang.System.*;

/**
 * Created by qishi on 2018/4/3.
 */
public class Solution {

    /*
    * Test Method
    * */
    public static void main(String[] args) {
        Solution solution = new Solution();
        int[][] tmp = {{1,5,9},{10,11,13},{12,13,15}};
        solution.kthSmallestBinarySearch(tmp, 8);
        //out.println(solution.combinationSum4(tmp, 4));
    }

    /*
    * Leetcode 378. Kth Smallest Element in a Sorted Matrix
    * */
    class Tuple implements Comparable<Tuple> {
        int x, y, val;
        public Tuple(int x, int y, int val) {
            this.x = x;
            this.y = y;
            this.val = val;
        }

        @Override
        public int compareTo(Tuple o) {
            return this.val - o.val;
        }
    }
    public int kthSmallest(int[][] matrix, int k) {
        PriorityQueue<Tuple> heap = new PriorityQueue<>();
        for (int j = 0; j < matrix.length; j ++) {
            heap.offer(new Tuple(0, j, matrix[0][j]));
        }

        for (int i = 0; i < k - 1; i ++) {
            Tuple t = heap.poll();
            if (t.x == matrix.length - 1) {
                continue;
            }
            heap.offer(new Tuple(t.x + 1, t.y, matrix[t.x + 1][t.y]));
        }

        return heap.poll().val;
    }

    public int kthSmallestBinarySearch(int[][] matrix, int k) {
        int lo = matrix[0][0], hi = matrix[matrix.length - 1][matrix[0].length - 1] + 1;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            int count = 0, j = matrix[0].length - 1;
            for (int i = 0; i < matrix.length; i ++) {
                while (j >= 0 && matrix[i][j] > mid) {
                    j --;
                }
                count += (j + 1);
            }
            if (count < k) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        return lo;
    }

    public int lengthOfLongestSubstring(String s) {
        int n = s.length();
        Set<Character> set = new HashSet<>();
        int res = 0, i = 0, j = 0;
        while (i < n && j < n) {
            if (! set.contains(s.charAt(j))) {
                set.add(s.charAt(j));
                j ++;
                res = Math.max(res, j - i);
            } else {
                set.remove(s.charAt(i));
                i ++;
            }
        }

        return res;
    }

    public int lengthOfLongestSubstring2(String s) {
        int n = s.length(), res = 0;
        Map<Character, Integer> map = new HashMap<>();
        for (int j = 0, i = 0; j < n; j ++) {
            if (map.containsKey(s.charAt(j))) {
                i = Math.max(map.get(s.charAt(j)), i);
            }

            res = Math.max(res, j - i + 1);
            map.put(s.charAt(j), j + 1);
        }

        return res;
    }

    /*
    * Leetcode 377. Combination Sum IV
    * */
    public int combinationSum4(int[] nums, int target) {
        int[] dp = new int[target + 1];
        dp[0] = 1;
        for (int i = 1; i <= target; i ++) {
            for (int j = 0; j < nums.length; j ++) {
                if (i - nums[j] >= 0) {
                    dp[i] += dp[i - nums[j]];
                }
            }
        }

        return dp[target];
    }

    // this will cause time limit exceeded
    int cS4Sum = 0;
    public void subCombinationSum4(int[] nums, int target, int curSum) {
        if (target == curSum) {
            cS4Sum ++;
            return ;
        } else if (target < curSum) {
            return ;
        }

        for (int i = 0; i < nums.length; i ++) {
            subCombinationSum4(nums, target, curSum + nums[i]);
        }
    }

    /*
    * Leetcode 376. Wiggle Subsequence
    * */
    public int wiggleMaxLength(int[] nums) {
        if (nums.length <= 1) {
            return nums.length;
        }
        int k = 0;
        while (k < nums.length - 1 && nums[k] == nums[k + 1]) {
            k ++;
        }
        if (k == nums.length - 1) {
            return 1;
        }

        int res = 2;
        boolean needBig = nums[k] < nums[k + 1];
        for (int i = k + 1; i < nums.length - 1; i ++) {
            if (needBig && nums[i] > nums[i + 1]) {
                nums[res] = nums[i + 1];
                res ++;
                needBig = ! needBig;
            } else {
                if (! needBig && nums[i] < nums[i + 1]) {
                    nums[res] = nums[i + 1];
                    res ++;
                    needBig = ! needBig;
                }
            }
        }
        return res;
    }

    /*
    * Leetcode 375. Guess Number Higher or Lower II
    * */
    //out.println(solution.getMoneyAmount(10));
    public int getMoneyAmount(int n) {
        if (n == 1) {
            return 0;
        }
        int[][] dp = new int[n + 1][n + 1];
        for (int len = 1; len < n; len ++) {
            for (int i = 0; i + len <= n; i ++) {
                int j = i + len;
                dp[i][j] = Integer.MAX_VALUE;
                for (int k = i; k <= j; k ++) {
                    dp[i][j] = Math.min(dp[i][j], k + Math.max(k - 1 >= i ? dp[i][k-1] : 0, j >= k + 1 ? dp[k+1][j] : 0));
                }
            }
        }

        return dp[1][n];
    }

    /*
    * Leetcode 373. Find K Pairs with Smallest Sums
    * */
    //int[] tmp = {1,1,2};
    //int[] tmp1 = {1,2,3};
    //solution.kSmallestPairs(tmp, tmp1, 10);
    public List<int[]> kSmallestPairs(int[] nums1, int[] nums2, int k) {
        PriorityQueue<int[]> queue = new PriorityQueue<>((a, b) -> a[0] + a[1] - b[0] - b[1]);
        List<int[]> list = new ArrayList<>();

        if (k > nums1.length * nums2.length) {
            k = nums1.length * nums2.length;
        }
        if (k == 0) {
            return list;
        }

        for (int i = 0; i < nums1.length && i < k; i ++) {
            queue.offer(new int[]{ nums1[i], nums2[0], 0 });
        }

        while (k-- > 0 && !queue.isEmpty()) {
            int[] cur = queue.poll();
            list.add(new int[]{ cur[0], cur[1] });
            if (cur[2] == nums2.length - 1) {
                continue;
            }
            queue.offer(new int[]{ cur[0], nums2[cur[2]+1], cur[2] + 1 });
        }

        return list;
    }

    /*
    * Leetcode 371. Sum of Two Integers
    * */
    public int getSum(int a, int b) {
        int t = 0;
        while (b != 0) {
            t = a & b;
            a = a ^ b;
            b = t << 1;
        }

        return a;
    }

    /* 单调栈总结 */
    //int[] tmp = {3,1,6,4,5,2};
    //out.println(solution.maxResult(tmp));
    public int[] maxResult(int[] arr) {
        int[] res = new int[3];
        int[] work = new int[arr.length + 1];
        int[] sum = new int[arr.length + 1];
        sum[0] = arr[0];
        for (int i = 0; i <= arr.length; i ++) {
            if (i == arr.length) {
                work[i] = -1;
            } else {
                if (i > 0) {
                    sum[i] = sum[i - 1] + arr[i];
                }
                work[i] = arr[i];
            }
        }

        Stack<Integer> stack = new Stack<>();
        int topPos = 0;
        for (int i = 0; i < work.length; i ++) {
            if (stack.empty() || work[stack.peek()] <= work[i]) {
                stack.push(i);
            } else {
                while (!stack.empty() && work[stack.peek()] > work[i]) {
                    topPos = stack.pop();
                    int tmp = 0;
                    if (topPos == 0) {
                        tmp = sum[0];
                    } else {
                        tmp = sum[i - 1] - sum[topPos - 1];
                    }
                    tmp *= work[topPos];
                    if (tmp > res[0]) {
                        res[0] = tmp;
                        res[1] = topPos;
                        res[2] = i;
                    }
                }
                stack.push(topPos);
                work[topPos] = work[i];
            }
        }

        return res;
    }

    public int[] firstGreaterOne(int[] arr) {
        int[] res = new int[arr.length];
        Arrays.fill(res, -1);
        Stack<Integer> stack = new Stack<>();

        for (int i = 0; i < arr.length; i ++) {
            if (stack.empty() || arr[i] <= arr[stack.peek()]) {
                stack.push(i);
            } else {
                while (!stack.empty() && arr[i] > arr[stack.peek()]) {
                    int pos = stack.pop();
                    res[pos] = i;
                }
                stack.push(i);
            }
        }
        for (int i : res) {
            out.println(i);
        }

        return res;
    }

    public int maxArea(int[] heights) {
        int[] work = new int[heights.length + 1];
        for (int i = 0; i <= heights.length; i ++) {
            if (i == heights.length) {
                work[i] = -1;
            } else {
                work[i] = heights[i];
            }
        }
        Stack<Integer> stack = new Stack<>();
        int ans = 0, topPos = 0;
        for (int i = 0; i < work.length; i ++) {
            if (stack.empty() || work[i] >= work[stack.peek()]) {
                stack.push(i);
            } else {
                while (!stack.empty() && work[i] < work[stack.peek()]) {
                    topPos = stack.pop();
                    int tmp = (i - topPos) * work[topPos];
                    if (tmp > ans) {
                        ans = tmp;
                    }
                }
                stack.push(topPos);
                work[topPos] = work[i];
            }
        }

        return ans;
    }

    /*
    * Leetcode 372. Super Pow
    * */
    //int[] tmp = {1,0};
    //out.println(solution.superPower(2,tmp));
    final int BASE = 1337;
    public int superPower(int a, int[] b) {
        LinkedList<Integer> bList = new LinkedList<>();
        for (int i : b) {
            bList.add(i);
        }
        return subSuperPower(a, bList);
    }

    public int subSuperPower(int a, LinkedList<Integer> b) {
        if (b.size() == 0) {
            return 1;
        }
        int last_digit = b.removeLast();

        return powMod(subSuperPower(a, b), 10) * powMod(a, last_digit) % BASE;
    }

    public int powMod(int a, int k) {
        a %= BASE;
        int res = 1;
        for (int i = 0; i < k; i ++) {
            res = (res * a) % BASE;
        }

        return res;
    }

    /*
    * Leetcode 368. Largest Divisible Subset
    * */
    public List<Integer> largestDivisibleSubset(int[] nums) {
        int[] dp = new int[nums.length];
        int[] parent = new int[nums.length];
        int m = 0, mi = 0;
        Arrays.sort(nums);

        for (int i = nums.length - 1; i >= 0; i --) {
            for (int j = i; j < nums.length; j ++) {
                if (nums[j] % nums[i] == 0 && dp[i] < dp[j] + 1) {
                    dp[i] = dp[j] + 1;
                    parent[i] = j;

                    if (dp[i] > m) {
                        m = dp[i];
                        mi = i;
                    }
                }
            }
        }

        List<Integer> list = new ArrayList<>();
        for (int i = 0; i < m; i ++) {
            list.add(nums[mi]);
            mi = parent[mi];
        }

        return list;
    }

    /*
    * Leetcode 367. Valid Perfect Square
    * */
    public boolean isPerfectSquare(int num) {
        if (num == 1) {
            return true;
        }
        for (int i = 1; i <= num / 2; i ++) {
            if (i * i == num) {
                return true;
            } else if (i * i > num) {
                return false;
            }
        }

        return false;
    }

    /*
    * Leetcode 357. Count Numbers with Unique Digits
    * */
    public int countNumbersWithUniqueDigits(int n) {
        if (n == 0) {
            return 1;
        }
        int sum = 10, last = 9;
        for (int i = 2; i <= n && i <= 10; i ++) {
            last = last * (9 - i + 2);
            sum += last;
        }

        return sum;
    }

    /*
    * Leetcode 350. Intersection of Two Arrays II
    * */
    public int[] intersect(int[] nums1, int[] nums2) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i : nums1) {
            if (map.containsKey(i)) {
                map.put(i, map.get(i) + 1);
            } else {
                map.put(i, 1);
            }
        }
        List<Integer> list = new ArrayList<>();
        for (int i : nums2) {
            if (map.containsKey(i) && map.get(i) != 0) {
                list.add(i);
                map.put(i, map.get(i) - 1);
            }
        }

        int[] t = new int[list.size()];
        Iterator it = list.iterator();
        for (int i = 0; it.hasNext(); i ++) {
            t[i] = (int) it.next();
        }

        return t;
    }

    /*
    * Leetcode 349. Intersection of Two Arrays
    * */
    public int[] intersection(int[] nums1, int[] nums2) {
        /* This also can use HashSet directly...or sort then compare... */
        Map<Integer, Boolean> map = new HashMap<>();
        for (int i : nums1) {
            map.put(i, true);
        }
        List<Integer> list = new ArrayList<>();
        for (int i : nums2) {
            if (map.containsKey(i) && map.get(i)) {
                list.add(i);
                map.put(i, false);
            }
        }

        int[] t = new int[list.size()];
        Iterator it = list.iterator();
        for (int i = 0; it.hasNext(); i ++) {
            t[i] = (int) it.next();
        }

        return t;
    }

    /*
    * Leetcode 347. Top K Frequent Elements
    * */
    //out.println(solution.topKFrequent(tmp, 1));
    public List<Integer> topKFrequent(int[] nums, int k) {
        Map<Integer, Integer> map = new HashMap<>();
        Arrays.sort(nums);
        int start = 0;
        while (nums[start] != nums[nums.length - 1]) {
            int pos = myBinarySearch(nums, start, nums.length - 1);
            map.put(nums[start], pos - start);
            start = pos + 1;
        }
        map.put(nums[start], nums.length - 1 - start);

        PriorityQueue<Map.Entry<Integer, Integer>> heap = new PriorityQueue<>((a, b) -> a.getValue() - b.getValue());
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            heap.offer(entry);
            if (heap.size() > k) {
                heap.poll();
            }
        }

        List<Integer> res = new LinkedList<>();
        while (!heap.isEmpty()) {
            res.add(0, heap.poll().getKey());
        }

        return res;
    }

    public int myBinarySearch(int[] nums, int from, int to) {
        int target = nums[from++];
        while (from <= to) {
            int mid = (from + to) / 2;
            if (nums[mid] > target) {
                to = mid - 1;
            } else {
                from = mid + 1;
            }
        }

        if (to < nums.length && nums[to] == target) {
            return to;
        }

        return -1;
    }

    /*
    * Leetcode 345. Reverse Vowels of a String
    * */
    //out.println(solution.reverseString("abcde"));
    public String reverseVowels(String s) {
        char[] chs = s.toCharArray();
        int left = 0, right = chs.length - 1;
        while (left <= right) {
            while (left <= right
                    && chs[left] != 'a' && chs[left] != 'e' && chs[left] != 'i' && chs[left] != 'o' && chs[left] != 'u'
                    && chs[left] != 'A' && chs[left] != 'E' && chs[left] != 'I' && chs[left] != 'O' && chs[left] != 'U') {
                left ++;
            }
            while (left <= right
                    && chs[right] != 'a' && chs[right] != 'e' && chs[right] != 'i' && chs[right] != 'o' && chs[right] != 'u'
                    && chs[right] != 'A' && chs[right] != 'E' && chs[right] != 'I' && chs[right] != 'O' && chs[right] != 'U') {
                right --;
            }
            if (left > right) {
                break;
            }
            char t = chs[left];
            chs[left] = chs[right];
            chs[right] = t;
            left ++;
            right --;
        }

        return String.valueOf(chs);
    }

    /*
    * Leetcode 344. Reverse String
    * */
    public String reverseString(String s) {
        char[] chs = s.toCharArray();
        int left = 0, right = chs.length - 1;
        while (left <= right) {
            char c = chs[left];
            chs[left] = chs[right];
            chs[right] = c;
            left ++;
            right --;
        }

        return String.valueOf(chs);
    }

    /*
    * Leetcode 343. Integer Break
    * */
    int integerBreakMax = 1;
    //out.println(solution.integerBreak(10));
    public int integerBreak(int n) {
        /* WARNING this problem has other smart math method to solve...see leetcode... */
        /* DP */
        int[] dp = new int[n+1];
        dp[1] = 1;
        dp[2] = 1;
        for (int i = 3; i <= n; i ++) {
            for (int j = 1; j < i; j ++) {
                dp[i] = Math.max(j * dp[i - j], Math.max(dp[i], j * (i - j)));
            }
        }

        return dp[n];
        /*
        Violence Solution
        Traverse all situations...

        subIntegerBreak(n, 1, true);
        return integerBreakMax;
        */
    }

    public void subIntegerBreak(int remain, int multi, boolean isFirst) {
        if (remain <= 1) {
            if (integerBreakMax < multi) {
                integerBreakMax = multi;
            }
            return ;
        }

        for (int i = 1; i <= remain; i ++) {
            if (isFirst && remain == i) {
                break;
            }
            subIntegerBreak(remain - i, multi * i, false);
        }
    }

    /*
    * Leetcode 342. Power of Four
    * */
    //out.println(solution.isPowerOfFour(8));
    public boolean isPowerOfFour(int num) {
        return num > 0 && (num & (num - 1)) == 0 && (num & 0x55555555) != 0;
    }

    /*
    * Leetcode 338. Counting Bits
    * */
    public int[] countBits(int num) {
        int[] f = new int[num + 1];
        for (int i = 1; i <= num; i ++) {
            f[i] = f[i >> 1] + (i & 1);
        }

        return f;
    }

    /*
    * Leetcode 337. House Robber III
    * */
    public int rob(TreeNode root) {
        /*
        First method... if use root, then search its children's children
        else, search its children. Finally get max.
        if (root == null) {
            return 0;
        }

        int val = 0;

        if (root.left != null) {
            val += rob(root.left.left) + rob(root.left.right);
        }

        if (root.right != null) {
            val += rob(root.right.left) + rob(root.right.right);
        }

        return Math.max(val + root.val, rob(root.left) + rob(root.right));
        */
        /*
        * Second method...
        * Only need an array which contains two elements
        * the first one means if we do not use root's result
        * the second one means if we use this root's result.
        * */
        int[] res = subRob337(root);
        return Math.max(res[0], res[1]);
    }

    public int[] subRob337(TreeNode root) {
        if (root == null) {
            return new int[2];
        }

        int[] left = subRob337(root.left);
        int[] right = subRob337(root.right);
        int[] res = new int[2];

        res[0] = Math.max(left[0], left[1]) + Math.max(right[0], right[1]);
        res[1] = root.val + left[0] + right[0];

        return res;
    }

    /*
    * Leetcode 328. Odd Even Linked List
    * */
    //ListNode a = new ListNode(1);
    //ListNode b = new ListNode(2);
    //ListNode c = new ListNode(3);
    //a.next = b;
    //b.next = c;
    //ListNode dd = solution.oddEvenList(a);
    //out.println(dd.val);
    public ListNode oddEvenList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }

        ListNode first = head, firstHead = head;
        ListNode second = head.next, secondHead = head.next;
        while (second.next != null && second.next.next != null) {
            first.next = second.next;
            first = second.next;
            second.next = first.next;
            second = first.next;
        }

        // second.next.next == null
        if (second.next != null) {
            first.next = second.next;
            first.next.next = secondHead;
            // WARNING!!! here must set to null 1->2->3
            second.next = null;
            return firstHead;
        } else {
            // fully linked
            first.next = secondHead;
            return firstHead;
        }
    }

    /*
    * Leetcode 334. Increasing Triplet Subsequence
    * */
    //int[] tmp = {1,8,3,2,3};
    //out.println(solution.increasingTriplet(tmp));
    public boolean increasingTriplet(int[] nums) {
        if (nums.length == 0) {
            return false;
        }

        int small = Integer.MAX_VALUE, big = Integer.MAX_VALUE;
        for (int n : nums) {
            if (n <= small) {
                small = n;
            } else if (n <= big) {
                big = n;
            } else {
                return true;
            }
        }

        return false;
    }

    /* Tree Traverse By Loop */
    /* Important! Interview Asked.. */
    public List<Integer> traversePreorder(TreeNode root) {
        ArrayList<Integer> list = new ArrayList<>();
        if (root == null) {
            return list;
        }

        Stack<TreeNode> stack = new Stack<>();
        while (root != null) {
            list.add(root.val);
            if (root.right != null) {
                stack.push(root.right);
            }
            root = root.left;
            if (root == null && !stack.empty()) {
                root = stack.pop();
            }
        }
        return list;
    }

    public List<Integer> traverseInorder(TreeNode root) {
        ArrayList<Integer> list = new ArrayList<>();
        if (root == null) {
            return list;
        }

        Stack<TreeNode> stack = new Stack<>();
        while (true) {
            while (root != null) {
                stack.push(root);
                root = root.left;
            }
            if (stack.empty()) {
                return list;
            }
            TreeNode t = stack.pop();
            list.add(t.val);
            root = t.right;
        }
    }

    public List<Integer> traversePostorder(TreeNode root) {
        ArrayList<Integer> list = new ArrayList<>();
        if (root == null) {
            return list;
        }

        Stack<TreeNode> stack = new Stack<>();
        TreeNode pre = null;
        while (true) {
            while (root != null) {
                stack.push(root);
                root = root.left;
            }
            if (stack.empty()) {
                break;
            } else {
                root = stack.peek();
                if (root.right != null && root.right != pre) {
                    root = root.right;
                } else {
                    list.add(root.val);
                    pre = root;
                    root = null;
                    stack.pop();
                }
            }
        }

        return list;
    }

    /* Microsoft Topic */

    /*
    * Leetcode 213. House Robber II
    * */
    public int rob(int[] nums) {
        if (nums.length == 1) {
            return nums[0];
        }
        return Math.max(subRob(nums, 0, nums.length - 2), subRob(nums, 1, nums.length - 1));
    }

    public int subRob(int[] nums, int left, int right) {
        int include = 0, exclude = 0;
        for (int i = left; i <= right; i ++) {
            int it = include, et = exclude;
            include = et + nums[i];
            exclude = Math.max(et, it);
        }

        return Math.max(include, exclude);
    }

    /*
    * Leetcode 47. Permutations II
    * */
    public List<List<Integer>> permuteUnique(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        boolean[] used = new boolean[nums.length];
        Arrays.sort(nums);
        subPermuteUnique(res, new ArrayList<>(), nums, used);
        return res;
    }

    public void subPermuteUnique(List<List<Integer>> res, List<Integer> cur, int[] nums, boolean[] used) {
        if (cur.size() == nums.length) {
            res.add(new ArrayList<Integer>(cur));
            return ;
        }

        for (int i = 0; i < nums.length; i ++) {
            if (used[i] || i > 0 && !used[i - 1] && nums[i] == nums[i-1]) {
                continue;
            }
            used[i] = true;
            cur.add(nums[i]);
            subPermuteUnique(res, cur, nums, used);
            cur.remove(cur.size() - 1);
            used[i] = false;
        }

    }


    /*
    * Leetcode 114. Flatten Binary Tree to Linked List
    * */
    TreeNode prev = null;
    public void flatten(TreeNode root) {
        if (root == null) {
            return ;
        }

        flatten(root.right);
        flatten(root.left);
        root.right = prev;
        root.left = null;
        prev = root;
    }

    public void flattenByLoop(TreeNode root) {
        if (root == null) {
            return ;
        }

        TreeNode node = root, pre = null;
        while (node != null) {
            if (node.left != null) {
                pre = node.left;
                while (pre.right != null) {
                    pre = pre.right;
                }

                pre.right = node.right;
                node.right = node.left;
                node.left = null;
            }
            node = node.right;
        }
    }

    /*
    * Leetcode 300. Longest Increasing Subsequence
    * */
    //int[] tmp = {1,3,5,6,6,8,9};
    //out.println(solution.lengthOfLIS(tmp));
    public int lengthOfLIS(int[] nums) {
        ArrayList<Integer> res = new ArrayList<>();
        for (int i : nums) {
            if (res.size() == 0 || i > res.get(res.size() - 1)) {
                res.add(i);
            } else if (i < res.get(0)) {
                res.set(0, i);
            } else {
                int pos = lowerBound(res, i);
                res.set(pos, i);
            }
        }

        return res.size();
    }

    // binarySearch return pos if target exists else return first pos where num >= target
    public int lowerBound(ArrayList<Integer> nums, int target) {
        int left = 0, right = nums.size(), mid = 0;
        while (left <= right) {
            mid = (left + right) / 2;
            if (nums.get(mid) >= target) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }

        return left;
    }

    /*
    * Leetcode 55. Jump Game
    * */
    public boolean canJump(int[] nums) {
        int max = 0;

        for (int i = 0; i < nums.length; i ++) {
            if (i > max) {
                return false;
            }
            max = Math.max(i + nums[i], max);
        }

        return true;
    }

    /*
    * Leetcode 365. Water and Jug Problem
    * */
    public boolean canMeasureWater(int x, int y, int z) {
        if (x + y < z) {
            return false;
        }

        if (x == z || y == z || x + y == z) {
            return true;
        }

        return z % GCD(x, y) == 0;
    }

    public int GCD(int a, int b) {
        while (b != 0) {
            int t = a % b;
            a = b;
            b = t;
        }
        return a;
    }

    /*
    * Leetcode 56. Merge Intervals
    * */
    public List<Interval> merge(List<Interval> intervals) {
        ArrayList<Interval> res = new ArrayList<>();

        intervals.sort((i1, i2) -> {
           return Integer.compare(i1.start, i2.start);
        });

        Interval t = null;
        while (! intervals.isEmpty()) {
            Interval first = intervals.get(0);
            intervals.remove(0);
            if (t == null) {
                t = new Interval();
                t.start = first.start;
                t.end = first.end;
                if (intervals.isEmpty()) {
                    res.add(t);
                }
            } else {
                if (first.start > t.end) {
                    res.add(t);
                    t = new Interval(first.start, first.end);
                    if (intervals.isEmpty()) {
                        res.add(t);
                    }
                } else {
                    t.end = Math.max(t.end, first.end);
                    if (intervals.isEmpty()) {
                        res.add(t);
                    }
                }
            }
        }

        return res;
    }

    /*
    * Leetcode 124. Binary Tree Maximum Path Sum
    * */
    int maxPathSumResult = Integer.MIN_VALUE;
    public int maxPathSum(TreeNode root) {
        subMaxPathSum(root);
        return maxPathSumResult;
    }

    public int subMaxPathSum(TreeNode root) {
        if (root == null) {
            return 0;
        }

        int left = Math.max(0, subMaxPathSum(root.left));
        int right = Math.max(0, subMaxPathSum(root.right));
        maxPathSumResult = Math.max(root.val + left + right, maxPathSumResult);
        return root.val + Math.max(left, right);
    }

    /*
    * Leetcode 112. Path Sum
    * */
    public boolean hasPathSum(TreeNode root, int sum) {
        if (root == null) {
            return false;
        }

        if (root.left == null && root.right == null) {
            return root.val == sum;
        }

        return hasPathSum(root.left, sum - root.val) || hasPathSum(root.right, sum - root.val);
    }

    /*
    * Leetcode 125. Valid Palindrome
    * */
    public boolean isPalindrome(String s) {
        String trimedString = getTrimedString(s);
        int i = 0, j = trimedString.length() - 1;
        while (i < j) {
            if (trimedString.charAt(i) != trimedString.charAt(j)) {
                return false;
            }
            i ++;
            j --;
        }

        return true;
    }

    public String getTrimedString(String s) {
        StringBuilder sb = new StringBuilder();
        char[] chs = s.toCharArray();
        for (char ch : chs) {
            if (ch >= 'a' && ch <= 'z') {
                sb.append(ch);
            } else if (ch >= 'A' && ch <= 'Z') {
                sb.append((char) (ch - 'A' + 'a'));
            } else if (ch >= '0' && ch <= '9') {
                sb.append(ch);
            }
        }
        return sb.toString();
    }

    /*
    * Leetcode 162. Find Peak Element
    * */
    //int[] tmp = {};
    //out.println(solution.findPeakElement(tmp));
    public int findPeakElement(int[] nums) {
        if (nums.length == 1) {
            return 0;
        }

        for (int i = 0; i < nums.length; i ++) {
            if (i == 0) {
                if (nums[0] > nums[i + 1]) {
                    return 0;
                }
            } else if (i == nums.length - 1) {
                if (nums[nums.length - 1] > nums[nums.length - 2]) {
                    return nums.length - 1;
                }
            } else {
                if (nums[i] > nums[i - 1] && nums[i] > nums[i + 1]) {
                    return i;
                }
            }
        }

        return -1;
    }

    /*
    * Leetcode 165. Compare Version Numbers
    * */
    //out.println(solution.compareVersion("1.1.2", "1.1.2.0"));
    public int compareVersion(String version1, String version2) {
        ArrayList<String> v1s = new ArrayList<>(Arrays.asList(version1.split("\\.")));
        trimRightZeros(v1s);
        ArrayList<String> v2s = new ArrayList<>(Arrays.asList(version2.split("\\.")));
        trimRightZeros(v2s);
        int i = 0;
        while (i < v1s.size() && i < v2s.size()) {
            int v1Code = Integer.parseInt(v1s.get(i));
            int v2Code = Integer.parseInt(v2s.get(i));
            if (v1Code == v2Code) {
                i ++;
            } else {
                return v1Code < v2Code ? -1 : 1;
            }
        }
        if (i < v1s.size()) {
            return 1;
        }
        if (i < v2s.size()) {
            return -1;
        }
        return 0;
    }

    public void trimRightZeros(ArrayList<String> list) {
        for (int i = list.size() - 1; i >= 0; i --) {
            int code = Integer.parseInt(list.get(i));
            if (code == 0) {
                list.remove(i);
            } else {
                break;
            }
        }
    }

    /*
    * Leetcode 91. Decode Ways
    * */
    //String s = "00";
    //out.println(solution.numDecodings(s));
    public int numDecodings(String s) {
        if (s.length() == 0) {
            return 0;
        }
        int[] dp = new int[s.length() + 1];
        // here is the point!
        // from right to left and init condition!!
        dp[s.length()] = 1;
        dp[s.length() - 1] = s.charAt(s.length() - 1) == '0' ? 0 : 1;
        for (int i = s.length() - 2; i >= 0; i --) {
            if (s.charAt(i) == '0') {
                continue;
            } else {
                if (Integer.parseInt(s.substring(i, i + 2)) <= 26) {
                    dp[i] = dp[i + 1] + dp[i + 2];
                } else {
                    dp[i] = dp[i + 1];
                }
            }
        }

        return dp[0];
    }

    /*
    * Leetcode 46. Permutations
    * */
    //char[][] board = {{'A', 'B', 'C', 'E'}, {'S', 'F', 'C', 'S'}, {'A', 'D', 'E', 'E'}};
    //out.println(solution.exist(board, "ABCCF"));
    public List<List<Integer>> permute(int[] nums) {
        ArrayList<List<Integer>> ret = new ArrayList<>();
        subPermute(ret, new ArrayList<>(), nums);
        return ret;
    }

    public void subPermute(List<List<Integer>> ret, List<Integer> tempList, int[] nums) {
        if (tempList.size() == nums.length) {
            ret.add(new ArrayList<Integer>(new ArrayList<Integer>(tempList)));
        } else {
            for (int i = 0; i < nums.length; i ++) {
                if (tempList.contains(nums[i])) {
                    continue;
                }
                tempList.add(nums[i]);
                subPermute(ret, tempList, nums);
                tempList.remove(tempList.size() - 1);
            }
        }
    }

    /*
    * Leetcode 79. Word Search
    * */
    public boolean exist(char[][] board, String word) {
        for (int i = 0; i < board.length; i ++) {
            for (int j = 0; j < board[0].length; j ++) {
                if (board[i][j] == word.charAt(0)) {
                    if (subExist(board, word, 0, i, j)) {
                        return true;
                    }
                }
            }
        }

        return false;
    }

    public boolean subExist(char[][] board, String word, int pos, int i, int j) {
        int m = board.length - 1;
        int n = board[0].length - 1;

        if (pos == word.length()) {
            return true;
        }
        if (i < 0 || j < 0 || i > m || j > n || word.charAt(pos) != board[i][j]) {
            return false;
        }
        board[i][j] = '#';
        boolean ok;
        ok = subExist(board, word, pos + 1, i + 1, j)
                || subExist(board, word, pos + 1, i, j + 1)
                || subExist(board, word, pos + 1, i - 1, j)
                || subExist(board, word, pos + 1, i, j - 1);
        board[i][j] = word.charAt(pos);
        return ok;
    }

    /*
    * Leetcode 258. Add Digits
    * */
    //out.println(solution.addDigits(12345));
    public int addDigits(int num) {
        int ret = 0;
        boolean firstLoop = true;
        while (ret >= 10 || firstLoop) {
            firstLoop = false;
            ret = 0;
            while (num != 0) {
                ret += num % 10;
                num = num / 10;
            }
            num = ret;
        }

        return ret;
    }

    /*
    * Leetcode 25. Reverse Nodes in k-Group
    * */
    //ListNode a = new ListNode(1);
    //ListNode b = new ListNode(2);
    //ListNode c = new ListNode(3);
    //ListNode d = new ListNode(4);
    //ListNode e = new ListNode(5);
    //a.next = b;
    //b.next = c;
    //c.next = d;
    //d.next = e;
    //solution.reverseKGroup(a, 2);
    public ListNode reverseKGroup(ListNode head, int k) {
        if (head == null || head.next == null || k < 2) {
            return head;
        }
        ListNode newHead = head, node = head, lastTail = head;
        while (hasKGroup(node, k)) {
            ListNode last = node, cur = node.next, t = null;
            for (int i = 0; i < k - 1; i ++) {
                t = cur.next;
                cur.next = last;
                last = cur;
                cur = t;
            }
            if (newHead != head) {
                lastTail.next = last;
            }
            if (newHead == head) {
                newHead = last;
            }
            lastTail = node;
            node = cur;
        }
        if (newHead != head) {
            lastTail.next = node;
        }

        return newHead;
    }

    public boolean hasKGroup(ListNode node, int k) {
        while (node != null && k != 0) {
            node = node.next;
            k --;
        }
        if (k > 0) {
            return false;
        } else {
            return true;
        }
    }

    /*
    * Leetcode 5. Longest Palindromic Substring
    * */
    //out.println(solution.longestPalindrome("ccc"));
    public String longestPalindrome(String s) {
        boolean[][] dp = new boolean[s.length()][s.length()];

        String max = "";
        for (int i = 0; i < s.length(); i ++) {
            dp[i][i] = true;
            if (max.length() < 1) {
                max = s.substring(i, i + 1);
            }
        }
        for (int i = 0; i < s.length() - 1; i ++) {
            if (s.charAt(i) == s.charAt(i + 1)) {
                dp[i][i + 1] = true;
                if (max.length() < 2) {
                    max = s.substring(i, i + 2);
                }
            }
        }

        for (int i = 2; i < s.length(); i ++) {
            for (int j = 0; j < s.length() - i; j ++) {
                if (dp[j + 1][j + i - 1] && s.charAt(j) == s.charAt(j + i)) {
                    dp[j][j + i] = true;
                    if (max.length() < i + 1) {
                        max = s.substring(j, j + i + 1);
                    }
                }
            }
        }

        return max;
    }

    /*
    * Leetcode 15. 3Sum
    * */
    //int[] tmp = {-1,0,1,2,-1,-4};
    //solution.threeSum(tmp);
    public List<List<Integer>> threeSum(int[] nums) {
        ArrayList<List<Integer>> res = new ArrayList<>();
        if (nums.length < 3) {
            return res;
        }
        Arrays.sort(nums);
        for (int i = 0; i < nums.length - 2; i ++) {
            if (i > 0 && nums[i] == nums[i-1]) {
                continue;
            }

            if (nums[i] == 0) {
                if (i + 1 < nums.length && i + 2 < nums.length) {
                    if (nums[i + 1] == 0 && nums[i + 2] == 0) {
                        ArrayList<Integer> t = new ArrayList<>();
                        t.add(0);
                        t.add(0);
                        t.add(0);
                        res.add(t);
                    }
                }
                break;
            }

            if (nums[i] > 0) {
                break;
            }

            int left = i + 1, right = nums.length - 1;
            while (left < right) {
                int sum =  nums[i] + nums[left] + nums[right];
                if (sum == 0) {
                    ArrayList<Integer> t = new ArrayList<>();
                    t.add(nums[i]);
                    t.add(nums[left]);
                    t.add(nums[right]);
                    res.add(t);
                    while (left < right && nums[left] == nums[left + 1]) {
                        left ++;
                    }
                    if (left < right && nums[right] == nums[right - 1]) {
                        right --;
                    }
                    left ++;
                    right --;
                } else if (sum < 0) {
                    left ++;
                } else {
                    right --;
                }

            }
        }

        return res;
    }

    /*
    * Leetcode 204. Count Primes
    * */
    //out.println(solution.countPrimes(10));
    public int countPrimes(int n) {
        boolean[] notPrime = new boolean[n];
        int count = 0;
        int lastPrime = 2;
        for (int i = 2; i < n; i ++) {
            if (notPrime[i] == false) {
                lastPrime = i;
                count ++;
                // This will reduce for times...
                if (lastPrime * lastPrime >= n) {
                    continue;
                }
                for (int j = 2; i * j < n; j ++) {
                    notPrime[i * j] = true;
                }
            }
        }

        return count;
    }

    public boolean isPrime(int n) {
        for (int i = 2; i < Math.sqrt(n); i ++) {
            if (n % i == 0) {
                return false;
            }
        }
        return true;
    }

    /*
    * Leetcode 106. Construct Binary Tree from Inorder and Postorder Traversal
    * */
    //int[] tmp = {9,3,15,20,7};
    //int[] tmp1 = {9,15,7,20,3};
    //solution.buildTree(tmp, tmp1);
    public TreeNode buildTree(int[] inorder, int[] postorder) {
        return subBuildTree(inorder, postorder, 0, inorder.length - 1, postorder.length - 1);
    }

    public TreeNode subBuildTree(int[] inorder, int[] postorder, int inStart, int inEnd, int postPos) {
        if (inStart > inEnd || postPos < 0) {
            return null;
        }
        int pos = findPos(inorder, postorder[postPos]);
        TreeNode node = new TreeNode(postorder[postPos]);
        node.left = subBuildTree(inorder, postorder, inStart, pos - 1, postPos - (inEnd - pos) - 1);
        node.right = subBuildTree(inorder, postorder, pos + 1, inEnd, postPos - 1);
        return node;
    }

    public int findPos(int[] order, int target) {
        for (int i = 0; i < order.length; i ++) {
            if (order[i] == target) {
                return i;
            }
        }

        return -1;
    }

    /*
    * Leetcode 387. First Unique Character in a String
    * */
    //out.println(solution.firstUniqChar("lll"));
    public int firstUniqChar(String s) {
        int freq[] = new int[26];
        for (int i = 0; i < s.length(); i ++) {
            freq[s.charAt(i) - 'a'] ++;
        }
        for (int i = 0; i < s.length(); i ++) {
            if (freq[s.charAt(i) - 'a'] == 1) {
                return i;
            }
        }
        return -1;
        /*
        Practice API...

        LinkedHashMap<Character, Integer> map = new LinkedHashMap<>();
        char[] arr = s.toCharArray();

        for (char c : arr) {
            if (map.containsKey(c)) {
                map.put(c, map.get(c) + 1);
            } else {
                map.put(c, 0);
            }
        }

        for (Map.Entry<Character, Integer> entry : map.entrySet()) {
            if (entry.getValue() == 0) {
                return s.indexOf(entry.getKey());
            }
        }

        return -1;
        */
    }

    /*
    * Leetcode 4. Median of Two Sorted Arrays
    * */
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        /* O(log(m + n)) Solution */
        return 0.0;

        /*
        O(m + n) Solution, just like sorted linked-list merge

        int len1 = nums1.length, len2 = nums2.length;
        int mid = (len1 + len2) / 2;
        int[] ret = new int[2];
        if ((len1 + len2) % 2 == 0) {
            ret[0] = mid;
            ret[1] = mid + 1;
        } else {
            ret[0] = mid + 1;
            ret[1] = mid + 1;
        }
        int i = 0, j = 0, count = 0;
        int res = 0;
        while (i < len1 && j < len2) {
            count ++;
            if (nums1[i] < nums2[j]) {
                if (count == ret[0]) {
                    res += nums1[i];
                }
                if (count == ret[1]) {
                    count ++;
                    res += nums1[i];
                    break;
                }
                i ++;
            } else {
                if (count == ret[0]) {
                    res += nums2[j];
                }
                if (count == ret[1]) {
                    count ++;
                    res += nums2[j];
                    break;
                }
                j ++;
            }
        }
        if (count < ret[1]) {
            if (i >= len1) {
                // find nums2
                while (j < len2) {
                    count ++;
                    if (count == ret[0]) {
                        res += nums2[j];
                    }
                    if (count == ret[1]) {
                        res += nums2[j];
                        break;
                    }
                    j ++;
                }
            }
            if (j >= len2) {
                // find nums1
                while (i < len1) {
                    count ++;
                    if (count == ret[0]) {
                        res += nums1[i];
                    }
                    if (count == ret[1]) {
                        res += nums1[i];
                        break;
                    }
                    i ++;
                }
            }
        }

        return res / 2.0;
        */
    }

    /*
    * Leetcode 71. Simplify Path
    * */
    //out.println(solution.simplifyPath("/a/b//../c"));
    public String simplifyPath(String path) {
        String[] paths = path.split("/");
        LinkedList<String> rtn = new LinkedList<>();
        for (String s : paths) {
            if (s.equals("") || s.equals(".")) {
                continue;
            }
            if (s.equals("..")) {
                rtn.pollLast();
                continue;
            }
            rtn.addLast(s);
        }

        return "/" + String.join("/", rtn);
    }

    /*
    * Leetcode 26. Remove Duplicates from Sorted Array
    * */
    //int[] tmp = {1,1,1,2,3,3,4};
    //solution.removeDuplicates(tmp);
    public int removeDuplicates(int[] nums) {
        int n = nums.length;
        if (n <= 1) {
            return n;
        }

        int id = 1;
        for (int i = 1; i < n; i ++) {
            if (nums[i] != nums[i-1]) {
                nums[id ++] = nums[i];
            }
        }

        return id;
    }

    /*
    * Leetcode 215. Kth Largest Element in an Array
    * */
    //int[] tmp = {3,1,2,4};
    //out.println(solution.findKthLargest(tmp, 2));
    public int findKthLargest(int[] nums, int k) {
        int left = 0, right = nums.length - 1;
        int pos = lomutoPartition(nums, left, right);
        k = nums.length - k;
        while (pos != k) {
            if (pos > k) {
                right = pos - 1;
                pos = lomutoPartition(nums, left, right);
            } else {
                left = pos + 1;
                pos = lomutoPartition(nums, left, right);
            }
        }

        return nums[pos];
    }

    /* Recommend this one */
    public int bookPartition(int[] nums, int start, int end) {
        int pivot = nums[start];
        while (start < end) {
            while (start < end && nums[end] >= pivot) {
                end --;
            }
            nums[start] = nums[end];
            while (start < end && nums[start] <= pivot) {
                start ++;
            }
            nums[end] = nums[start];
        }
        nums[start] = pivot;
        return start;
    }

    public int lomutoPartition(int[] nums, int start, int end) {
        int pivot = nums[end];
        int i = start - 1;
        for (int j = start; j < end; j ++) {
            if (nums[j] <= pivot) {
                i ++;
                int t = nums[i];
                nums[i] = nums[j];
                nums[j] = t;
            }
        }
        int t = nums[i + 1];
        nums[i + 1] = pivot;
        nums[end] = t;
        return i + 1;
    }

    public int hoarePartition(int[] nums, int start, int end) {
        int pivot = nums[start];
        int i = start - 1, j = end + 1;
        while (true) {
            do {
                i ++;
            } while (nums[i] < pivot);

            do {
                j --;
            } while (nums[j] > pivot);

            if (i >= j) {
                return j;
            }

            int t = nums[i];
            nums[i] = nums[j];
            nums[j] = t;
        }

    }

    /*
    * Leetcode 160. Intersection of Two Linked Lists
    * */
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode n1 = headA, n2 = headB;

        while (n1 != n2) {
            n1 = n1 == null ? headB : n1.next;
            n2 = n2 == null ? headA : n2.next;
        }

        return n1;
    }

    /*
    * Leetcode 191. Number of 1 Bits
    * */
    public int hammingWeight(int n) {
        int count = 0;
        while (n != 0) {
            count += (n & 1);
            n  = n >>> 1;
        }
        return count;
    }

    /*
    * Leetcode 24. Swap Nodes in Pairs
    * */
    //ListNode a = new ListNode(1);
    //ListNode b = new ListNode(2);
    //ListNode c = new ListNode(3);
    //ListNode d = new ListNode(4);
    //a.next = b;
    //b.next = c;
    //c.next = d;
    //solution.swapPairs(a);
    public ListNode swapPairs(ListNode head) {
        /*
        Iterative
        */
        if (head == null || head.next == null) {
            return head;
        }

        ListNode cur = head, newHead = head.next;
        while (cur != null) {
            ListNode t = cur;
            cur = cur.next;
            t.next = cur.next;
            cur.next = t;
            cur = t.next;
            // change to right pos
            if (cur != null && cur.next != null) {
                t.next = cur.next;
            }
        }

        return newHead;

        /*
        Recursive
        if (head == null || head.next == null) {
            return head;
        }

        ListNode node = head.next;
        head.next = swapPairs(node.next);
        node.next = head;
        return node;
        */
    }

    /*
    * Leetcode 33. Search in Rotated Sorted Array
    * */
    //int[] tmp = { 4,5,6,7,0,1,2 };
    //int[] tmp = { 1,3 };
    //int[] tmp = { 5,1,2,3,4 };
    //out.println(solution.search(tmp, 1));
    public int search(int[] nums, int target) {
        int left = 0, right = nums.length;
        while (left <= right) {
            int mid = (left + right) / 2;
            if (mid < 0 || mid >= nums.length) {
                return -1;
            }
            if (nums[mid] == target) {
                return mid;
            }
            // primitive right part
            // !!! here must have =
            if (nums[mid] >= nums[0]) {
                // must search right
                if (nums[mid] < target) {
                    left = mid + 1;
                } else {
                    // maybe left or right
                    if (nums[0] == target) {
                        return 0;
                    }
                    // search right
                    if (nums[0] > target) {
                        left = mid + 1;
                    } else {
                        // search left
                        right = mid - 1;
                    }
                }
            } else {
                // primitive left part
                // must search left
                if (nums[mid] > target) {
                    right = mid - 1;
                } else {
                    // maybe left or right
                    if (nums[nums.length - 1] == target) {
                        return nums.length - 1;
                    }
                    if (nums[nums.length - 1] > target) {
                        // search right
                        left = mid + 1;
                    } else {
                        // search left
                        right = mid - 1;
                    }
                }
            }
        }

        return -1;
    }

    /*
    * Leetcode 13. Roman to Integer
    * */
    //out.println(solution.romanToInt("III"));
    public int romanToInt(String s) {
        Map<String, Integer> charToInt = new HashMap<>();
        charToInt.put("I", 1);
        charToInt.put("V", 5);
        charToInt.put("X", 10);
        charToInt.put("L", 50);
        charToInt.put("C", 100);
        charToInt.put("D", 500);
        charToInt.put("M", 1000);
        int result = 0;
        int index = s.length() - 1;
        int preInt = 0;
        while (index >= 0) {
            int curInt = charToInt.get(s.substring(index, index + 1));
            if (curInt >= preInt) {
                result += curInt;
            } else {
                result -= curInt;
            }

            preInt = curInt;
            index--;
        }

        return result;
    }

    /*
    * Leetcode 98. Validate Binary Search Tree
    * */
    public boolean isValidBST(TreeNode root) {
        if (root == null) {
            return true;
        }
        Stack<TreeNode> stack = new Stack<>();
        Integer lastVal = null;
        while (true) {
            while (root != null) {
                stack.push(root);
                root = root.left;
            }
            if (stack.empty()) {
                return true;
            }
            TreeNode t = stack.pop();
            if (lastVal == null) {
                lastVal = t.val;
            } else {
                if (lastVal >= t.val) {
                    return false;
                }
                lastVal = t.val;
            }
            root = t.right;
        }
    }

    public boolean isValidBSTRecursive(TreeNode root, TreeNode min, TreeNode max) {
        if (root == null) {
            return true;
        }
        return (min == null || root.val > min.val) && (max == null || root.val < max.val)
                && isValidBSTRecursive(root.left, min, root) && isValidBSTRecursive(root.right, root, max);
    }

    /*
    * Leetcode 20. Valid Parentheses
    * */
    //out.println(solution.isValid("()[]{}"));
    public boolean isValid(String s) {
        Stack<String> stack = new Stack<>();
        for (int i = 0; i < s.length(); i ++) {
            String cur = s.substring(i, i + 1);
            if (cur.equals("(") || cur.equals("{") || cur.equals("[")) {
                stack.push(cur);
            } else {
                if (stack.empty()) {
                    return false;
                }
                String top = stack.pop();
                if (top.equals("(")) {
                    if (! cur.equals(")")) {
                        return false;
                    }
                } else if (top.equals("{")) {
                    if (! cur.equals("}")) {
                        return false;
                    }
                } else if (top.equals("[")) {
                    if (! cur.equals("]")) {
                        return false;
                    }
                } else {
                    return false;
                }
            }
        }

        return stack.empty();
    }

    /*
    * Leetcode 174. Dungeon Game
    * */
    //int[][] tmp = {{-2,-3,3},{-5,-10,1},{10,30,-5}};
    //out.println(solution.calculateMinimumHP(tmp));
    public int calculateMinimumHP(int[][] dungeon) {
        int m = dungeon.length;
        if (m == 0) {
            return 0;
        }
        int n = dungeon[0].length;

        // init dp array
        int dp[][] = new int[m][n];
        dp[m-1][n-1] = Math.max(1 - dungeon[m-1][n-1], 1);
        for (int i = m - 2; i >= 0; i --) {
            dp[i][n-1] = Math.max(dp[i+1][n-1] - dungeon[i][n-1], 1);
        }
        for (int i = n - 2; i >= 0; i --) {
            dp[m-1][i] = Math.max(dp[m-1][i + 1] - dungeon[m-1][i], 1);
        }

        // dp
        for (int i = m - 2; i >= 0; i --) {
            for (int j = n - 2; j >= 0; j --) {
                int lastRow = Math.max(dp[i+1][j] - dungeon[i][j], 1);
                int lastCol = Math.max(dp[i][j+1] - dungeon[i][j], 1);
                dp[i][j] = Math.min(lastCol, lastRow);
            }
        }

        return dp[0][0];
    }

    /*
    * Leetcode 8. String to Integer (atoi)
    * */
    public int myAtoi(String str) {
        String trimStr = trimData(str);
        if (trimStr == null) {
            return 0;
        }
        if (trimStr.length() > 11) {
            if (trimStr.charAt(0) == '+') {
                return Integer.MAX_VALUE;
            } else {
                return Integer.MIN_VALUE;
            }
        }

        int num = 0;
        boolean positive = true;
        if (trimStr.charAt(0) == '-') {
            positive = false;
        }
        for (int i = 1; i < trimStr.length(); i ++) {
            if (i == 10) {
                // test if outbound
                if (checkOutofBound(num, trimStr.charAt(i) - '0')) {
                    if (positive) {
                        return Integer.MAX_VALUE;
                    } else {
                        return Integer.MIN_VALUE;
                    }
                } else {
                    if (positive) {
                        num = 10 * num + (trimStr.charAt(i) - '0');
                    } else {
                        num = 10 * num - (trimStr.charAt(i) - '0');
                    }
                }
            } else {
                if (positive) {
                    num = 10 * num + (trimStr.charAt(i) - '0');
                } else {
                    num = 10 * num - (trimStr.charAt(i) - '0');
                }
            }
        }

        return num;
    }

    public boolean checkOutofBound(int num, int extra) {
        if (num < 0) {
            int t = num * 10 - extra;
            if (t < 0) {
                return false;
            } else {
                return true;
            }
        } else {
            int t = num * 10 + extra;
            if (t > 0) {
                return false;
            } else {
                return true;
            }
        }
    }

    public String trimData(String str) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < str.length(); i ++) {
            char ch = str.charAt(i);
            if (ch == ' ') {
                continue;
            }
            if ((ch <= '9' && ch >= '0') || (ch == '-') || (ch == '+')) {
                sb.append(ch);
            } else {
                break;
            }
        }
        if (sb.length() == 0) {
            return null;
        }
        char ch = sb.charAt(sb.length() - 1);
        if (ch >= '0' && ch <= '9') {
            boolean negative = false;
            for (int i = 0; i < sb.length(); i ++) {
                if (sb.charAt(i) == '-') {
                    negative = ! negative;
                } else if (sb.charAt(i) == '+') {
                } else {
                    if (negative) {
                        return "-" + sb.substring(i);
                    } else {
                        return "+" + sb.substring(i);
                    }
                }
            }
            return sb.toString();
        } else {
            return null;
        }
    }

    /*
    * Leetcode 2. Add Two Numbers
    * */
    public ListNode addTwoNumbers2(ListNode l1, ListNode l2) {
        ListNode n1 = l1, n2 = l2;
        ListNode head = null, last = null, cur = null;
        int over = 0;
        while (n1 != null && n2 != null) {
            int sum = n1.val + n2.val + over;
            over = sum / 10;
            cur = new ListNode(sum % 10);
            if (last != null) {
                last.next = cur;
            } else {
                head = cur;
            }
            last = cur;
            n1 = n1.next;
            n2 = n2.next;
        }

        while (n1 != null) {
            int sum = n1.val + over;
            over = sum / 10;
            cur = new ListNode(sum % 10);
            if (last != null) {
                last.next = cur;
            } else {
                head = cur;
            }
            last = cur;
            n1 = n1.next;
        }

        while (n2 != null) {
            int sum = n2.val + over;
            over = sum / 10;
            cur = new ListNode(sum % 10);
            if (last != null) {
                last.next = cur;
            } else {
                head = cur;
            }
            last = cur;
            n2 = n2.next;
        }

        if (over > 0) {
            cur = new ListNode(over);
            if (last != null) {
                last.next = cur;
            } else {
                head = cur;
            }
            last = cur;
        }

        return head;
    }

    /*
    * Leetcode 168. Excel Sheet Column Title
    * */
    public String convertToTitle(int n) {
        String ret = "";
        while (n > 0) {
            int i = n % 26;
            ret = String.valueOf((char) ('A' + i - 1)) + ret;
            if (i == 0) {
                n --;
            }
            n /= 26;
        }
        return ret;

        /*
        Recursive
        if (n == 0) {
            return "";
        }
        return convertToTitle((n - 1) / 26) + String.valueOf((char) ('A' + (n - 1) % 26));
        */
    }

    /*
    * Leetcode 445. Add Two Numbers II
    * */
    //ListNode a = new ListNode(7);
    //ListNode b = new ListNode(2);
    //ListNode d = new ListNode(4);
    //ListNode e = new ListNode(3);
    //a.next = b;
    //b.next = d;
    //d.next = e;
    //ListNode f = new ListNode(5);
    //ListNode g = new ListNode(6);
    //ListNode h = new ListNode(4);
    //f.next = g;
    //g.next = h;
    //solution.addTwoNumbers(a, f);
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode n1 = l1, n2 = l2;
        if (l1 == null) {
            return l2;
        }
        if (l2 == null) {
            return l1;
        }
        /*
        * Find length diff then
        * Mark start and end
        * start      end
        *   |         |
        *   1 -> 2 -> 3 -> 4
        *             2 -> 3
        * */
        while (n1 != null && n2 != null) {
            n1 = n1.next;
            n2 = n2.next;
        }
        ListNode start = null, end = null;
        if (n1 == null && n2 != null) {
            // l2 is longer
            n1 = l2;
            start = l2;
            while (n2 != null) {
                n1 = n1.next;
                n2 = n2.next;
            }
            end = n1;
            // rewind n2 to l1
            n2 = l1;
        } else if (n1 != null && n2 == null) {
            // l1 is longer
            n2 = l1;
            start = l1;
            while (n1 != null) {
                n1 = n1.next;
                n2 = n2.next;
            }
            end = n2;
            // rewind n1 to l2
            n1 = l2;
        } else {
            // same length, rewind both
            n1 = l1;
            n2 = l2;
        }

        Object[] rtns = subAddTwoNumbers(n1, n2);
        ListNode node = (ListNode) rtns[0];
        /*
        * Now we should add returned val to longer linked-list start-end part
        * */
        Object[] leftRtns = subAddOneNumber(start, end, node, (int) rtns[1]);
        if (leftRtns[0] != null) {
            node = (ListNode) leftRtns[0];
            if ((int) leftRtns[1] != 0) {
                node = new ListNode((int) leftRtns[1]);
                node.next = (ListNode) leftRtns[0];
            }
        } else {
            if ((int) rtns[1] != 0) {
                node = new ListNode((int) rtns[1]);
                node.next = (ListNode) rtns[0];
            }
        }

        return node;
    }

    public Object[] subAddOneNumber(ListNode start, ListNode end, ListNode newNode, int val) {
        Object[] rtns = new Object[2];
        rtns[0] = null;
        rtns[1] = 0;

        if (start == end) {
            rtns[0] = newNode;
            rtns[1] = val;
            return rtns;
        }

        rtns = subAddOneNumber(start.next, end, newNode, val);
        int sum = (int) rtns[1] + start.val;
        ListNode node = new ListNode(sum % 10);
        node.next = (ListNode) rtns[0];
        rtns[0] = node;
        rtns[1] = sum / 10;
        return rtns;
    }

    public Object[] subAddTwoNumbers(ListNode n1, ListNode n2) {
        Object[] rtns = new Object[2];
        rtns[0] = null;
        rtns[1] = 0;
        if (n1 == null && n2 == null) {
            return rtns;
        }

        if (n1 != null) {
            if (n2 == null) {
                rtns = subAddTwoNumbers(n1.next, n2);
            } else {
                rtns = subAddTwoNumbers(n1.next, n2.next);
            }
        } else {
            if (n2 != null) {
                rtns = subAddTwoNumbers(n1, n2.next);
            }
        }

        // backtrace...
        int sum = 0;
        if (n1 != null) {
            sum += n1.val;
        }
        if (n2 != null) {
            sum += n2.val;
        }
        sum += (int) rtns[1];
        ListNode cur = new ListNode(sum % 10);
        cur.next = (ListNode) rtns[0];
        rtns[0] = cur;
        rtns[1] = sum / 10;
        return rtns;
    }

    /*
    * Leetcode 237. Delete Node in a Linked List
    * */
    public void deleteNode(ListNode node) {
        if (node == null) {
            return ;
        }
        if (node.next != null) {
            node.val = node.next.val;
            node.next = node.next.next;
        } else {
            node = null;
        }
    }

    /*
    * Leetcode 141. Linked List Cycle
    * */
    public boolean hasCycle(ListNode head) {
        if (head == null) {
            return false;
        }
        ListNode slow = head, fast = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
            if (slow == fast) {
                return true;
            }
        }

        return false;
    }

    /*
    * Leetcode 121. Best Time to Buy and Sell Stock
    * */
    public int maxProfit121(int[] prices) {
        if (prices.length == 0) {
            return 0;
        }

        int min = prices[0], max = 0;
        for (int i = 1; i < prices.length; i ++) {
            if (min > prices[i]) {
                min = prices[i];
            }
            if (prices[i] - min > max) {
                max = prices[i] - min;
            }
        }

        return max;
    }

    /*
    * Leetcode 73. Set Matrix Zeroes
    * 思路: 把中间位置出现 0 的地方映射到对应行的头(或者列的头,只需要一个维度即可)
    * 然后只需根据第一行和第一列就可以对矩阵进行 0 填充即可
    * 用一个变量 firstCol 标志第一列是否需要变成 0,
    * 因为我们把内部的 0 映射到了行首, 导致没有这个变量无法判断行首的 0 是本来就有的, 还是映射过来才出现的
    * */
    public void setZeroes(int[][] matrix) {
        int m = matrix.length;
        if (m == 0) {
            return ;
        }
        int n = matrix[0].length;
        boolean firstCol = false;

        for (int i = 0; i < m; i ++) {
            if (matrix[i][0] == 0) {
                firstCol = true;
            }
            for (int j = 1; j < n; j ++) {
                if (matrix[i][j] == 0) {
                    matrix[i][0] = 0;
                    matrix[0][j] = 0;
                }
            }
        }

        for (int i = m - 1; i >= 0; i --) {
            for (int j = n - 1; j >= 1; j --) {
                if (matrix[i][0] == 0 || matrix[0][j] == 0) {
                    matrix[i][j] = 0;
                }
            }
            if (firstCol) {
                matrix[i][0] = 0;
            }
        }

    }

    /*
    * Leetcode 238. Product of Array Except Self
    * */
    //int[] tmp = {1, 2, 3, 4, 5};
    //out.println(Arrays.toString(solution.productExceptSelf(tmp)));
    public int[] productExceptSelf(int[] nums) {
        int[] leftProducts = new int[nums.length];
        if (nums.length == 0) {
            return leftProducts;
        }

        /* Method 1 */
        int left = 1, right = 1, n = nums.length;
        Arrays.fill(leftProducts, 1);
        for (int i = 0; i < n; i ++) {
            leftProducts[i] *= left;
            left *= nums[i];
            leftProducts[n-1-i] *= right;
            right *= nums[n-1-i];
        }

        /*
        Method 2
        leftProducts[0] = 1;
        for (int i = 1; i < nums.length; i ++) {
            leftProducts[i] = leftProducts[i - 1] * nums[i - 1];
        }

        int right = 1;
        for (int i = nums.length - 1; i >= 0; i --) {
            leftProducts[i] *= right;
            right *= nums[i];
        }
        */

        return leftProducts;
    }

    /*
    * Leetcode 103. Binary Tree Zigzag Level Order Traversal
    * */
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>> ret = new LinkedList<>();
        if (root == null) {
            return ret;
        }
        LinkedList<TreeNode> curNodes = new LinkedList<>();
        List<Integer> curVals = null;
        curNodes.add(root);
        boolean reverse = false;

        while (curNodes.size() != 0) {
            LinkedList<TreeNode> nextNodes = new LinkedList<>();
            curVals = new LinkedList<>();
            if (reverse) {
                TreeNode node = curNodes.pollLast();
                while (node != null) {
                    curVals.add(node.val);
                    if (node.right != null) {
                        nextNodes.addFirst(node.right);
                    }
                    if (node.left != null) {
                        nextNodes.addFirst(node.left);
                    }
                    node = curNodes.pollLast();
                }
            } else {
                TreeNode node = curNodes.poll();
                while (node != null) {
                    curVals.add(node.val);
                    if (node.left != null) {
                        nextNodes.add(node.left);
                    }
                    if (node.right != null) {
                        nextNodes.add(node.right);
                    }
                    node = curNodes.poll();
                }
            }
            curNodes = nextNodes;
            ret.add(curVals);

            reverse = ! reverse;
        }

        return ret;
    }

    /*
    * Leetcode 75. Sort Colors
    * */
    public void sortColors(int[] nums) {
        int left = 0, mid = 0, right = nums.length - 1;
        while (mid <= right) {
            if (nums[mid] == 0) {
                int t = nums[mid];
                nums[mid] = nums[left];
                nums[left] = t;
                left ++;
                mid ++;
            } else if (nums[mid] == 2) {
                int t = nums[right];
                nums[right] = nums[mid];
                nums[mid] = t;
                right --;
            } else {
                mid ++;
            }
        }
    }

    /*
    * Leetcode 116. Populating Next Right Pointers in Each Node
    * */
    public class TreeLinkNode {
        int val;
        TreeLinkNode left, right, next;
        TreeLinkNode(int x) { val = x; }
    }
    public void connect(TreeLinkNode root) {
        if (root == null) {
            return ;
        }

        TreeLinkNode pre = root, last_pre = root, cur = root;
        boolean firstLoop = true;
        while (firstLoop || pre != last_pre) {
            firstLoop = false;
            last_pre = pre;
            cur = pre;
            TreeLinkNode left = null;
            while (cur != null) {
                if (cur.left != null) {
                    if (left == null) {
                        left = cur.left;
                        pre = left;
                    } else {
                        left.next = cur.left;
                        left = cur.left;
                    }
                }
                if (cur.right != null) {
                    if (left == null) {
                        left = cur.right;
                        pre = left;
                    } else {
                        left.next = cur.right;
                        left = cur.right;
                    }
                }

                cur = cur.next;
            }
        }
    }

    /*
    * Leetcode 151. Reverse Words in a String
    * Sorry, this can not Accept...because shit input and shit output
    * */
    //String t = " hello world hiho ";
    //out.println(solution.reverseWords(t));
    public String reverseWords(String s) {
        char[] rev = s.toCharArray();
        subReverseWords(rev, 0, s.length() - 1);
        int wordStart = 0;
        for (int i = 0; i < rev.length; i ++) {
            if (rev[i] == ' ') {
                if (i > wordStart) {
                    subReverseWords(rev, wordStart, i - 1);
                }
                wordStart = i + 1;
            }
        }
        if (wordStart < rev.length) {
            subReverseWords(rev, wordStart, rev.length - 1);
        }
        return String.valueOf(rev);
    }

    public void subReverseWords(char[] chs, int start, int end) {
        while (start <= end) {
            char t = chs[start];
            chs[start] = chs[end];
            chs[end] = t;
            start ++;
            end --;
        }
    }

    /*
    * Leetcode 53. Maximum Subarray
    * */
    //int[] tmp = {-2,1,-3,4,-1,2,1,-5,4};
    //int[] tmp = {-1,-2};
    //out.println(solution.maxSubArray(tmp));
    public int maxSubArray(int[] nums) {
        if (nums.length == 0) {
            return 0;
        }
        int[] dp = new int[nums.length];
        int max = nums[0];
        dp[0] = nums[0];

        for (int i = 1; i < nums.length; i ++) {
            dp[i] = nums[i] + (dp[i - 1] > 0 ? dp[i - 1] : 0);
            max = Math.max(max, dp[i]);
        }

        return max;
    }

    /*
    * Leetcode 268. Missing Number
    * */
    //int[] tmp = {0,2,3,4};
    //out.println(solution.missingNumber(tmp));
    public int missingNumber(int[] nums) {
        int sum = 0, max = Integer.MIN_VALUE;
        for (int i = 0; i < nums.length; i ++) {
            sum += nums[i];
            if (max < nums[i]) {
                max = nums[i];
            }
        }
        if (max == nums.length - 1) {
            return nums.length;
        } else {
            return (nums.length + 1) * (max) / 2 - sum;
        }
    }

    /*
    * Leetcode 200. Number of Islands
    * */
    /*
    * Data Structure: Union-Find Set
    * */
    class UnionFindSet {
        private int[] id;
        private int[] idSize;
        private int count;

        public UnionFindSet(int n) {
            count = n;
            id = new int[n];
            idSize = new int[n];
            for (int i = 0; i < n; i ++) {
                id[i] = i;
                idSize[i] = 1;
            }
        }

        // used for leetcode 200
        public UnionFindSet(char[][] grid) {
            int n = grid.length * grid[0].length;
            id = new int[n];
            idSize = new int[n];

            for (int i = 0; i < grid.length; i ++) {
                for (int j = 0; j < grid[i].length; j ++) {
                    if (grid[i][j] == '1') {
                        int pos = i * grid[0].length + j;
                        id[pos] = pos;
                        idSize[pos] = 1;
                        count ++;
                    }
                }
            }
        }

        public int count() {
            return count;
        }

        public boolean connected(int p, int q) {
            return find(p) == find(q);
        }

        public int find(int p) {
            // set p's parent to p's grandfather to compact path length
            while (p != id[p]) {
                id[p] = id[id[p]];
                p = id[p];
            }

            return p;
        }

        public void union(int p, int q) {
            // find p and q's root
            int pID = find(p);
            int qID = find(q);
            // if root is equal, just return
            if (pID == qID) {
                return ;
            }

            // choose smaller tree and set it's parent to larger tree
            // then the resulted tree will be weighted
            if (idSize[pID] < idSize[qID]) {
                id[pID] = qID;
                idSize[qID] += idSize[pID];
            } else {
                id[qID] = pID;
                idSize[pID] += idSize[qID];
            }
            count --;
        }
    }
    //char[][] tmp = {{'1','1','1','1','0'},{'1','1','0','1','0'},{'1','1','0','0','0'},{'0','0','0','0','0'}};
    //out.println(solution.numIslands(tmp));
    public int numIslands(char[][] grid) {
        int[][] distances = { {1, 0}, {-1, 0}, {0, 1}, {0, -1} };
        int rows = grid.length;
        if (rows == 0) {
            return 0;
        }
        int cols = grid[0].length;
        UnionFindSet ufs = new UnionFindSet(grid);
        for (int i = 0; i < grid.length; i ++) {
            for (int j = 0; j < grid[i].length; j ++) {
                if (grid[i][j] == '1') {
                    for (int[] d : distances) {
                        int x = i + d[0];
                        int y = j + d[1];
                        if (x >= 0 && x < rows && y >= 0 && y < cols && grid[x][y] == '1') {
                            int id1 = i * cols + j;
                            int id2 = x * cols + y;
                            ufs.union(id1, id2);
                        }
                    }
                }
            }
        }

        return ufs.count();
    }

    /*
    * Leetcode 171. Excel Sheet Column Number
    * */
    //out.println(solution.titleToNumber("BA"));
    public int titleToNumber(String s) {
        int res = 0;
        for (int i = 0; i < s.length(); i ++) {
            res = res + (int) Math.pow(26, s.length() - i - 1) * (s.charAt(i) - 'A' + 1);
        }
        return res;
    }

    /*
    * Leetcode 235. Lowest Common Ancestor of a Binary Search Tree
    * */
    public TreeNode lowestCommonAncestorBST(TreeNode root, TreeNode p, TreeNode q) {
        if (p.val > q.val) {
            TreeNode t = p;
            p = q;
            q = t;
        }

        if (root == null) {
            return null;
        }
        if (root.val >= p.val && root.val <= q.val) {
            return root;
        }
        if (root.val > p.val && root.val > q.val) {
            return lowestCommonAncestorBST(root.left, p, q);
        }
        if (root.val < p.val && root.val < q.val) {
            return lowestCommonAncestorBST(root.right, p, q);
        }

        return null;
    }

    /*
    * Leetcode 236. Lowest Common Ancestor of a Binary Tree
    * */
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == p || root == q || root == null) {
            return root;
        }

        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        if (left != null && right != null) {
            return root;
        }
        if (left != null) {
            return left;
        } else {
            return right;
        }
    }

    /*
    * Leetcode 54. Spiral Matrix
    * */
    //int[][] tmp = new int[2][5];
    //int count = 1;
    //for (int i = 0; i < tmp.length; i ++) {
    //    for (int j = 0; j < tmp[i].length; j ++) {
    //        tmp[i][j] = count ++;
    //    }
    //}
    //out.println(solution.spiralOrder(tmp));
    public List<Integer> spiralOrder(int[][] matrix) {
        ArrayList<Integer> res = new ArrayList<>();
        if (matrix.length == 0) {
            return res;
        }
        int top = 0, bottom = matrix.length - 1, left = 0, right = matrix[0].length - 1;
        int i = 0, j = 0;
        while (left <= right && top <= bottom) {
            for (j = left; j <= right; j ++) {
                res.add(matrix[top][j]);
            }
            top ++;

            for (i = top; i <= bottom; i ++) {
                res.add(matrix[i][right]);
            }
            right --;

            if (top <= bottom) {
                for (j = right; j >= left; j --) {
                    res.add(matrix[bottom][j]);
                }
            }
            bottom --;

            if (left <= right) {
                for (i = bottom; i >= top; i --) {
                    res.add(matrix[i][left]);
                }
            }
            left ++;
        }

        return res;
    }

    /*
    * Leetcode 419. Battleships in a Board
    * */
    public int countBattleships(char[][] board) {
        int count = 0;
        for (int i = 0; i < board.length; i ++) {
            for (int j = 0; j < board[i].length; j ++) {
                if (board[i][j] == 'X') {
                    int left = j - 1;
                    int top = i - 1;
                    if (left < 0) {
                        if (top < 0) {
                            count ++;
                        } else {
                            if (board[top][j] == '.') {
                                count ++;
                            }
                        }
                    } else {
                        if (top < 0) {
                            if (board[i][left] == '.') {
                                count ++;
                            }
                        } else {
                            if (board[i][left] == '.' && board[top][j] == '.') {
                                count ++;
                            }
                        }
                    }
                }
            }
        }

        return count;
    }

    /*
    * Alibaba Test
    * Compare to LeetCode.273
    * */
    //BigInteger num = new BigInteger("168212345678");
    //solution.alibabaTest(num);
    String[] s1 = { "零", "一", "二", "三", "四", "五", "六", "七", "八", "九" };
    String[] s2 = { "", "十", "百", "千", "万", "十", "百", "千", "亿", "十", "百", "千", "万" };
    public void alibabaTest(BigInteger number) {
        if (number.equals(0)) {
            out.println(s1[0]);
            return ;
        }

        int i = 0;
        String words = "";
        int hasGap = 0;
        BigInteger zero = new BigInteger("0");
        BigInteger ten = new BigInteger("10");
        while (number.compareTo(zero) == 1) {
            if (number.mod(ten).compareTo(zero) != 0) {
                if (hasGap >= 3) {
                    words = helper(number.mod(ten).intValue()) + s2[i] + "零" + words;
                    hasGap = 0;
                } else {
                    words = helper(number.mod(ten).intValue()) + s2[i] + words;
                }
            } else {
                if (i == 0) {
                } else if (i == 4 || i == 8) {
                    words = s2[i] + words;
                } else {
                    hasGap ++;
                }
            }
            number = number.divide(ten);
            i ++;
        }

        out.println(words);
    }

    public String helper(int num) {
        return s1[num];
    }

    /*
    * 138. Copy List with Random Pointer
    * */
    class RandomListNode {
        int label;
        RandomListNode next, random;
        RandomListNode(int x) { this.label = x; }
    }
    public RandomListNode copyRandomList(RandomListNode head) {
        if (head == null) {
            return null;
        }

        Map<RandomListNode, RandomListNode> map = new HashMap<>();
        RandomListNode node = head;
        while (node != null) {
            map.put(node, new RandomListNode(node.label));
            node = node.next;
        }

        node = head;
        while (node != null) {
            map.get(node).next = map.get(node.next);
            if (node.random != null) {
                map.get(node).random = map.get(node.random);
            }
            node = node.next;
        }

        return map.get(head);
    }

    /*
    * Leetcode 332. Reconstruct Itinerary
    * */
    Map<String, PriorityQueue<String>> targets = new HashMap<>();
    List<String> route = new LinkedList();
    public List<String> findItinerary(String[][] tickets) {
        for (String[] ticket : tickets) {
            targets.computeIfAbsent(ticket[0], k -> new PriorityQueue()).add(ticket[1]);
        }
        visit("JFK");
        return route;
    }

    void visit(String airport) {
        while(targets.containsKey(airport) && !targets.get(airport).isEmpty()) {
            visit(targets.get(airport).poll());
        }
        route.add(0, airport);
    }

    /*
    * Leetcode 331. Verify Preorder Serialization of a Binary Tree
    * */
    //String preorder = "9,#,3,#,#";
    //String preorder = "9,3,4,#,#,1,#,#,2,#,6,#,#";
    //out.println(solution.isValidSerialization(preorder));
    public boolean isValidSerialization(String preorder) {
        if (preorder == null) {
            return false;
        }
        /*
        * Tree Perspective
        * */
        String[] nodes = preorder.split(",");
        Stack<String> stack = new Stack<>();
        for (String node : nodes) {
            while (node.equals("#") && !stack.isEmpty() && stack.peek().equals("#")) {
                stack.pop();
                if (stack.isEmpty()) {
                    return false;
                }
                stack.pop();
            }
            stack.push(node);
        }
        return stack.size() == 1 && stack.peek().equals("#");

        /*
        * Degree Perspective
        * Every # node supply 1 indegree and 0 outdegree
        * Every not # node supply 1 indegree and 2 outdegree
        * Use diff to record outdegree - indegree
        * If we scan from left to right and diff never < 0
        * and at last diff = 0
        * It means OK.
        String[] nodes = preorder.split(",");
        int diff = 1;
        for (String node : nodes) {
            if (--diff < 0) {
                return false;
            }
            if (! node.equals("#")) {
                diff += 2;
            }
        }
        return diff == 0;
        */
    }

    /*
    * Three Way Partition
    * Dutch National Flag Problem
    * Related: Leetcode 75
    * */
    //int[] nums = {5,3,1,2,6,7,8,5,5};
    //solution.threeWayPartition(nums, 5);
    public void threeWayPartition(int[] a, int mid) {
        int i = 0, j = 0, n = a.length - 1;

        while (j <= n) {
            if (a[j] < mid) {
                swap(a, i, j);
                i ++;
                j ++;
            } else if (a[j] > mid) {
                swap(a, j, n);
                n --;
            } else {
                j ++;
            }
        }
        out.println(Arrays.toString(a));
    }

    /*
    * Leetcode 324. Wiggle Sort II
    * */
    //int[] nums = {1,3,2,2,3,1};
    //int[] nums = {2,1};
    //int[] nums = {1,1,1,2,2,2};
    //int[] nums = {1,5,1,1,6,4};
    //int[] nums = {1,1,1,6,4};
    //solution.wiggleSort(nums);
    //out.println(Arrays.toString(nums));
    public void wiggleSort(int[] nums) {
        /*
        * Assume that quickSelect use O(n) time
        * Actually we should use BFPRT algorithm
        * */
        int n = nums.length;
        int median = nums[n / 2];
        if (n > 1) {
            median = quickSelect(nums, 0, n - 1, (n + 1) / 2);
            median = nums[median];
        }
        int left = 0, i = 0, right = n - 1;

        while (i <= right) {
            if (nums[newIndex(i, n)] > median) {
                swap(nums, newIndex(left++, n), newIndex(i++, n));
            } else if (nums[newIndex(i, n)] < median) {
                swap(nums, newIndex(right--, n), newIndex(i, n));
            } else {
                i++;
            }
        }
    }

    public int newIndex(int index, int n) {
        return (1 + 2 * index) % (n | 1);
    }

    public int quickSelect(int[] nums, int lo, int hi, int k) {
        int i = lo, j = hi, pivot = nums[hi];
        while (true) {
            while (i < j && nums[i] < pivot) {
                i ++;
            }
            while (i < j && nums[j] >= pivot) {
                j --;
            }
            if (i >= j) {
                break;
            }
            swap(nums, i, j);
        }
        swap(nums, i, hi);
        int m = i - lo + 1;
        if (m == k) {
            return i;
        } else if (m > k) {
            return quickSelect(nums, lo, i - 1, k);
        } else {
            return quickSelect(nums, i + 1, hi, k - m);
        }
    }

    public void swap(int[] nums, int i, int j) {
        int tmp = nums[i];
        nums[i] = nums[j];
        nums[j] = tmp;
    }

    /*
    * Leetcode 322. Coin Change
    * */
    //int[] coins = {1};
    //out.println(solution.coinChange(coins, 1));
    public int coinChange(int[] coins, int amount) {
        int[] dp = new int[amount + 1];
        for (int i = 1; i <= amount; i ++) {
            dp[i] = Integer.MAX_VALUE;
        }
        Arrays.sort(coins);

        for (int i = 1; i <= amount; i ++) {
            for (int j = 0; j < coins.length; j ++) {
                if (coins[j] <= i && i - coins[j] >= 0 && dp[i-coins[j]] != Integer.MAX_VALUE) {
                    dp[i] = Math.min(dp[i-coins[j]] + 1, dp[i]);
                }
            }
        }
        /*
        for (int i = 1; i <= amount; i ++) {
            for (int j = 0; j < i; j ++) {
                if (dp[j] == Integer.MAX_VALUE) {
                    continue;
                }
                for (int k = 0; k < coins.length; k ++) {
                    if (j + coins[k] > i) {
                        break;
                    } else if (j + coins[k] == i) {
                        if (dp[j] + 1 < dp[i]) {
                            dp[i] = dp[j] + 1;
                        }
                        break;
                    }
                }
            }
        }
        */

        if (dp[amount] == Integer.MAX_VALUE) {
            return -1;
        } else {
            return dp[amount];
        }
    }

    /*
    * Leetcode 319. Bulb Switcher
    * */
    //out.println(solution.bulbSwitch(3));
    public int bulbSwitch(int n) {
        return (int) Math.sqrt(n);
        /*
        if (n == 0) {
            return 0;
        } else if (n == 1) {
            return n;
        }

        BitSet bulbs = new BitSet(n);
        // init
        for (int i = 0; i < n; i ++) {
            bulbs.set(i, true);
        }
        // do loop
        for (int i = 2; i <= n; i ++) {
            int p = i - 1;
            while (p < n) {
                boolean v = bulbs.get(p);
                bulbs.set(p, !v);
                p += i;
            }
        }

        int nums = 0;
        for (int i = 0; i < n; i ++) {
            if (bulbs.get(i)) {
                nums ++;
            }
        }

        return nums;
        */
    }

    /*
    * Leetcode 313. Super Ugly Number
    * */
    public int maxProduct(String[] words) {
        if (words == null || words.length == 0) {
            return 0;
        }
        int len = words.length;
        int[] value = new int[len];

        // hash
        for (int i = 0; i < len; i ++) {
            String tmp = words[i];
            value[i] = 0;
            for (int j = 0; j < tmp.length(); j ++) {
                value[i] |= 1 << (tmp.charAt(j) - 'a');
            }
        }

        // compare and calculate
        int maxProduct = 0;
        for (int i = 0; i < len; i ++) {
            for (int j = i + 1; j < len; j ++) {
                if ((value[i] & value[j]) == 0 && (words[i].length() * words[j].length() > maxProduct)) {
                    maxProduct = words[i].length() * words[j].length();
                }
            }
        }

        return maxProduct;
    }

    /*
    * Leetcode 313. Super Ugly Number
    * */
    //int[] test = {2,7,13,19};
    //out.println(solution.nthSuperUglyNumber(7, test));
    public int nthSuperUglyNumber(int n, int[] primes) {
        class Num implements Comparable<Num> {
            int val;
            int idx;
            int p;

            public Num(int val, int idx, int p) {
                this.val = val;
                this.idx = idx;
                this.p = p;
            }

            @Override
            public int compareTo(Num o) {
                return this.val - o.val;
            }
        }
        int[] ugly = new int[n];
        PriorityQueue<Num> heap = new PriorityQueue<Num>();
        for (int i = 0; i < primes.length; i ++) {
            heap.add(new Num(primes[i], 1, primes[i]));
        }
        ugly[0] = 1;

        for (int i = 1; i < n; i ++) {
            ugly[i] = heap.peek().val;
            while (heap.peek().val == ugly[i]) {
                Num next = heap.poll();
                heap.add(new Num(next.p * ugly[next.idx], next.idx + 1, next.p));
            }
        }

        return ugly[n - 1];

        /*
        * 方法二
        * val 数组记录了新的一轮循环每个位置所达到的比当前 ugly 大的最小值
        * 通过每次选举最小的数作为 ugly 的下一个数，同时在循环中把与这个数相同位置的 val 进行更新
        * 实现对ugly的查找动作
        * */
        /*
        int[] ugly = new int[n];
        int[] idx = new int[primes.length];
        int[] val = new int[primes.length];
        Arrays.fill(val, 1);

        int next = 1;
        for (int i = 0; i < n; i ++) {
            ugly[i] = next;

            next = Integer.MAX_VALUE;
            for (int j = 0; j < primes.length; j ++) {
                if (val[j] == ugly[i]) {
                    val[j] = ugly[idx[j]] * primes[j];
                    idx[j] ++;
                }
                next = Math.min(next, val[j]);
            }
        }

        return ugly[n - 1];
        */

        /*
        * 方法一
        * idx[i] 表示的是 primes[i] 可以乘的ugly的位置，使这个值可以参与下一轮ugly数的选举
        * 因此我们每选出一个ugly数，就要对所有的idx进行检查，把乘积后小于等于当前ugly数的idx加到恰好大于当前ugly数以进行下一轮选举
        * */
        /*
        int[] ugly = new int[n];
        int[] idx = new int[primes.length];

        ugly[0] = 1;
        for (int i = 1; i < n; i++) {
            //find next
            ugly[i] = Integer.MAX_VALUE;
            for (int j = 0; j < primes.length; j++)
                ugly[i] = Math.min(ugly[i], primes[j] * ugly[idx[j]]);

            //slip duplicate
            for (int j = 0; j < primes.length; j++) {
                while (primes[j] * ugly[idx[j]] <= ugly[i]) idx[j]++;
            }
        }

        return ugly[n - 1];
        */
    }

    /*
    * Leetcode 310. Minimum Height Trees
    * */
    //int[][] test = {
    //        {1, 0},
    //        {1, 2},
    //        {1, 3},
    //};
    //int[][] test1 = {
    //        {0, 3},
    //        {1, 3},
    //        {2, 3},
    //        {4, 3},
    //        {5, 4},
    //};
    //out.println(solution.findMinHeightTrees(4, test));
    //out.println(solution.findMinHeightTrees(6, test1));
    public List<Integer> findMinHeightTrees(int n, int[][] edges) {
        if (n == 1) {
            return Collections.singletonList(0);
        }

        // format data
        // adj[i] store all other nodes linked with i
        ArrayList<Set<Integer>> adj = new ArrayList<>(n);
        for (int i = 0; i < n; i ++) {
            adj.add(new HashSet<>());
        }
        for (int[] edge : edges) {
            adj.get(edge[0]).add(edge[1]);
            adj.get(edge[1]).add(edge[0]);
        }

        // get leaves
        ArrayList<Integer> leaves = new ArrayList<>();
        for (int i = 0; i < n; i ++) {
            if (adj.get(i).size() == 1) {
                leaves.add(i);
            }
        }

        // walk from all leaves, when n <= 2, it means a middle pos got!
        while (n > 2) {
            n -= leaves.size();
            ArrayList<Integer> newLeaves = new ArrayList<>();
            for (int i : leaves) {
                // get node j linked with i
                // then delete i from j's adj
                int j = adj.get(i).iterator().next();
                adj.get(j).remove(i);
                if (adj.get(j).size() == 1) {
                    newLeaves.add(j);
                }
            }
            leaves = newLeaves;
        }

        return leaves;
    }

    // this is too slow O(n^2)
    public List<Integer> findMinHeightTreesSlow(int n, int[][] edges) {
        ArrayList<Integer> res = new ArrayList<>();
        int min = Integer.MAX_VALUE;
        for (int i = 0; i < n; i ++) {
            int height = getTreeHeight(i, edges, n);
            if (height < min) {
                min = height;
                res.clear();
                res.add(i);
            } else if (height == min) {
                res.add(i);
            }
        }

        return res;
    }

    public int getTreeHeight(int root, int[][] edges, int nodeSum) {
        LinkedList<Integer> queue = new LinkedList<>();
        LinkedList<Integer> bakQueue = new LinkedList<>();
        HashSet<Integer> walked = new HashSet<>();
        queue.add(root);
        int level = 0;
        while (! queue.isEmpty()) {
            int cur = queue.poll();
            walked.add(cur);
            for (int[] t : edges) {
                if ((t[0] == cur || t[1] == cur) && ! walked.contains(t[0] + t[1] - cur)) {
                    bakQueue.add(t[0] + t[1] - cur);
                }
            }

            if (queue.isEmpty()) {
                queue.addAll(bakQueue);
                bakQueue.clear();
                level ++;
                if (nodeSum == walked.size() + queue.size()) {
                    break;
                }
            }
        }

        return level;
    }

    /*
    * Leetcode 309. Best Time to Buy and Sell Stock with Colldown
    * */
    //int[] test = {1, 2, 3, 0, 2};
    //solution.maxProfit(test);
    public int maxProfit(int[] prices) {
        int pricesLen = prices.length;
        if (pricesLen < 2) {
            return 0;
        }
        int sell = 0, prev_sell = 0, buy = Integer.MIN_VALUE, prev_buy;
        for (int price : prices) {
            prev_buy = buy;
            buy = Math.max(prev_sell - price, prev_buy);
            prev_sell = sell;
            sell = Math.max(prev_buy + price, prev_sell);
        }

        out.println(sell);
        return sell;
        /*
        int[] buy = new int[pricesLen];
        int[] sell = new int[pricesLen];
        int[] rest = new int[pricesLen];
        // second version
        buy[1] = 0;
        sell[1] = prices[1] - prices[0];
        for (int i = 2; i < pricesLen; i ++) {
            buy[i] = Math.max(sell[i-2] - prices[i], buy[i-1]);
            sell[i] = Math.max(buy[i-1] + prices[i], sell[i-1]);
        }
        out.println(sell[pricesLen - 1]);
        return sell[pricesLen - 1];


        // first version
        for (int i = 1; i < pricesLen; i ++) {
            buy[i] = Math.max(rest[i-1] - prices[i], buy[i-1]);
            sell[i] = Math.max(buy[i-1] + prices[i], sell[i-1]);
            rest[i] = Math.max(Math.max(buy[i-1], sell[i-1]), rest[i-1]);
        }
        out.println(rest[pricesLen - 1]);
        return Math.max(rest[pricesLen - 1], sell[pricesLen - 1]);
        */
    }

    /*
    * Leetcode 306. Additive Number
    * */
    public boolean isAdditiveNumber(String num) {
        int n = num.length();
        for (int i = 1; i <= n / 2; i ++) {
            if (num.charAt(0) == '0' && i > 1) {
                return false;
            }
            BigInteger num1 = new BigInteger(num.substring(0, i));
            for (int j = 1; Math.max(i, j) <= n - i - j; j ++) {
                if (num.charAt(i) == '0' && j > 1) {
                    break;
                }
                BigInteger num2 = new BigInteger(num.substring(i, i + j));
                if (subIsAdditiveNumber(num, num1, num2, i + j)) {
                    return true;
                }
            }
        }
        return false;
    }

    public boolean subIsAdditiveNumber(String num, BigInteger num1, BigInteger num2, Integer curPos) {
        if (curPos == num.length()) {
            return true;
        }
        String shouldPrefix = num1.add(num2).toString();
        return num.startsWith(shouldPrefix, curPos) && subIsAdditiveNumber(num, num2, num1.add(num2), curPos + shouldPrefix.length());
    }
}
