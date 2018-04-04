package com.nick;

import java.lang.reflect.Array;
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
        int[] test = {2,7,13,19};
        out.println(solution.nthSuperUglyNumber(7, test));
        //int[] test = {1, 2, 3, 0, 2};
        //solution.maxProfit(test);
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
