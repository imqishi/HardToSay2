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
        //int[] test = {1, 2, 3, 0, 2};
        //solution.maxProfit(test);
        int[][] test = {
                {1, 0},
                {1, 2},
                {1, 3},
        };
        int[][] test1 = {
                {0, 3},
                {1, 3},
                {2, 3},
                {4, 3},
                {5, 4},
        };
        out.println(solution.findMinHeightTrees(4, test));
        out.println(solution.findMinHeightTrees(6, test1));
    }
    /*
    * Leetcode 310. MinimumHeightTrees
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
