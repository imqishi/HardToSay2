package com.nick;

import static java.lang.System.*;

/**
 * Created by qishi on 2018/4/2.
 * Leetcode 307.Range Sum Query - Mutable
 */
public class NumArray {

    class SegmentTreeNode {
        int start, end;
        SegmentTreeNode left, right;
        int sum;

        public SegmentTreeNode(int start, int end) {
            this.start = start;
            this.end = end;
            this.left = null;
            this.right = null;
            this.sum = 0;
        }
    }

    SegmentTreeNode root = null;
    int NumArrayLength = 0;

    public NumArray(int[] nums) {
        root = buildTree(nums, 0, nums.length - 1);
        NumArrayLength = nums.length;
    }

    public SegmentTreeNode buildTree(int[] nums, int start, int end) {
        if (start > end) {
            return null;
        } else {
            SegmentTreeNode cur = new SegmentTreeNode(start, end);
            if (start == end) {
                cur.sum = nums[start];
            } else {
                int mid = start + (end - start) / 2;
                cur.left = buildTree(nums, start, mid);
                cur.right = buildTree(nums, mid + 1, end);
                cur.sum = cur.left.sum + cur.right.sum;
            }

            return cur;
        }
    }

    public void update(int i, int val) {
        update(root, i, val);
    }

    public void update(SegmentTreeNode root, int pos, int val) {
        if (root.start == root.end) {
            root.sum = val;
        } else {
            int mid = root.start + (root.end - root.start) / 2;
            if (pos <= mid) {
                update(root.left, pos, val);
            } else {
                update(root.right, pos, val);
            }
            root.sum = root.left.sum + root.right.sum;
        }
    }

    public int sumRange(int i, int j) {
        if (NumArrayLength == 0) {
            return 0;
        }
        return sumRange(root, i, j);
    }

    public int sumRange(SegmentTreeNode root, int start, int end) {
        if (root == null || start > root.end || end < root.start) {
            return 0;
        }

        if (start <= root.start && end >= root.end) {
            return root.sum;
        }

        return sumRange(root.right, start, end) + sumRange(root.left, start, end);
    }

    public static void main(String[] args) {
        int[] test = {0, 9, 5, 7, 3};
        NumArray obj = new NumArray(test);
        out.println(obj.sumRange(4, 4));
        out.println(obj.sumRange(2, 4));
        out.println(obj.sumRange(3, 3));
        out.println(obj.sumRange(4, 5));
        out.println(obj.sumRange(1, 7));
    }
}
