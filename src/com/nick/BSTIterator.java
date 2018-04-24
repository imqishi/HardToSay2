package com.nick;

import java.util.Stack;

/**
 * Created by qishi on 2018/4/21.
 */
public class BSTIterator {
    private Stack<TreeNode> stack = new Stack<>();

    public BSTIterator(TreeNode root) {
        pushAll(root);
    }

    /** @return whether we have a next smallest number */
    public boolean hasNext() {
        return !stack.isEmpty();
    }

    /** @return the next smallest number */
    public int next() {
        TreeNode t = stack.pop();
        pushAll(t.right);
        return t.val;
    }

    private void pushAll(TreeNode root) {
        while (root != null) {
            stack.push(root);
            root = root.left;
        }
    }
}
