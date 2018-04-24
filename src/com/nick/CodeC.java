package com.nick;

/**
 * Created by qishi on 2018/4/18.
 * Leetcode 297. Serialize and Deserialize Binary Tree
 * */

//Codec cdc = new Codec();
//TreeNode a = new TreeNode(1);
//TreeNode b = new TreeNode(2);
//TreeNode c = new TreeNode(3);
//a.left = b;
//a.right = c;
//String t = cdc.serialize(a);
//TreeNode tr = cdc.deserialize(t);
//out.println(tr.right.val);
public class Codec {
    String code = "";

    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        subSerialize(root);
        return code;
    }

    public void subSerialize(TreeNode root) {
        if (root == null) {
            if (code.length() == 0) {
                code += "#";
            } else {
                code += " #";
            }
            return ;
        }

        if (code.length() == 0) {
            code += root.val;
        } else {
            code += " " + root.val;
        }
        subSerialize(root.left);
        subSerialize(root.right);
    }

    // Decodes your encoded data to tree.
    int pos = 0;
    public TreeNode deserialize(String data) {
        String[] nums = data.split(" ");
        pos = 0;
        return subDeserialize(nums);
    }

    public TreeNode subDeserialize(String[] nums) {
        if (nums.length == pos || nums[pos].equals("#")) {
            pos ++;
            return null;
        }

        TreeNode node = new TreeNode(Integer.valueOf(nums[pos]));
        pos ++;
        node.left = subDeserialize(nums);
        node.right = subDeserialize(nums);
        return node;
    }
}
