package com.nick;

import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;

/**
 * Created by qishi on 2018/4/19.
 */
public class LRUCache {

    private int size;
    long timeCounter = 0;  //模拟时间点
    private HashMap<Integer, Long> keyTime = new HashMap<>(); //不需要有序，这个地方改成TreeMap会变慢（要维持顺序）
    private TreeMap<KeyTime, Integer> caches = new TreeMap<>(); //有序

    public LRUCache(int capacity) {
        this.size = capacity;
    }

    public int get(int key) {
        if(keyTime.containsKey(key)){
            long newTime = timeCounter++;
            KeyTime kt = new KeyTime(key,keyTime.get(key).longValue());
            int value = caches.get(kt);
            keyTime.put(key, newTime);
            caches.remove(kt);
            caches.put(new KeyTime(key,newTime), value);
            return value;
        }
        return -1;
    }

    public void put(int key, int value) {
        long newTime = timeCounter ++;
        if (keyTime.containsKey(key)) {                     //已经存在
            long oldTime = keyTime.get(key);
            keyTime.put(key, newTime);
            caches.remove(new KeyTime(key, oldTime));
            caches.put(new KeyTime(key, newTime), value);
        } else if (keyTime.size() < this.size) {               //不存在但未满
            keyTime.put(key, newTime);
            caches.put(new KeyTime(key, newTime), value);
        } else {                                            //不存在，而且满了
            Map.Entry<KeyTime, Integer> firstEntry = caches.firstEntry();
            int k = firstEntry.getKey().key;
            long time = firstEntry.getKey().time;
            caches.remove(new KeyTime(k, time));
            keyTime.remove(k);
            keyTime.put(key, newTime);
            caches.put(new KeyTime(key, newTime), value);
        }
    }

    class KeyTime implements Comparable<KeyTime>{
        int key;
        long time;

        public KeyTime(int key,long time){
            this.key = key;
            this.time = time;
        }

        @Override
        public int compareTo(KeyTime o) {          //重写compareTo方法
            if (this.time < o.time) {
                return -1;
            } else if (this.time > o.time) {
                return 1;
            }
            return 0;
        }
    }
}
