package structure;

import java.util.Arrays;
import java.util.Comparator;

public class IndexValuePair {
    double value;
    int index;

    IndexValuePair(double value, int index) {
        this.value = value;
        this.index = index;
    }

    // 对数组从大到小排序并返回原来的数对应的下标
    public static int[] sortIndicesByValue(double[] variance) {
        int n = variance.length;
        IndexValuePair[] pairArray = new IndexValuePair[n];

        for (int i = 0; i < n; i++) {
            pairArray[i] = new IndexValuePair(variance[i], i);
        }

        // 从大到小排序
        Arrays.sort(pairArray, new Comparator<IndexValuePair>() {
            @Override
            public int compare(IndexValuePair o1, IndexValuePair o2) {
                return Double.compare(o2.value, o1.value);
            }
        });

        int[] sortedIndices = new int[n];
        for (int i = 0; i < n; i++) {
            sortedIndices[i] = pairArray[i].index;
        }
        return sortedIndices;
    }
}
