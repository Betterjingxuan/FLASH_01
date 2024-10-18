package Global;


import java.util.*;

public class GenerateDataset {

    String[] element = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q",
            "R", "S", "T", "U", "V", "W", "X", "Y", "Z"};

    LinkedHashMap<String, Double>[] maps;

    /* 适用于key为String的数据类型
    * num: 特征的数量*/
    public String[] initializeFeature(int num){
        String[] featureSet = new String[num];
        for(int ind = 0; ind < num; ind++){
            featureSet[ind]= this.element[ind];
        }
        return featureSet;
    }

    /* TODO 生成数据集MAP
      num: 特征的数量
      featureSet: 所有特征的集合
      */
    public LinkedHashMap<String, Double>[] generateDataSet(int num, String[] featureSet) {
        LinkedHashMap<String, Double>[] maps = new LinkedHashMap[num];  //数组下标代表树的层数，map<itemset, values>
        Queue<String> queue = new LinkedList<>();

        //0) 初始化maps
        for(int i=0; i<featureSet.length; i++){
            maps[i] = new LinkedHashMap<>();
        }

        //1)首先把单个特征放入
        for (String element : featureSet) {
            queue.offer(element);
            maps[0].put(element, Math.random() * 1);
        }

        //2)然后从队列取出生成特征子集
        while (!queue.isEmpty()) {
            String currentCombination = queue.poll();
            int level = currentCombination.length();

            //2) 找到继续拓展的位置
            char c = currentCombination.charAt(currentCombination.length()-1);
            int indexOfc = -1;
            for(int ind=0; ind< featureSet.length; ind++){
                if(featureSet[ind].equals(String.valueOf(c))){
                    indexOfc = ind;
                    break;
                }
            }
            //3）在当前组合的基础上添加每个元素，生成新的组合
            for(int i=indexOfc+1; i<featureSet.length; i++){
                String newCombination = currentCombination + featureSet[i];  //生成新的
                queue.offer(newCombination);
                maps[level].put(newCombination, Math.random() * 100);
            }
        }
        return maps;
    }

        //1）生成所有特征子集
//        //第一层循环，层数（项的数量）
//        for(int level = 0; level < num_of_features; level++){
//            Map<String, Double> level_dataset = new HashMap<>();
//            //第二层循环：遍历集合中的每个特征
//            for(int j = 0; j < num_of_features; j++){
//                String subset = featureSet[j];
//                for (int length = 0; length<level; length++) {   //i表示长度，控制这个新生成特征子集subset的长度
//
//                }
//                level_dataset.put(subset, Math.random() * 100);
//            }
//            this.maps[level] = level_dataset;
//        }


        //2）
//        for (int i = 0; i < (1 << n); i++) {
//        Map<List<String>, Double> dataSet = new HashMap<>();
//        for (List<String> subset : allSubsets) {
//            double value = Math.random() * 100; // 示例：使用随机数作为价值函数
//            dataSet.put(subset, value);
//        }
//        return dataSet;
//    }

    public static void main(String[] args) {
        GenerateDataset gene = new GenerateDataset();
        String[] features = gene.initializeFeature(Info.num_of_features);
        LinkedHashMap<String, Double>[] maps = gene.generateDataSet(Info.num_of_features, features);
        System.out.println();
    }

}
