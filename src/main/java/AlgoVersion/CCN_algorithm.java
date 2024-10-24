package AlgoVersion;

import Game.GameClass;
import Global.Comparer;
import Global.Info;
import structure.ShapMatrixEntry;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;

public class CCN_algorithm {

    int num_features;  // the number of features
    double[] given_weights;
    double[] exact;  // the exact shapley value
    double halfSum;  //for Voting game
    int num_samples;
    int total_num_evaluations;
    int initial_m;


    //更换 computeShapBySampling_eva_3();
    public void CCN_Shap(boolean gene_weight, String model){

        GameClass game = new GameClass();
        initialization_eva(game, gene_weight, model);

        long ave_runtime = 0;
        double ave_error_max = 0; //计算最大误差
        double ave_error_ave = 0;
//        double ave_mse = 0;

        for(int i=0; i< Info.timesRepeat; i++) {
            Random random = new Random(game.seedSet[i]);
            int total_evaluateNum = this.total_num_evaluations;  //因为一个sample需要predict两次

            //3）计算shapley value
            long time_1 = System.currentTimeMillis();
//            ShapMatrixEntry[] shap_matrix = computeShapBySampling_3(game, initial_m, model, random);
            ShapMatrixEntry[] shap_matrix = computeShapBySampling_eva_3(game, this.initial_m, model, random, total_evaluateNum);
            long time_2 = System.currentTimeMillis();

            //4）计算误差
            Comparer comparator = new Comparer();
            double error_ave = comparator.computeAverageError(shap_matrix, this.exact, this.num_features);  //计算平均误差
            double error_max = comparator.computeMaxError(shap_matrix, this.exact, this.num_features); //计算最大误差
//            double mse = comparator.computeMSE(shap_matrix, this.exact, this.num_features);
            ave_runtime += time_2 - time_1;
            ave_error_max += error_max;
            ave_error_ave += error_ave;
//            ave_mse += mse;
//            System.out.println("run: " + i + " " + "error_ave: " + error_ave + " \t"  +  "error_max: " + error_max + " \t"  +  "mse: " + mse);
            System.out.println("run: " + i + " " + "error_ave: " + error_ave + " \t"  +  "error_max: " + error_max);
        }

        // 5）打印输出结果
        System.out.println(model + " Game:  " + "error_ave: " + ave_error_ave/Info.timesRepeat + " \t"  +  "error_max: " + ave_error_max/Info.timesRepeat);
        System.out.println("CCN time : " + (ave_runtime * 0.001)/ Info.timesRepeat );  //+ "S"
//        HashMap<Integer, double[]> s = new HashMap<>();
//        for(int i=0; i<num_features; i++){
//            s.put(i, new double[2]);
//        }
    }

    public void CCN_scale(boolean gene_weight, String model){
        ShapMatrixEntry[][] sv_values = new ShapMatrixEntry[Info.timesRepeat][];  //[runs][feature]
        GameClass game = new GameClass();
        initialization_eva(game, gene_weight, model, sv_values);
        Comparer comparator = new Comparer();

        long ave_runtime = 0;
        double ave_error_max = 0; //计算最大误差
        double ave_error_ave = 0;
//        double ave_mse = 0;

        for(int i=10; i< Info.timesRepeat; i++) {
            Random random = new Random(game.seedSet[i]);
            int total_evaluateNum = this.total_num_evaluations;  //因为一个sample需要predict两次

            //3）计算shapley value
            long time_1 = System.currentTimeMillis();
//            ShapMatrixEntry[] shap_matrix = computeShapBySampling_3(game, initial_m, model, random);
            ShapMatrixEntry[] shap_matrix = computeShapBySampling_eva_3(game, this.initial_m, model, random, total_evaluateNum);
            long time_2 = System.currentTimeMillis();
            sv_values[i] = shap_matrix;

            //4）计算误差
            double error_ave = comparator.computeAverageError(shap_matrix, this.exact, this.num_features);  //计算平均误差
            double error_max = comparator.computeMaxError(shap_matrix, this.exact, this.num_features); //计算最大误差
//            double mse = comparator.computeMSE(shap_matrix, this.exact, this.num_features);
            ave_runtime += time_2 - time_1;
            ave_error_max += error_max;
            ave_error_ave += error_ave;
//            ave_mse += mse;
//            System.out.println("run: " + i + " " + "error_ave: " + error_ave + " \t"  +  "error_max: " + error_max + " \t"  +  "mse: " + mse);
            System.out.println("run: " + i + " " + "error_ave: " + error_ave + " \t"  +  "error_max: " + error_max);
        }

        double acv = comparator.computeACV(sv_values, this.num_features);
        // 5）打印输出结果
        System.out.println(model + " Game:  " + "error_ave: " + ave_error_ave/Info.timesRepeat + " \t"  +  "error_max: " + ave_error_max/Info.timesRepeat);
        System.out.println("average cv :" + acv);
        System.out.println("CCN time : " + (ave_runtime * 0.001)/ Info.timesRepeat );  //+ "S"
//        HashMap<Integer, double[]> s = new HashMap<>();
//        for(int i=0; i<num_features; i++){
//            s.put(i, new double[2]);
//        }
    }

    private void initialization_eva(GameClass game, boolean gene_weight, String modelName) {
        game.gameInit(gene_weight, modelName);
        this.num_features = game.num_features; //the number of features
        this.total_num_evaluations = Info.total_samples_num;  //因为一个sample需要predict两次
        this.num_samples = this.total_num_evaluations/2;  //因为一个sample需要predict两次
        this.exact = game.exact;   // the exact shapley value
        this.given_weights = game.given_weights;
        this.halfSum = game.halfSum;  //for Voting game
        this.initial_m = initialm(this.num_samples/2, this.num_features);
    }

    //TODO game.seedSet();
    private void initialization_eva(GameClass game, boolean gene_weight, String modelName, ShapMatrixEntry[][] sv_values) {
        game.gameInit(gene_weight, modelName);
        this.num_features = game.num_features; //the number of features
        this.total_num_evaluations = Info.total_samples_num;  //因为一个sample需要predict两次
        this.num_samples = this.total_num_evaluations/2;  //因为一个sample需要predict两次
        this.exact = game.exact;   // the exact shapley value
        this.given_weights = game.given_weights;
        this.halfSum = game.halfSum;  //for Voting game
        this.initial_m = initialm(this.num_samples/2, this.num_features);
        for(ShapMatrixEntry[] ele : sv_values){
            ele = new ShapMatrixEntry[this.num_features];
        }
    }


    //TODO 去掉向上取整 arr_m[j] = (int) (total_evaluateNum * Math.sqrt(sigma_k[j] + sigma_n_k[j]) / var_sum/2));
    private ShapMatrixEntry[] computeShapBySampling_eva_3(GameClass game, int initial_m, String model, Random random, int total_evaluateNum) {

        ShapMatrixEntry[][] utility = new ShapMatrixEntry[this.num_features][]; // 需要换成带计数器的大数组
        for(int i=0; i<utility.length; i++){   //外层循环：features对应每个长度的collations (n个feature，n+1层)
            utility[i] = new ShapMatrixEntry[this.num_features];
            for(int j=0; j<this.num_features; j++) {  //内层循坏：每个features
                utility[i][j] = new ShapMatrixEntry();
                utility[i][j].record = new ArrayList<>();  //记录每层的方差
            }
        }
        long[] coef = combination(this.num_features-1,1);

        int evaluations_num = 0;
        int count = 0;
        while(true) {
            int temp_count = count;

            ArrayList<Integer> subset_0 = new ArrayList<>();
            ArrayList<Integer> subset_n = new ArrayList<>();
            for(int i=0; i<this.num_features; i++){
                subset_n.add(i);
            }
            double value_0 = game.gameValue(model, subset_0);
            double value_n = game.gameValue(model, subset_n);
            evaluations_num += 2;

            for (int i = 0; i < this.num_features; i++) {  // [外层] 遍历每个feature
                utility[i][this.num_features -1].sum += value_n - value_0;
                utility[i][this.num_features -1].count ++;
                utility[i][this.num_features -1].record.add(value_n - value_0);

                ArrayList<Integer> idxs = new ArrayList<>();   //构造一个不包含i的permutation ：idxs
                for (int ele = 0; ele < i; ele++) {  // 添加 range(i)
                    idxs.add(ele);
                }
                for (int ele = i + 1; ele < this.num_features; ele++) {  // 添加 range(i)
                    idxs.add(ele);
                }
                for (int len = 0; len < this.num_features; len++) {  //[内层] 遍历每个length
                    if (utility[i][len].count >= initial_m || utility[i][len].count >= coef[len]) {  //跳过当前网格，进入下一个len
                        continue;
                    }
                    idxs = permutation(idxs, random);
                    count ++;

                    //2）利用序列p, 求每个feature 的 marginal contribution
                    ArrayList<Integer> subset_1 = new ArrayList<>();   //*subset_1 第一个特征子集
                    ArrayList<Integer> subset_2 = new ArrayList<>();   //*subset_2 第二个特征子集

                    // 3）构造两个子集subset_1 & subset_2
                    for (int ind = 0; ind < len; ind++) { // ele 对应一个特征
                        subset_1.add(idxs.get(ind));
                    }
                    subset_1.add(i);
                    for (int ind = len; ind < idxs.size(); ind++) { // ele 对应一个特征
                        subset_2.add(idxs.get(ind));
                    }
                    //4)分别求函数值
                    double value_1 = game.gameValue(model, subset_1);
                    double value_2 = game.gameValue(model, subset_2);
                    utility[i][len].sum += value_1 - value_2;
                    utility[i][len].count ++;
                    utility[i][len].record.add(value_1 - value_2);
                    evaluations_num += 2;
//                    System.out.println(subset_1.toString() + ": " + value_1 + ";  " + subset_2.toString() + ": " + value_2);
//                    System.out.println(i + ", " + len + ":  " + utility[i][len].record.toString());
                    for (int l = 0; l < this.num_features - 1; l++) {
                        if (l < len) {
                            utility[idxs.get(l)][len].sum += value_1 - value_2;
                            utility[idxs.get(l)][len].count++;
                            utility[idxs.get(l)][len].record.add(value_1 - value_2);
                        }
                        else {
                            utility[idxs.get(l)][this.num_features - len - 2].sum += value_2 - value_1;
                            utility[idxs.get(l)][this.num_features - len - 2].count++;
                            utility[idxs.get(l)][this.num_features - len - 2].record.add(value_2 - value_1);
                        }
                    }
                }
            }
            if (count == temp_count) {
                break;
            }
        }
//        System.out.println(initial_m + " : " + count + " : " + evaluations_num);

        // 计算方差
        double[][] variance = new double[this.num_features][this.num_features];
        for(int i=0; i < this.num_features; i++){
            for(int j=0; j < this.num_features; j++){
                variance[i][j] = varComputation(utility[i][j].record);
            }
        }

        //计算分配样本
        double var_sum = 0;
        double[] sigma_k = new double[this.num_features];
        double[] sigma_n_k = new double[this.num_features];
        for(int j = (int) Math.ceil(1.0f * this.num_features / 2) - 1; j < this.num_features; j++){
            for (int i = 0; i < this.num_features; i++) {
                sigma_k[j] += variance[i][j] / (j + 1);
                if (this.num_features - j - 2 < 0) {
                    sigma_n_k[j] += 0;
                }
                else {
                    sigma_n_k[j] += variance[i][this.num_features - j - 2] / (this.num_features - j - 1);
                }
            }
            var_sum += Math.sqrt(sigma_k[j] + sigma_n_k[j]);
        }

//        this.num_samples = (int) (0.5 * this.num_samples - count);
        total_evaluateNum = total_evaluateNum - evaluations_num;
        int[] arr_m = new int[this.num_features];
        Arrays.fill(arr_m, 0); // Initialize m to zeros
        for (int j = (int) Math.ceil(this.num_features / 2.0) - 1; j < this.num_features; j++) {
            arr_m[j] =  Math.max(0, (int) (total_evaluateNum * Math.sqrt(sigma_k[j] + sigma_n_k[j]) / var_sum/2));
            //            System.out.print(j + ":" + arr_m[j] + " \t ");
        }

        // --------------------- second stage --------------------------
        double[][] new_utility = new double[this.num_features][this.num_features];
        int[][] arr_count = new int[this.num_features][this.num_features];
        for (int i = 0; i < this.num_features; i++) {
            for (int j = 0; j < this.num_features; j++) {
                new_utility[i][j] = sumArray(utility[i][j].record); // Calculate sum for simplicity
                arr_count[i][j] = utility[i][j].record.size(); // Length of the inner array
            }
        }

        ArrayList<Integer> idxs = new ArrayList<>();
        for (int i = 0; i < this.num_features; i++) {
            idxs.add(i);
        }
        for(int len = 0; len < this.num_features; len++) {
            for (int k = 0; k < arr_m[len]; k++) {
                idxs = permutation(idxs, random);
                // 3）构造两个子集subset_1 & subset_2
                ArrayList<Integer> subset_1 = new ArrayList<>();   //*subset_1 第一个特征子集
                ArrayList<Integer> subset_2 = new ArrayList<>();   //*subset_2 第二个特征子集
                for (int ind = 0; ind < len+1; ind++) { // ele 对应一个特征
                    subset_1.add(idxs.get(ind));
                }
                for (int ind = len+1; ind < idxs.size(); ind++) { // ele 对应一个特征
                    subset_2.add(idxs.get(ind));
                }
                //4)分别求函数值
                double value_1 = game.gameValue(model, subset_1);
                double value_2 = game.gameValue(model, subset_2);
                evaluations_num += 2;

                //temp = np.zeros(n)
                double[] temp_1 = new double[this.num_features];
                for (int i = 0; i < len+1; i++) {
                    temp_1[idxs.get(i)] = 1;
                }
                for (int i = 0; i < this.num_features; i++) {
                    new_utility[i][len] += temp_1[i] * (value_1 - value_2);
                    arr_count[i][len] += temp_1[i];
                }

                double[] temp_2 = new double[this.num_features];
                for (int i = len + 1; i < this.num_features; i++) {
                    temp_2[idxs.get(i)] = 1;
                }
                for (int i = 0; i < this.num_features; i++) {
                    new_utility[i][this.num_features - len - 2] += temp_2[i] * (value_2 - value_1);
                    arr_count[i][this.num_features - len - 2] += temp_2[i];
                }
            }
        }

//        System.out.println("evaluations_num: " + evaluations_num);

        // 5) 求shapley value的平均值，对于每个特征&每个长度求均值
        ShapMatrixEntry[] resultShap = new ShapMatrixEntry[this.num_features];  //这是最后返回的大数组
        for (int i = 0; i < this.num_features; i++) {
            resultShap[i] = new ShapMatrixEntry();
            for (int j = 0; j < this.num_features; j++) {
                if (arr_count[i][j] != 0) {
                    resultShap[i].sum +=  new_utility[i][j] / arr_count[i][j];
                    resultShap[i].count ++;
                }
//                else{
//                    resultShap[i].sum += 0;
//                    resultShap[i].count ++;
//                }
            }
            //6）求一个整体的均值
            if(resultShap[i].count != 0){
                resultShap[i].sum = resultShap[i].sum / resultShap[i].count;
            }
            else{
                resultShap[i].sum = 0;
            }
        }
        return resultShap;
    }

    private int initialm(int m, int n) {
        return Math.max(2, m/n/n/2);
    }
    private ArrayList<Integer> permutation(ArrayList<Integer> list, Random PermutationGene) {
        //1) 初始化perm序列（initial 对应的位置是对应的值）
        ArrayList<Integer> perm = new ArrayList<>(list);

        // 生成一个新的种子
        int sequenceSeed = PermutationGene.nextInt();
        Random random = new Random(sequenceSeed);

        // 使用 Collections.shuffle 打乱序列
        Collections.shuffle(perm, random);

        return perm;
    }

    //TODO 计算方差
    private double varComputation(ArrayList<Double> record) {
        if(record.size() <= 1){
            return 0;
        }
        double sum = 0.0;
        // Calculate the mean
        for (double value : record) {
            sum += value;
        }
        double mean = sum / record.size();
        // Calculate the variance
        double sumSquares = 0.0;
        for (double value : record) {
            sumSquares += Math.pow(value - mean, 2);
        }
        return sumSquares / (record.size() - 1);  //特别指定ddof=1 分母是(n-1) 而不是1
    }

    private long[] combination(int num_features, int s) {
        long[] coef = new long[this.num_features];
        BigInteger maxLimit = BigInteger.valueOf(Long.MAX_VALUE);
        for (int i = 0; i < num_features; i++) {
            BigInteger bigInteger = CombinationCalculator(num_features - 1, i);
            if (bigInteger.compareTo(maxLimit) > 0) {
                coef[i] = maxLimit.longValue();
            } else if (bigInteger.compareTo(BigInteger.valueOf(Long.MAX_VALUE)) > 0) {
                coef[i] = Long.MAX_VALUE;
            } else {
                coef[i] = bigInteger.longValue();
            }
        }
        return coef;
    }

    private BigInteger CombinationCalculator (int num_features, int s) {
        BigInteger numerator = BigInteger.ONE;
        BigInteger denominator = BigInteger.ONE;
        for (int i = 0; i < s; i++) {
            numerator = numerator.multiply(BigInteger.valueOf(num_features - i));
            denominator = denominator.multiply(BigInteger.valueOf(i + 1));
        }
        return numerator.divide(denominator);
    }

    private long[] comb(int num_features, int s) {
        long[] coef = new long[this.num_features];
        for(int ind =0; ind<Math.ceil(this.num_features/2.0); ind++) {
            if(ind==0){
                coef[ind] = 1;
                coef[this.num_features-ind-1] = 1;
            }
            else if(ind==1){
                coef[ind] = num_features -1;
                coef[this.num_features-ind-1] = num_features -1;
            }
            else if(ind==2){
                coef[ind] = (long) (num_features * (num_features-1) * 0.5);
                coef[this.num_features-ind-1] = (long) (num_features * (num_features-1) * 0.5);
            }
            else if(ind==3){
                coef[ind] = ((long) num_features * (num_features-1) * (num_features-2) / (2*3));
                coef[this.num_features-ind-1] = ((long) num_features * (num_features-1) * (num_features-2) / (2*3));
            }
            else if(ind==4){
                coef[ind] = ((long) num_features * (num_features-1) * (num_features-2) * (num_features-4)/ (2*3*4));
                coef[this.num_features-ind-1] =((long) num_features * (num_features-1) * (num_features-2) * (num_features-4)/ (2*3*4));
            }
            else if(ind==5){
                coef[ind] = ((long) num_features * (num_features-1) * (num_features-2) * (num_features-4)* (num_features-5)/ (2*3*4*5));
                coef[this.num_features-ind-1] = ((long) num_features * (num_features-1) * (num_features-2) * (num_features-4)* (num_features-5)/ (2*3*4*5));
            }
            else if(ind==6){
                coef[ind] = ((long) num_features * (num_features-1) * (num_features-2) * (num_features-4)* (num_features-5)* (num_features-6)/ (2*3*4*5*6));
                coef[this.num_features-ind-1] = ((long) num_features * (num_features-1) * (num_features-2) * (num_features-4)* (num_features-5)* (num_features-6)/ (2*3*4*5*6));
            }
            else {
                coef[ind] = 1999999999;
                coef[this.num_features-ind-1] = 1999999999;
            }
        }
        return coef;
    }

    private static double sumArray(ArrayList<Double> array) {
        double sum = 0.0;
        for (double value : array) {
            sum += value;
        }
        return sum;
    }
}
