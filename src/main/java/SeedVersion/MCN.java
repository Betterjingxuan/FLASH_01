package SeedVersion;

import Game.GameClass;
import Global.Comparer;
import Global.Info;
import structure.ShapMatrixEntry;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

/*TODO Compute Shapley value by sampling m marginal contributions by optimum allocation
*  MCN-2017-Castro et. al.*/
public class MCN {

    int num_features;  //the number of features
    double[] exact;   // the exact shapley value
    int num_samples;

    int initial_m;

    public void MCN_scale(boolean gene_weight, String model){

        //1)初始化
        GameClass game = new GameClass();
        initialization(game, gene_weight, model);
        Comparer comparator = new Comparer();

        long ave_runtime = 0;
        double ave_error_max = 0; //计算最大误差
        double ave_error_ave = 0;
        double[][] sv_values = new double[Info.timesRepeat][this.num_features];  //[runs][feature]
//        double ave_mse = 0;
        for(int r=0; r< Info.timesRepeat; r++) {
            ShapMatrixEntry[][] utility = new ShapMatrixEntry[this.num_features][]; // 需要换成带计数器的大数组
            for(int i=0; i<utility.length; i++){   //外层循环：features对应每个长度的collations (n个feature，n+1层)
                utility[i] = new ShapMatrixEntry[this.num_features];
                for(int j=0; j<this.num_features; j++) {  //内层循坏：每个features
                    utility[i][j] = new ShapMatrixEntry();
                    utility[i][j].record = new ArrayList<>();  //记录每层的方差
                }
            }
            Random PermutationGene = new Random(game.seedSet[r]);
            long time_1 = System.currentTimeMillis();
//            double[] shap_matrix = computeShapBySampling(game, this.num_features, this.num_samples, model, PermutationGene, utility);   //实验版
            double[] shap_matrix = computeShapBySampling_real(game, this.num_features, this.num_samples, model, PermutationGene);   //实验版
            long time_2 = System.currentTimeMillis();

            //4）计算误差
            double error_max = comparator.computeMaxError(shap_matrix, this.exact); //计算最大误差
            double error_ave = comparator.computeAverageError(shap_matrix, this.exact, this.num_features);  //计算平均误差
//            double mse = comparator.computeMSE(shap_matrix, this.exact, this.num_features);

            sv_values[r] = shap_matrix;
            ave_runtime += time_2 - time_1;
            ave_error_max += error_max;
            ave_error_ave += error_ave;
//            ave_mse += mse;
//            System.out.println("run: " + i + " " + "error_ave: " + error_ave + " \t"  +  "error_max: " + error_max + " \t"  +  "mse: " + mse);
            System.out.println("run: " + r + " " + "error_ave: " + error_ave + " \t"  +  "error_max: " + error_max);
        }
        double acv = comparator.computeACV(sv_values, this.num_features);

        // 5）输出时间
//        System.out.println(model + " Game:  " + "error_ave: " + ave_error_ave/Info.timesRepeat + " \t"  + "error_max: " + ave_error_max/Info.timesRepeat + " \t"  +  "mse: " + ave_mse/Info.timesRepeat);
        System.out.println(model + " Game:  " + "error_ave: " + ave_error_ave/Info.timesRepeat + " \t"  + "error_max: " + ave_error_max/Info.timesRepeat);
        System.out.println("average cv :" + acv);
        System.out.println("MC_V4 time : " + (ave_runtime * 0.001)/ Info.timesRepeat);  //+ "S"
    }

    //TODO 通过Monte Carlo random sampling 计算 shapley value
    private double[] computeShapBySampling(GameClass game, int num_features, int num_sample, String model, Random random, ShapMatrixEntry[][] utility) {
        int con = 0;
        double[] shap_matrix = new double[num_features];  //大数组
        ShapMatrixEntry[][] utilArr = new ShapMatrixEntry[this.num_features][]; // 需要换成带计数器的大数组
        for(int i=0; i<utilArr.length; i++){   //外层循环：features对应每个长度的collations (n个feature，n+1层)
            utilArr[i] = new ShapMatrixEntry[this.num_features];
            for(int j=0; j<this.num_features; j++) {  //内层循坏：每个features
                utilArr[i][j] = new ShapMatrixEntry();
                utilArr[i][j].record = new ArrayList<>();  //记录每层的方差
            }
        }
        //评估阶段：填充矩阵
        for(int i=0; i<num_features; i++) {
            ArrayList<Integer> idxs = new ArrayList<>();   //构造一个不包含i的permutation ：idxs
            for (int ele = 0; ele < i; ele++) {
                idxs.add(ele);
            }
            for (int ele = i + 1; ele < this.num_features; ele++) {  // 添加 range(i)
                idxs.add(ele);
            }
            for (int j = 0; j < num_features; j++) {  //遍历每个长度
                for (int count = 0; count < this.initial_m; count++) {
                    idxs = permutation(idxs, random);   //生成一个打乱的序列

                    //2）利用序列p, 求每个feature 的 marginal contribution
                    ArrayList<Integer> subset_1 = new ArrayList<>();   //*subset_1 第一个特征子集
                    ArrayList<Integer> subset_2 = new ArrayList<>();   //*subset_2 第二个特征子集

                    // 3）构造两个子集subset_1 & subset_2
                    for (int ind = 0; ind < j; ind++) { // ele 对应一个特征
                        subset_1.add(idxs.get(ind));
                        subset_2.add(idxs.get(ind));
                    }
                    subset_2.add(i);
                    //4)分别求函数值
                    double value_1 = game.gameValue(model, subset_1);
                    double value_2 = game.gameValue(model, subset_2);
                    utilArr[i][j].sum += value_2 - value_1;
                    utilArr[i][j].count++;
                    utilArr[i][j].record.add(value_2 - value_1);
                    con += 2;
                }
            }
        }
        System.out.println("count: " + con);

        // 计算方差
        double[][] var = new double[this.num_features][this.num_features];
        double total_var = 0;
        for(int i=0; i < this.num_features; i++){
            for(int j=0; j < this.num_features; j++){
                var[i][j] = varComputation(utilArr[i][j].record);
                total_var += var[i][j];
            }
        }

        //分配样本
        int[][] mst = new int[this.num_features][this.num_features];
        num_sample = num_sample - 2 * this.initial_m * this.num_features * this.num_features;
        if(num_sample <= 0){
            System.out.println("sample is no enough");
        }
        for(int i=0; i<num_features; i++) {
            for(int j=0; j<num_features; j++) {
                mst[i][j] = (int) Math.max(0, num_sample * var[i][j] / total_var/ 2);
            }
        }

        // second stage
        for(int i=0; i<num_features; i++) {
            ArrayList<Integer> idxs = new ArrayList<>();   //构造一个不包含i的permutation ：idxs
            for (int ele = 0; ele < i; ele++) {
                idxs.add(ele);
            }
            for (int ele = i + 1; ele < this.num_features; ele++) {  // 添加 range(i)
                idxs.add(ele);
            }
            for (int j = 0; j < num_features; j++) {  //遍历每个长度
                for (int count = 0; count < mst[i][j]; count++) {
                    idxs = permutation(idxs, random);   //生成一个打乱的序列

                    //2）利用序列p, 求每个feature 的 marginal contribution
                    ArrayList<Integer> subset_1 = new ArrayList<>();   //*subset_1 第一个特征子集
                    ArrayList<Integer> subset_2 = new ArrayList<>();   //*subset_2 第二个特征子集

                    // 3）构造两个子集subset_1 & subset_2
                    for (int ind = 0; ind < j; ind++) { // ele 对应一个特征
                        subset_1.add(idxs.get(ind));
                        subset_2.add(idxs.get(ind));
                    }
                    subset_2.add(i);
                    //4)分别求函数值
                    double value_1 = game.gameValue(model, subset_1);
                    double value_2 = game.gameValue(model, subset_2);
                    utility[i][j].sum += value_2 - value_1;
                    utility[i][j].count++;
                    utility[i][j].record.add(value_2 - value_1);
                    con += 2;
                }
            }
        }
        System.out.println("count: " + con);

        // 5) 求r次采样的平均值
        for(int i=0; i<num_features; i++) {
            for(int j=0; j<num_features; j++) {
                if(utility[i][j].count != 0){
                    utility[i][j].sum = utility[i][j].sum / utility[i][j].count;
                    shap_matrix[i] += utility[i][j].sum;
                }
            }
            shap_matrix[i] = shap_matrix[i] / num_features;
        }
        return shap_matrix;
    }

    private double[] computeShapBySampling_real(GameClass game, int num_features, int num_sample, String model, Random random) {
        int con = 0;
        double[] shap_matrix = new double[num_features];  //大数组
        ShapMatrixEntry[][] utility = new ShapMatrixEntry[this.num_features][]; // 需要换成带计数器的大数组
        for(int i=0; i<utility.length; i++){   //外层循环：features对应每个长度的collations (n个feature，n+1层)
            utility[i] = new ShapMatrixEntry[this.num_features];
            for(int j=0; j<this.num_features; j++) {  //内层循坏：每个features
                utility[i][j] = new ShapMatrixEntry();
                utility[i][j].record = new ArrayList<>();  //记录每层的方差
            }
        }
        //评估阶段：填充矩阵
        for(int i=0; i<num_features; i++) {
            ArrayList<Integer> idxs = new ArrayList<>();   //构造一个不包含i的permutation ：idxs
            for (int ele = 0; ele < i; ele++) {
                idxs.add(ele);
            }
            for (int ele = i + 1; ele < this.num_features; ele++) {  // 添加 range(i)
                idxs.add(ele);
            }
            for (int j = 0; j < num_features; j++) {  //遍历每个长度
                for (int count = 0; count < this.initial_m; count++) {
                    idxs = permutation(idxs, random);   //生成一个打乱的序列

                    //2）利用序列p, 求每个feature 的 marginal contribution
                    ArrayList<Integer> subset_1 = new ArrayList<>();   //*subset_1 第一个特征子集
                    ArrayList<Integer> subset_2 = new ArrayList<>();   //*subset_2 第二个特征子集

                    // 3）构造两个子集subset_1 & subset_2
                    for (int ind = 0; ind < j; ind++) { // ele 对应一个特征
                        subset_1.add(idxs.get(ind));
                        subset_2.add(idxs.get(ind));
                    }
                    subset_2.add(i);
                    //4)分别求函数值
                    double value_1 = game.gameValue(model, subset_1);
                    double value_2 = game.gameValue(model, subset_2);
                    utility[i][j].sum += value_2 - value_1;
                    utility[i][j].count++;
                    utility[i][j].record.add(value_2 - value_1);
                    con += 2;
                }
            }
        }
//        System.out.println("count: " + con);

        // 计算方差
        double[][] var = new double[this.num_features][this.num_features];
        double total_var = 0;
        for(int i=0; i < this.num_features; i++){
            for(int j=0; j < this.num_features; j++){
                var[i][j] = varComputation(utility[i][j].record);
                total_var += var[i][j];
            }
        }

        //分配样本
        int[][] mst = new int[this.num_features][this.num_features];
        num_sample = num_sample - 2 * this.initial_m * this.num_features * this.num_features;
        if(num_sample <= 0){
            System.out.println("sample is no enough");
        }
        for(int i=0; i<num_features; i++) {
            for(int j=0; j<num_features; j++) {
                mst[i][j] = (int) Math.max(0, num_sample * var[i][j] / total_var/ 2);
            }
        }

        // second stage
        for(int i=0; i<num_features; i++) {
            ArrayList<Integer> idxs = new ArrayList<>();   //构造一个不包含i的permutation ：idxs
            for (int ele = 0; ele < i; ele++) {
                idxs.add(ele);
            }
            for (int ele = i + 1; ele < this.num_features; ele++) {  // 添加 range(i)
                idxs.add(ele);
            }
            for (int j = 0; j < num_features; j++) {  //遍历每个长度
                for (int count = 0; count < mst[i][j]; count++) {
                    idxs = permutation(idxs, random);   //生成一个打乱的序列

                    //2）利用序列p, 求每个feature 的 marginal contribution
                    ArrayList<Integer> subset_1 = new ArrayList<>();   //*subset_1 第一个特征子集
                    ArrayList<Integer> subset_2 = new ArrayList<>();   //*subset_2 第二个特征子集

                    // 3）构造两个子集subset_1 & subset_2
                    for (int ind = 0; ind < j; ind++) { // ele 对应一个特征
                        subset_1.add(idxs.get(ind));
                        subset_2.add(idxs.get(ind));
                    }
                    subset_2.add(i);
                    //4)分别求函数值
                    double value_1 = game.gameValue(model, subset_1);
                    double value_2 = game.gameValue(model, subset_2);
                    utility[i][j].sum += value_2 - value_1;
                    utility[i][j].count++;
                    utility[i][j].record.add(value_2 - value_1);
                    con += 2;
                }
            }
        }
        System.out.println("count: " + con);

        // 5) 求r次采样的平均值
        for(int i=0; i<num_features; i++) {
            for(int j=0; j<num_features; j++) {
                if(utility[i][j].count != 0){
                    utility[i][j].sum = utility[i][j].sum / utility[i][j].count;
                    shap_matrix[i] += utility[i][j].sum;
                }
            }
            shap_matrix[i] = shap_matrix[i] / num_features;
        }
        return shap_matrix;
    }

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

    //TODO 生成一个随机序列
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

    private void initialization(GameClass game, boolean gene_weight, String modelName) {
        game.gameInit(gene_weight, modelName);
        this.num_features = game.num_features; //the number of features
        this.exact = game.exact;   // the exact shapley value
//        this.given_weights = game.given_weights;
//        this.halfSum = game.halfSum;  //for Voting game
        this.num_samples = Info.total_samples_num;  //一条permutation 要predict (n+1)次
        this.initial_m = Math.max(2, this.num_samples/ (2 * 2 * this.num_features * this.num_features));
    }
}
