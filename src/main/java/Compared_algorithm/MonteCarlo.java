package Compared_algorithm;

import Global.*;
import structure.*;
import java.io.DataInputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

public class MonteCarlo {

    /* TODO [MC Algorithm]: 通过Monte Carlo sampling 计算 shapley value
       num_sample: 采样的数量
       model : 当前进行的game */
    public void MCShap(int num_sample, String model){
        //1)初始化
        int num_features = 0;  // the number of features
        double[] exact;  // the exact shapley value
        //2）根据game初始化num_features & exact
        if(model.equals("airport")){
            num_features = Info.num_of_features_airport;
            exact = new double[num_features];
            exact = Info.airport_exact;
        }
        else if(model.equals("voting")){
            num_features = Info.num_of_features_voting;
            exact = new double[num_features];
            exact = Info.voting_exact;
        }
        else{
            exact = new double[0];
        }

        //3）计算shapley value
        long time_1 = System.currentTimeMillis();
        double[] shap_matrix = computeShapBySampling(num_features, num_sample, model);
        long time_2 = System.currentTimeMillis();

        //4）计算误差
        double error_max = computeMaxError(shap_matrix, exact); //计算最大误差
        double error_ave = computeAverageError(shap_matrix, exact, num_features);  //计算平均误差
        System.out.println(model + " Game:  " + "error_ave: " + error_ave + " \t"  +  "error_max: " + error_max );

        // 5）输出时间
        System.out.println("MC time : " + (time_2 - time_1) * 0.001 );  //+ "S"
//        HashMap<Integer, double[]> s = new HashMap<>();
//        for(int i=0; i<num_features; i++){
//            s.put(i, new double[2]);
//        }

    }

    //TODO 通过Monte Carlo random sampling 计算 shapley value
    private double[] computeShapBySampling(int num_features, int num_sample, String model) {
        double[] shap_matrix = new double[num_features];  //大数组

        //进行若干次采样（num_sample：采样的次数）
        for(int r=0; r<num_sample; r++){
            //1）生成一个打乱的序列
            int[] p = permutation(num_features);
//            int[] tmpExample = new int[num_features]; //构建一个tmpExample
//            tmpExample = Arrays.copyOf(example, num_features);

            //2）利用序列p, 求每个feature 的 marginal contribution
            /* P = [ABC]: A - 0; AB -A; ABC-AB
             *  P = [BAC]: B - 0; AB -B; ABC-AB */
            for(int ind = 0; ind <p.length; ind ++){ // ele 对应一个特征
                int ele = p[ind];  //ele就是对应的特征i
                //*subset_1 第一个特征子集
                ArrayList<Integer> subset_1 = new ArrayList<>();
                for(int i =0; i<=ind; i++){
                    subset_1.add(p[i]);
                }
                double value_1 = 0;
                //*subset_2 第二个特征子集
                ArrayList<Integer> subset_2 = new ArrayList<>(subset_1);
                subset_2.remove(ind);
                double value_2 = 0;

                //*将两个list重新排列
                Collections.sort(subset_1);
                Collections.sort(subset_2);

                //4)分别求函数值
                if(model.equals("airport")){
                    value_1 = value_airport(subset_1, Info.given_weights_airport);
                    value_2 = value_airport(subset_2, Info.given_weights_airport);
                }
                else if(model.equals("voting")){
                    double halfSum = Arrays.stream(Info.given_weights_voting).sum() / 2;  //对given_weights中的数据求和
                    value_1 = value_voting(subset_1, Info.given_weights_voting, halfSum);
                    value_2 = value_voting(subset_2, Info.given_weights_voting, halfSum);
                }
                shap_matrix[ele] += value_1 - value_2;
            }
        }
        // 5) 求r次采样的平均值
        for(int i=0; i < shap_matrix.length; i++){
            shap_matrix[i] = shap_matrix[i] / num_sample;
        }
        return shap_matrix;
    }

    //TODO 通过Monte Carlo random sampling 计算 shapley value
    private double[] computeShapBySampling_scale(int num_features, int num_sample, String model, double[] given_weights) {
        double[] shap_matrix = new double[num_features];  //大数组

        //进行若干次采样（num_sample：采样的次数）
        for(int r=0; r<num_sample; r++){
            //1）生成一个打乱的序列
            int[] p = permutation(num_features);
//            int[] tmpExample = new int[num_features]; //构建一个tmpExample
//            tmpExample = Arrays.copyOf(example, num_features);

            //2）利用序列p, 求每个feature 的 marginal contribution
            /* P = [ABC]: A - 0; AB -A; ABC-AB
             *  P = [BAC]: B - 0; AB -B; ABC-AB */
            for(int ind = 0; ind <p.length; ind ++){ // ele 对应一个特征
                int ele = p[ind];  //ele就是对应的特征i
                //*subset_1 第一个特征子集
                ArrayList<Integer> subset_1 = new ArrayList<>();
                for(int i =0; i<=ind; i++){
                    subset_1.add(p[i]);
                }
                double value_1 = 0;
                //*subset_2 第二个特征子集
                ArrayList<Integer> subset_2 = new ArrayList<>(subset_1);
                subset_2.remove(ind);
                double value_2 = 0;

                //*将两个list重新排列
                Collections.sort(subset_1);
                Collections.sort(subset_2);

                //4)分别求函数值
                if(model.equals("airport")){
                    value_1 = value_airport_2(subset_1, given_weights);
                    value_2 = value_airport_2(subset_2, given_weights);
                }
                else if(model.equals("voting")){
                    double halfSum = 1.0 * Arrays.stream(Info.given_weights_voting).sum() / 2;  //对given_weights中的数据求和
                    value_1 = value_voting(subset_1, given_weights, halfSum);
                    value_2 = value_voting(subset_2, given_weights, halfSum);
                }
                shap_matrix[ele] += value_1 - value_2;
            }
//            System.out.println("finish: " + r + "  /  " + num_sample);
        }
        // 5) 求r次采样的平均值
        for(int i=0; i < shap_matrix.length; i++){
            shap_matrix[i] = shap_matrix[i] / num_sample;
        }
        return shap_matrix;
    }

    //TODO 通过Monte Carlo random sampling 计算 shapley value
    // benchmark == scale 一模一样，只是benchmark() 版本会的打印进度
    private double[] computeShapBySampling_benchmark(int num_features, int num_sample, String model, double[] given_weights) {
        double[] shap_matrix = new double[num_features];  //大数组

        //进行若干次采样（num_sample：采样的次数）
        for(int r=0; r<num_sample; r++){
            //1）生成一个打乱的序列
            int[] p = permutation(num_features);
//            int[] tmpExample = new int[num_features]; //构建一个tmpExample
//            tmpExample = Arrays.copyOf(example, num_features);

            //2）利用序列p, 求每个feature 的 marginal contribution
            /* P = [ABC]: A - 0; AB -A; ABC-AB
             *  P = [BAC]: B - 0; AB -B; ABC-AB */
            for(int ind = 0; ind <p.length; ind ++){ // ele 对应一个特征
                int ele = p[ind];  //ele就是对应的特征i
                //*subset_1 第一个特征子集
                ArrayList<Integer> subset_1 = new ArrayList<>();
                for(int i =0; i<=ind; i++){
                    subset_1.add(p[i]);
                }
                double value_1 = 0;
                //*subset_2 第二个特征子集
                ArrayList<Integer> subset_2 = new ArrayList<>(subset_1);
                subset_2.remove(ind);
                double value_2 = 0;

                //*将两个list重新排列
                Collections.sort(subset_1);
                Collections.sort(subset_2);

                //4)分别求函数值
                if(model.equals("airport")){
                    value_1 = value_airport_2(subset_1, given_weights);
                    value_2 = value_airport_2(subset_2, given_weights);
                }
                else if(model.equals("voting")){
                    double halfSum = Arrays.stream(Info.given_weights_voting).sum() / 2;  //对given_weights中的数据求和
                    value_1 = value_voting(subset_1, given_weights, halfSum);
                    value_2 = value_voting(subset_2, given_weights, halfSum);
                }
                shap_matrix[ele] += value_1 - value_2;
            }
            System.out.println("finish: " + r + "  /  " + num_sample);
        }
        // 5) 求r次采样的平均值
        for(int i=0; i < shap_matrix.length; i++){
            shap_matrix[i] = shap_matrix[i] / num_sample;
        }
        return shap_matrix;
    }

    //TODO 生成一个长度为n随机序列，序列中的值是[0， n-1]
    private int[] permutation(int numFeatures) {
        //1) 初始化perm序列（initial 对应的位置是对应的值）
        int[] perm = new int[numFeatures];
        for(int i=0; i<numFeatures; i++){
            perm[i] = i;
        }

        //2）打乱perm序列
        Random rand = new Random();
        /* Fisher-Yates洗牌算法（Knuth洗牌算法）来对数组进行打乱顺序。
        该算法的思想是从数组末尾开始，依次将当前位置的元素与前面随机位置的元素交换，直到数组的第一个位置。
        这样可以保证每个元素被随机置换的概率相同。*/
        for(int i=numFeatures-1; i>0; i--){
            int j = rand.nextInt(i+1); //从[0, i+1)中随机选取一个int
            int temp = perm[i];  // 交换位置
            perm[i] = perm[j];
            perm[j] = temp;
        }
        return perm;
    }

    //TODO 定义 Airport game函数v(S)
    private int value_airport(ArrayList<Integer> subset, double[] given_weights) {
        int subset_max_ele = -1;
        if(subset.size() == 0){
            return 0;
        }
        else if(subset.size() == 1){
            return (int)given_weights[subset.get(0)];  //返回值是最大下标对应的weights
        }
        else{
            for(int ele : subset){
                if(subset_max_ele < ele){
                    subset_max_ele = ele;
                }
            }
            return (int)given_weights[subset_max_ele];  //返回值是最大下标对应的weights
        }
    }

    private double value_airport_2(ArrayList<Integer> subset, double[] given_weights) {
        double maxValue = 0;
        for(int ele : subset){
            maxValue = Math.max(given_weights[ele], maxValue);
        }
        return maxValue;  //返回值是最大值
    }
    //TODO 定义 Voting game函数v(S)
    private int value_voting(ArrayList<Integer> subset, double[] given_weights, double halfSum) {
        int weights_sum = 0;
        //求weights_sum
        for(int ele : subset){
            weights_sum += given_weights[ele];
        }
        if(weights_sum > halfSum){
            return 1;
        }
        else return 0;
    }

    //TODO 计算最大误差 (Voting game & Airport game)
    private double computeMaxError(double[] shap_matrix, double[] exact) {
        double error_max = 0;  //误差总量，最后除以特征数
        for(int i=0; i<exact.length; i++){
            error_max = Math.max(Math.abs((shap_matrix[i] - exact[i]) / exact[i]), error_max);
        }
        return error_max;
    }

    private double computeAverageError(double[] shap_matrix, double[] exact, int num_of_features) {
        double error_ave = 0;  //误差总量，最后除以特征数
        for(int i=0; i<exact.length; i++){
            error_ave += Math.abs((shap_matrix[i] - exact[i]) / exact[i]);
        }
        error_ave = 1.0f * error_ave / num_of_features;
        return error_ave;
    }

    /* TODO [Scale_MC_Algorithm]: 通过Monte Carlo sampling 计算 shapley value (Benchmark-不计算误差)
       num_sample: 采样的数量
       model : 当前进行的game */
    public void MCShap_scale(int num_sample, String model){
        //1) 初始化num_features & exact
        int num_features = Info.num_of_features;  // the number of features
        double[] exact = new double[num_features] ;  // the benchmark of shapley value (此实验不计算误差，输出的结果作为benchmark)
        double[] given_weights = new double[num_features];
        String path = null;
        DataInputStream dataInputStream = null;

        /*【改】：改用given_weight*/
        if(model.equals("airport")){
            path = Info.ROOT + "airport_" + num_features + ".npy";
        }
        else if (model.equals("voting")) {
            path = Info.ROOT + "voting_" + num_features + ".npy";
        }
        else{
            path = null;
        }

        try {
            dataInputStream = new DataInputStream(Files.newInputStream(Paths.get(path)));
            NumpyReader reader = new NumpyReader();
            given_weights = reader.readIntArray(dataInputStream);  //读入.npy文件中存储的数组
        }
        catch (Exception e) {
            throw new RuntimeException(e);
        }

        //2）计算shapley value
        long time_1 = System.currentTimeMillis();
        double[] shap_matrix = computeShapBySampling_scale(num_features, num_sample, model, given_weights);
        long time_2 = System.currentTimeMillis();
        System.out.println("time : " + (time_2 - time_1) * 0.001 );  //+ "S"

        //3）读入误差
        String benchmark_path = Info.benchmark_path;
        FileOpera opera = new FileOpera();
        double[] benchmark_sv = opera.file_read(benchmark_path, num_features);  //读入benchmark

        //4）计算误差
        double error_max = computeMaxError(shap_matrix, benchmark_sv); //计算最大误差
        double error_ave = computeAverageError(shap_matrix, benchmark_sv, num_features);  //计算平均误差
        System.out.println(model + " Game:  " + "error_ave: " + error_ave + " \t"  +  "error_max: " + error_max );

    }


    /* TODO [Benchmark-MC Algorithm]: 通过Monte Carlo sampling 计算 shapley value (Benchmark-不计算误差)
       num_sample: 采样的数量
       model : 当前进行的game */
    public void MCShap_Benchmark(int num_sample, String model){
        //1) 初始化num_features & exact
        int num_features = Info.num_of_features;  // the number of features
        double[] exact = new double[num_features] ;  // the benchmark of shapley value (此实验不计算误差，输出的结果作为benchmark)
        double[] given_weights = new double[num_features];
        String path = null;
        DataInputStream dataInputStream = null;

        /*【改】：改用given_weight*/
        if(model.equals("airport")){
            path = Info.ROOT + "airport_" + num_features + ".npy";
        }
        else if (model.equals("voting")) {
            path = Info.ROOT + "voting_" + num_features + ".npy";
        }
        else{
            path = null;
        }

        try {
            dataInputStream = new DataInputStream(Files.newInputStream(Paths.get(path)));
            NumpyReader reader = new NumpyReader();
            given_weights = reader.readIntArray(dataInputStream);  //读入.npy文件中存储的数组
        }
        catch (Exception e) {
            throw new RuntimeException(e);
        }

        //2）计算shapley value
        long time_1 = System.currentTimeMillis();
        double[] shap_matrix = computeShapBySampling_benchmark(num_features, num_sample, model, given_weights);
        long time_2 = System.currentTimeMillis();
        System.out.println("time : " + (time_2 - time_1) * 0.001 );  //+ "S"

        //3）Benchmark写到文件中
        String file = Info.ROOT + "Benchmark/benchmark_" + model + "_" + num_features + "_" + Info.total_samples_num + ".tex";
        FileOpera opera = new FileOpera();
        opera.outputExUtilTable(file, shap_matrix);

    }

//    public void MCShap(int num_sumples){
//        int[][] args = split_premutation(num_sumples);
//
//        //创建线程池
//        ExecutorService pool = Executors.newCachedThreadPool();
//
//        //创建函数
//        Function<int[], Double> func = new Function<int[], Double>() {
//            @Override
//            public Double apply(int[] ints) {
//                return null;
//            }
//        }
//        Function<int[], Double> func = arr -> mc_shap_task(game, arr);
//
//        pool.invokeAll(Arrays.stream(args)).map()
//
//    }

//    private int[][] split_premutation(int num_sumples, int proc_num) {
//
//    }
}
