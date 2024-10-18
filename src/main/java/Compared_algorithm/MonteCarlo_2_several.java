package Compared_algorithm;

import Global.*;
import okhttp3.OkHttpClient;
import structure.*;
import java.io.DataInputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import Game.Health;

/*TODO subset不排序的airport*/
public class MonteCarlo_2_several {

    /* TODO [MC Algorithm]: 通过Monte Carlo sampling 计算 shapley value
       num_sample: 采样的数量
       model : 当前进行的game */
    public void MCShap(int num_sample, String model){

        //1)初始化
        int num_features = 0;  // the number of features
        double[] exact = new double[Info.num_of_features];  // the exact shapley value
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
        //TODO [Shoes GAME]
        else if(model.equals("shoes")){
            num_features = Info.num_of_features_shoes;
            exact = new double[Info.num_of_features_shoes];
            Arrays.fill(exact, Info.shoes_exact);
        }
        else if(model.equals("health")){
            num_features = Info.num_of_features_health;
            String benchmark_path = Info.benchmark_path;
            FileOpera opera = new FileOpera();
            exact = new double[Info.num_of_features_health];
            exact = opera.file_read(benchmark_path, Info.num_of_features_health);
        }

        long ave_runtime = 0;
        double ave_error_max = 0; //计算最大误差
        double ave_error_ave = 0;
        for(int i=0; i< Info.timesRepeat; i++) {
            long time_1 = System.currentTimeMillis();
            double[] shap_matrix = computeShapBySampling(num_features, num_sample, model);
            long time_2 = System.currentTimeMillis();
            //4）计算误差
            double error_max = computeMaxError(shap_matrix, exact); //计算最大误差
            double error_ave = computeAverageError(shap_matrix, exact, num_features);  //计算平均误差

            ave_runtime += time_2 - time_1;
            ave_error_max += error_max;
            ave_error_ave += error_ave;
        }

        // 5）输出时间
        System.out.println(model + " Game:  " + "error_ave: " + ave_error_ave/Info.timesRepeat + " \t"  +  "error_max: " + ave_error_max/Info.timesRepeat );
        System.out.println("MC_V2 time : " + (ave_runtime * 0.001)/ Info.timesRepeat);  //+ "S"

    }

    //TODO 通过Monte Carlo random sampling 计算 shapley value
    private double[] computeShapBySampling(int num_features, int num_sample, String model) {
        double[] shap_matrix = new double[num_features];  //大数组
        OkHttpClient client = new OkHttpClient();
        Health healthGame = new Health();
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
                ArrayList<Integer> subset_2 = new ArrayList<>();  //*subset_2 第二个特征子集
                for(int i =0; i<ind; i++){
                    subset_2.add(p[i]);
                }
                double value_2 = 0;
                 //*subset_2 第二个特征子集
                ArrayList<Integer> subset_1 = new ArrayList<>(subset_2);
                subset_1.add(ele);
                double value_1 = 0;

                //*将两个list重新排列
//                Collections.sort(subset_1);
//                Collections.sort(subset_2);

                //4)分别求函数值
                if(model.equals("airport")){
                    value_1 = value_airport_2(subset_1, Info.given_weights_airport);
                    value_2 = value_airport_2(subset_2, Info.given_weights_airport);
                }
                else if(model.equals("voting")){
                    double halfSum = Arrays.stream(Info.given_weights_voting).sum() / 2;  //对given_weights中的数据求和
                    value_1 = value_voting(subset_1, Info.given_weights_voting, halfSum);
                    value_2 = value_voting(subset_2, Info.given_weights_voting, halfSum);
                }
                else if(model.equals("shoes")){
                    value_1 = value_shoes(subset_1);
                    value_2 = value_shoes(subset_2);
                }
                else if(model.equals("health")){
                value_1 = healthGame.Health_value(Info.instance_health, subset_1, client);
                value_2 = healthGame.Health_value(Info.instance_health, subset_2, client);
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
//                Collections.sort(subset_1);
//                Collections.sort(subset_2);

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
                else if(model.equals("shoes")){
                    value_1 = value_shoes(subset_1);
                    value_2 = value_shoes(subset_2);
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
//                Collections.sort(subset_1);
//                Collections.sort(subset_2);

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
                else if(model.equals("shoes")){
                    value_1 = value_shoes(subset_1);
                    value_2 = value_shoes(subset_2);
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

    //TODO 因为scale时生成的weight无序，所以不能直接用下标判断对应的返回值
    private int value_airport_2(ArrayList<Integer> subset, double[] given_weights) {
        int maxValue = 0;
        for(int ele : subset){
            maxValue = Math.max((int)given_weights[ele], maxValue);
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

    //TODO [Shoes Game]
    private int value_shoes(ArrayList<Integer> subset) {
        int left_shoes = 0;    //left shoes 的个数
        int right_shoes = 0;  //right shoes 的个数
        for(int ele : subset){
            if(ele < 50){
                left_shoes ++;
            }
            else{
                right_shoes ++;
            }
        }
        return Math.min(left_shoes, right_shoes);  //返回值
    }

    public double value_modelPrediction(double[] input) {
        return 0.6964815986513586 + Math.exp(-0.013050887686940627 * (Math.pow(1.0 - input[0], 2.0) + Math.pow(1.0 - input[1], 2.0) + Math.pow(1.0 - input[2], 2.0) + Math.pow(1.0 - input[3], 2.0) + Math.pow(10.0 - input[4], 2.0) + Math.pow(1.0 - input[5], 2.0) + Math.pow(1.0 - input[6], 2.0) + Math.pow(1.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * -0.3919284854007987 + Math.exp(-0.013050887686940627 * (Math.pow(5.0 - input[0], 2.0) + Math.pow(3.0 - input[1], 2.0) + Math.pow(4.0 - input[2], 2.0) + Math.pow(3.0 - input[3], 2.0) + Math.pow(4.0 - input[4], 2.0) + Math.pow(5.0 - input[5], 2.0) + Math.pow(4.0 - input[6], 2.0) + Math.pow(7.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * -1.0 + Math.exp(-0.013050887686940627 * (Math.pow(3.0 - input[0], 2.0) + Math.pow(1.0 - input[1], 2.0) + Math.pow(1.0 - input[2], 2.0) + Math.pow(1.0 - input[3], 2.0) + Math.pow(2.0 - input[4], 2.0) + Math.pow(2.0 - input[5], 2.0) + Math.pow(7.0 - input[6], 2.0) + Math.pow(1.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * -1.0 + Math.exp(-0.013050887686940627 * (Math.pow(1.0 - input[0], 2.0) + Math.pow(1.0 - input[1], 2.0) + Math.pow(1.0 - input[2], 2.0) + Math.pow(1.0 - input[3], 2.0) + Math.pow(2.0 - input[4], 2.0) + Math.pow(10.0 - input[5], 2.0) + Math.pow(3.0 - input[6], 2.0) + Math.pow(1.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * -1.0 + Math.exp(-0.013050887686940627 * (Math.pow(1.0 - input[0], 2.0) + Math.pow(1.0 - input[1], 2.0) + Math.pow(1.0 - input[2], 2.0) + Math.pow(2.0 - input[3], 2.0) + Math.pow(1.0 - input[4], 2.0) + Math.pow(3.0 - input[5], 2.0) + Math.pow(1.0 - input[6], 2.0) + Math.pow(1.0 - input[7], 2.0) + Math.pow(7.0 - input[8], 2.0))) * -0.03557088758728521 + Math.exp(-0.013050887686940627 * (Math.pow(8.0 - input[0], 2.0) + Math.pow(4.0 - input[1], 2.0) + Math.pow(6.0 - input[2], 2.0) + Math.pow(3.0 - input[3], 2.0) + Math.pow(3.0 - input[4], 2.0) + Math.pow(1.0 - input[5], 2.0) + Math.pow(4.0 - input[6], 2.0) + Math.pow(3.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * -1.0 + Math.exp(-0.013050887686940627 * (Math.pow(5.0 - input[0], 2.0) + Math.pow(3.0 - input[1], 2.0) + Math.pow(4.0 - input[2], 2.0) + Math.pow(1.0 - input[3], 2.0) + Math.pow(4.0 - input[4], 2.0) + Math.pow(1.0 - input[5], 2.0) + Math.pow(3.0 - input[6], 2.0) + Math.pow(1.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * -0.47740525187457483 + Math.exp(-0.013050887686940627 * (Math.pow(7.0 - input[0], 2.0) + Math.pow(1.0 - input[1], 2.0) + Math.pow(2.0 - input[2], 2.0) + Math.pow(3.0 - input[3], 2.0) + Math.pow(2.0 - input[4], 2.0) + Math.pow(1.0 - input[5], 2.0) + Math.pow(2.0 - input[6], 2.0) + Math.pow(1.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * -0.5817795889369591 + Math.exp(-0.013050887686940627 * (Math.pow(5.0 - input[0], 2.0) + Math.pow(4.0 - input[1], 2.0) + Math.pow(4.0 - input[2], 2.0) + Math.pow(5.0 - input[3], 2.0) + Math.pow(7.0 - input[4], 2.0) + Math.pow(10.0 - input[5], 2.0) + Math.pow(3.0 - input[6], 2.0) + Math.pow(2.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * -1.0 + Math.exp(-0.013050887686940627 * (Math.pow(6.0 - input[0], 2.0) + Math.pow(3.0 - input[1], 2.0) + Math.pow(3.0 - input[2], 2.0) + Math.pow(5.0 - input[3], 2.0) + Math.pow(3.0 - input[4], 2.0) + Math.pow(10.0 - input[5], 2.0) + Math.pow(3.0 - input[6], 2.0) + Math.pow(5.0 - input[7], 2.0) + Math.pow(3.0 - input[8], 2.0))) * -1.0 + Math.exp(-0.013050887686940627 * (Math.pow(1.0 - input[0], 2.0) + Math.pow(3.0 - input[1], 2.0) + Math.pow(3.0 - input[2], 2.0) + Math.pow(2.0 - input[3], 2.0) + Math.pow(2.0 - input[4], 2.0) + Math.pow(1.0 - input[5], 2.0) + Math.pow(7.0 - input[6], 2.0) + Math.pow(2.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * -0.39369432798882814 + Math.exp(-0.013050887686940627 * (Math.pow(4.0 - input[0], 2.0) + Math.pow(6.0 - input[1], 2.0) + Math.pow(5.0 - input[2], 2.0) + Math.pow(6.0 - input[3], 2.0) + Math.pow(7.0 - input[4], 2.0) + Math.pow(1.0 - input[5], 2.0) + Math.pow(4.0 - input[6], 2.0) + Math.pow(9.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * -1.0 + Math.exp(-0.013050887686940627 * (Math.pow(6.0 - input[0], 2.0) + Math.pow(6.0 - input[1], 2.0) + Math.pow(6.0 - input[2], 2.0) + Math.pow(9.0 - input[3], 2.0) + Math.pow(6.0 - input[4], 2.0) + Math.pow(1.0 - input[5], 2.0) + Math.pow(7.0 - input[6], 2.0) + Math.pow(8.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * -1.0 + Math.exp(-0.013050887686940627 * (Math.pow(4.0 - input[0], 2.0) + Math.pow(4.0 - input[1], 2.0) + Math.pow(2.0 - input[2], 2.0) + Math.pow(1.0 - input[3], 2.0) + Math.pow(2.0 - input[4], 2.0) + Math.pow(5.0 - input[5], 2.0) + Math.pow(2.0 - input[6], 2.0) + Math.pow(1.0 - input[7], 2.0) + Math.pow(2.0 - input[8], 2.0))) * -1.0 + Math.exp(-0.013050887686940627 * (Math.pow(4.0 - input[0], 2.0) + Math.pow(3.0 - input[1], 2.0) + Math.pow(1.0 - input[2], 2.0) + Math.pow(1.0 - input[3], 2.0) + Math.pow(2.0 - input[4], 2.0) + Math.pow(1.0 - input[5], 2.0) + Math.pow(4.0 - input[6], 2.0) + Math.pow(8.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * -1.0 + Math.exp(-0.013050887686940627 * (Math.pow(1.0 - input[0], 2.0) + Math.pow(1.0 - input[1], 2.0) + Math.pow(1.0 - input[2], 2.0) + Math.pow(1.0 - input[3], 2.0) + Math.pow(2.0 - input[4], 2.0) + Math.pow(5.0 - input[5], 2.0) + Math.pow(5.0 - input[6], 2.0) + Math.pow(1.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * -0.19694769478769497 + Math.exp(-0.013050887686940627 * (Math.pow(5.0 - input[0], 2.0) + Math.pow(4.0 - input[1], 2.0) + Math.pow(5.0 - input[2], 2.0) + Math.pow(1.0 - input[3], 2.0) + Math.pow(8.0 - input[4], 2.0) + Math.pow(1.0 - input[5], 2.0) + Math.pow(3.0 - input[6], 2.0) + Math.pow(6.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * -1.0 + Math.exp(-0.013050887686940627 * (Math.pow(5.0 - input[0], 2.0) + Math.pow(2.0 - input[1], 2.0) + Math.pow(2.0 - input[2], 2.0) + Math.pow(4.0 - input[3], 2.0) + Math.pow(2.0 - input[4], 2.0) + Math.pow(4.0 - input[5], 2.0) + Math.pow(1.0 - input[6], 2.0) + Math.pow(1.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * -1.0 + Math.exp(-0.013050887686940627 * (Math.pow(1.0 - input[0], 2.0) + Math.pow(1.0 - input[1], 2.0) + Math.pow(1.0 - input[2], 2.0) + Math.pow(1.0 - input[3], 2.0) + Math.pow(2.0 - input[4], 2.0) + Math.pow(1.0 - input[5], 2.0) + Math.pow(1.0 - input[6], 2.0) + Math.pow(1.0 - input[7], 2.0) + Math.pow(8.0 - input[8], 2.0))) * -0.9517107779602391 + Math.exp(-0.013050887686940627 * (Math.pow(5.0 - input[0], 2.0) + Math.pow(3.0 - input[1], 2.0) + Math.pow(6.0 - input[2], 2.0) + Math.pow(1.0 - input[3], 2.0) + Math.pow(2.0 - input[4], 2.0) + Math.pow(1.0 - input[5], 2.0) + Math.pow(1.0 - input[6], 2.0) + Math.pow(1.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * -0.9465779520555 + Math.exp(-0.013050887686940627 * (Math.pow(8.0 - input[0], 2.0) + Math.pow(2.0 - input[1], 2.0) + Math.pow(1.0 - input[2], 2.0) + Math.pow(1.0 - input[3], 2.0) + Math.pow(5.0 - input[4], 2.0) + Math.pow(1.0 - input[5], 2.0) + Math.pow(1.0 - input[6], 2.0) + Math.pow(1.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * -1.0 + Math.exp(-0.013050887686940627 * (Math.pow(3.0 - input[0], 2.0) + Math.pow(4.0 - input[1], 2.0) + Math.pow(5.0 - input[2], 2.0) + Math.pow(3.0 - input[3], 2.0) + Math.pow(7.0 - input[4], 2.0) + Math.pow(3.0 - input[5], 2.0) + Math.pow(4.0 - input[6], 2.0) + Math.pow(6.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * -1.0 + Math.exp(-0.013050887686940627 * (Math.pow(6.0 - input[0], 2.0) + Math.pow(9.0 - input[1], 2.0) + Math.pow(7.0 - input[2], 2.0) + Math.pow(5.0 - input[3], 2.0) + Math.pow(5.0 - input[4], 2.0) + Math.pow(8.0 - input[5], 2.0) + Math.pow(4.0 - input[6], 2.0) + Math.pow(2.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * -1.0 + Math.exp(-0.013050887686940627 * (Math.pow(3.0 - input[0], 2.0) + Math.pow(3.0 - input[1], 2.0) + Math.pow(2.0 - input[2], 2.0) + Math.pow(6.0 - input[3], 2.0) + Math.pow(3.0 - input[4], 2.0) + Math.pow(3.0 - input[5], 2.0) + Math.pow(3.0 - input[6], 2.0) + Math.pow(5.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * -1.0 + Math.exp(-0.013050887686940627 * (Math.pow(4.0 - input[0], 2.0) + Math.pow(4.0 - input[1], 2.0) + Math.pow(4.0 - input[2], 2.0) + Math.pow(4.0 - input[3], 2.0) + Math.pow(6.0 - input[4], 2.0) + Math.pow(5.0 - input[5], 2.0) + Math.pow(7.0 - input[6], 2.0) + Math.pow(3.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * -1.0 + Math.exp(-0.013050887686940627 * (Math.pow(8.0 - input[0], 2.0) + Math.pow(3.0 - input[1], 2.0) + Math.pow(3.0 - input[2], 2.0) + Math.pow(1.0 - input[3], 2.0) + Math.pow(2.0 - input[4], 2.0) + Math.pow(2.0 - input[5], 2.0) + Math.pow(3.0 - input[6], 2.0) + Math.pow(2.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * -1.0 + Math.exp(-0.013050887686940627 * (Math.pow(8.0 - input[0], 2.0) + Math.pow(4.0 - input[1], 2.0) + Math.pow(4.0 - input[2], 2.0) + Math.pow(5.0 - input[3], 2.0) + Math.pow(4.0 - input[4], 2.0) + Math.pow(7.0 - input[5], 2.0) + Math.pow(7.0 - input[6], 2.0) + Math.pow(8.0 - input[7], 2.0) + Math.pow(2.0 - input[8], 2.0))) * -1.0 + Math.exp(-0.013050887686940627 * (Math.pow(3.0 - input[0], 2.0) + Math.pow(1.0 - input[1], 2.0) + Math.pow(1.0 - input[2], 2.0) + Math.pow(3.0 - input[3], 2.0) + Math.pow(8.0 - input[4], 2.0) + Math.pow(1.0 - input[5], 2.0) + Math.pow(5.0 - input[6], 2.0) + Math.pow(8.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * -1.0 + Math.exp(-0.013050887686940627 * (Math.pow(6.0 - input[0], 2.0) + Math.pow(8.0 - input[1], 2.0) + Math.pow(8.0 - input[2], 2.0) + Math.pow(1.0 - input[3], 2.0) + Math.pow(3.0 - input[4], 2.0) + Math.pow(4.0 - input[5], 2.0) + Math.pow(3.0 - input[6], 2.0) + Math.pow(7.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * -1.0 + Math.exp(-0.013050887686940627 * (Math.pow(5.0 - input[0], 2.0) + Math.pow(5.0 - input[1], 2.0) + Math.pow(5.0 - input[2], 2.0) + Math.pow(6.0 - input[3], 2.0) + Math.pow(3.0 - input[4], 2.0) + Math.pow(10.0 - input[5], 2.0) + Math.pow(3.0 - input[6], 2.0) + Math.pow(1.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * 0.5620289295324898 + Math.exp(-0.013050887686940627 * (Math.pow(6.0 - input[0], 2.0) + Math.pow(1.0 - input[1], 2.0) + Math.pow(3.0 - input[2], 2.0) + Math.pow(1.0 - input[3], 2.0) + Math.pow(4.0 - input[4], 2.0) + Math.pow(5.0 - input[5], 2.0) + Math.pow(5.0 - input[6], 2.0) + Math.pow(10.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * 0.6016186634059852 + Math.exp(-0.013050887686940627 * (Math.pow(4.0 - input[0], 2.0) + Math.pow(1.0 - input[1], 2.0) + Math.pow(1.0 - input[2], 2.0) + Math.pow(3.0 - input[3], 2.0) + Math.pow(1.0 - input[4], 2.0) + Math.pow(5.0 - input[5], 2.0) + Math.pow(2.0 - input[6], 2.0) + Math.pow(1.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * 1.0 + Math.exp(-0.013050887686940627 * (Math.pow(9.0 - input[0], 2.0) + Math.pow(4.0 - input[1], 2.0) + Math.pow(5.0 - input[2], 2.0) + Math.pow(10.0 - input[3], 2.0) + Math.pow(6.0 - input[4], 2.0) + Math.pow(10.0 - input[5], 2.0) + Math.pow(4.0 - input[6], 2.0) + Math.pow(8.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * 0.06412084138409357 + Math.exp(-0.013050887686940627 * (Math.pow(8.0 - input[0], 2.0) + Math.pow(10.0 - input[1], 2.0) + Math.pow(3.0 - input[2], 2.0) + Math.pow(2.0 - input[3], 2.0) + Math.pow(6.0 - input[4], 2.0) + Math.pow(4.0 - input[5], 2.0) + Math.pow(3.0 - input[6], 2.0) + Math.pow(10.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * 0.44748568592050775 + Math.exp(-0.013050887686940627 * (Math.pow(7.0 - input[0], 2.0) + Math.pow(5.0 - input[1], 2.0) + Math.pow(10.0 - input[2], 2.0) + Math.pow(10.0 - input[3], 2.0) + Math.pow(10.0 - input[4], 2.0) + Math.pow(10.0 - input[5], 2.0) + Math.pow(4.0 - input[6], 2.0) + Math.pow(10.0 - input[7], 2.0) + Math.pow(3.0 - input[8], 2.0))) * 0.00684083017813916 + Math.exp(-0.013050887686940627 * (Math.pow(3.0 - input[0], 2.0) + Math.pow(10.0 - input[1], 2.0) + Math.pow(3.0 - input[2], 2.0) + Math.pow(10.0 - input[3], 2.0) + Math.pow(6.0 - input[4], 2.0) + Math.pow(10.0 - input[5], 2.0) + Math.pow(5.0 - input[6], 2.0) + Math.pow(1.0 - input[7], 2.0) + Math.pow(4.0 - input[8], 2.0))) * 0.18625556270974028 + Math.exp(-0.013050887686940627 * (Math.pow(5.0 - input[0], 2.0) + Math.pow(3.0 - input[1], 2.0) + Math.pow(3.0 - input[2], 2.0) + Math.pow(1.0 - input[3], 2.0) + Math.pow(3.0 - input[4], 2.0) + Math.pow(3.0 - input[5], 2.0) + Math.pow(3.0 - input[6], 2.0) + Math.pow(3.0 - input[7], 2.0) + Math.pow(3.0 - input[8], 2.0))) * 1.0 + Math.exp(-0.013050887686940627 * (Math.pow(10.0 - input[0], 2.0) + Math.pow(10.0 - input[1], 2.0) + Math.pow(10.0 - input[2], 2.0) + Math.pow(3.0 - input[3], 2.0) + Math.pow(10.0 - input[4], 2.0) + Math.pow(10.0 - input[5], 2.0) + Math.pow(9.0 - input[6], 2.0) + Math.pow(10.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * 0.05685426806653855 + Math.exp(-0.013050887686940627 * (Math.pow(10.0 - input[0], 2.0) + Math.pow(10.0 - input[1], 2.0) + Math.pow(10.0 - input[2], 2.0) + Math.pow(10.0 - input[3], 2.0) + Math.pow(3.0 - input[4], 2.0) + Math.pow(10.0 - input[5], 2.0) + Math.pow(10.0 - input[6], 2.0) + Math.pow(6.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * 0.09618511360209893 + Math.exp(-0.013050887686940627 * (Math.pow(10.0 - input[0], 2.0) + Math.pow(8.0 - input[1], 2.0) + Math.pow(4.0 - input[2], 2.0) + Math.pow(4.0 - input[3], 2.0) + Math.pow(4.0 - input[4], 2.0) + Math.pow(10.0 - input[5], 2.0) + Math.pow(3.0 - input[6], 2.0) + Math.pow(10.0 - input[7], 2.0) + Math.pow(4.0 - input[8], 2.0))) * 0.015568868875913218 + Math.exp(-0.013050887686940627 * (Math.pow(3.0 - input[0], 2.0) + Math.pow(4.0 - input[1], 2.0) + Math.pow(5.0 - input[2], 2.0) + Math.pow(2.0 - input[3], 2.0) + Math.pow(6.0 - input[4], 2.0) + Math.pow(8.0 - input[5], 2.0) + Math.pow(4.0 - input[6], 2.0) + Math.pow(1.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * 1.0 + Math.exp(-0.013050887686940627 * (Math.pow(5.0 - input[0], 2.0) + Math.pow(10.0 - input[1], 2.0) + Math.pow(10.0 - input[2], 2.0) + Math.pow(5.0 - input[3], 2.0) + Math.pow(4.0 - input[4], 2.0) + Math.pow(5.0 - input[5], 2.0) + Math.pow(4.0 - input[6], 2.0) + Math.pow(4.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * 0.6660316602976392 + Math.exp(-0.013050887686940627 * (Math.pow(5.0 - input[0], 2.0) + Math.pow(10.0 - input[1], 2.0) + Math.pow(10.0 - input[2], 2.0) + Math.pow(10.0 - input[3], 2.0) + Math.pow(10.0 - input[4], 2.0) + Math.pow(10.0 - input[5], 2.0) + Math.pow(10.0 - input[6], 2.0) + Math.pow(1.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * 0.09098168000872013 + Math.exp(-0.013050887686940627 * (Math.pow(10.0 - input[0], 2.0) + Math.pow(10.0 - input[1], 2.0) + Math.pow(10.0 - input[2], 2.0) + Math.pow(1.0 - input[3], 2.0) + Math.pow(6.0 - input[4], 2.0) + Math.pow(1.0 - input[5], 2.0) + Math.pow(2.0 - input[6], 2.0) + Math.pow(8.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * 0.18308435781299645 + Math.exp(-0.013050887686940627 * (Math.pow(10.0 - input[0], 2.0) + Math.pow(10.0 - input[1], 2.0) + Math.pow(10.0 - input[2], 2.0) + Math.pow(10.0 - input[3], 2.0) + Math.pow(10.0 - input[4], 2.0) + Math.pow(10.0 - input[5], 2.0) + Math.pow(4.0 - input[6], 2.0) + Math.pow(10.0 - input[7], 2.0) + Math.pow(10.0 - input[8], 2.0))) * 0.1397254666382331 + Math.exp(-0.013050887686940627 * (Math.pow(8.0 - input[0], 2.0) + Math.pow(6.0 - input[1], 2.0) + Math.pow(4.0 - input[2], 2.0) + Math.pow(3.0 - input[3], 2.0) + Math.pow(5.0 - input[4], 2.0) + Math.pow(9.0 - input[5], 2.0) + Math.pow(3.0 - input[6], 2.0) + Math.pow(1.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * 0.037936806277875115 + Math.exp(-0.013050887686940627 * (Math.pow(2.0 - input[0], 2.0) + Math.pow(5.0 - input[1], 2.0) + Math.pow(3.0 - input[2], 2.0) + Math.pow(3.0 - input[3], 2.0) + Math.pow(6.0 - input[4], 2.0) + Math.pow(7.0 - input[5], 2.0) + Math.pow(7.0 - input[6], 2.0) + Math.pow(5.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * 1.0 + Math.exp(-0.013050887686940627 * (Math.pow(10.0 - input[0], 2.0) + Math.pow(5.0 - input[1], 2.0) + Math.pow(6.0 - input[2], 2.0) + Math.pow(10.0 - input[3], 2.0) + Math.pow(6.0 - input[4], 2.0) + Math.pow(10.0 - input[5], 2.0) + Math.pow(7.0 - input[6], 2.0) + Math.pow(7.0 - input[7], 2.0) + Math.pow(10.0 - input[8], 2.0))) * 0.016942241777662216 + Math.exp(-0.013050887686940627 * (Math.pow(5.0 - input[0], 2.0) + Math.pow(3.0 - input[1], 2.0) + Math.pow(3.0 - input[2], 2.0) + Math.pow(4.0 - input[3], 2.0) + Math.pow(2.0 - input[4], 2.0) + Math.pow(4.0 - input[5], 2.0) + Math.pow(3.0 - input[6], 2.0) + Math.pow(4.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * 1.0 + Math.exp(-0.013050887686940627 * (Math.pow(4.0 - input[0], 2.0) + Math.pow(2.0 - input[1], 2.0) + Math.pow(3.0 - input[2], 2.0) + Math.pow(5.0 - input[3], 2.0) + Math.pow(3.0 - input[4], 2.0) + Math.pow(8.0 - input[5], 2.0) + Math.pow(7.0 - input[6], 2.0) + Math.pow(6.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * 0.25225681810710976 + Math.exp(-0.013050887686940627 * (Math.pow(8.0 - input[0], 2.0) + Math.pow(6.0 - input[1], 2.0) + Math.pow(4.0 - input[2], 2.0) + Math.pow(10.0 - input[3], 2.0) + Math.pow(10.0 - input[4], 2.0) + Math.pow(1.0 - input[5], 2.0) + Math.pow(3.0 - input[6], 2.0) + Math.pow(5.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * 0.6020764048869371 + Math.exp(-0.013050887686940627 * (Math.pow(5.0 - input[0], 2.0) + Math.pow(8.0 - input[1], 2.0) + Math.pow(4.0 - input[2], 2.0) + Math.pow(10.0 - input[3], 2.0) + Math.pow(5.0 - input[4], 2.0) + Math.pow(8.0 - input[5], 2.0) + Math.pow(9.0 - input[6], 2.0) + Math.pow(10.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * 0.04958251228336161 + Math.exp(-0.013050887686940627 * (Math.pow(5.0 - input[0], 2.0) + Math.pow(2.0 - input[1], 2.0) + Math.pow(3.0 - input[2], 2.0) + Math.pow(1.0 - input[3], 2.0) + Math.pow(6.0 - input[4], 2.0) + Math.pow(10.0 - input[5], 2.0) + Math.pow(5.0 - input[6], 2.0) + Math.pow(1.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * 0.47752287174535935 + Math.exp(-0.013050887686940627 * (Math.pow(1.0 - input[0], 2.0) + Math.pow(4.0 - input[1], 2.0) + Math.pow(3.0 - input[2], 2.0) + Math.pow(10.0 - input[3], 2.0) + Math.pow(4.0 - input[4], 2.0) + Math.pow(10.0 - input[5], 2.0) + Math.pow(5.0 - input[6], 2.0) + Math.pow(6.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * 0.23312572216152702 + Math.exp(-0.013050887686940627 * (Math.pow(5.0 - input[0], 2.0) + Math.pow(3.0 - input[1], 2.0) + Math.pow(5.0 - input[2], 2.0) + Math.pow(5.0 - input[3], 2.0) + Math.pow(3.0 - input[4], 2.0) + Math.pow(3.0 - input[5], 2.0) + Math.pow(4.0 - input[6], 2.0) + Math.pow(10.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * 1.0 + Math.exp(-0.013050887686940627 * (Math.pow(9.0 - input[0], 2.0) + Math.pow(5.0 - input[1], 2.0) + Math.pow(8.0 - input[2], 2.0) + Math.pow(1.0 - input[3], 2.0) + Math.pow(2.0 - input[4], 2.0) + Math.pow(3.0 - input[5], 2.0) + Math.pow(2.0 - input[6], 2.0) + Math.pow(1.0 - input[7], 2.0) + Math.pow(5.0 - input[8], 2.0))) * 0.029939986818475015 + Math.exp(-0.013050887686940627 * (Math.pow(6.0 - input[0], 2.0) + Math.pow(10.0 - input[1], 2.0) + Math.pow(2.0 - input[2], 2.0) + Math.pow(8.0 - input[3], 2.0) + Math.pow(10.0 - input[4], 2.0) + Math.pow(2.0 - input[5], 2.0) + Math.pow(7.0 - input[6], 2.0) + Math.pow(8.0 - input[7], 2.0) + Math.pow(10.0 - input[8], 2.0))) * 0.08422289811241174 + Math.exp(-0.013050887686940627 * (Math.pow(6.0 - input[0], 2.0) + Math.pow(3.0 - input[1], 2.0) + Math.pow(4.0 - input[2], 2.0) + Math.pow(1.0 - input[3], 2.0) + Math.pow(5.0 - input[4], 2.0) + Math.pow(2.0 - input[5], 2.0) + Math.pow(3.0 - input[6], 2.0) + Math.pow(9.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * 1.0 + Math.exp(-0.013050887686940627 * (Math.pow(5.0 - input[0], 2.0) + Math.pow(4.0 - input[1], 2.0) + Math.pow(6.0 - input[2], 2.0) + Math.pow(7.0 - input[3], 2.0) + Math.pow(9.0 - input[4], 2.0) + Math.pow(7.0 - input[5], 2.0) + Math.pow(8.0 - input[6], 2.0) + Math.pow(10.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * 0.21273203268543417 + Math.exp(-0.013050887686940627 * (Math.pow(9.0 - input[0], 2.0) + Math.pow(5.0 - input[1], 2.0) + Math.pow(5.0 - input[2], 2.0) + Math.pow(2.0 - input[3], 2.0) + Math.pow(2.0 - input[4], 2.0) + Math.pow(2.0 - input[5], 2.0) + Math.pow(5.0 - input[6], 2.0) + Math.pow(1.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * 1.0 + Math.exp(-0.013050887686940627 * (Math.pow(5.0 - input[0], 2.0) + Math.pow(10.0 - input[1], 2.0) + Math.pow(10.0 - input[2], 2.0) + Math.pow(10.0 - input[3], 2.0) + Math.pow(5.0 - input[4], 2.0) + Math.pow(2.0 - input[5], 2.0) + Math.pow(8.0 - input[6], 2.0) + Math.pow(5.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * 0.0853801631475682 + Math.exp(-0.013050887686940627 * (Math.pow(8.0 - input[0], 2.0) + Math.pow(3.0 - input[1], 2.0) + Math.pow(5.0 - input[2], 2.0) + Math.pow(4.0 - input[3], 2.0) + Math.pow(5.0 - input[4], 2.0) + Math.pow(10.0 - input[5], 2.0) + Math.pow(1.0 - input[6], 2.0) + Math.pow(6.0 - input[7], 2.0) + Math.pow(2.0 - input[8], 2.0))) * 0.20450970050452283 + Math.exp(-0.013050887686940627 * (Math.pow(4.0 - input[0], 2.0) + Math.pow(10.0 - input[1], 2.0) + Math.pow(8.0 - input[2], 2.0) + Math.pow(5.0 - input[3], 2.0) + Math.pow(4.0 - input[4], 2.0) + Math.pow(1.0 - input[5], 2.0) + Math.pow(10.0 - input[6], 2.0) + Math.pow(1.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * 0.01633520520593213 + Math.exp(-0.013050887686940627 * (Math.pow(3.0 - input[0], 2.0) + Math.pow(3.0 - input[1], 2.0) + Math.pow(6.0 - input[2], 2.0) + Math.pow(4.0 - input[3], 2.0) + Math.pow(5.0 - input[4], 2.0) + Math.pow(8.0 - input[5], 2.0) + Math.pow(4.0 - input[6], 2.0) + Math.pow(4.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * 1.0 + Math.exp(-0.013050887686940627 * (Math.pow(8.0 - input[0], 2.0) + Math.pow(4.0 - input[1], 2.0) + Math.pow(4.0 - input[2], 2.0) + Math.pow(1.0 - input[3], 2.0) + Math.pow(6.0 - input[4], 2.0) + Math.pow(10.0 - input[5], 2.0) + Math.pow(2.0 - input[6], 2.0) + Math.pow(5.0 - input[7], 2.0) + Math.pow(2.0 - input[8], 2.0))) * 0.0016034220659953214 + Math.exp(-0.013050887686940627 * (Math.pow(8.0 - input[0], 2.0) + Math.pow(2.0 - input[1], 2.0) + Math.pow(3.0 - input[2], 2.0) + Math.pow(1.0 - input[3], 2.0) + Math.pow(6.0 - input[4], 2.0) + Math.pow(3.0 - input[5], 2.0) + Math.pow(7.0 - input[6], 2.0) + Math.pow(1.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * 0.7373595960989454 + Math.exp(-0.013050887686940627 * (Math.pow(9.0 - input[0], 2.0) + Math.pow(10.0 - input[1], 2.0) + Math.pow(10.0 - input[2], 2.0) + Math.pow(10.0 - input[3], 2.0) + Math.pow(10.0 - input[4], 2.0) + Math.pow(10.0 - input[5], 2.0) + Math.pow(10.0 - input[6], 2.0) + Math.pow(10.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * 0.07748906155989697 + Math.exp(-0.013050887686940627 * (Math.pow(7.0 - input[0], 2.0) + Math.pow(4.0 - input[1], 2.0) + Math.pow(6.0 - input[2], 2.0) + Math.pow(4.0 - input[3], 2.0) + Math.pow(6.0 - input[4], 2.0) + Math.pow(1.0 - input[5], 2.0) + Math.pow(4.0 - input[6], 2.0) + Math.pow(3.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * 1.0 + Math.exp(-0.013050887686940627 * (Math.pow(5.0 - input[0], 2.0) + Math.pow(7.0 - input[1], 2.0) + Math.pow(4.0 - input[2], 2.0) + Math.pow(1.0 - input[3], 2.0) + Math.pow(6.0 - input[4], 2.0) + Math.pow(1.0 - input[5], 2.0) + Math.pow(7.0 - input[6], 2.0) + Math.pow(10.0 - input[7], 2.0) + Math.pow(3.0 - input[8], 2.0))) * 0.8985969985629755 + Math.exp(-0.013050887686940627 * (Math.pow(10.0 - input[0], 2.0) + Math.pow(10.0 - input[1], 2.0) + Math.pow(6.0 - input[2], 2.0) + Math.pow(3.0 - input[3], 2.0) + Math.pow(3.0 - input[4], 2.0) + Math.pow(10.0 - input[5], 2.0) + Math.pow(4.0 - input[6], 2.0) + Math.pow(3.0 - input[7], 2.0) + Math.pow(2.0 - input[8], 2.0))) * 0.036097420062460965 + Math.exp(-0.013050887686940627 * (Math.pow(7.0 - input[0], 2.0) + Math.pow(3.0 - input[1], 2.0) + Math.pow(4.0 - input[2], 2.0) + Math.pow(4.0 - input[3], 2.0) + Math.pow(3.0 - input[4], 2.0) + Math.pow(3.0 - input[5], 2.0) + Math.pow(3.0 - input[6], 2.0) + Math.pow(2.0 - input[7], 2.0) + Math.pow(7.0 - input[8], 2.0))) * 1.0 + Math.exp(-0.013050887686940627 * (Math.pow(10.0 - input[0], 2.0) + Math.pow(4.0 - input[1], 2.0) + Math.pow(3.0 - input[2], 2.0) + Math.pow(10.0 - input[3], 2.0) + Math.pow(4.0 - input[4], 2.0) + Math.pow(10.0 - input[5], 2.0) + Math.pow(10.0 - input[6], 2.0) + Math.pow(1.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * 0.005051218314311532 + Math.exp(-0.013050887686940627 * (Math.pow(5.0 - input[0], 2.0) + Math.pow(2.0 - input[1], 2.0) + Math.pow(3.0 - input[2], 2.0) + Math.pow(4.0 - input[3], 2.0) + Math.pow(2.0 - input[4], 2.0) + Math.pow(7.0 - input[5], 2.0) + Math.pow(3.0 - input[6], 2.0) + Math.pow(6.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * 1.0 + Math.exp(-0.013050887686940627 * (Math.pow(5.0 - input[0], 2.0) + Math.pow(4.0 - input[1], 2.0) + Math.pow(6.0 - input[2], 2.0) + Math.pow(8.0 - input[3], 2.0) + Math.pow(4.0 - input[4], 2.0) + Math.pow(1.0 - input[5], 2.0) + Math.pow(8.0 - input[6], 2.0) + Math.pow(10.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * 1.0 + Math.exp(-0.013050887686940627 * (Math.pow(8.0 - input[0], 2.0) + Math.pow(10.0 - input[1], 2.0) + Math.pow(10.0 - input[2], 2.0) + Math.pow(10.0 - input[3], 2.0) + Math.pow(6.0 - input[4], 2.0) + Math.pow(10.0 - input[5], 2.0) + Math.pow(10.0 - input[6], 2.0) + Math.pow(10.0 - input[7], 2.0) + Math.pow(10.0 - input[8], 2.0))) * 0.061802649895822126 + Math.exp(-0.013050887686940627 * (Math.pow(8.0 - input[0], 2.0) + Math.pow(2.0 - input[1], 2.0) + Math.pow(4.0 - input[2], 2.0) + Math.pow(1.0 - input[3], 2.0) + Math.pow(5.0 - input[4], 2.0) + Math.pow(1.0 - input[5], 2.0) + Math.pow(5.0 - input[6], 2.0) + Math.pow(4.0 - input[7], 2.0) + Math.pow(4.0 - input[8], 2.0))) * 1.0 + Math.exp(-0.013050887686940627 * (Math.pow(8.0 - input[0], 2.0) + Math.pow(10.0 - input[1], 2.0) + Math.pow(10.0 - input[2], 2.0) + Math.pow(1.0 - input[3], 2.0) + Math.pow(3.0 - input[4], 2.0) + Math.pow(6.0 - input[5], 2.0) + Math.pow(3.0 - input[6], 2.0) + Math.pow(9.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * 0.2414381410372649 + Math.exp(-0.013050887686940627 * (Math.pow(6.0 - input[0], 2.0) + Math.pow(3.0 - input[1], 2.0) + Math.pow(2.0 - input[2], 2.0) + Math.pow(1.0 - input[3], 2.0) + Math.pow(3.0 - input[4], 2.0) + Math.pow(4.0 - input[5], 2.0) + Math.pow(4.0 - input[6], 2.0) + Math.pow(1.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * 1.0 + Math.exp(-0.013050887686940627 * (Math.pow(5.0 - input[0], 2.0) + Math.pow(10.0 - input[1], 2.0) + Math.pow(10.0 - input[2], 2.0) + Math.pow(10.0 - input[3], 2.0) + Math.pow(10.0 - input[4], 2.0) + Math.pow(2.0 - input[5], 2.0) + Math.pow(10.0 - input[6], 2.0) + Math.pow(10.0 - input[7], 2.0) + Math.pow(10.0 - input[8], 2.0))) * 0.1562568111582785 + Math.exp(-0.013050887686940627 * (Math.pow(5.0 - input[0], 2.0) + Math.pow(6.0 - input[1], 2.0) + Math.pow(5.0 - input[2], 2.0) + Math.pow(6.0 - input[3], 2.0) + Math.pow(10.0 - input[4], 2.0) + Math.pow(1.0 - input[5], 2.0) + Math.pow(3.0 - input[6], 2.0) + Math.pow(1.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * 1.0 + Math.exp(-0.013050887686940627 * (Math.pow(10.0 - input[0], 2.0) + Math.pow(4.0 - input[1], 2.0) + Math.pow(4.0 - input[2], 2.0) + Math.pow(6.0 - input[3], 2.0) + Math.pow(2.0 - input[4], 2.0) + Math.pow(10.0 - input[5], 2.0) + Math.pow(2.0 - input[6], 2.0) + Math.pow(3.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * 0.16589609535436806 + Math.exp(-0.013050887686940627 * (Math.pow(5.0 - input[0], 2.0) + Math.pow(6.0 - input[1], 2.0) + Math.pow(7.0 - input[2], 2.0) + Math.pow(8.0 - input[3], 2.0) + Math.pow(8.0 - input[4], 2.0) + Math.pow(10.0 - input[5], 2.0) + Math.pow(3.0 - input[6], 2.0) + Math.pow(10.0 - input[7], 2.0) + Math.pow(3.0 - input[8], 2.0))) * 0.004386823561162363 + Math.exp(-0.013050887686940627 * (Math.pow(10.0 - input[0], 2.0) + Math.pow(8.0 - input[1], 2.0) + Math.pow(8.0 - input[2], 2.0) + Math.pow(2.0 - input[3], 2.0) + Math.pow(8.0 - input[4], 2.0) + Math.pow(10.0 - input[5], 2.0) + Math.pow(4.0 - input[6], 2.0) + Math.pow(8.0 - input[7], 2.0) + Math.pow(10.0 - input[8], 2.0))) * 0.0050905891701812705 + Math.exp(-0.013050887686940627 * (Math.pow(7.0 - input[0], 2.0) + Math.pow(3.0 - input[1], 2.0) + Math.pow(2.0 - input[2], 2.0) + Math.pow(10.0 - input[3], 2.0) + Math.pow(5.0 - input[4], 2.0) + Math.pow(10.0 - input[5], 2.0) + Math.pow(5.0 - input[6], 2.0) + Math.pow(4.0 - input[7], 2.0) + Math.pow(4.0 - input[8], 2.0))) * 0.19178557613230768 + Math.exp(-0.013050887686940627 * (Math.pow(10.0 - input[0], 2.0) + Math.pow(10.0 - input[1], 2.0) + Math.pow(10.0 - input[2], 2.0) + Math.pow(8.0 - input[3], 2.0) + Math.pow(2.0 - input[4], 2.0) + Math.pow(10.0 - input[5], 2.0) + Math.pow(4.0 - input[6], 2.0) + Math.pow(1.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * 0.07910129250404292 + Math.exp(-0.013050887686940627 * (Math.pow(8.0 - input[0], 2.0) + Math.pow(4.0 - input[1], 2.0) + Math.pow(5.0 - input[2], 2.0) + Math.pow(1.0 - input[3], 2.0) + Math.pow(2.0 - input[4], 2.0) + Math.pow(1.0 - input[5], 2.0) + Math.pow(7.0 - input[6], 2.0) + Math.pow(3.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * 0.8178070179092138 + Math.exp(-0.013050887686940627 * (Math.pow(9.0 - input[0], 2.0) + Math.pow(10.0 - input[1], 2.0) + Math.pow(10.0 - input[2], 2.0) + Math.pow(1.0 - input[3], 2.0) + Math.pow(10.0 - input[4], 2.0) + Math.pow(8.0 - input[5], 2.0) + Math.pow(3.0 - input[6], 2.0) + Math.pow(3.0 - input[7], 2.0) + Math.pow(1.0 - input[8], 2.0))) * 0.07650696105738106;
    }

}
