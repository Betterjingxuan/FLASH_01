package Compared_algorithm;

import Game.GameClass;
import Global.*;
import java.io.DataInputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

/*TODO subset不排序的airport*/
public class MonteCarlo_3 {

    int num_features;  //the number of features
    double[] exact;   // the exact shapley value
    int num_samples;

    /* TODO [MC Algorithm]: 通过Monte Carlo sampling 计算 shapley value
       num_sample: 采样的数量; model : 当前进行的game */
    public void MCShap(boolean gene_weight, String model){

        //1)初始化
        GameClass game = new GameClass();
        initialization(game, gene_weight, model);

        long ave_runtime = 0;
        double ave_error_max = 0; //计算最大误差
        double ave_error_ave = 0;
        double ave_mse = 0;
        for(int i=0; i< Info.timesRepeat; i++) {
            long time_1 = System.currentTimeMillis();
//            double[] shap_matrix = computeShapBySampling(this.num_features, num_sample, model);   //正常
            double[] shap_matrix = computeShapBySampling_2(game, this.num_features, this.num_samples, model);   //实验版
//            double[] shap_matrix = computeShapBySampling_3(this.num_features, num_sample, model);   //超高随机性（）
            long time_2 = System.currentTimeMillis();
            //4）计算误差
            Comparer comparator = new Comparer();
            double error_max = comparator.computeMaxError(shap_matrix, this.exact); //计算最大误差
            double error_ave = comparator.computeAverageError(shap_matrix, this.exact, this.num_features);  //计算平均误差
            double mse = comparator.computeMSE(shap_matrix, this.exact, this.num_features);

            ave_runtime += time_2 - time_1;
            ave_error_max += error_max;
            ave_error_ave += error_ave;
            ave_mse += mse;
            System.out.println("run: " + i + " " + "error_ave: " + error_ave + " \t"  +  "error_max: " + error_max + " \t"  +  "mse: " + mse);
        }

        // 5）输出时间
        System.out.println(model + " Game:  " + "error_ave: " + ave_error_ave/Info.timesRepeat + " \t"  +
                "error_max: " + ave_error_max/Info.timesRepeat + " \t"  +  "mse: " + ave_mse/Info.timesRepeat);
        System.out.println("MC_V3 time : " + (ave_runtime * 0.001)/ Info.timesRepeat);  //+ "S"
    }


    //TODO 通过Monte Carlo random sampling 计算 shapley value
    private double[] computeShapBySampling(GameClass game, int num_features, int num_sample, String model) {
        double[] shap_matrix = new double[num_features];  //大数组

        //进行若干次采样（num_sample：采样的次数）
        for(int r=0; r<num_sample; r++){
            //1）生成一个打乱的序列
            int[] p = permutation(num_features);

            //2）利用序列p, 求每个feature 的 marginal contribution
            /* P = [ABC]: A - 0; AB -A; ABC-AB
             *  P = [BAC]: B - 0; AB -B; ABC-AB */
            for(int ind = 0; ind <p.length; ind ++){ // ele 对应一个特征
                int ele = p[ind];  //ele就是对应的特征i
                //*subset_1 第1个特征子集
                ArrayList<Integer> subset_1 = new ArrayList<>();
                for(int i =0; i<ind; i++){
                    subset_1.add(p[i]);
                }
                double value_1 = 0;

                //*subset_2 第二个特征子集
                ArrayList<Integer> subset_2 = new ArrayList<>(subset_1);
                subset_2.add(ele);
                double value_2 = 0;

                //4)分别求函数值
                value_1 = game.gameValue(model, subset_1);
                value_2 = game.gameValue(model, subset_2);
                shap_matrix[ele] += value_2 - value_1;
            }
        }
        // 5) 求r次采样的平均值
        for(int i=0; i < shap_matrix.length; i++){
            shap_matrix[i] = shap_matrix[i] / num_sample;
        }
        return shap_matrix;
    }

    //TODO 通过Monte Carlo random sampling 计算 shapley value
    private double[] computeShapBySampling_2(GameClass game, int num_features, int num_sample, String model) {
        double[] shap_matrix = new double[num_features];  //大数组
        //进行若干次采样（num_sample：采样的次数）
        for(int r=0; r<num_sample; r++){
            //1）生成一个打乱的序列
            ArrayList<Integer> p = new ArrayList<>();
            p = permutation_2(num_features);

            //2）利用序列p, 求每个feature 的 marginal contribution
            /* P = [ABC]: A - 0; AB -A; ABC-AB
             *  P = [BAC]: B - 0; AB -B; ABC-AB */
            for(int ind = 0; ind <p.size(); ind ++){ // ele 对应一个特征
                int ele = p.get(ind);  //ele就是对应的特征i
                //*subset_1 第1个特征子集
                ArrayList<Integer> subset_1 = new ArrayList<>();
                double value_1 = 0;
                ArrayList<Integer> subset_2 = new ArrayList<>();
                double value_2 = 0;
                for(int i =0; i<ind; i++){
                    subset_1.add(p.get(i));
                }
                //*subset_2 第二个特征子集
                for(int i =0; i<ind; i++){
                    subset_2.add(p.get(i));
                }
                subset_2.add(ele);
                //4)分别求函数值
                value_1 = game.gameValue(model, subset_1);
                value_2 = game.gameValue(model, subset_2);
                shap_matrix[ele] += value_2 - value_1;
            }
        }
        // 5) 求r次采样的平均值
        for(int i=0; i < shap_matrix.length; i++){
            shap_matrix[i] = shap_matrix[i] / num_sample;
        }
        return shap_matrix;
    }

    //TODO 通过Monte Carlo random sampling 计算 shapley value
    private double[] computeShapBySampling_scale(GameClass game, int num_features, int num_sample, String model) {
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
                value_1 = game.gameValue(model, subset_1);
                value_2 = game.gameValue(model, subset_2);
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
    private double[] computeShapBySampling_benchmark(GameClass game, int num_features, int num_sample, String model, double[] given_weights) {
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
                value_1 = game.gameValue(model, subset_1);
                value_2 = game.gameValue(model, subset_2);
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

    private ArrayList<Integer> permutation_2(int numFeatures) {
        //1) 初始化perm序列（initial 对应的位置是对应的值）
        ArrayList<Integer> perm = new ArrayList<>();
        for(int i=0; i<numFeatures; i++){
            perm.add(i);
        }

        //2）打乱perm序列
        Random rand = new Random(System.currentTimeMillis());
        /* Fisher-Yates洗牌算法（Knuth洗牌算法）来对数组进行打乱顺序。
        该算法的思想是从数组末尾开始，依次将当前位置的元素与前面随机位置的元素交换，直到数组的第一个位置。
        这样可以保证每个元素被随机置换的概率相同。*/
//        for(int i=numFeatures/2; i>0; i--){
//            int ind_1 = rand.nextInt(numFeatures); //从[0, i+1)中随机选取一个int
//            int ind_2 = rand.nextInt(numFeatures); //从[0, i+1)中随机选取一个int
//            int temp = perm.get(ind_1);  // 交换位置
//            perm.set(ind_1, perm.get(ind_2));
//            perm.set(ind_2,temp);
//        }
        for(int i= numFeatures/4; i>1; i--){
            int j = rand.nextInt(numFeatures); //从[0, i+1)中随机选取一个int
            int temp = perm.get(j);  // 交换位置
            perm.set(j, perm.get(i));
            perm.set(i, temp);
        }
        return perm;
    }

    private ArrayList<Integer> permutation_3(int numFeatures) {
        //1) 初始化perm序列（initial 对应的位置是对应的值）
        ArrayList<Integer> perm = new ArrayList<>();
        for(int i=0; i<numFeatures; i++){
            perm.add(i);
        }

        // 打乱序列
        Collections.shuffle(perm);
        return perm;
    }

    /* TODO [Scale_MC_Algorithm]: 通过Monte Carlo sampling 计算 shapley value (Benchmark-不计算误差)
       num_sample: 采样的数量
       model : 当前进行的game */
    public void MCShap_scale(GameClass game, int num_sample, String model){
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
        double[] shap_matrix = computeShapBySampling_scale(game, num_features, num_sample, model);
        long time_2 = System.currentTimeMillis();
        System.out.println("time : " + (time_2 - time_1) * 0.001 );  //+ "S"

        //3）读入误差
        String benchmark_path = Info.benchmark_path;
        FileOpera opera = new FileOpera();
        double[] benchmark_sv = opera.file_read(benchmark_path, num_features);  //读入benchmark

        //4）计算误差
        Comparer comparer = new Comparer();
        double error_max = comparer.computeMaxError(shap_matrix, benchmark_sv); //计算最大误差
        double error_ave = comparer.computeAverageError(shap_matrix, benchmark_sv, num_features);  //计算平均误差
        System.out.println(model + " Game:  " + "error_ave: " + error_ave + " \t"  +  "error_max: " + error_max );

    }

    private void initialization(GameClass game, boolean gene_weight, String modelName) {
        game.gameInit(gene_weight, modelName);
        this.num_features = game.num_features; //the number of features
        this.exact = game.exact;   // the exact shapley value
//        this.given_weights = game.given_weights;
//        this.halfSum = game.halfSum;  //for Voting game
        this.num_samples = Info.setting;
    }

    /* TODO [Benchmark-MC Algorithm]: 通过Monte Carlo sampling 计算 shapley value (Benchmark-不计算误差)
       num_sample: 采样的数量
       model : 当前进行的game */
    public void MCShap_Benchmark(GameClass game, int num_sample, String model){
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
        double[] shap_matrix = computeShapBySampling_benchmark(game, num_features, num_sample, model, given_weights);
        long time_2 = System.currentTimeMillis();
        System.out.println("time : " + (time_2 - time_1) * 0.001 );  //+ "S"

        //3）Benchmark写到文件中
        String file = Info.ROOT + "Benchmark/benchmark_" + model + "_" + num_features + "_" + Info.total_samples_num + ".tex";
        FileOpera opera = new FileOpera();
        opera.outputExUtilTable(file, shap_matrix);

    }




}
