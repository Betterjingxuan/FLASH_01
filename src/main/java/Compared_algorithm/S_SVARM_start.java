package Compared_algorithm;

import Game.ModelGame;
import Global.*;

import java.io.DataInputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;

public class S_SVARM_start {
    int num_features;  //数据集中包含的特征数量

    double[] given_weights;
    double[] exact;  // the exact shapley value
    double halfSum;  //for Voting game

    int num_samples;
    public void model_game(boolean gene_weight, String type, String model_name){
        //1、初始化：根据game初始化 1)num_features； 2）given_weights； 3）exact； 4）num_samples；5）halfSum
        gameInit(gene_weight, model_name);

        //全局初始化(为了节省初始化的时间)
        //1）ModelGame：为了不同的game 调用不同value_function
        ModelGame game = new ModelGame(model_name);

        //3、开始计算shapley value：
        long ave_runtime = 0;
        double ave_error_max = 0; //计算最大误差
        double ave_error_ave = 0;
        for(int t = 0; t< Info.timesRepeat; t++) {

            //1) 初始化
            double[][] phi_i_l_plus = new double[this.num_features][this.num_features];
            double[][] phi_i_l_minus = new double[this.num_features][this.num_features];
            double[][] c_i_l_plus = new double[this.num_features][this.num_features];
            double[][] c_i_l_minus = new double[this.num_features][this.num_features];

            //2) robability distribution over sizes (2,...,n-2) for sampling
            double[] distribution = generateDistribution(type, this.num_features);
            double[] probs = new double[this.num_features + 1];
            for (int s = 0; s <= this.num_features; s++) {
                probs[s] = distribution[s];
            }


            //2)然后给每层分配样本
            Allocation allo = new Allocation();
            //        allo.sampleAllocation_2(Info.num_of_features_voting-1); //sample_num_level 每一层采样几个 【逻辑错误】
//            allo.sampleAllocation_4(this.num_features, level_index); //使用两个矩阵

            long time_1 = System.currentTimeMillis();
//            ShapleyApproximate_pattern(evaluateMatrix, levelMatrix, level_index, allo, game, this.given_weights, shap_matrix);
            long time_2 = System.currentTimeMillis();

            // 计算误差
            Comparer comp = new Comparer();
//            double error_max = comp.computeMaxError(shap_matrix, this.exact, this.num_features); //计算最大误差
//            double error_ave = comp.computeAverageError(shap_matrix, this.exact, this.num_features);  //计算平均误差
//
            ave_runtime += time_2 - time_1;
//            ave_error_max += error_max;
//            ave_error_ave += error_ave;
        }

        System.out.println(model_name + ":  " + "error_ave: " + ave_error_ave/Info.timesRepeat + " \t"  +  "error_max: " + ave_error_max/Info.timesRepeat );
        System.out.println("Shap_PSA time : " + (ave_runtime * 0.001)/ Info.timesRepeat );  //+ "S"
    }

    private double[] generateDistribution(String name, int n) {
        switch (name) {
            case "paper":
                return generatePaperDistribution(n);
            case "uniform":
                return generateUniformDistribution(n);
            default:
                return new double[0];
        }
    }

    private double[] generateLinearlyDescendingDistribution(int n, double v) {
        return new double[0];
    }

    private double[] generateUniformDistribution(int n) {
        return new double[0];
    }

    // TODO probability distribution over sizes for sampling according to paper
    private double[] generatePaperDistribution(int n) {
        double[] dist = new double[n + 1];
        for (int i = 0; i <= n; i++) {
            dist[i] = 0;
        }

        if (n % 2 == 0) {
            double nlogn = n * Math.log(n);
            double H = calculateDistribution(1, n);  //计算从 1 到 n/2 - 1 的倒数之和
            double nominator = nlogn - 1;
            double denominator = 2 * nlogn * (H - 1);
            double frac = nominator / denominator;
        }
//        for (int s = 2; s < n / 2; s++) {
//            dist[s] = frac / s;
//            dist[n - s] = frac / s;
//        }
//        dist[n / 2] = 1 / nlogn;
        return dist;
    }

    //TODO 计算从 a 到 b 的倒数之和
    private double calculateDistribution(double a, double b) {
        return 0;
    }

    //全局参数的初始化
    private void gameInit(boolean gene_weight, String modelName) {
        //TODO [Airport GAME]
        switch (modelName) {
            case "airport":
                //Case1: 使用生成的weight
                if (gene_weight) {
                    this.num_features = Info.num_of_features;  //1)num_features
                    //2)初始化given_weights
                    String path = Info.ROOT + "airport_" + this.num_features + ".npy";
                    try {
                        DataInputStream dataInputStream = null;
                        dataInputStream = new DataInputStream(Files.newInputStream(Paths.get(path)));
                        NumpyReader reader = new NumpyReader();
                        this.given_weights = reader.readIntArray(dataInputStream);  //读入.npy文件中存储的数组
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                    //3）初始化 exact[]
                    String benchmark_path = Info.benchmark_path;
                    FileOpera opera = new FileOpera();
                    this.exact = new double[num_features];
                    this.exact = opera.file_read(benchmark_path, num_features);  //从文件读入benchmark
                    //4）初始化 num_samples
                    this.num_samples = Info.total_samples_num;
                }
                //Case2: 使用默认的weight
                else {
                    this.num_features = Info.num_of_features_airport;  //1)num_features
                    this.given_weights = Info.given_weights_airport;  //2）given_weights
                    this.exact = Info.airport_exact;  //3）exact
                    this.num_samples = Info.total_samples_num;  // 4）num_samples
                }
                break;

            //TODO [Voting GAME]
            case "voting":
                //Case1: 使用生成的weight
                if (gene_weight) {
                    // 1) num_features
                    this.num_features = Info.num_of_features;
                    //2）初始化given_weights
                    String path = Info.ROOT + "voting_" + this.num_features + ".npy";
                    try {
                        DataInputStream dataInputStream = null;
                        dataInputStream = new DataInputStream(Files.newInputStream(Paths.get(path)));
                        NumpyReader reader = new NumpyReader();
                        this.given_weights = reader.readIntArray(dataInputStream);  //读入.npy文件中存储的数组
                        this.halfSum = Arrays.stream(this.given_weights).sum() / 2;  //对given_weights中的数据求和
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                    //3）初始化 exact[]
                    String benchmark_path = Info.benchmark_path;
                    FileOpera opera = new FileOpera();
                    this.exact = opera.file_read(benchmark_path, num_features);  //从文件读入benchmark
                    //4）num_samples
                    this.num_samples = Info.total_samples_num;
                }
                //Case2: 使用默认的weight
                else {
                    this.num_features = Info.num_of_features_voting;
                    this.given_weights = Info.given_weights_voting;
                    this.halfSum = Arrays.stream(this.given_weights).sum() / 2;  //对given_weights中的数据求和
                    this.exact = Info.voting_exact;
                    this.num_samples = Info.total_samples_num;
                }
                break;

            //TODO [Shoes GAME]
            case "shoes":
                this.num_features = Info.num_of_features_shoes;
                this.exact = new double[Info.num_of_features_shoes];
                Arrays.fill(this.exact, Info.shoes_exact);
                this.num_samples = Info.total_samples_num;
                break;
            //TODO 没有固定模型，就是train的model
            case "model": {
                this.num_features = Info.num_of_features;  //1）num_features

                this.given_weights = Info.model_instance_ave; //2）given_weights = model_instance_ave

                String benchmark_path = Info.benchmark_path;
                FileOpera opera = new FileOpera();
                this.exact = new double[this.num_features];  //3）exact

                this.exact = opera.file_read(benchmark_path, this.num_features);  //从文件读入benchmark

                this.num_samples = Info.total_samples_num;
                break;
            }
            case "svm_model": {
                this.num_features = Info.num_of_features;
                this.given_weights = Info.model_instance_ave_2;
                String benchmark_path = Info.benchmark_path;
                FileOpera opera = new FileOpera();
                this.exact = new double[this.num_features];
                this.exact = opera.file_read(benchmark_path, this.num_features);  //从文件读入benchmark

                this.num_samples = Info.total_samples_num;
                break;
            }
            case "iot": {
                this.num_features = Info.num_of_features;
                this.given_weights = Info.instance_iot_org;
                String benchmark_path = Info.benchmark_path;
                FileOpera opera = new FileOpera();
                this.exact = new double[this.num_features];
                this.exact = opera.file_read(benchmark_path, this.num_features);  //从文件读入benchmark

                this.num_samples = Info.total_samples_num;
                break;
            }
        }
    }

}
