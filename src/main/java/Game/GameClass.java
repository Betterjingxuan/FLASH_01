package Game;

import Global.FileOpera;
import Global.Info;
import Global.NumpyReader;
import okhttp3.OkHttpClient;

import java.io.DataInputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class GameClass {
    public int num_features;  //the number of features
    public double[] exact;   // the exact shapley value
    public double[] given_weights;
    public double halfSum;  //for Voting game
    private OkHttpClient client;
    public ModelGame model;  //ModelGame：为了不同的game 调用不同value_function

    public double key_features_weight = 0.5;
    public double check_weight = 0.1;   //最后阶段方差计算的比例
    public double number_weight = 0.2;  //在样本分配时，数量特征所占的比例
    public double variance_weight; //在样本分配时，方差特征所占的比例
    public long[] seedSet;
    public boolean isRealData;   //true: real dataset; // false：game theorem dataset
    public int start_level;
    public int end_level;


    //TODO [版本2] 全局参数的初始化
    public void gameInit(boolean gene_weight, String modelName) {
        String path;
        String benchmark_path;
        FileOpera opera = new FileOpera();
        this.model = new ModelGame(Info.model_name);  //ModelGame：为了不同的game 调用不同value_function
        this.client = new OkHttpClient();

        //TODO [Airport GAME]
        switch (modelName) {
            case "airport":
                //Case1: 使用生成的weight
                if (gene_weight) {
                    this.num_features = Info.num_of_features;  //1)num_features
                    //2)初始化given_weights
                    path = Info.ROOT + "ScaleDateset/" + modelName + "_" + this.num_features + ".npy";
                    try {
                        DataInputStream dataInputStream = null;
                        dataInputStream = new DataInputStream(Files.newInputStream(Paths.get(path)));
                        NumpyReader reader = new NumpyReader();
                        this.given_weights = reader.readIntArray(dataInputStream);  //读入.npy文件中存储的数组
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                    //3）初始化 exact[]
                    benchmark_path = Info.ROOT + "benchmark/" + "benchmark_airport_" + this.num_features + "_" + this.num_features + "0000.tex";
                    this.exact = new double[num_features];
                    this.exact = opera.file_read(benchmark_path, num_features);  //从文件读入benchmark
                    this.check_weight = 0.0;
                    this.isRealData = false;
                    this.seedSet = seedSet(Info.seed);
                }
                //Case2: 使用默认的weight
                else {
                    this.num_features = Info.num_of_features_airport;  //1)num_features
                    this.given_weights = Info.given_weights_airport;  //2）given_weights
                    this.exact = Info.airport_exact;  //3）exact
                    this.check_weight = 0.1;
                    this.key_features_weight = 0.5;
                    this.isRealData = false;
                    this.seedSet = seedSet(Info.seed);
                }
                break;

            //TODO [Voting GAME]
            case "voting":
                //Case1: 使用生成的weight
                if (gene_weight) {
                    // 1) num_features
                    this.num_features = Info.num_of_features;
                    //2）初始化given_weights
                    path = Info.ROOT + "ScaleDateset/" + "voting_" + this.num_features + ".npy";
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
                    this.seedSet = seedSet(Info.seed);
                    benchmark_path = Info.benchmark_path;
                    this.exact = opera.file_read(benchmark_path, num_features);  //从文件读入benchmark
                    this.halfSum = Arrays.stream(this.given_weights).sum() / 2;  //对given_weights中的数据求和;
                    this.check_weight = 0.0;
                    this.isRealData = false;
                }
                //Case2: 使用默认的weight
                else {
                    this.seedSet(2024L);
                    this.num_features = Info.num_of_features_voting;
                    this.given_weights = Info.given_weights_voting;
                    this.halfSum = Arrays.stream(this.given_weights).sum() / 2;  //对given_weights中的数据求和
                    this.exact = Info.voting_exact;
                    this.isRealData = false;
                    this.check_weight = 0.1;  //0.35  //0.2  //0.18  //0.15
                    this.key_features_weight = 0.5;    // 0.8  //0.85
                    /* voting-2024L-100n- 0.1 - 0.5*/
                }
                break;

            //TODO [Shoes GAME]
            case "shoes":
                this.num_features = Info.num_of_features_shoes;
                this.exact = new double[Info.num_of_features_shoes];
                Arrays.fill(this.exact, Info.shoes_exact);
                break;
            //TODO 没有固定模型，就是train的model
            case "model":
                this.num_features = Info.num_of_features;  //1）num_features
                this.given_weights = Info.model_instance_ave; //2）given_weights = model_instance_ave
                benchmark_path = Info.benchmark_path;
                this.exact = new double[this.num_features];  //3）exact
                this.exact = opera.file_read(benchmark_path, this.num_features);  //从文件读入benchmark
                break;
            case "svm_model":
                this.num_features = Info.num_of_features;
                this.given_weights = Info.model_instance_ave_2;
                benchmark_path = Info.benchmark_path;
                this.exact = new double[this.num_features];
                this.exact = opera.file_read(benchmark_path, this.num_features);  //从文件读入benchmark
                break;
            case "iot":
                this.num_features = Info.num_of_features;
                this.given_weights = Info.instance_iot_org;
                benchmark_path = Info.benchmark_path;
                this.exact = new double[this.num_features];
                this.exact = opera.file_read(benchmark_path, this.num_features);  //从文件读入benchmark
                break;
            case "bank":
                this.num_features = Info.num_of_features_bank;
                benchmark_path = Info.benchmark_path;
                this.exact = new double[this.num_features];
                this.exact = opera.file_read(benchmark_path, this.num_features);  //从文件读入benchmark
                this.seedSet = seedSet(Info.seed);
                this.check_weight = 0.1;
                this.key_features_weight = 0.5;
                this.isRealData = true;
                break;
            case "health":
                this.num_features = Info.num_of_features_health;
                this.given_weights = Info.instance_health;
                benchmark_path = Info.benchmark_path;
                this.exact = new double[this.num_features];
                this.exact = opera.file_read(benchmark_path, this.num_features);  //从文件读入benchmark
                this.seedSet = seedSet(Info.seed);
                this.check_weight = 0.1;
                this.key_features_weight = 0.5;
                this.isRealData = true;
                break;
        }
        this.variance_weight = Math.max(0.0, 1-this.number_weight-this.check_weight); //在样本分配时，数量特征所占的比例
        this.start_level = 2;
        this.end_level = this.num_features - 1;
    }

    //TODO 求解函数值
    public double gameValue(String modelName, ArrayList<Integer> subset){
        double value = 0;
        switch (modelName) {
            case "airport":
                value = this.model.value_airport(subset, this.given_weights);
                break;
            case "voting":
                value = this.model.value_voting(subset, this.given_weights, this.halfSum);
                break;
            case "shoes":
                value = this.model.value_shoes(subset);
                break;
            case "model":
                value = this.model.value_modelPrediction(given_weights, subset);
            case "svm_model":
                value = this.model.value_darwin(given_weights, subset);  //复制两份
                break;
            case "iot":
                value = this.model.IOT_value(given_weights, subset);  //复制两份
                break;
            case "health":
                value = this.model.Health_value(subset, client);
                break;
            case "bank":
                value = this.model.BankMarketing_value(subset, client);
                break;
        }
        return value;
    }

    public long[] seedSet(long seed){
        long[] localSet = new long[Info.timesRepeat];
        Random random = new Random(seed);
        for(int i=0; i<Info.timesRepeat; i++){
            localSet[i] = random.nextLong();    // random.nextLong()可能是正也是负
        }
        return localSet;
    }

//    public long[] seedSet(long seed, int start){
//        long[] localSet = new long[Info.timesRepeat];
//        long[] temp = new long[Info.timesRepeat*start*start];
//        Random random = new Random(seed);
//        for(int i=0; i<temp.length; i++){
//            temp[i] = random.nextLong();    // random.nextLong()可能是正也是负
//        }
//        System.arraycopy(temp, start * Info.timesRepeat, localSet, 0, Info.timesRepeat);
//        return localSet;
//    }

}
