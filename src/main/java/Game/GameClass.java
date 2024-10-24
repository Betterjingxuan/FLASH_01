package Game;

import Global.FileOpera;
import config.Info;
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
    public double halfSum;
    private OkHttpClient client;
    public ModelGame model;  //ModelGame：utility function

    public double mu_n = 0.5;  //the proportion of the number of features in feature-wise evaluation
    public double mu_f = 0.1;   //the proportion of the number of evaluations in feature-wise evaluation

    public long[] seedSet;
    public boolean isRealData;   //true: real dataset (bank, health); // false：game theory dataset (airport, voting)
    public int start_level;
    public int end_level;


    //TODO [版本2] 全局参数的初始化
    public void gameInit(boolean gene_weight, String modelName) {
        String path;
        String benchmark_path;
        FileOpera opera = new FileOpera();
        this.model = new ModelGame(Info.model_name);  //game definition
        this.client = new OkHttpClient();

        //TODO [Airport GAME]
        switch (modelName) {
            case "airport":
                //Case1: use the generated weight (scale)
                if (gene_weight) {
                    this.num_features = Info.num_of_features;
                    path = Info.ROOT + "ScaleDateset/" + modelName + "_" + this.num_features + ".npy";
                    try {
                        DataInputStream dataInputStream = null;
                        dataInputStream = new DataInputStream(Files.newInputStream(Paths.get(path)));
                        NumpyReader reader = new NumpyReader();
                        this.given_weights = reader.readIntArray(dataInputStream);
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                    // initialize the exact[]
                    benchmark_path = Info.ROOT + "benchmark/" + "benchmark_airport_" + this.num_features + "_" +
                            this.num_features + "0000.tex";
                    this.exact = new double[num_features];
                    this.exact = opera.file_read(benchmark_path, num_features);
                    this.mu_f = 0.0;
                    this.isRealData = false;
                    this.seedSet = seedSet(Info.seed);
                }
                //Case2: Use the default weight (evaluation)
                else {
                    this.num_features = Info.num_of_features_airport;
                    this.given_weights = Info.given_weights_airport;
                    this.exact = Info.airport_exact;
                    this.mu_f = 0.1;
                    this.mu_n = 0.5;
                    this.isRealData = false;
                    this.seedSet = seedSet(Info.seed);
                }
                break;

            //TODO [Voting GAME]
            case "voting":
                //Case1: use the generated weight (scale)
                if (gene_weight) {
                    this.num_features = Info.num_of_features;
                    path = Info.ROOT + "ScaleDateset/" + "voting_" + this.num_features + ".npy";
                    try {
                        DataInputStream dataInputStream = null;
                        dataInputStream = new DataInputStream(Files.newInputStream(Paths.get(path)));
                        NumpyReader reader = new NumpyReader();
                        this.given_weights = reader.readIntArray(dataInputStream);
                        this.halfSum = Arrays.stream(this.given_weights).sum() / 2;
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                    this.seedSet = seedSet(Info.seed);
                    // initialize the exact[]
                    benchmark_path = Info.ROOT + "benchmark/" + "benchmark_voting_" + this.num_features + "_" +
                            this.num_features + "0000.tex";
                    this.exact = opera.file_read(benchmark_path, num_features);
                    this.halfSum = Arrays.stream(this.given_weights).sum() / 2;
                    this.mu_f = 0.0;
                    this.isRealData = false;
                }
                //Case2: Use the default weight (evaluation)
                else {
                    this.seedSet = seedSet(2024L);
                    this.num_features = Info.num_of_features_voting;
                    this.given_weights = Info.given_weights_voting;
                    this.halfSum = Arrays.stream(this.given_weights).sum() / 2;
                    this.exact = Info.voting_exact;
                    this.isRealData = false;
                    this.mu_f = 0.1;
                    this.mu_n = 0.5;
                }
                break;
            case "model":
                this.num_features = Info.num_of_features;
                this.given_weights = Info.model_instance_ave;
                benchmark_path = Info.benchmark_path;
                this.exact = new double[this.num_features];
                this.exact = opera.file_read(benchmark_path, this.num_features);
                break;
            case "svm_model":
                this.num_features = Info.num_of_features;
                this.given_weights = Info.model_instance_ave_2;
                benchmark_path = Info.benchmark_path;
                this.exact = new double[this.num_features];
                this.exact = opera.file_read(benchmark_path, this.num_features);
                break;
            case "bank":
                this.num_features = Info.num_of_features_bank;
                benchmark_path = Info.benchmark_path;
                this.exact = new double[this.num_features];
                this.exact = opera.file_read(benchmark_path, this.num_features);
                this.seedSet = seedSet(Info.seed);
                this.mu_f = 0.1;
                this.mu_n = 0.5;
                this.isRealData = true;
                break;
            case "health":
                this.num_features = Info.num_of_features_health;
                this.given_weights = Info.instance_health;
                benchmark_path = Info.benchmark_path;
                this.exact = new double[this.num_features];
                this.exact = opera.file_read(benchmark_path, this.num_features);
                this.seedSet = seedSet(Info.seed);
                this.mu_f = 0.1;
                this.mu_n = 0.5;
                this.isRealData = true;
                break;
        }
        this.start_level = 2;
        this.end_level = this.num_features - 1;
    }

    //TODO Utility function
    public double gameValue(String modelName, ArrayList<Integer> subset){
        double value = 0;
        switch (modelName) {
            case "airport":
                value = this.model.value_airport(subset, this.given_weights);
                break;
            case "voting":
                value = this.model.value_voting(subset, this.given_weights, this.halfSum);
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
            localSet[i] = random.nextLong();
        }
        return localSet;
    }


}
