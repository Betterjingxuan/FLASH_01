package Compared_algorithm;

import Game.Health;
import Game.IOT_83F_sm1_iter2;
import Game.ModelGame;
import Global.Comparer;
import Global.FileOpera;
import Global.Info;
import Global.NumpyReader;
import okhttp3.OkHttpClient;
import structure.ShapMatrixEntry;

import java.io.DataInputStream;
import java.math.BigInteger;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;

public class CCN_algorithm {

    int num_features;  // the number of features
    double[] given_weights;
    double[] exact;  // the exact shapley value
    double halfSum;  //for Voting game
    public void CCN_Shap(int num_sample, boolean gene_weight, String model){

        initialize(gene_weight, model);
        int initial_m = Math.max(2, num_sample / (this.num_features * this.num_features * 2));
        long ave_runtime = 0;
        double ave_error_max = 0; //计算最大误差
        double ave_error_ave = 0;
        for(int i=0; i< Info.timesRepeat; i++) {
            //3）计算shapley value
            long time_1 = System.currentTimeMillis();
            ShapMatrixEntry[] shap_matrix = computeShapBySampling_2(num_sample, initial_m, model, halfSum);
            long time_2 = System.currentTimeMillis();

            //4）计算误差
            Comparer comparator = new Comparer();
            double error_ave = comparator.computeAverageError(shap_matrix, this.exact, this.num_features);  //计算平均误差
            double error_max = comparator.computeMaxError(shap_matrix, this.exact, this.num_features); //计算最大误差
            ave_runtime += time_2 - time_1;
            ave_error_max += error_max;
            ave_error_ave += error_ave;
        }

        System.out.println(model + " Game:  " + "error_ave: " + ave_error_ave/Info.timesRepeat + " \t"  +  "error_max: " + ave_error_max/Info.timesRepeat );
        // 5）输出时间
        System.out.println("CCN time : " + (ave_runtime * 0.001)/ Info.timesRepeat );  //+ "S"
//        HashMap<Integer, double[]> s = new HashMap<>();
//        for(int i=0; i<num_features; i++){
//            s.put(i, new double[2]);
//        }
    }

    //TODO 根据game初始化num_features & exact
    private void initialize(boolean gene_weight, String model) {
        //TODO [Airport GAME]
        if(model.equals("airport")){
            //Case1: 使用生成的weight
            if(gene_weight){
                this.num_features = Info.num_of_features;
                //1)初始化given_weights
                String path = Info.ROOT + "airport_" + this.num_features + ".npy";
                try {
                    DataInputStream dataInputStream = null;
                    dataInputStream = new DataInputStream(Files.newInputStream(Paths.get(path)));
                    NumpyReader reader = new NumpyReader();
                    this.given_weights = reader.readIntArray(dataInputStream);  //读入.npy文件中存储的数组
                }
                catch (Exception e) {
                    throw new RuntimeException(e);
                }
                //2）初始化 exact[]
                String benchmark_path = Info.benchmark_path;
                FileOpera opera = new FileOpera();
                this.exact = new double[num_features];
                this.exact = opera.file_read(benchmark_path, num_features);  //从文件读入benchmark
            }
            //Case2: 使用默认的weight
            else{
                this.num_features = Info.num_of_features_airport;
                this.given_weights = Info.given_weights_airport;
                this.exact = Info.airport_exact;
            }
        }

        //TODO [Voting GAME]
        else if(model.equals("voting")){
            //Case1: 使用生成的weight
            if(gene_weight){
                this.num_features = Info.num_of_features;
                //1)初始化given_weights
                String path = Info.ROOT + "voting_" + this.num_features + ".npy";
                try {
                    DataInputStream dataInputStream = null;
                    dataInputStream = new DataInputStream(Files.newInputStream(Paths.get(path)));
                    NumpyReader reader = new NumpyReader();
                    this.given_weights = reader.readIntArray(dataInputStream);  //读入.npy文件中存储的数组
                    this.halfSum = Arrays.stream(this.given_weights).sum() / 2;  //对given_weights中的数据求和
                }
                catch (Exception e) {
                    throw new RuntimeException(e);
                }
                //2）初始化 exact[]
                String benchmark_path = Info.benchmark_path;
                FileOpera opera = new FileOpera();
                this.exact = opera.file_read(benchmark_path, num_features);  //从文件读入benchmark
            }
            //2）初始化 exact[]
            else{
                this.num_features = Info.num_of_features_voting;
                this.given_weights = Info.given_weights_voting;
                this.halfSum = Arrays.stream(this.given_weights).sum() / 2;  //对given_weights中的数据求和
                this.exact = Info.voting_exact;
            }
        }

        //TODO [Shoes GAME]
        else if(model.equals("shoes")){
            this.num_features = Info.num_of_features_shoes;
            this.exact = new double[Info.num_of_features_shoes];
            Arrays.fill(this.exact, Info.shoes_exact);
        }

        //TODO 没有固定模型，就是train的model
        else if(model.equals("model")){
            this.num_features = Info.num_of_features;
            this.given_weights = Info.model_instance_ave;
            String benchmark_path = Info.benchmark_path;
            FileOpera opera = new FileOpera();
            this.exact = new double[this.num_features];
            this.exact = opera.file_read(benchmark_path, this.num_features);  //从文件读入benchmark
        }
        else if(model.equals("svm_model")){
            this.num_features = Info.num_of_features;
            this.given_weights = Info.model_instance_ave_2;
            String benchmark_path = Info.benchmark_path;
            FileOpera opera = new FileOpera();
            this.exact = new double[this.num_features];
            this.exact = opera.file_read(benchmark_path, this.num_features);  //从文件读入benchmark
        }
        else if(model.equals("iot")){
            this.num_features = Info.num_of_features;
            this.given_weights = Info.instance_iot_org;
            String benchmark_path = Info.benchmark_path;
            FileOpera opera = new FileOpera();
            this.exact = new double[this.num_features];
            this.exact = opera.file_read(benchmark_path, this.num_features);  //从文件读入benchmark
        }
        else if(model.equals("health")){
            this.num_features = Info.num_of_features;
            this.given_weights = Info.instance_health;
            String benchmark_path = Info.benchmark_path;
            FileOpera opera = new FileOpera();
            this.exact = new double[this.num_features];
            this.exact = opera.file_read(benchmark_path, this.num_features);  //从文件读入benchmark
        }

    }

    //TODO 通过CCN algorithm 计算 shapley value
    private ShapMatrixEntry[] computeShapBySampling(int num_sample, int initial_m, String model, double halfSum) {
        ModelGame game = new ModelGame(model);

//        double[] shap_matrix = new double[num_features];  //大数组
        ShapMatrixEntry[][] utility = new ShapMatrixEntry[this.num_features+1][]; // 需要换成带计数器的大数组
        for(int i=0; i<utility.length; i++){   //外层循环：features对应每个长度的collations (n个feature，n+1层)
            utility[i] = new ShapMatrixEntry[this.num_features];
            for(int j=0; j<this.num_features; j++) {  //内层循坏：每个features
                utility[i][j] = new ShapMatrixEntry();
            }
        }
        // 创建随机数生成器
        Random localState = new Random(); // 使用系统时间作为种子
        long[] coef = comb(this.num_features-1,1);

        int count = 0;
        while(true){
            int temp_count = count;
            // [外层] 遍历每个feature
            for (int i = 0; i < this.num_features; i++) {
                //构造一个不包含i的permutation ：idxs
                ArrayList<Integer> idxs = new ArrayList<>();
                for (int ele = 0; ele < i; ele++) {  // 添加 range(i)
                    idxs.add(ele);
                }
                for (int ele = i + 1; ele < this.num_features; ele++) {  // 添加 range(i)
                    idxs.add(ele);
                }
                //[内层] 遍历每个length
                for (int len = 0; len < this.num_features; len++) {
                    if (utility[i][len].count >= initial_m || utility[i][len].count >= coef[len]) {
                        continue;  //跳过当前层，进入下一个len
                    }
                    Collections.shuffle(idxs, localState);  //打乱idxs顺序
                    count ++;

                    //2）利用序列p, 求每个feature 的 marginal contribution
                    ArrayList<Integer> subset_1 = new ArrayList<>();   //*subset_1 第一个特征子集
                    double value_1 = 0;
                    ArrayList<Integer> subset_2 = new ArrayList<>();   //*subset_2 第二个特征子集
                    double value_2 = 0;
                    // 3）构造两个子集subset_1 & subset_2
                    for(int ind = 0; ind <idxs.size(); ind ++) { // ele 对应一个特征
                        subset_1.add(idxs.get(ind));
                    }
                    subset_1.add(i);
                    for(int ind = len; ind <idxs.size(); ind ++) { // ele 对应一个特征
                        subset_2.add(idxs.get(ind));
                    }
                    //4)分别求函数值
                    switch (model) {
                        case "airport":
                            value_1 = game.value_airport(subset_1, this.given_weights);
                            value_2 = game.value_airport(subset_2, this.given_weights);
                            break;
                        case "voting":
                            value_1 = game.value_voting(subset_1, this.given_weights, halfSum);
                            value_2 = game.value_voting(subset_2, this.given_weights, halfSum);
                            break;
                        case "shoes":
                            value_1 = game.value_shoes(subset_1);
                            value_2 = game.value_shoes(subset_2);
                            break;
                        case "model":
                            value_1 = game.value_modelPrediction(given_weights, subset_1);
                            value_2 = game.value_modelPrediction(given_weights, subset_2);
                            break;
                        case "svm_model":
                            ModelGame model_svm = new ModelGame("svm_model");
                            if (subset_1.size() == 0) {
                                value_1 = model_svm.value_darwin(given_weights, new ArrayList<Integer>());
                                value_2 = model_svm.value_darwin(given_weights, subset_2);
                            } else if (subset_2.size() == 0) {
                                value_2 = model_svm.value_darwin(given_weights, new ArrayList<Integer>());
                                value_1 = model_svm.value_darwin(given_weights, subset_1);
                            } else {
                                value_1 = model_svm.value_darwin(given_weights, subset_1);
                                value_2 = model_svm.value_darwin(given_weights, subset_2);
                            }
                            break;
                        case "iot":
                            IOT_83F_sm1_iter2 iot_model = new IOT_83F_sm1_iter2();
                            value_1 = iot_model.IOT_value(given_weights, subset_1);
                            value_2 = iot_model.IOT_value(given_weights, subset_2);
                            break;
                    }

                    utility[i][len].sum += value_1 - value_2;
                    for(int l = 0; l < this.num_features - 1; l++){
                        if(l<len){
                            utility[idxs.get(l)][len].sum += value_1 - value_2;
                            utility[idxs.get(l)][len].count ++;
                        }
                        else{
                            utility[idxs.get(l)][this.num_features-len-2].sum += value_2 - value_1;
                            utility[idxs.get(l)][this.num_features-len-2].count ++;
                        }
                    }
                }
            }
            if (count == temp_count)
                break;

            // 计算方差
            for(int i=0; i < this.num_features; i++){
                for(int j=0; j < this.num_features; j++){

                }
            }

            //计算分配样本
            double var_sum = 0;
            double[] sigma_j = new double[this.num_features];
            double[] sigma_n_j = new double[this.num_features];
        }

//        //进行若干次采样（num_sample：采样的次数）
//        for(int r=0; r<num_sample; r++){
//            //1）生成一个打乱的序列 & 一个随机数
//            int[] p = permutation(this.num_features);
//            Random rand = new Random();
//            int random_i = rand.nextInt(this.num_features);  //随机生成一个数,表示feature subset的长度
//
//            //2）利用序列p, 求每个feature 的 marginal contribution
//            ArrayList<Integer> subset_1 = new ArrayList<>();   //*subset_1 第一个特征子集
//            double value_1 = 0;
//            ArrayList<Integer> subset_2 = new ArrayList<>();   //*subset_2 第二个特征子集
//            double value_2 = 0;
//
//            /* S = [ABC];
//            *  P = N-S = [DEF] */
//            for(int ind = 0; ind <random_i; ind ++) { // ele 对应一个特征
//                subset_1.add(p[ind]);
//            }
//            for(int ind = random_i; ind <p.length; ind ++) { // ele 对应一个特征
//                subset_2.add(p[ind]);
//            }
//
//            //*将两个list重新排列
////            Collections.sort(subset_1);
////            Collections.sort(subset_2);
//
//            //4)分别求函数值
//            switch (model) {
//                case "airport":
//                    value_1 = value_airport_2(subset_1, this.given_weights);
//                    value_2 = value_airport_2(subset_2, this.given_weights);
//                    break;
//                case "voting":
//                    value_1 = value_voting(subset_1, this.given_weights, halfSum);
//                    value_2 = value_voting(subset_2, this.given_weights, halfSum);
//                    break;
//                case "shoes":
//                    value_1 = value_shoes(subset_1);
//                    value_2 = value_shoes(subset_2);
//                    break;
//                case "model":
//                    value_1 = value_modelPrediction(given_weights, subset_1);
//                    value_2 = value_modelPrediction(given_weights, subset_2);
//                    break;
//                case "svm_model":
//                    ModelGame model_svm = new ModelGame("svm_model");
//                    if (subset_1.size() == 0) {
//                        value_1 = model_svm.value_darwin(given_weights, new ArrayList<>());
//                        value_2 = model_svm.value_darwin(given_weights, subset_2);
//                    } else if (subset_2.size() == 0) {
//                        value_2 = model_svm.value_darwin(given_weights, new ArrayList<>());
//                        value_1 = model_svm.value_darwin(given_weights, subset_1);
//                    } else {
//                        value_1 = model_svm.value_darwin(given_weights, subset_1);
//                        value_2 = model_svm.value_darwin(given_weights, subset_2);
//                    }
//                    break;
//                case "iot":
//                    IOT_83F_sm1_iter2 iot_model = new IOT_83F_sm1_iter2();
//                    value_1 = iot_model.IOT_value(given_weights, subset_1);
//                    value_2 = iot_model.IOT_value(given_weights, subset_2);
//                    break;
//            }
//
//            //5）求complementary contribution
//            double comp_contrib = value_1 - value_2;
//
//            //6)存入对应的集合中
//            for(Integer ele : subset_1){
//                shap_matrix[random_i][ele].sum += comp_contrib;
//                shap_matrix[random_i][ele].count ++;
//            }
//            for(Integer ele : subset_2){
//                shap_matrix[this.num_features - random_i][ele].sum -= comp_contrib;
////                shap_matrix[num_features - random_i][ele].sum += value_2 - value_1;
//                shap_matrix[this.num_features - random_i][ele].count ++;
//            }
//
//        }
//        // 5) 求shapley value的平均值，对于每个特征&每个长度求均值
        ShapMatrixEntry[] resultShap = new ShapMatrixEntry[this.num_features];  //这是最后返回的大数组
        for(int i=0; i<this.num_features; i++) {  //内层循环：features对应每个长度的collations
            resultShap[i] = new ShapMatrixEntry();
        }
//
//        for(ShapMatrixEntry[] featureSet : shap_matrix){   //外层循环：features对应每个长度的collations
//            for(int fea=0; fea<featureSet.length ; fea++){  //内层循坏：每个features
//                ShapMatrixEntry entry = featureSet[fea];   //相同长度的一层中，每个特征对应的sv.
//                if(entry.count != 0){
//                    entry.sum = entry.sum / entry.count;
//                    resultShap[fea].sum += entry.sum;  //对于每个长度求和
//                    resultShap[fea].count ++;
//                }
//            }
//        }

        //6）求一个整体的均值
        for(ShapMatrixEntry entry : resultShap){
            if(entry.count != 0){
                entry.sum = entry.sum / entry.count;
            }
            else{
                entry.sum =0;
            }
        }
        return resultShap;
    }

    //TODO 通过CCN algorithm 计算 shapley value
    private ShapMatrixEntry[] computeShapBySampling_2(int num_sample, int initial_m, String model, double halfSum) {
        ModelGame game = new ModelGame(model);
        OkHttpClient client=new OkHttpClient();
        long[] coef = combination(this.num_features-1,1);

//        double[] shap_matrix = new double[num_features];  //大数组
        ShapMatrixEntry[][] utility = new ShapMatrixEntry[this.num_features][]; // 需要换成带计数器的大数组
        for(int i=0; i<utility.length; i++){   //外层循环：features对应每个长度的collations (n个feature，n+1层)
            utility[i] = new ShapMatrixEntry[this.num_features];
            for(int j=0; j<this.num_features; j++) {  //内层循坏：每个features
                utility[i][j] = new ShapMatrixEntry();
                utility[i][j].record = new ArrayList<>();  //记录每层的方差
            }
        }

        int count = 0;
//        while(true) {
        while(true) {
            int temp_count = count;
            for (int i = 0; i < this.num_features; i++) {  // [外层] 遍历每个feature
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
                    else{
//                        System.out.println(utility[i][len].count + "     " + initial_m + "     " + coef[len]);
                    }
                    Collections.shuffle(idxs);  //打乱idxs顺序
                    count++;

                    //2）利用序列p, 求每个feature 的 marginal contribution
                    ArrayList<Integer> subset_1 = new ArrayList<>();   //*subset_1 第一个特征子集
                    double value_1 = 0;
                    ArrayList<Integer> subset_2 = new ArrayList<>();   //*subset_2 第二个特征子集
                    double value_2 = 0;
                    // 3）构造两个子集subset_1 & subset_2
                    for (int ind = 0; ind < len; ind++) { // ele 对应一个特征
                        subset_1.add(idxs.get(ind));
                    }
                    subset_1.add(i);
                    for (int ind = len; ind < idxs.size(); ind++) { // ele 对应一个特征
                        subset_2.add(idxs.get(ind));
                    }
                    //4)分别求函数值
                    switch (model) {
                        case "airport":
                            value_1 = game.value_airport(subset_1, this.given_weights);
                            value_2 = game.value_airport(subset_2, this.given_weights);
                            break;
                        case "voting":
                            value_1 = game.value_voting(subset_1, this.given_weights, halfSum);
                            value_2 = game.value_voting(subset_2, this.given_weights, halfSum);
                            break;
                        case "shoes":
                            value_1 = game.value_shoes(subset_1);
                            value_2 = game.value_shoes(subset_2);
                            break;
                        case "model":
                            value_1 = game.value_modelPrediction(given_weights, subset_1);
                            value_2 = game.value_modelPrediction(given_weights, subset_2);
                            break;
                        case "svm_model":
                            ModelGame model_svm = new ModelGame("svm_model");
                            if (subset_1.size() == 0) {
                                value_1 = model_svm.value_darwin(given_weights, new ArrayList<Integer>());
                                value_2 = model_svm.value_darwin(given_weights, subset_2);
                            } else if (subset_2.size() == 0) {
                                value_2 = model_svm.value_darwin(given_weights, new ArrayList<Integer>());
                                value_1 = model_svm.value_darwin(given_weights, subset_1);
                            } else {
                                value_1 = model_svm.value_darwin(given_weights, subset_1);
                                value_2 = model_svm.value_darwin(given_weights, subset_2);
                            }
                            break;
                        case "iot":
                            IOT_83F_sm1_iter2 iot_model = new IOT_83F_sm1_iter2();
                            value_1 = iot_model.IOT_value(given_weights, subset_1);
                            value_2 = iot_model.IOT_value(given_weights, subset_2);
                            break;
                        case "health":
                            Health healthGame = new Health();
                            value_1 = healthGame.Health_value(given_weights, subset_1, client);
                            value_2 = healthGame.Health_value(given_weights, subset_1, client);
                            break;
                    }
                    utility[i][len].sum += value_1 - value_2;
                    for (int l = 0; l < this.num_features - 1; l++) {
                        if (l < len) {
                            utility[idxs.get(l)][len].sum += value_1 - value_2;
                            utility[idxs.get(l)][len].count++;
                            utility[idxs.get(l)][len].record.add(value_1 - value_2);
                        } else {
                            utility[idxs.get(l)][this.num_features - len - 2].sum += value_2 - value_1;
                            utility[idxs.get(l)][this.num_features - len - 2].count++;
                            utility[idxs.get(l)][this.num_features - len - 2].record.add(value_2 - value_1);
                        }
                    }
//                    for(int ind=0; ind < subset_1.size(); ind ++){
//                        utility[idxs.get(ind)][len].sum += value_1 - value_2;
//                        utility[idxs.get(ind)][this.num_features-len-1].sum += value_1 - value_2;
//                    }
                }
            }
//            System.out.println("count " + count + "   temp_count: " + temp_count);
            if (count == temp_count)
                break;
        }

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
        for(int j=(int) Math.ceil(this.num_features / 2.0) - 1; j < this.num_features; j++){
            for (int i = 0; i < this.num_features; i++) {
                sigma_k[j] += variance[i][j] / (j + 1);
                if (this.num_features - j - 2 < 0) {
                    sigma_n_k[j] += 0;
                } else {
                    sigma_n_k[j] += variance[i][this.num_features - j - 2] / (this.num_features - j - 1);
                }
                var_sum += Math.sqrt(sigma_k[j] + sigma_n_k[j]);
            }
        }

        num_sample = num_sample - count;
        double[] arr_m = new double[this.num_features];
        // Initialize m to zeros
        Arrays.fill(arr_m, 0.0);
        for (int j = (int) Math.ceil(this.num_features / 2.0) - 1; j < this.num_features; j++) {
            arr_m[j] = Math.max(0, (int) Math.ceil(num_sample * Math.sqrt(sigma_k[j] + sigma_n_k[j]) / var_sum));
        }

        // --------------------- second stage --------------------------
        double[][] new_utility = new double[this.num_features][this.num_features];
        double[][] arr_count = new double[this.num_features][this.num_features];

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
        for (int j = 0; j < this.num_features; j++) {
            for (int k = 0; k < (int)arr_m[j]; k++) {
                Collections.shuffle(idxs);  // Shuffle idxs array (custom method needed)
                // 3）构造两个子集subset_1 & subset_2
                ArrayList<Integer> subset_1 = new ArrayList<>();   //*subset_1 第一个特征子集
                double value_1 = 0;
                ArrayList<Integer> subset_2 = new ArrayList<>();   //*subset_2 第二个特征子集
                double value_2 = 0;
                for (int ind = 0; ind < j+1; ind++) { // ele 对应一个特征
                    subset_1.add(idxs.get(ind));
                }
                for (int ind = j+1; ind < idxs.size(); ind++) { // ele 对应一个特征
                    subset_2.add(idxs.get(ind));
                }
                //4)分别求函数值
                switch (model) {
                    case "airport":
                        value_1 = game.value_airport(subset_1, this.given_weights);
                        value_2 = game.value_airport(subset_2, this.given_weights);
                        break;
                    case "voting":
                        value_1 = game.value_voting(subset_1, this.given_weights, halfSum);
                        value_2 = game.value_voting(subset_2, this.given_weights, halfSum);
                        break;
                    case "shoes":
                        value_1 = game.value_shoes(subset_1);
                        value_2 = game.value_shoes(subset_2);
                        break;
                    case "model":
                        value_1 = game.value_modelPrediction(given_weights, subset_1);
                        value_2 = game.value_modelPrediction(given_weights, subset_2);
                        break;
                    case "svm_model":
                        ModelGame model_svm = new ModelGame("svm_model");
                        if (subset_1.size() == 0) {
                            value_1 = model_svm.value_darwin(given_weights, new ArrayList<Integer>());
                            value_2 = model_svm.value_darwin(given_weights, subset_2);
                        } else if (subset_2.size() == 0) {
                            value_2 = model_svm.value_darwin(given_weights, new ArrayList<Integer>());
                            value_1 = model_svm.value_darwin(given_weights, subset_1);
                        } else {
                            value_1 = model_svm.value_darwin(given_weights, subset_1);
                            value_2 = model_svm.value_darwin(given_weights, subset_2);
                        }
                        break;
                    case "iot":
                        IOT_83F_sm1_iter2 iot_model = new IOT_83F_sm1_iter2();
                        value_1 = iot_model.IOT_value(given_weights, subset_1);
                        value_2 = iot_model.IOT_value(given_weights, subset_2);
                        break;
                    case "health":
                        Health healthGame = new Health();
                        value_1 = healthGame.Health_value(given_weights, subset_1, client);
                        value_2 = healthGame.Health_value(given_weights, subset_1, client);
                        break;
                }

                //temp = np.zeros(n)
                double[] temp = new double[this.num_features];
                for (int i = 0; i <= j; i++) {
                    temp[idxs.get(i)] = 1;
                }
                for (int i = 0; i < this.num_features; i++) {
                    new_utility[i][j] += temp[i] * (value_1 - value_2);
                    arr_count[i][j] += temp[i];
                }

                //temp = np.zeros(n)
                temp = new double[this.num_features];
                for (int i = j + 1; i < this.num_features; i++) {
                    temp[idxs.get(i)] = 1;
                }
                for (int i = 0; i < this.num_features; i++) {
                    new_utility[i][this.num_features - j - 2] += temp[i] * (value_2 - value_1);
                    arr_count[i][this.num_features - j - 2] += temp[i];
                }
            }
        }

        // 5) 求shapley value的平均值，对于每个特征&每个长度求均值
        ShapMatrixEntry[] resultShap = new ShapMatrixEntry[this.num_features];  //这是最后返回的大数组
        double[] sv = new double[this.num_features]; // Example array for sv values
        for (int i = 0; i < this.num_features; i++) {
            resultShap[i] = new ShapMatrixEntry();
            for (int j = 0; j < this.num_features; j++) {
                if (arr_count[i][j] == 0) {
                    resultShap[i].sum += 0;
                    resultShap[i].count ++;
                }
                else{
                    resultShap[i].sum += (double) new_utility[i][j] / arr_count[i][j];
                    resultShap[i].count ++;
                }
            }
        }

        //6）求一个整体的均值
        for(ShapMatrixEntry entry : resultShap){
            if(entry.count != 0){
                entry.sum = entry.sum / entry.count;
            }
            else{
                entry.sum =0;
            }
        }
        return resultShap;
    }

    //TODO 计算方差
    private double varComputation(ArrayList<Double> record) {
        double sum = 0.0;
        // Calculate the mean
        for (double value : record) {
            sum += value;
        }
        double mean = sum / record.size();
        // Calculate the variance
        double sumSquares = 0.0;
        for (double value : record) {
            sumSquares += (value - mean) * (value - mean);
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
