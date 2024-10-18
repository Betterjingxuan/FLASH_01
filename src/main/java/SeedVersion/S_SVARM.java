package SeedVersion;

import Game.GameClass;
import Game.ModelGame;
import Global.*;
import structure.ShapMatrixEntry;

import java.io.DataInputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.sql.SQLOutput;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;
import java.util.stream.IntStream;

public class S_SVARM {
    int num_features;  //数据集中包含的特征数量

    double[] given_weights;
    double[] exact;  // the exact shapley value
    double halfSum;  //for Voting game

    int num_samples;
    double[][] phi_i_l_plus;
    double[][] phi_i_l_minus;
    int[][] c_i_l_plus;
    int[][] c_i_l_minus;

    public void SSVARM_algorithm(boolean gene_weight, String model){

        GameClass game = new GameClass();
        initialization(game, gene_weight, model);
        Comparer comp = new Comparer();

        //3、开始计算shapley value：
        long ave_runtime = 0;
        double ave_error_max = 0; //计算最大误差
        double ave_error_ave = 0;
        double[][] sv_values = new double[Info.timesRepeat][this.num_features];  //[runs][feature]
        double ave_mse = 0;
        for(int t = 0; t < Info.timesRepeat; t++) {
            Random random = new Random(game.seedSet[t]);
            this.num_samples = Info.total_samples_num;
            this.phi_i_l_plus = new double[this.num_features][];
            this.phi_i_l_minus = new double[this.num_features][];
            this.c_i_l_plus = new int[this.num_features][];
            this.c_i_l_minus = new int[this.num_features][];

            for(int i = 0; i< this.num_features; i++){  // s范围是[0,n+1)
                this.phi_i_l_plus[i] = new double[this.num_features];
                this.phi_i_l_minus[i] = new double[this.num_features];
                this.c_i_l_plus[i] = new int[this.num_features];
                this.c_i_l_minus[i] = new int[this.num_features];
            }

            long time_1 = System.currentTimeMillis();
            double[] shap_matrix = computeSVBySSVARM(game, model, random);
            long time_2 = System.currentTimeMillis();
            sv_values[t] = shap_matrix;

            // 计算误差
            double error_max = comp.computeMaxError(shap_matrix, this.exact); //计算最大误差
            double error_ave = comp.computeAverageError(shap_matrix, this.exact, this.num_features);  //计算平均误差
//            double mse = comp.computeMSE(shap_matrix, this.exact, this.num_features);
//
            ave_runtime += time_2 - time_1;
            ave_error_max += error_max;
            ave_error_ave += error_ave;
//            ave_mse += mse;
            System.out.println("run: " + t + " " + "error_ave: " + error_ave + " \t"  +  "error_max: " + error_max);
        }
//        double acv = comp.computeACV(sv_values, this.num_features);
        System.out.println(model + ":  " + "error_ave: " + ave_error_ave/Info.timesRepeat + " \t"  +  "error_max: " +
                ave_error_max/Info.timesRepeat + " \t"  +  "error_MSE: " + ave_mse/ Info.timesRepeat);
//        System.out.println("average cv :" + acv);
        System.out.println("S-SVARM time : " + (ave_runtime * 0.001)/ Info.timesRepeat );  //+ "S"
    }

    //TODO S-SVARM algorithm
    private double[] computeSVBySSVARM(GameClass game, String model, Random random) {

        ArrayList<Integer> allPlayers = new ArrayList<>();
        for(int i=0; i<this.num_features; i++){
            allPlayers.add(i);
        }

        //1) robability distribution over sizes (2,...,n-2) for sampling
        double[] distribution = generatePaperDistribution(this.num_features);
        double[] probs = new double[this.num_features + 1];
        double probs_sum = 0.0;
        for (int s = 0; s <= this.num_features; s++) {
            probs[s] = distribution[s];
            probs_sum += distribution[s];
        }

        // 2) Exact_calculation: plus[i][0], plus[i][n-1], plus[i][n-2];  minus[i][1], minus[i][n-1]
        exact_calculation(game, model, allPlayers);

        // 3) [Warmup] paper version: set normalize=False, warm_up=True, rich_warm_up=False, paired_sampling=False
        positiveWarmup(game, model, random);
        negativeWarmup(game, model, random);

        //4) Calculation the contributions
        while(this.num_samples > 0){
            int s = getRandomChoice(probs, probs_sum, random);  //随机选择一个长度，按权重prob[], s范围是[0,n+1)
            ArrayList<Integer> A = getRandomSubset(allPlayers, s, random);
            updateProcedure(A, allPlayers, game, model);
            if(this.num_samples <= 0){
                break;
            }
        }

        //5) Calculate the mean value
        double[] shap_matrix = new double[num_features];  //返回的大数组
        for(int i : allPlayers){
            for(int l=0; l<this.num_features; l++){  //S的范围是[0, n+1), 长度是0-n
                shap_matrix[i] += phi_i_l_plus[i][l] - phi_i_l_minus[i][l];
            }
            shap_matrix[i] = shap_matrix[i] / this.num_features;
//            System.out.print(shap_matrix[i] + "\t");
        }
        return shap_matrix;
    }

    //TODO 计算A的value, 并更新数组
    private void updateProcedure(ArrayList<Integer> A, ArrayList<Integer> players, GameClass game, String model) {
        double value = game.gameValue(model, A);
        this.num_samples --;
        int s = A.size();

        for(Integer i : A){
            this.phi_i_l_plus[i][s - 1] = (this.phi_i_l_plus[i][s - 1] * this.c_i_l_plus[i][s - 1] + value) / (c_i_l_plus[i][s - 1] + 1);
            c_i_l_plus[i][s - 1] ++;
        }

        //找A中不存在的元素
        ArrayList<Integer> notA = new ArrayList<>(players);
        notA.removeAll(A);

        for (Integer i : notA) {
            phi_i_l_minus[i][s] = (phi_i_l_minus[i][s] * c_i_l_minus[i][s] + value) / (c_i_l_minus[i][s] + 1);
            c_i_l_minus[i][s] ++;
        }
    }

    //TODO:从集合中随机选择指定数量的元素(不重复元素)
    private ArrayList<Integer> getRandomSubset(ArrayList<Integer> players, int len, Random random) {
        ArrayList<Integer> A = new ArrayList<>();
        ArrayList<Integer> list = permutation(players, random);  //生成一个随机打乱的序列
        for(int ind=0; ind<len; ind++){
            A.add(list.get(ind));
        }
        return A;
    }

    //TODO: 按照probs[] 的概率分布随机选取一个int
    private int getRandomChoice(double[] probs, double probs_sum, Random random) {
        int returnInt = -1; // 返回的结果

        if(probs_sum == 0){
            System.out.println("ERR: The sum of probabilities should be greater than 0.");
        }
        else{
            //1）生成一个范围是[0.0, probs_sum)的随机数
            double randomValue = random.nextDouble() * probs_sum;
            //2）根据随机数值找到对应的权重区间
            double cumulativeSum = 0.0;
            for (int i = 0; i < probs.length; i++) {
                cumulativeSum += probs[i];
                if (randomValue <= cumulativeSum) {
                    return i;
                }
            }
        }
        return returnInt;
    }

    private void positiveWarmup(GameClass game, String model, Random random) {
        //初始化 & 赋值
        int n = this.num_features;
        ArrayList<Integer> players = new ArrayList<>();
        for(int i=0; i<n; i++){
            players.add(i);
        }

        for(int s=2; s<n-1; s++){
            ArrayList<Integer> pi = permutation(players, random);  //生成一个随机打乱的序列
            for (int k = 0; k < n / s; k++) {
                ArrayList<Integer> A = new ArrayList<>();
                for (int r = 1; r <= s; r++) {
                    A.add(pi.get(r + k * s - 1));
                }
                double value = game.gameValue(model, A);
                this.num_samples --;
                for (int i : A) {
                    phi_i_l_plus[i][s - 1] = value;
                    c_i_l_plus[i][s - 1] = 1;
                }
            }

            if (n % s != 0) {
                ArrayList<Integer> A = new ArrayList<>();
                ArrayList<Integer> set = new ArrayList<>(players);
                for (int r = n - (n % s) + 1; r <= n; r++) {
                    A.add(pi.get(r - 1));
                    set.remove(pi.get(r - 1));
                }
                ArrayList<Integer> B = new ArrayList<>();
                for (int j = 0; j < s - (n % s); j++) {
                    int element  = random.nextInt(set.size());
                    B.add(element);
                    set.remove(element);
                }
                A.addAll(B);
                double value = game.gameValue(model, A);
                this.num_samples --;
                for (int i : A) {
                    phi_i_l_plus[i][s - 1] = value;
                    c_i_l_plus[i][s - 1] = 1;
                }
            }
        }
    }

    private void negativeWarmup(GameClass game, String model, Random random) {
        //初始化 & 赋值
        int n = this.num_features;
        ArrayList<Integer> players = new ArrayList<>();
        for(int i=0; i<n; i++){
            players.add(i);
        }

        for(int s=2; s<n-1; s++){
            ArrayList<Integer> pi = permutation(players, random);  //生成一个随机打乱的序列
            for (int k = 0; k < n / s; k++) {
                ArrayList<Integer> A = new ArrayList<>();
                for (int r = 1; r <= s; r++) {
                    A.add(pi.get(r + k * s - 1));
                }
                ArrayList<Integer> set = new ArrayList<>(players);
                set.removeAll(A);
                double value = game.gameValue(model, set);
                this.num_samples --;
                for (int i : A) {
                    phi_i_l_plus[i][n - s] = value;
                    c_i_l_plus[i][n - s] = 1;
                }
            }

            if (n % s != 0) {
                ArrayList<Integer> A = new ArrayList<>();
                ArrayList<Integer> set = new ArrayList<>(players);
                for (int r = n - (n % s) + 1; r <= n; r++) {
                    A.add(pi.get(r - 1));
                    set.remove(pi.get(r - 1));
                }
                ArrayList<Integer> B = new ArrayList<>();
                for (int j = 0; j < s - (n % s); j++) {
                    int element  = random.nextInt(set.size());
                    B.add(element);
                    set.remove(element);
                }
                double value = game.gameValue(model, set);
                this.num_samples --;
                for (int i : A) {
                    phi_i_l_plus[i][n - s] = value;
                    c_i_l_plus[i][n - s] = 1;
                }
            }
        }
    }

    //TODO 随机打乱的序列 list
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

    //TODO  Exact_calculation:
    //plus[i][0], plus[i][n-1], plus[i][n-2];  minus[i][1], minus[i][n-1]  exact_calculation();
    private void exact_calculation(GameClass game, String model, ArrayList<Integer> allPlayers) {
        int n = this.num_features;

        double empty_v = game.gameValue(model, new ArrayList<>());  //空集
        this.num_samples --;

        // negative strata
        for(int i=0; i<n; i++){
            this.phi_i_l_minus[i][0] = empty_v;
            this.c_i_l_minus[i][0] = 1;
        }

        double grand_co_value = game.gameValue(model, allPlayers); //全集
        this.num_samples --;

        // plus[i][n-1]: positive n-1 strata
        for(int i=0; i<n; i++){
            this.phi_i_l_plus[i][n-1] = grand_co_value;
            this.c_i_l_plus[i][n-1] = 1;
        }

        for(int i=0; i<n; i++){
            ArrayList<Integer> single = new ArrayList<>();
            single.add(i);
            ArrayList<Integer> set = new ArrayList<>(allPlayers);
            set.remove(i);

            double v_plus = game.gameValue(model, single);
            this.num_samples --;

            // plus[i][0]: positive 0 strata
            this.phi_i_l_plus[i][0] = v_plus;
            this.c_i_l_plus[i][0] = 1;

            // minus[i][1]: negative 1 strata
            for(Integer j : set){
                this.phi_i_l_minus[j][1] = (this.phi_i_l_minus[j][1] * this.c_i_l_minus[j][1] + v_plus) / (this.c_i_l_minus[j][1] + 1);
                this.c_i_l_minus[j][1] += 1;
            }

            double v_minus = game.gameValue(model, set);
            this.num_samples --;

            // minus[i][n-1]: negative n-1 strata
            this.phi_i_l_minus[i][n - 1] = v_minus;
            this.c_i_l_minus[i][n - 1] = 1;

            // plus[i][n-2]: positive n-2 strata
            for(Integer j : set) {
                this.phi_i_l_plus[j][n - 2] = (this.phi_i_l_plus[j][n - 2] * this.c_i_l_plus[j][n - 2] + v_minus) / (this.c_i_l_plus[j][n - 2] + 1);
                this.c_i_l_plus[j][n - 2] += 1;
            }
        }

    }

    private double[] generateDistribution(String name, int n) {
        switch (name) {
            case "paper":
                return generatePaperDistribution(n);
            case "uniform":
                return generateUniformDistribution(n);
            case "descending":
                return generateLinearlyDescendingDistribution(n, 3.0);
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

        double frac = 0;
        double H = 0;
        if (n % 2 == 0) {
            double nlogn = n * Math.log(n);
            H = calculateDistributionSum(1,n/2);  //计算从 1 到 n/2 - 1 的倒数之和
            double nominator = nlogn - 1;
            double denominator = 2 * nlogn * (H - 1);
            frac = nominator / denominator;
            for (int s = 2; s < n / 2; s++) {
                dist[s] = frac / s;
                dist[n - s] = frac / s;
            }
            dist[n / 2] = 1 / nlogn;
        }
        else{
            H = calculateDistributionSum(1,(n - 1) / 2 + 1);
            frac = 1 / (2 * (H - 1));
            for (int s = 2; s < (n - 1) / 2 + 1; s++) {
                dist[s] = frac / s;
                dist[n - s] = frac / s;
            }
        }
        return dist;
    }

    //TODO 计算从 a 到 b 的倒数之和
    private double calculateDistributionSum(int a, int b) {
        double sum = 0;
        for (int s = a; s < b; s++) {
            sum += 1.0 / s;
        }
        return sum;
    }

    //全局参数的初始化
    private void initialization(GameClass game, boolean gene_weight, String modelName) {
        game.gameInit(gene_weight, modelName);
        this.num_features = game.num_features; //the number of features
        this.exact = game.exact;   // the exact shapley value
        this.given_weights = game.given_weights;
        this.halfSum = game.halfSum;  //for Voting game
        this.num_samples = Info.total_samples_num;
    }

}
