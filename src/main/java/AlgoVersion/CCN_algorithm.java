package AlgoVersion;

import Game.GameClass;
import Global.Comparer;
import config.Info;
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

    public void CCN(boolean gene_weight, String model){
        ShapMatrixEntry[][] sv_values = new ShapMatrixEntry[Info.timesRepeat][];
        GameClass game = new GameClass();
        initialization(game, gene_weight, model, sv_values);
        Comparer comparator = new Comparer();

        long ave_runtime = 0;
        double ave_error_max = 0;
        double ave_error_ave = 0;

        for(int i=0; i< Info.timesRepeat; i++) {
            Random random = new Random(game.seedSet[i]);
            int total_evaluateNum = this.total_num_evaluations;

            long time_1 = System.currentTimeMillis();
            ShapMatrixEntry[] shap_matrix = computeShapBySampling(game, this.initial_m, model, random, total_evaluateNum);
            long time_2 = System.currentTimeMillis();
            sv_values[i] = shap_matrix;

            double error_ave = comparator.computeAverageError(shap_matrix, this.exact, this.num_features);
            double error_max = comparator.computeMaxError(shap_matrix, this.exact, this.num_features);
            ave_runtime += time_2 - time_1;
            ave_error_max += error_max;
            ave_error_ave += error_ave;
//            System.out.println("run: " + i + " " + "error_ave: " + error_ave + " \t"  +  "error_max: " + error_max);
        }

        double acv = comparator.computeACV(sv_values, this.num_features);
        System.out.println(model + " Game:  " + "error_ave: " + ave_error_ave/Info.timesRepeat + " \t"  +  "error_max: "
                + ave_error_max/Info.timesRepeat);
//        System.out.println("average cv :" + acv);
        System.out.println("CCN time : " + (ave_runtime * 0.001)/ Info.timesRepeat );
    }

    private void initialization(GameClass game, boolean gene_weight, String modelName, ShapMatrixEntry[][] sv_values) {
        game.gameInit(gene_weight, modelName);
        this.num_features = game.num_features; //the number of features
        this.total_num_evaluations = Info.total_samples_num;  //the number of evaluations
        this.num_samples = this.total_num_evaluations/2;  //A sample requires pairwise evaluations
        this.exact = game.exact;   // the exact shapley value
        this.given_weights = game.given_weights;
        this.halfSum = game.halfSum;  //for Voting game
        this.initial_m = initialm(this.num_samples/2, this.num_features);
        for(ShapMatrixEntry[] ele : sv_values){
            ele = new ShapMatrixEntry[this.num_features];
        }
    }

    private ShapMatrixEntry[] computeShapBySampling(GameClass game, int initial_m, String model, Random random,
                                                    int total_evaluateNum) {

        ShapMatrixEntry[][] utility = new ShapMatrixEntry[this.num_features][];
        for(int i=0; i<utility.length; i++){
            utility[i] = new ShapMatrixEntry[this.num_features];
            for(int j=0; j<this.num_features; j++) {
                utility[i][j] = new ShapMatrixEntry();
                utility[i][j].record = new ArrayList<>();
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

            for (int i = 0; i < this.num_features; i++) {
                utility[i][this.num_features -1].sum += value_n - value_0;
                utility[i][this.num_features -1].count ++;
                utility[i][this.num_features -1].record.add(value_n - value_0);

                ArrayList<Integer> idxs = new ArrayList<>();   //idxs: generate a permutation excluding feature i
                for (int ele = 0; ele < i; ele++) {
                    idxs.add(ele);
                }
                for (int ele = i + 1; ele < this.num_features; ele++) {
                    idxs.add(ele);
                }
                for (int len = 0; len < this.num_features; len++) {
                    if (utility[i][len].count >= initial_m || utility[i][len].count >= coef[len]) {
                        continue;
                    }
                    idxs = permutation(idxs, random);
                    count ++;

                    //compute the complementary contribution of each feature by using p
                    ArrayList<Integer> subset_1 = new ArrayList<>();
                    ArrayList<Integer> subset_2 = new ArrayList<>();

                    for (int ind = 0; ind < len; ind++) {
                        subset_1.add(idxs.get(ind));
                    }
                    subset_1.add(i);
                    for (int ind = len; ind < idxs.size(); ind++) {
                        subset_2.add(idxs.get(ind));
                    }
                    double value_1 = game.gameValue(model, subset_1);
                    double value_2 = game.gameValue(model, subset_2);
                    utility[i][len].sum += value_1 - value_2;
                    utility[i][len].count ++;
                    utility[i][len].record.add(value_1 - value_2);
                    evaluations_num += 2;
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

        // calculate the variance
        double[][] variance = new double[this.num_features][this.num_features];
        for(int i=0; i < this.num_features; i++){
            for(int j=0; j < this.num_features; j++){
                variance[i][j] = varComputation(utility[i][j].record);
            }
        }

        //allocate the number of samples
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

        total_evaluateNum = total_evaluateNum - evaluations_num;
        int[] arr_m = new int[this.num_features];
        Arrays.fill(arr_m, 0); // Initialize m to zeros
        for (int j = (int) Math.ceil(this.num_features / 2.0) - 1; j < this.num_features; j++) {
            arr_m[j] =  Math.max(0, (int) (total_evaluateNum * Math.sqrt(sigma_k[j] + sigma_n_k[j]) / var_sum/2));
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
                ArrayList<Integer> subset_1 = new ArrayList<>();
                ArrayList<Integer> subset_2 = new ArrayList<>();
                for (int ind = 0; ind < len+1; ind++) {
                    subset_1.add(idxs.get(ind));
                }
                for (int ind = len+1; ind < idxs.size(); ind++) {
                    subset_2.add(idxs.get(ind));
                }
                double value_1 = game.gameValue(model, subset_1);
                double value_2 = game.gameValue(model, subset_2);
                evaluations_num += 2;

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


        // compute the shapley value
        ShapMatrixEntry[] resultShap = new ShapMatrixEntry[this.num_features];
        for (int i = 0; i < this.num_features; i++) {
            resultShap[i] = new ShapMatrixEntry();
            for (int j = 0; j < this.num_features; j++) {
                if (arr_count[i][j] != 0) {
                    resultShap[i].sum +=  new_utility[i][j] / arr_count[i][j];
                    resultShap[i].count ++;
                }
            }
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
        ArrayList<Integer> perm = new ArrayList<>(list);
        int sequenceSeed = PermutationGene.nextInt();
        Random random = new Random(sequenceSeed);
        Collections.shuffle(perm, random);
        return perm;
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
        return sumSquares / (record.size() - 1);
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

    private static double sumArray(ArrayList<Double> array) {
        double sum = 0.0;
        for (double value : array) {
            sum += value;
        }
        return sum;
    }
}
