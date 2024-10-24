package AlgoVersion;

import Game.GameClass;
import Global.Comparer;
import config.Info;
import java.util.ArrayList;
import java.util.Random;

public class MC_algorithm {

    int num_features;  //the number of features
    double[] exact;   // the exact shapley value
    int num_samples;

    // [MC Algorithm]: approximate the shapley value by Monte Carlo sampling
    public void MC(boolean gene_weight, String model){

        //Initialization
        GameClass game = new GameClass();
        initialization(game, gene_weight, model);
        Comparer comparator = new Comparer();

        long ave_runtime = 0;
        double ave_error_max = 0;
        double ave_error_ave = 0;
        double[][] sv_values = new double[Info.timesRepeat][this.num_features];
        for(int i=0; i< Info.timesRepeat; i++) {
            Random PermutationGene = new Random(game.seedSet[i]);
            long time_1 = System.currentTimeMillis();
            double[] shap_matrix = computeShapBySampling(game, this.num_features, this.num_samples, model, PermutationGene);
            long time_2 = System.currentTimeMillis();

            double error_max = comparator.computeMaxError(shap_matrix, this.exact);
            double error_ave = comparator.computeAverageError(shap_matrix, this.exact, this.num_features);

            sv_values[i] = shap_matrix;
            ave_runtime += time_2 - time_1;
            ave_error_max += error_max;
            ave_error_ave += error_ave;
//            System.out.println("run: " + i + " " + "error_ave: " + error_ave + " \t"  +  "error_max: " + error_max);
        }
        double acv = comparator.computeACV(sv_values, this.num_features);
        System.out.println(model + " Game:  " + "error_ave: " + ave_error_ave/Info.timesRepeat + " \t"  + "error_max: "
                + ave_error_max/Info.timesRepeat);
//        System.out.println("average cv :" + acv);
        System.out.println("MC time : " + (ave_runtime * 0.001)/ Info.timesRepeat);
    }

    private double[] computeShapBySampling(GameClass game, int num_features, int num_sample, String model, Random PermutationGene) {
        double[] shap_matrix = new double[num_features];

        for(int r=0; r<num_sample; r++){
            // randomly generate a permutation and shuffle the sequence
            ArrayList<Integer> p = permutation(PermutationGene);

            for(int ind = 0; ind <p.size(); ind ++){
                int ele = p.get(ind);
                ArrayList<Integer> subset_1 = new ArrayList<>();
                double value_1 = 0;
                ArrayList<Integer> subset_2 = new ArrayList<>();
                double value_2 = 0;
                for(int i =0; i<ind; i++){
                    subset_1.add(p.get(i));
                }
                for(int i =0; i<ind; i++){
                    subset_2.add(p.get(i));
                }
                subset_2.add(ele);
                value_1 = game.gameValue(model, subset_1);
                value_2 = game.gameValue(model, subset_2);
                shap_matrix[ele] += value_2 - value_1;
            }
        }
        for(int i=0; i < shap_matrix.length; i++){
            shap_matrix[i] = shap_matrix[i] / num_sample;
        }
        return shap_matrix;
    }

    private ArrayList<Integer> permutation(Random PermutationGene) {
        //1) using features to generate a permutation sequence
        ArrayList<Integer> perm = new ArrayList<>();
        for(int i=0; i<this.num_features; i++){
            perm.add(i);
        }
        int sequenceSeed = PermutationGene.nextInt();
        Random random = new Random(sequenceSeed);
        double step = Math.min(Math.max(1.0, 4.6 - Math.log10(Info.setting)), 4);

        // Randomly shuffle the sequence
        for(double i = 0; i<this.num_features; i += step){
            int index = (int) Math.min(this.num_features-1, Math.round(i));
            int j = random.nextInt(this.num_features);
            int temp = perm.get(j);
            perm.set(j, perm.get(index));
            perm.set(index, temp);
        }
        return perm;
    }

    private void initialization(GameClass game, boolean gene_weight, String modelName) {
        game.gameInit(gene_weight, modelName);
        this.num_features = game.num_features; //the number of features
        this.exact = game.exact;   // the exact shapley value
        this.num_samples = Info.total_samples_num / (game.num_features + 1);
    }
}
