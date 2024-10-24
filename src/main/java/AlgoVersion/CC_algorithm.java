package AlgoVersion;

import Game.GameClass;
import Global.Comparer;
import config.Info;
import structure.ShapMatrixEntry;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

public class CC_algorithm {

    int num_features;  // the number of features
    double[] given_weights;
    double[] exact;  // the exact shapley value
    double halfSum;  //for Voting game
    int num_samples;

    // [CC Algorithm]: approximate the shapley value by complementary contributions
    public void CC(boolean gene_weight, String model){
        ShapMatrixEntry[][] sv_values = new ShapMatrixEntry[Info.timesRepeat][];
        GameClass game = new GameClass();
        initialization(game, gene_weight, model, sv_values);
        Comparer comparator = new Comparer();

        long ave_runtime = 0;
        double ave_error_max = 0;
        double ave_error_ave = 0;
        for(int i=0; i< Info.timesRepeat; i++) {
            Random PermutationGene = new Random(game.seedSet[i]);
            long time_1 = System.currentTimeMillis();
            ShapMatrixEntry[] shap_matrix = computeShapBySampling(game, this.num_samples, model, PermutationGene);
            long time_2 = System.currentTimeMillis();
            sv_values[i] = shap_matrix;

            double error_ave = comparator.computeAverageError(shap_matrix, this.exact, this.num_features);
            double error_max = comparator.computeMaxError(shap_matrix, this.exact, this.num_features);

            ave_runtime += time_2 - time_1;
            ave_error_max += error_max;
            ave_error_ave += error_ave;
//            System.out.println("run: " + i + " " + "error_ave: " + error_ave + " \t"  +  "error_max: " + error_max);
        }
//        double acv = comparator.computeACV(sv_values, this.num_features);
        System.out.println(model + " Game:  " + "error_ave: " + ave_error_ave/Info.timesRepeat + " \t"  +  "error_max: "
                + ave_error_max/Info.timesRepeat + " \t");
//        System.out.println("average cv :" + acv);
        System.out.println("CC time : " + (ave_runtime * 0.001)/ Info.timesRepeat);
    }

    private ShapMatrixEntry[] computeShapBySampling(GameClass game, int num_sample, String model, Random random) {

        ShapMatrixEntry[][] shap_matrix = new ShapMatrixEntry[this.num_features+1][];
        for(int i=0; i<shap_matrix.length; i++){
            shap_matrix[i] = new ShapMatrixEntry[this.num_features];
            for(int j=0; j<this.num_features; j++) {
                shap_matrix[i][j] = new ShapMatrixEntry();
            }
        }

        for(int r=0; r<num_sample; r+=2){
            // randomly generate a permutation and shuffle it
            ArrayList<Integer> p = permutation(this.num_features, random);
            int random_i = random.nextInt(this.num_features);
            ArrayList<Integer> subset_1 = new ArrayList<>();
            double value_1 = 0;
            ArrayList<Integer> subset_2 = new ArrayList<>();
            double value_2 = 0;

            for(int ind = 0; ind <random_i; ind ++) {
                subset_1.add(p.get(ind));
            }
            for(int ind = random_i; ind <p.size(); ind ++) {
                subset_2.add(p.get(ind));
            }

            value_1 = game.gameValue(model, subset_1);
            value_2 = game.gameValue(model, subset_2);
            double comp_contrib = value_1 - value_2;

            // update the complementary contributions
            for(Integer ele : subset_1){
                shap_matrix[random_i][ele].sum += comp_contrib;
                shap_matrix[random_i][ele].count ++;
            }
            for(Integer ele : subset_2){
                shap_matrix[this.num_features - random_i][ele].sum -= comp_contrib;
                shap_matrix[this.num_features - random_i][ele].count ++;
            }

        }
        ShapMatrixEntry[] resultShap = new ShapMatrixEntry[this.num_features];
        for(int i=0; i<this.num_features; i++) {
            resultShap[i] = new ShapMatrixEntry();
        }

        for(ShapMatrixEntry[] featureSet : shap_matrix){
            for(int fea=0; fea<featureSet.length ; fea++){
                ShapMatrixEntry entry = featureSet[fea];
                if(entry.count != 0){
                    entry.sum = entry.sum / entry.count;
                    resultShap[fea].sum += entry.sum;
                    resultShap[fea].count ++;
                }
            }
        }

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

    private ArrayList<Integer> permutation(int numFeatures, Random PermutationGene) {
        ArrayList<Integer> perm = new ArrayList<>();
        for(int i=0; i<numFeatures; i++){
            perm.add(i);
        }
        int sequenceSeed = PermutationGene.nextInt();
        Random random = new Random(sequenceSeed);
        Collections.shuffle(perm, random);
        return perm;
    }

    private void initialization(GameClass game, boolean gene_weight, String modelName, ShapMatrixEntry[][] sv_values) {
        game.gameInit(gene_weight, modelName);
        this.num_features = game.num_features; //the number of features
        this.exact = game.exact;   // the exact shapley value
        this.given_weights = game.given_weights;
        this.halfSum = game.halfSum;  //for Voting game
        this.num_samples = Info.total_samples_num;
        for(ShapMatrixEntry[] ele : sv_values){
            ele = new ShapMatrixEntry[this.num_features];
        }
    }

}
