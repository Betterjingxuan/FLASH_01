package Global;

import Game.GameClass;
import structure.ShapMatrixEntry;
import java.util.*;

public class Allocation {

    public double[] weigh_for_sample;

    public int[] num_sample;

    public double total_weights;
    public int total_samples;
    public double mu_f_weight;

    public Allocation(GameClass game){
        this.mu_f_weight = game.mu_f;
    }

    public void sampleAllocation_uni(int num_features, int num_samples, int startLev, int endLev) {
        this.num_sample = new int[num_features];
        int sample = (int) Math.ceil(1.0f * num_samples / (endLev-startLev));
        for(int ind = startLev; ind <endLev; ind ++){
            this.num_sample[ind] += sample;
            this.total_samples += sample;
        }
    }

    //TODO: allocate the number of evaluations by the variance of each layer
    public int sampleAllocation(int num_features, double[][] newLevelMatrix, int limit, int allSamples, GameClass game,
                                                                     Random random, ShapMatrixEntry[][] shap_matrix) {
        this.weigh_for_sample = new double[num_features];
        this.num_sample = new int[num_features];
        this.total_weights = 0.0;

        // initialize varianceArr & utility
        ShapMatrixEntry[][] utility = new ShapMatrixEntry[num_features][];
        for(int j=0; j<utility.length; j++){
            utility[j] = new ShapMatrixEntry[num_features];
            for(int i=0; i<num_features; i++) {
                utility[j][i] = new ShapMatrixEntry();
            }
        }

        for(int l = game.start_level; l<=game.end_level; l++){
            for(int fea=0; fea<num_features; fea++){
               if(game.isRealData){
                   double value = 0;
                   if(fea == num_features - 1){
                       value = newLevelMatrix[l][fea] - newLevelMatrix[l-1][0];
                   }
                   else{
                       value = newLevelMatrix[l][fea] - newLevelMatrix[l-1][fea+1];
                   }
                   shap_matrix[l][fea].record.add(value);
                   shap_matrix[l][fea].sum += value;
                   shap_matrix[l][fea].count ++;
               }

                for(int count = 0; count <limit; count ++){
                    Set<Integer> hashSet = new HashSet<>();
                    while (hashSet.size() <= l) {
                        int number = random.nextInt(num_features);
                        hashSet.add(number);
                    }
                    ArrayList<Integer> subset_1 = new ArrayList<>(hashSet);
                    if(!subset_1.contains(fea)){
                        int removeInd = random.nextInt(subset_1.size());
                        subset_1.set(removeInd, fea);
                    }
                    ArrayList<Integer> subset_2 = new ArrayList<>(subset_1);
                    subset_2.remove(Integer.valueOf(fea));
                    double value_1 = game.gameValue(game.model.gameName, subset_1);
                    double value_2 = game.gameValue(game.model.gameName, subset_2);
                    utility[l][fea].record.add(value_1 - value_2);
                    utility[l][fea].sum += value_1 - value_2;
                    utility[l][fea].count ++;
                    //save the marginal contribution
                    shap_matrix[l][fea].record.add(value_1 - value_2);
                    shap_matrix[l][fea].sum += value_1 - value_2;
                    shap_matrix[l][fea].count ++;
                }
            }
        }

        // calculate the variance and allocate the number of evaluations
        double[] varianceArr = new double[num_features];
        for(int lev = 0; lev<num_features; lev++){
            ShapMatrixEntry[] level_u = utility[lev];
            double variance = 0;
            for(ShapMatrixEntry ele_u : level_u){
                variance += varComputation(ele_u.record);
            }
            varianceArr[lev] = variance;
            this.total_weights +=variance;
        }
        //allocate the number of evaluations (samples)
        int apartSam = (int) Math.ceil(allSamples/this.total_weights);
        for(int lev = 0; lev<num_features; lev++){
            int sample = 0;
            if(varianceArr[lev] != 0){
                sample = (int) Math.ceil(varianceArr[lev] * apartSam);
            }
            this.num_sample[lev] += sample;
        }
        int return_lev = game.start_level;
        for(int lev = game.start_level; lev<num_features; lev++){
            if(this.num_sample[lev] != 0){
                return_lev = lev;
                break;
            }
        }
        return return_lev;
    }

    private double varComputation(ArrayList<Double> record) {
        if(record.size() <= 1){
            return 0;
        }
        double sum = 0.0;
        // Calculate the mean value
        for (double value : record) {
            sum += value;
        }
        double mean = sum / record.size();
        // Calculate the variance
        double sumSquares = 0.0;
        for (double value : record) {
            sumSquares += Math.pow(value - mean, 2);
        }
        return sumSquares / record.size();
    }


}
