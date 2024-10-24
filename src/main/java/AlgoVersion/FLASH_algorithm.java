package AlgoVersion;

import Game.GameClass;
import Global.Allocation;
import Global.Comparer;
import structure.FeatureSubset;
import config.Info;
import structure.ShapMatrixEntry;
import java.util.*;

public class FLASH_algorithm {
    private int num_features;  //the number of features
    private double[] exact;  // the exact shapley value
    private int num_samples;
    private FeatureSubset[][] allCoalitions;
    private String model;
    private double[][] evaluateMatrix;
    private double[][] levelMatrix;
    private int cheLevelInd;

    public void FLASH(boolean gene_weight, String model_name) {

        GameClass game = new GameClass();
        game.gameInit(gene_weight, model_name);
        ArrayList<Integer>[][] coaSet = new ArrayList[game.num_features][game.num_features];
        initialization(game, gene_weight, model_name, coaSet);
        Comparer comp = new Comparer();

        long ave_runtime = 0;
        double ave_error_max = 0;
        double ave_error_ave = 0;
        double[][] sv_values = new double[Info.timesRepeat][this.num_features];
        for(int i=0; i< Info.timesRepeat; i++) {
            this.num_samples = Info.total_samples_num;
            Random random = new Random(game.seedSet[i]);
            Allocation allo = new Allocation(game);
            ShapMatrixEntry[][] shap_matrix = new ShapMatrixEntry[this.num_features][];
            for (int ii = 0; ii < this.num_features; ii++) {
                shap_matrix[ii] = new ShapMatrixEntry[this.num_features];
                for (int j = 0; j < this.num_features; j++) {
                    shap_matrix[ii][j] = new ShapMatrixEntry();
                }
            }

            long time_1 = System.currentTimeMillis();
            double[] sv = ShapleyApproximateFLASH(game, shap_matrix, coaSet, allo, random);
            long time_2 = System.currentTimeMillis();
            sv_values[i] = sv;

            double error_max = comp.computeMaxError(sv, this.exact);
            double error_ave = comp.computeAverageError(sv, this.exact, this.num_features);
            ave_runtime += time_2 - time_1;
            ave_error_max += error_max;
            ave_error_ave += error_ave;
//            System.out.println("run: " + i + " " + "error_ave: " + error_ave + " \t"  +  "error_max: " + error_max );
        }
        double acv = comp.computeACV(sv_values, this.num_features);
        System.out.println(model_name + ":  " + "error_ave: " + ave_error_ave/Info.timesRepeat + " \t"  +  "error_max: "
                + ave_error_max/Info.timesRepeat);
//        System.out.println("average cv :" + acv);
        System.out.println("FLASH time : " + (ave_runtime * 0.001)/ Info.timesRepeat );
    }

    private void initialEstimation(ShapMatrixEntry[][] shap_matrix, ArrayList<Integer>[][] coaSet, Allocation allo,
                                   GameClass game, Random random) {
        game.start_level = Math.max(game.start_level, checkLevel_start(this.levelMatrix) + 1);
        game.end_level = checkLevel_end(this.levelMatrix);
        int m_star = Math.max(2, this.num_samples / (this.num_features * this.num_features * 2));
        int allSamples = this.num_samples;

        switch (this.model) {
            case "airport":   // (#feature = 100)
                allSamples -= Info.num_of_features_airport*Info.num_of_features_airport/2 - Info.num_of_features_airport/2;
                if(Info.is_gene_weight){  //for scalability test: the number of features is setting by [this.num_features]
                    allSamples = (int) Math.ceil((1.0 - game.mu_f) * (allSamples - m_star *
                            (game.end_level - game.start_level) * this.num_features));
                }
                else{
                    allSamples = (int) Math.ceil((1.0 - game.mu_f) * (allSamples - m_star *
                            (game.end_level - game.start_level) * Info.num_of_features_airport));
                }
                break;
            case "voting":   //(#feature = 51)
                if (Info.setting <= 100) {
                    allSamples = (int) Math.ceil((1.0 - game.mu_f) * allSamples);   m_star++;
                } else {
                    allSamples = (int) Math.ceil((1.0 - game.mu_f) * (allSamples - m_star *
                            (game.end_level - game.start_level) * Info.num_of_features_voting));
                }
                break;
            case "bank":   // (#feature = 16)
                allSamples = (int) Math.ceil((1.0 - game.mu_f) * (allSamples - m_star *
                        (game.end_level - game.start_level) * Info.num_of_features_bank));
                break;
            case "health":   //(#feature = 39)
                allSamples = (int) Math.ceil((1.0 - game.mu_f) * (allSamples - m_star *
                        (game.end_level - game.start_level) * Info.num_of_features_health));
                break;
            default:
                allSamples = (int) Math.ceil((1.0 - game.mu_f) * (allSamples - m_star *
                        (game.end_level - game.start_level) * Info.num_of_features));
                break;
        }
        allSamples = Math.max(1,  allSamples);

        //allocate the number of evaluations by the variance of each layer
        this.cheLevelInd = allo.sampleAllocation(this.num_features, this.levelMatrix, m_star, allSamples, game, random, shap_matrix);

        // save the record
        this.allCoalitions = coalitionsArr(game, allo, coaSet);

        for(int ind=2; ind<game.start_level;ind++){
            for(int i=0; i<this.num_features; i++) {
                shap_matrix[ind][i].sum += 0.0;
                shap_matrix[ind][i].count++;
                shap_matrix[ind][i].record.add(0.0);
            }
        }
        for(int ind=this.num_features-game.start_level; ind<this.num_features; ind++){
            for(int i=0; i<this.num_features; i++) {
                shap_matrix[ind][i].sum += 0.0;
                shap_matrix[ind][i].count++;
                shap_matrix[ind][i].record.add(0.0);
            }
        }
    }

    private double[][] initLevMat(double[][] evaluateMatrix, GameClass game, ArrayList[][] coalSet) {
        double[][] levelMatrix = new double[this.num_features][];
        for(int ind =0; ind<this.num_features; ind ++){
            levelMatrix[ind] = new double[this.num_features];
        }
        for(int i=0; i<this.num_features; i++){
            levelMatrix[0][i] = evaluateMatrix[i][i];
        }
        for(int step=0; step<this.num_features-1; step++){
            levelMatrix[1][step] = evaluateMatrix[step][step+1];
        }
        for(int ind=0; ind<this.num_features; ind++){
            int layer = 2;
            ArrayList<Integer> subset = new ArrayList<>();
            subset.add(ind);
            int ele = ind+1;
            if(ele < this.num_features){
                subset.add(ele);
            }
            else{
                ele = ind+1-this.num_features;
                subset.add(ele);
            }
            ele ++;
            for(int j=layer; j < this.num_features; j++){
                if(ele >= this.num_features){
                    ele = ele-this.num_features;
                }
                subset.add(ele);
                ele ++;
                levelMatrix[j][ind] = game.gameValue(this.model, subset);
                coalSet[j][ind] = new ArrayList(subset);
                layer ++;
            }
        }
        return levelMatrix;
    }

    private double calculateVariance(ArrayList<Double> list) {
        int n = list.size();
        if (n == 0) {
            return 0;
        }
        double mean = 0;
        for (double num : list) {
            mean += num;
        }
        mean /= n;
        double variance = 0;
        if(mean == 0){
            for (double num : list) {
                variance += Math.abs((num - mean));
            }
        }
        else{
            for (double num : list) {
                variance += Math.abs((num - mean) / mean);
            }
        }
        variance /= n;
        return variance;
    }

    private void initialization(GameClass game, boolean gene_weight, String modelName, ArrayList<Integer>[][] coaSet) {
        this.num_features = game.num_features; //the number of features
        this.exact = game.exact;   // the exact shapley value
        this.num_samples = Info.total_samples_num;  //the number of evaluations
        this.model = modelName;  // the game class
        this.cheLevelInd = 2;
        this.evaluateMatrix = initEvalMat(game);
        this.levelMatrix = initLevMat(this.evaluateMatrix, game, coaSet);
        this.allCoalitions = new FeatureSubset[this.num_features][];
    }

    private double[] ShapleyApproximateFLASH(GameClass game, ShapMatrixEntry[][] shap_matrix,
                                             ArrayList<Integer>[][] coaSet, Allocation allo, Random random) {
        // 0.the initial estimation
        initialEstimation(shap_matrix, coaSet, allo, game, random);

        // 1.the layer-wise evaluation
        layerWise(game, shap_matrix, allo, random);

        // 2.the feature-wise evaluation
        double[] sv = featureWise(shap_matrix, game, random);

        return sv;
    }

    // the layer-wise evaluation
    private void layerWise(GameClass game, ShapMatrixEntry[][] shap_matrix, Allocation allo, Random random) {
        //compute the contributions of the first two layers in matrix
        computeMatrix(game, shap_matrix, this.evaluateMatrix);

        FeatureSubset[] coalitionSet = initialLevel(game, this.evaluateMatrix, this.cheLevelInd, allo, random);
        this.allCoalitions[this.cheLevelInd-1] = coalitionSet;
        FeatureSubset[] coalitionSet_start = new FeatureSubset[0];

        //initialize and compute the contributions of features within layers
        for(int ind= game.start_level; ind <this.cheLevelInd; ind++){   // for each layer
            computeNextLevel(game, ind, coalitionSet_start, shap_matrix, allo, random);   //initialization
        }
        for(int ind= this.cheLevelInd; ind <this.num_features; ind++){  // for each layer
            if(coalitionSet.length == 0){
                coalitionSet = initialLevel(game, this.evaluateMatrix, ind, allo, random);  //find the first layer
            }
            coalitionSet = computeNextLevel(game, ind, coalitionSet, shap_matrix, allo, random);  //compute next layer
        }
    }

    //the feature-wise evaluation
    private double[] featureWise(ShapMatrixEntry[][] shap_matrix, GameClass game, Random random) {
        double[][] variance_level_fea = new double[this.num_features][this.num_features];
        for(int lev = game.start_level; lev<game.end_level; lev++){
            for(int i=0; i<this.num_features; i++){
                variance_level_fea[lev][i] = calculateVariance(shap_matrix[lev][i].record);   //compute the variance
            }
        }
        double[] layer_variance = new double[this.num_features];  //the average variance of feature i in each layer
        double[] layer_variance_sum = new double[this.num_features];  //the variance sum of feature i in each layer
        for(int i = 0; i < this.num_features; i++){ // 遍历每个feature
            int count = 0;
            for(int len = game.start_level; len<game.end_level; len++){
                layer_variance_sum[i] += variance_level_fea[len][i];
                count ++;
            }
            layer_variance[i] = layer_variance_sum[i] / count;  //the variance of feature i in each layer
        }

        //determine the number of k (top-k features)
        int top_k_feature_num = (int) Math.max(1, Math.ceil(game.mu_n * this.num_features));
        //PriorityQueue for the top-k features
        PriorityQueue<Integer> checkFeaSet = findTopKFeatures(top_k_feature_num, layer_variance);
        //the number of evaluations for each features in k
        int evaluations_num = (int) Math.ceil(this.num_samples * game.mu_f / top_k_feature_num);
        for(Integer fea : checkFeaSet){  // for each feature in the top-k features

            //allocate the number of evaluations by the variance
            int[] alloSamples = allocationByVar(fea, layer_variance_sum, evaluations_num, variance_level_fea, game, layer_variance);

            // compute the marginal contributions
            compMarContribution(fea, alloSamples, game, shap_matrix, random);
        }
        return meanShapleyValue(shap_matrix);
    }

    private void compMarContribution(int fea, int[] alloSamples, GameClass game, ShapMatrixEntry[][] shap_matrix, Random random) {
        for(int level=game.start_level; level<game.end_level; level++){   //for each level
            int count = alloSamples[level];  //the number of evaluations in this layer (L_level)
            while(count > 0){
                FeatureSubset[] set = this.allCoalitions[level];
                int index = random.nextInt(set.length);
                ArrayList<Integer> subset_1 = set[index].name;
                double value_1 = set[index].value_fun;
                double value_2 = 0;
                if(subset_1.contains(fea)){
                    ArrayList<Integer> subset_2 = new ArrayList<>(subset_1);
                    subset_2.remove(subset_1.indexOf(fea));
                    value_2 = game.gameValue(this.model, subset_2);
                    shap_matrix[level][fea].sum += value_1 - value_2;
                    shap_matrix[level][fea].count ++;
                }
                else{
                    ArrayList<Integer> subset_2 = new ArrayList<>(subset_1);
                    int index_2 = random.nextInt(subset_1.size());
                    subset_2.remove(index_2);
                    subset_1.remove(index_2);
                    subset_1.add(fea);
                    value_1 = game.gameValue(this.model, subset_1);
                    value_2 = game.gameValue(this.model, subset_2);
                    shap_matrix[level][fea].sum += value_1 - value_2;
                    shap_matrix[level][fea].count ++;
                }
                count --;
            }

        }
    }

    // Allocate the number of evaluations by the variance
    private int[] allocationByVar(Integer fea, double[] layer_variance_sum, int evaluations_num,
                                  double[][] variance_level_fea, GameClass game, double[] layer_variance) {

        int[] alloSamples = new int[this.num_features];  //the number of evaluations in each layer

        if(layer_variance[fea] == 0 ){  //Exceptional Case: the variance is 0，uniform allocation
            int temp = (int) Math.ceil((double) evaluations_num / (game.end_level - game.start_level));
            for(int ind=game.start_level; ind< game.end_level; ind++){
                alloSamples[ind]  = temp;
            }
        }
        else{
            double totalVar = layer_variance_sum[fea];
            for(int ind = 1; ind < this.num_features-1; ind++) {
                alloSamples[ind] = (int) Math.ceil(evaluations_num * variance_level_fea[ind][fea] / totalVar);
            }
        }
        return alloSamples;
    }

    private PriorityQueue<Integer> findTopKFeatures(int k, double[] array) {
        // minHeap: the set for the top-k features with the highest variance
        PriorityQueue<Integer> minHeap = new PriorityQueue<>(k, Comparator.comparingDouble(i -> array[i]));
        for (int i = 0; i < array.length; i++) {
            if (minHeap.size() < k) {
                minHeap.offer(i);
            }
            else if (array[i] > array[minHeap.peek()]) {
                minHeap.poll();
                minHeap.offer(i);
            }
        }
        return minHeap;
    }

    //init and generate a matrix
    public double[][] initEvalMat(GameClass game) {
        double[][] matrix = new double[this.num_features][this.num_features];
        for(int i=0; i<this.num_features; i++){
            ArrayList<Integer> subset = new ArrayList<>();
            subset.add(i);  //单个特征联盟，1-联盟
            matrix[i][i] = game.gameValue(this.model, subset);
            for (int j = i + 1; j < this.num_features; j++) {
                ArrayList<Integer> twoCoalition = new ArrayList<>(subset);
                twoCoalition.add(j);
                matrix[i][j] = matrix[j][i] = game.gameValue(this.model, twoCoalition);
            }
        }
        return matrix;
    }

    private int checkLevel_start(double[][] levelMatrix) {
        int line_ind = 2;
        for(int step = 0; step < this.num_features-1; step ++){
            double line_max = levelMatrix[step+1][0] - levelMatrix[step][0];
            double line_min = levelMatrix[step+1][this.num_features-1] - levelMatrix[step][this.num_features-1];
            for(int i=0; i < this.num_features; i++){
                if(line_max < levelMatrix[step+1][i]- levelMatrix[step][i]){
                    line_max = levelMatrix[step+1][i]- levelMatrix[step][i];
                }
                else if(line_min > levelMatrix[step+1][i]- levelMatrix[step][i]){
                    line_min = levelMatrix[step+1][i]- levelMatrix[step][i];
                }
            }
            if(line_max != line_min){
                line_ind = step;
                break;
            }
        }
        return line_ind;
    }

    private int checkLevel_end(double[][] levelMatrix) {
        int line_ind = 2;
        for(int step = this.num_features-1; step >0; step --){
            double line_max = levelMatrix[step][0] - levelMatrix[step-1][0];
            double line_min = levelMatrix[step][this.num_features-1] - levelMatrix[step-1][this.num_features-1];
            for(int i=0; i < this.num_features; i++){
                if(line_max < levelMatrix[step][i]- levelMatrix[step-1][i]){
                    line_max = levelMatrix[step][i]- levelMatrix[step-1][i];
                }
                else if(line_min > levelMatrix[step][i]- levelMatrix[step-1][i]){
                    line_min = levelMatrix[step][i]- levelMatrix[step-1][i];
                }
            }
            if(line_max != line_min){
                line_ind = step;
                break;
            }
        }
        return line_ind;
    }

    private void computeMatrix(GameClass game, ShapMatrixEntry[][] shap_matrix, double[][] evaluateMatrix){
        double one_feature_sum = 0;
        double org_value = game.gameValue(this.model, new ArrayList<>());
        for (int i = 0; i < this.num_features; i++) {
            double value = evaluateMatrix[i][i] - org_value;
            shap_matrix[0][i].sum += value;
            shap_matrix[0][i].count++;
            shap_matrix[0][i].record.add(value);
            one_feature_sum += evaluateMatrix[i][i];
        }
        for (int i = 0; i < this.num_features; i++) {
            double line_sum = 0;
            for (int j = 0; j < this.num_features; j++) {
                line_sum += evaluateMatrix[i][j];
            }
            double value =  (line_sum - one_feature_sum) / (this.num_features - 1);
            shap_matrix[1][i].sum += value;
            shap_matrix[1][i].count ++;
            shap_matrix[1][i].record.add(value);
        }
    }

    // Define the initial layer
    private FeatureSubset[] initialLevel(GameClass game, double[][] evaluateMatrix, int level_index, Allocation allo, Random random) {
        FeatureSubset[] twoCoalition_set;
        if(level_index > 2){
            // randomly select m elements of length k from n elements
            twoCoalition_set = randomSubsetsArr(random, game, allo.num_sample[level_index], level_index);
        }
        else{
            twoCoalition_set = new FeatureSubset[this.num_features * (this.num_features-1) / 2];
            int index = 0;
            for (int i = 0; i < this.num_features; i++) {
                List<Integer> subSet = new ArrayList<>();
                subSet.add(i);
                for (int j = i + 1; j < this.num_features; j++) {
                    ArrayList<Integer> newSubset = new ArrayList<>(subSet);
                    newSubset.add(j);
                    FeatureSubset ele = new FeatureSubset(newSubset, evaluateMatrix[i][j]);
                    twoCoalition_set[index] = new FeatureSubset(new ArrayList<>(), 0);
                    twoCoalition_set[index] = ele;
                    index ++;
                }
            }
        }
        return twoCoalition_set;
    }

    private FeatureSubset[][] coalitionsArr(GameClass game, Allocation allo, ArrayList<Integer>[][] coaSet) {
        for(int level= game.start_level; level<this.num_features-1; level++){
            this.allCoalitions[level] = new FeatureSubset[allo.num_sample[level] + this.num_features];
            for(int i=0; i<allo.num_sample[level]; i++){
                this.allCoalitions[level][i] = new FeatureSubset(new ArrayList<Integer>(), 0.0);
            }
            for(int ind=0; ind< this.num_features; ind++){
                this.allCoalitions[level][allo.num_sample[level] + ind] = new FeatureSubset(coaSet[level][ind], this.levelMatrix[level][ind]);
            }
        }
        this.allCoalitions[this.num_features-1] = new FeatureSubset[1];
        this.allCoalitions[this.num_features-1][0] = new FeatureSubset(coaSet[this.num_features-1][0], this.levelMatrix[this.num_features-1][0]);
        return this.allCoalitions;
    }

    public FeatureSubset[] randomSubsetsArr(Random random, GameClass game, int m, int len){
        FeatureSubset[] subsets = new FeatureSubset[m];
        for (int i = 0; i < m; i++) {
            Set<Integer> subset = new HashSet<>();
            while (subset.size() < len) {
                subset.add(random.nextInt(this.num_features));
            }
            ArrayList<Integer> name = new ArrayList<>(subset);
            double value = game.gameValue(this.model, name);
            FeatureSubset ele = new FeatureSubset(name, value);
            subsets[i] = ele;
        }
        return subsets;
    }

    private FeatureSubset[] computeNextLevel(GameClass game, int level, FeatureSubset[] coalitionSet,
                                             ShapMatrixEntry[][] shap_matrix, Allocation all, Random random) {
        //1.define a set
        int GenCoalitionsNum = all.num_sample[level];
        FeatureSubset[] generationSet = new FeatureSubset[GenCoalitionsNum];   // coalitions in this layer
        ShapMatrixEntry[] temp = new ShapMatrixEntry[this.num_features];
        for(int i=0; i<this.num_features; i++){
            temp[i] = new ShapMatrixEntry();
        }
        if (GenCoalitionsNum == 0) {
            for(int i=0; i<this.num_features; i++) {
                shap_matrix[level][i].sum += 0.0;
                shap_matrix[level][i].count++;
                shap_matrix[level][i].record.add(0.0);
            }
        }
        else {
            //2.initialize the set
            for (int i = 0; i < GenCoalitionsNum; i++) {
                generationSet[i] = new FeatureSubset(new ArrayList<Integer>(), 0.0);
            }
            //3.sampling and calculating
            int count = 0;
            while (count < GenCoalitionsNum) {
                //1) select a coalition in the last layer
                FeatureSubset random_sample = randomGet(coalitionSet, random);
                //2) select a feature to form a new coalition
                int i = random.nextInt(this.num_features);
                if (!random_sample.name.contains(i)) {
                    ArrayList<Integer> name = new ArrayList<>(random_sample.name);
                    name.add(i);
                    double value = game.gameValue(this.model, name);
                    FeatureSubset newFeaSub = new FeatureSubset(name, value);
                    //3）compute the contribution and record in matrix
                    temp[i].sum += value - random_sample.value_fun;
                    temp[i].count++;
                    temp[i].record.add(value - random_sample.value_fun);
                    shap_matrix[level][i].sum += value - random_sample.value_fun;
                    shap_matrix[level][i].count++;
                    shap_matrix[level][i].record.add(value - random_sample.value_fun);
                    //4）put the coalitions in the generationSet for the next layer evaluation
                    generationSet[count] = newFeaSub;
                    if (count < this.allCoalitions[level].length) {
                        this.allCoalitions[level][count] = newFeaSub;
                    }
                    count++;
                }
            }
        }
        return generationSet;
    }

    //Randomly select a non-empty sample from the set
    public FeatureSubset randomGet(FeatureSubset[] aGrid, Random random){
        int index;
        do {
            index = random.nextInt(aGrid.length);
        }
        while (aGrid[index] == null);
        return aGrid[index];
    }

    private double[] meanShapleyValue(ShapMatrixEntry[][] shap_matrix) {
        double[] sv = new double[this.num_features];
        int[] count = new int[this.num_features];
        for(int len=0; len<this.num_features; len++){
            for(int fea=0; fea<this.num_features; fea++){
                ShapMatrixEntry entry = shap_matrix[len][fea];
                if(entry.count != 0){
                    entry.sum = entry.sum / entry.count;
                    sv[fea] += entry.sum;
                    count[fea] ++;
                }
            }
        }
        for(int fea=0; fea<this.num_features; fea++){
            sv[fea] = sv[fea] / count[fea];
        }
        return sv;
    }
}