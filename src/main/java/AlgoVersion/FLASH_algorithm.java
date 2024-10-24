package AlgoVersion;

import Game.GameClass;
import Global.Allocation;
import Global.Comparer;
import Global.FeatureSubset;
import Global.Info;
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
    private int check_level_index = 2;

    private double[][] initLevMat(double[][] evaluateMatrix, GameClass game, ArrayList[][] coalSet) {
        //1）构造矩阵
        double[][] levelMatrix = new double[this.num_features][];
        for(int ind =0; ind<this.num_features; ind ++){
            levelMatrix[ind] = new double[this.num_features];
        }
        //2)第一层赋值,长度为1
        for(int i=0; i<this.num_features; i++){
            levelMatrix[0][i] = evaluateMatrix[i][i];
        }
        //3)继续赋值
        for(int step=0; step<this.num_features-1; step++){
            levelMatrix[1][step] = evaluateMatrix[step][step+1];
        }
        //4) 按列竖着加item, 从第3行开始，所以star = 2
        for(int ind=0; ind<this.num_features; ind++){  //第0列，第2行
            int layer = 2;
            ArrayList<Integer> subset = new ArrayList<>();
            subset.add(ind);  //单个特征联盟，1-联盟
            int ele = ind+1;
            if(ele < this.num_features){
                subset.add(ele);  //单个特征联盟，1-联盟
            }
            else{  //反正给它凑够俩item，并且循环回去了
                ele = ind+1-this.num_features;
                subset.add(ele);  //单个特征联盟，1-联盟
            }
            ele ++;
            for(int j=layer; j < this.num_features; j++){
                if(ele >= this.num_features){
                    ele = ele-this.num_features; //重新开始
                }
                subset.add(ele);  //单个特征联盟，1-联盟
                ele ++;
                levelMatrix[j][ind] = game.gameValue(this.model, subset);
                coalSet[j][ind] = new ArrayList(subset);  //(ArrayList<Integer>) subset.clone()
                layer ++;
            }
        }
        return levelMatrix;
    }

    public void FLASH_Shap(boolean gene_weight, String model_name) {

        //Initialization
        GameClass game = new GameClass();
        game.gameInit(gene_weight, model_name);
        ArrayList<Integer>[][] coaSet = new ArrayList[game.num_features][game.num_features];
        initialization(game, gene_weight, model_name, coaSet);

        long ave_runtime = 0;
        double ave_error_max = 0;
        double ave_error_ave = 0;
        for(int i=0; i< Info.timesRepeat; i++) {
            this.num_samples = Info.total_samples_num;
            Random random = new Random(game.seedSet[i]);
            Allocation allo = new Allocation(game);
            ShapMatrixEntry[][] shap_matrix = new ShapMatrixEntry[this.num_features][]; //shap_matrix[len][feature]
            for (int ii = 0; ii < this.num_features; ii++) {
                shap_matrix[ii] = new ShapMatrixEntry[this.num_features];
                for (int j = 0; j < this.num_features; j++) {
                    shap_matrix[ii][j] = new ShapMatrixEntry();
                }
            }

            long time_1 = System.currentTimeMillis();
            double[] sv = ShapleyApproximateFLASH(game, shap_matrix, coaSet, allo, random);
            long time_2 = System.currentTimeMillis();
            ave_runtime += time_2 - time_1;

            // Estimate error
            Comparer comp = new Comparer();
            double error_max = comp.computeMaxError(sv, this.exact);
            double error_ave = comp.computeAverageError(sv, this.exact, this.num_features);
            ave_error_max += error_max;
            ave_error_ave += error_ave;
            System.out.println("run: " + i + " " + "error_ave: " + error_ave + " \t"  +  "error_max: " + error_max );
        }
        System.out.println(model_name + ":  " + "error_ave: " + ave_error_ave/Info.timesRepeat + " \t"  +  "error_max: " + ave_error_max/Info.timesRepeat);
        System.out.println("FLASH time : " + (ave_runtime * 0.001)/ Info.timesRepeat );
    }

    public void FLASH_scale(boolean gene_weight, String model_name) {

        GameClass game = new GameClass();
        game.gameInit(gene_weight, model_name);
        ArrayList<Integer>[][] coaSet = new ArrayList[game.num_features][game.num_features];
        initialization(game, gene_weight, model_name, coaSet);
        Comparer comp = new Comparer();

        long ave_runtime = 0;
        double ave_error_max = 0;
        double ave_error_ave = 0;
        double[][] sv_values = new double[Info.timesRepeat][this.num_features];  //[runs][feature]
        for(int i=0; i< Info.timesRepeat; i++) {
            this.num_samples = Info.total_samples_num;
            Random random = new Random(game.seedSet[i]);
            Allocation allo = new Allocation(game);
            ShapMatrixEntry[][] shap_matrix = new ShapMatrixEntry[this.num_features][]; //shap_matrix[len][feature]
            for (int ii = 0; ii < this.num_features; ii++) {   // 初始化二维数组的每个元素
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
            //-----------------------------------------------------------------------------------------------------------
            ave_runtime += time_2 - time_1;
            ave_error_max += error_max;
            ave_error_ave += error_ave;
            System.out.println("run: " + i + " " + "error_ave: " + error_ave + " \t"  +  "error_max: " + error_max );
        }
        double acv = comp.computeACV(sv_values, this.num_features);
        System.out.println(model_name + ":  " + "error_ave: " + ave_error_ave/Info.timesRepeat + " \t"  +  "error_max: " + ave_error_max/Info.timesRepeat);
        System.out.println("average cv :" + acv);
        System.out.println("FLASH time : " + (ave_runtime * 0.001)/ Info.timesRepeat );
    }

    private void initialEstimation(ShapMatrixEntry[][] shap_matrix, ArrayList<Integer>[][] coaSet, Allocation allo, GameClass game, Random random) {
        game.start_level = Math.max(game.start_level, checkLevel_start(this.levelMatrix) + 1);
        game.end_level = checkLevel_end(this.levelMatrix);
        int m_star = Math.max(2, this.num_samples / (this.num_features * this.num_features * 2));
        int allSamples = this.num_samples;

        switch (this.model) {
            case "airport":   // (#feature = 100)
                allSamples -= Info.num_of_features_airport*Info.num_of_features_airport/2 - Info.num_of_features_airport/2;
                if(Info.is_gene_weight){   //for scalability test: the number of features is setting by [this.num_features]
                    allSamples = (int) Math.ceil((1.0 - game.check_weight) * (allSamples - m_star * (game.end_level - game.start_level) * this.num_features));
                }
                else{
                    allSamples = (int) Math.ceil((1.0 - game.check_weight) * (allSamples - m_star * (game.end_level - game.start_level) * Info.num_of_features_airport));
                }
                break;
            case "voting":   //(#feature = 51)
                if (Info.setting <= 100) {
                    allSamples = (int) Math.ceil((1.0 - game.check_weight) * allSamples);   m_star++;
                } else {
                    allSamples = (int) Math.ceil((1.0 - game.check_weight) * (allSamples - m_star * (game.end_level - game.start_level) * Info.num_of_features_voting));
                }
                break;
            case "bank":   // (#feature = 16)
                allSamples = (int) Math.ceil((1.0 - game.check_weight) * (allSamples - m_star * (game.end_level - game.start_level) * Info.num_of_features_bank));
                break;
            case "health":   //(#feature = 39)
                allSamples = (int) Math.ceil((1.0 - game.check_weight) * (allSamples - m_star * (game.end_level - game.start_level) * Info.num_of_features_health));
                break;
            default:
                allSamples = (int) Math.ceil((1.0 - game.check_weight) * (allSamples - m_star * (game.end_level - game.start_level) * Info.num_of_features));
                break;
        }
        allSamples = Math.max(1,  allSamples);

        //allocate the number of evaluations by the variance of each layer
        this.check_level_index = allo.sampleAllocation(this.num_features, this.levelMatrix, m_star, allSamples, game, random, shap_matrix);

        // save the record
        this.allCoalitions = coalitionsArr(game, allo, coaSet);

        for(int ind=2; ind<game.start_level;ind++){
            for(int i=0; i<this.num_features; i++) {
                shap_matrix[ind][i].sum += 0.0;
                shap_matrix[ind][i].count++;
                shap_matrix[ind][i].record.add(0.0);
            }
        }
        for(int ind=this.num_features-game.start_level; ind<this.num_features; ind++){   //this.num_features-start_level-1
            for(int i=0; i<this.num_features; i++) {
                shap_matrix[ind][i].sum += 0.0;
                shap_matrix[ind][i].count++;
                shap_matrix[ind][i].record.add(0.0);
            }
        }
    }

    private double calculateVariance_2(ArrayList<Double> list) {
        int n = list.size();
        if (n == 0) {
            return 0;
        }
        // 计算平均值
        double mean = 0;
        for (double num : list) {
            mean += num;
        }
        mean /= n;
        // 计算每个元素与平均值的差的平方的平均值
        double variance = 0;
        if(mean == 0){
            for (double num : list) {
                variance += Math.abs((num - mean));  //传统方差公式
            }
        }
        else{
            for (double num : list) {
                variance += Math.abs((num - mean) / mean);  //新的方差公式
            }
        }
        variance /= n;
        return variance;
    }

    //全局参数的初始化
    private void initialization(GameClass game, boolean gene_weight, String modelName, ArrayList<Integer>[][] coaSet) {
//        game.gameInit(gene_weight, modelName);
        this.num_features = game.num_features; //the number of features
        this.exact = game.exact;   // the exact shapley value
        this.num_samples = Info.total_samples_num;
        this.model = modelName;
        this.evaluateMatrix = initEvalMat(game);
        this.levelMatrix = initLevMat(this.evaluateMatrix, game, coaSet);
        this.allCoalitions = new FeatureSubset[this.num_features][];
    }

    private double[] ShapleyApproximateFLASH(GameClass game, ShapMatrixEntry[][] shap_matrix, ArrayList<Integer>[][] coaSet, Allocation allo, Random random) {
        // 0) the initial estimation
        initialEstimation(shap_matrix, coaSet, allo, game, random);

        // 1) the layer-wise evaluation
        layerWise(game, shap_matrix, allo, random);

        // 2) the feature-wise evaluation
        double[] sv = featureWise(shap_matrix, game, random);

        return sv;
    }

    //TODO the layer-wise evaluation
    private void layerWise(GameClass game, ShapMatrixEntry[][] shap_matrix, Allocation allo, Random random) {
        computeMatrix(game, shap_matrix, this.evaluateMatrix);

        FeatureSubset[] coalitionSet = initialLevel(game, this.evaluateMatrix, this.check_level_index, allo, random);  //level =2 （存储长度为3的特征子集）
        this.allCoalitions[this.check_level_index-1] = coalitionSet;
        FeatureSubset[] coalitionSet_start = new FeatureSubset[0];

        //4)依次生成联盟 (ind / level 等于被减去的联盟的长度)
        for(int ind= game.start_level; ind <this.check_level_index; ind++){   // for each layer
            computeNextLevel(game, ind, coalitionSet_start, shap_matrix, allo, random);   //纯填充
        }
        for(int ind= this.check_level_index; ind <this.num_features; ind++){  // for each layer
            if(coalitionSet.length == 0){
                coalitionSet = initialLevel(game, this.evaluateMatrix, ind, allo, random);  //find the first layer
            }
            coalitionSet = computeNextLevel(game, ind, coalitionSet, shap_matrix, allo, random);  //compute next layer
        }
    }

    //TODO the feature-wise evaluation
    private double[] featureWise(ShapMatrixEntry[][] shap_matrix, GameClass game, Random random) {

        //计算 feature-i在不同层的方差 （计算方差大的feature 需要重新计算）
        double[][] variance_level_fea = new double[this.num_features][this.num_features];
//        int[][] variance_count = new int[this.num_features][this.num_features];
        for(int lev = game.start_level; lev<game.end_level; lev++){  //每层：表示不同长度
            for(int i=0; i<this.num_features; i++){   //层内的方差
                variance_level_fea[lev][i] = calculateVariance_2(shap_matrix[lev][i].record);   //计算方差
//                variance_count[lev][i] = shap_matrix[lev][i].count;
            }
        }

        //每个feature-i，在每层的方差，求均值（variance_level_fea[len][i]：L_k层中，i的方差）
        double[] layer_variance = new double[this.num_features];  // 1）i 在每层的方差的均值： 用于挑选key_features
        double[] layer_variance_sum = new double[this.num_features];  // 2）i 在每层的方差的和: findLargestFeatures()用于样本分配
        for(int i = 0; i < this.num_features; i++){ // 遍历每个feature
            int count = 0;  //计算了多少有效的层
            for(int len = game.start_level; len<game.end_level; len++){  //第一层和最后一层只有一个数，方差是0
                layer_variance_sum[i] += variance_level_fea[len][i];
                count ++;
            }
            layer_variance[i] = layer_variance_sum[i] / count;  //i在不同层的方差
        }


        //determine the number of k (top-k features)
        int top_k_feature_num = (int) Math.max(1, Math.ceil(game.key_features_weight * this.num_features));
        //PriorityQueue for the top-k features
        PriorityQueue<Integer> checkFeaSet = findTopKFeatures(top_k_feature_num, layer_variance);
        //the number of evaluations for each features in k
        int evaluations_num = (int) Math.ceil(this.num_samples * game.check_weight / top_k_feature_num);
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

    //TODO: allocate the number of evaluations by the variance
    private int[] allocationByVar(Integer fea, double[] layer_variance_sum, int evaluations_num, double[][] variance_level_fea, GameClass game, double[] layer_variance) {

        int[] alloSamples = new int[this.num_features];  //给每一层分配样本

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

    //TODO 构建evaluateMatrix, 记录1-联盟 & 2-联盟的值
    public double[][] initEvalMat(GameClass game) {
        double[][] matrix = new double[this.num_features][this.num_features];
        for(int i=0; i<this.num_features; i++){   //i: 横坐标 (也是第1个item)
            ArrayList<Integer> subset = new ArrayList<>();
            subset.add(i);  //单个特征联盟，1-联盟
            matrix[i][i] = game.gameValue(this.model, subset);
            for (int j = i + 1; j < this.num_features; j++) {   //j: 纵坐标(也是第2个item)
                ArrayList<Integer> twoCoalition = new ArrayList<>(subset);
                twoCoalition.add(j);
                matrix[i][j] = matrix[j][i] = game.gameValue(this.model, twoCoalition);  //复制两份
            }
        }
        return matrix;
    }


    //TODO 判断从哪层开始计算 //新的LM
    private int checkLevel_start(double[][] levelMatrix) {
        int line_ind = 2;
        //检查一行一行
        for(int step = 0; step < this.num_features-1; step ++){  //line_ind 是第几行
            double line_max = levelMatrix[step+1][0] - levelMatrix[step][0];   // 这层mc最小值
            double line_min = levelMatrix[step+1][this.num_features-1] - levelMatrix[step][this.num_features-1];    // 这层mc最大值
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
                break;  //只是为了检测是否为0
            }
        }
        return line_ind;   //返回line_ind = 9; 第9层，存储长度为10的，
        // level =10, 长度为11的发生了变换,在L9时用 10-9 被检测出来；
    }

    //TODO 判断哪层计算结束 //新的LM
    private int checkLevel_end(double[][] levelMatrix) {
        int line_ind = 2;
        //检查一行一行
        for(int step = this.num_features-1; step >0; step --){  //line_ind 是第几行
            double line_max = levelMatrix[step][0] - levelMatrix[step-1][0];   // 这层mc最小值
            double line_min = levelMatrix[step][this.num_features-1] - levelMatrix[step-1][this.num_features-1];    // 这层mc最大值
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
                break;  //只是为了检测是否为0
            }
        }
        return line_ind;   //返回line_ind = 9; 第9层，存储长度为10的，
        // level =10, 长度为11的发生了变换,在L9时用 10-9 被检测出来；
    }

    private void computeMatrix(GameClass game, ShapMatrixEntry[][] shap_matrix, double[][] evaluateMatrix){

        double one_feature_sum = 0; //记录对角线元素的值
        double org_value = game.gameValue(this.model, new ArrayList<Integer>()); //表示空集的值

        // 1）读取1-联盟的shapley value
        for (int i = 0; i < this.num_features; i++) {
            double value = evaluateMatrix[i][i] - org_value;
            shap_matrix[0][i].sum += value;  //对角线上的元素依次填入shapley value[]的矩阵中
            shap_matrix[0][i].count++;   //被减的coalition(后项)长度为0
            shap_matrix[0][i].record.add(value);
            one_feature_sum += evaluateMatrix[i][i];  //对角线求和
        }
        // 2) 读取和计算2-联盟的shapley value
        for (int i = 0; i < this.num_features; i++) {  //i 是横坐标  一行就对应一个特征
            double line_sum = 0;
//            ArrayList<Double> list = new ArrayList<>();
            for (int j = 0; j < this.num_features; j++) {  //j是纵坐标
                line_sum += evaluateMatrix[i][j];
//                list.add(evaluateMatrix[i][j] - evaluateMatrix[j][j]);
            }
//            this.variance_level_fea[1][i] = calculateVariance_2(list);  //表示后项被减coalition长度为1
//            this.variance_count[1][i] = this.num_features - 1;
            double value =  (line_sum - one_feature_sum) / (this.num_features - 1);  //第二层的shapley value
            shap_matrix[1][i].sum += value;  //第二层的shapley value
            shap_matrix[1][i].count ++;
            shap_matrix[1][i].record.add(value);
        }
    }

    //    TODO [省略构建网格] 计算初始层，并存储到网格中(3-联盟-2联盟的值) => 利用矩阵EvaluateMatrix
    private FeatureSubset[] initialLevel(GameClass game, double[][] evaluateMatrix, int level_index, Allocation allo, Random random) {
        //TODO 性质：矩阵是一个对称的矩阵，所以2-联盟只需要看一半，存一半就可以

        FeatureSubset[] twoCoalition_set;

        if(level_index > 2){
            //从某一层开始计算，在这层中随机生成若干个组合（按照分配方式）
            twoCoalition_set = randomSubsetsArr(random, game, allo.num_sample[level_index], level_index);  //n个元素中随机取出m个长度为k的元素
            //            randomSubsets(game, allo.num_sample[level_index], level_index, given_weights, halfSum);
        }

        // 情况2：正常的计算过程，没有可以被剪枝
        else{
            twoCoalition_set = new FeatureSubset[this.num_features * (this.num_features-1) / 2];
            // 1) 读取2-联盟的shapley value
            int index = 0;  //表示twoCoalition_set的数组下标
            for (int i = 0; i < this.num_features; i++) {  //i 是横坐标  一行就对应一个特征
                List<Integer> subSet = new ArrayList<>();
                subSet.add(i);  //第1个特征
                for (int j = i + 1; j < this.num_features; j++) {  //j是纵坐标
                    ArrayList<Integer> newSubset = new ArrayList<>(subSet);
                    newSubset.add(j);  //第2个特征
                    FeatureSubset ele = new FeatureSubset(newSubset, evaluateMatrix[i][j]);
                    twoCoalition_set[index] = new FeatureSubset(new ArrayList<Integer>(), 0); //初始化
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
        this.allCoalitions[this.num_features-1][0] = new FeatureSubset(coaSet[this.num_features-1][0], this.levelMatrix[this.num_features-1][0]); //赋值最后一层
        return this.allCoalitions;
    }

    //TODO 随机取出m个长度为len的元素
    public FeatureSubset[] randomSubsetsArr(Random random, GameClass game, int m, int len){
        FeatureSubset[] subsets = new FeatureSubset[m];  //所有生成的subset的集合
        for (int i = 0; i < m; i++) {
            Set<Integer> subset = new HashSet<>();  //一个生成的subset(其中元素不重复)
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

    private FeatureSubset[] computeNextLevel(GameClass game, int level, FeatureSubset[] coalitionSet, ShapMatrixEntry[][] shap_matrix, Allocation all, Random random) {
        //本层中抽样的数量
        int GenCoalitionsNum = all.num_sample[level];  //level从第2层开始，num_sample 表示每层的采样数量
        FeatureSubset[] generationSet = new FeatureSubset[GenCoalitionsNum];   // 记录当前层生成的特征子集
        // 1) 初始化
        // 记录这一层的shapley 值计算情况（因为每个特征对应的采样数可能不一致，需要结构体记录采样数量）
        ShapMatrixEntry[] temp = new ShapMatrixEntry[this.num_features];
        for(int i=0; i<this.num_features; i++){
            temp[i] = new ShapMatrixEntry();
        }

        //[添加]：先判断这层是否需要被计算
        if (GenCoalitionsNum == 0) {
            for(int i=0; i<this.num_features; i++) {
                shap_matrix[level][i].sum += 0.0;
                shap_matrix[level][i].count++;
                shap_matrix[level][i].record.add(0.0);
            }
        }
        else {  //没有被跳过，有样本就正常算
            // 2）一些初始化
            for (int i = 0; i < GenCoalitionsNum; i++) {
                generationSet[i] = new FeatureSubset(new ArrayList<Integer>(), 0.0);
            }
            //3）开始抽样计算
            int count = 0; //抽样计数器
            while (count < GenCoalitionsNum) {
                //1) 随机取出样本, 用于拓展成新的newFeaSub，并返回
                FeatureSubset random_sample = randomGet(coalitionSet, random);  //coalitionSet存储上一层的样本
                //2)选出来的这个FeatureSubset，与每个feature(不包含自己)构成一个新的FeatureSubset
                int i = random.nextInt(this.num_features);
                if (!random_sample.name.contains(i)) {   //挑选出的random_sample不包含当前的特征，就可以构成新的
                    ArrayList<Integer> name = new ArrayList<>(random_sample.name);
                    name.add(i);
                    double value = game.gameValue(this.model, name);
                    FeatureSubset newFeaSub = new FeatureSubset(name, value);
                    //3）计算shapley value 并填写到矩阵中
                    temp[i].sum += value - random_sample.value_fun; //V(S U i) - V(S), random_sample就是S
                    temp[i].count++;
                    temp[i].record.add(value - random_sample.value_fun);
                    //20240821补充：直接添加到shap_matrix
                    shap_matrix[level][i].sum += value - random_sample.value_fun; //V(S U i) - V(S), random_sample就是S
                    shap_matrix[level][i].count++;
                    shap_matrix[level][i].record.add(value - random_sample.value_fun);
                    //4）存储新生成特征子集S，为下一层选取做好准备 (只存储不重复的元素)
                    generationSet[count] = newFeaSub;
                    //将generationSet 拷贝到this.allCoalitions
                    if (count < this.allCoalitions[level].length) {
                        this.allCoalitions[level][count] = newFeaSub;
                    }
                    count++;
                }
            }
        }
        return generationSet;
    }

    //TODO [版本3]从集合中随机选出一个样本 (添加了不为空集的判断)
    public FeatureSubset randomGet(FeatureSubset[] aGrid, Random random){
        int index;
        // 从数组中随机选择一个非空的特征子集
        do {
            index = random.nextInt(aGrid.length);
        }
        while (aGrid[index] == null);
        return aGrid[index];
    }

    private double[] meanShapleyValue(ShapMatrixEntry[][] shap_matrix) {
        double[] sv = new double[this.num_features];
        int[] count = new int[this.num_features];
        //1）对ShapMatrixEntry[][]求均值
        for(int len=0; len<this.num_features; len++){
            for(int fea=0; fea<this.num_features; fea++){
                ShapMatrixEntry entry = shap_matrix[len][fea];
                if(entry.count != 0){
                    entry.sum = entry.sum / entry.count;
                    //                    System.out.println(fea + ": " + entry.count + ": " + entry.sum);
                    sv[fea] += entry.sum;
                    count[fea] ++;
                }
            }
        }
        //2）对sv[]求均值
        for(int fea=0; fea<this.num_features; fea++){
            sv[fea] = sv[fea] / count[fea];
        }
        return sv;
    }
}