package SeedVersion;

/* 记录时间，省略打印 */
import Game.GameClass;
import Global.Allocation;
import Global.Comparer;
import Global.FeatureSubset;
import Global.Info;
import structure.ShapMatrixEntry;

import java.util.*;

//优化了Voting的算法： computeNextLevelNOG_12() 跳过的层也需要被记录sv, sv=0

public class PSA_time {
    private int num_features;  //数据集中包含的特征数量
    private double[] exact;  // the exact shapley value
    private int num_samples;
    private FeatureSubset[][] allCoalitions;
    private String model;
    private double[][] evaluateMatrix;
    private double[][] newLevelMatrix;
    private int check_level_index = 2;

    private double[][] constructLevelMatrix_2(double[][] evaluateMatrix, GameClass game, ArrayList[][] coalSet) {
        //1）构造矩阵
        double[][] newLevelMatrix = new double[this.num_features][];  //存储对应长度的coalition
        for(int ind =0; ind<this.num_features; ind ++){
            newLevelMatrix[ind] = new double[this.num_features];
        }
        //2)第一层赋值,长度为1
        for(int i=0; i<this.num_features; i++){
            newLevelMatrix[0][i] = evaluateMatrix[i][i];
        }
        //3)继续赋值
        for(int step=0; step<this.num_features-1; step++){
            newLevelMatrix[1][step] = evaluateMatrix[step][step+1];
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
                newLevelMatrix[j][ind] = game.gameValue(this.model, subset);
                coalSet[j][ind] = new ArrayList(subset);  //(ArrayList<Integer>) subset.clone()
                layer ++;
            }
        }
        return newLevelMatrix;
    }

    //TODO 根据第一阶段计算得到的sv[]分配样本
    public void model_game_nog_0(boolean gene_weight, String model_name) {

        //【警告】：这四句话不要调换顺序！
        //1、初始化：根据game初始化 1)num_features； 2）given_weights； 3）exact； 4）num_samples；5）halfSum
        GameClass game = new GameClass();
        game.gameInit(gene_weight, model_name);
        ArrayList<Integer>[][] coaSet = new ArrayList[game.num_features][game.num_features];
        initialization(game, gene_weight, model_name, coaSet);

        //3、开始计算shapley value：
        long ave_runtime = 0;
        double ave_error_max = 0; //计算最大误差
        double ave_error_ave = 0;
        for(int i=0; i< Info.timesRepeat; i++) {
            this.num_samples = Info.total_samples_num;
            Random random = new Random(game.seedSet[i]); //this.localSet[i]
            Allocation allo = new Allocation(game);
            ShapMatrixEntry[][] shap_matrix = new ShapMatrixEntry[this.num_features][]; //shap_matrix[len][feature]
            for (int ii = 0; ii < this.num_features; ii++) {   // 初始化二维数组的每个元素
                shap_matrix[ii] = new ShapMatrixEntry[this.num_features];
                for (int j = 0; j < this.num_features; j++) {
                    shap_matrix[ii][j] = new ShapMatrixEntry();
                }
            }

            long time_1 = System.currentTimeMillis();
            double[] sv = ShapleyApproximateNog_0(game, shap_matrix, coaSet, allo, random);
            long time_2 = System.currentTimeMillis();

            // --------------------------------------------------计算误差------------------------------------------------
            Comparer comp = new Comparer();
            double error_max = comp.computeMaxError(sv, this.exact); //计算最大误差
            double error_ave = comp.computeAverageError(sv, this.exact, this.num_features);  //计算平均误差
//            double mse = comp.computeMSE(shap_matrix, this.exact, this.num_features);
            //-----------------------------------------------------------------------------------------------------------
            ave_runtime += time_2 - time_1;
            ave_error_max += error_max;
            ave_error_ave += error_ave;
//            ave_mse += mse;
//            System.out.println("run: " + i + " " + "error_ave: " + error_ave + " \t"  +  "error_max: " + error_max + " \t"  +  "mse: " + mse);
            System.out.println("run: " + i + " " + "error_ave: " + error_ave + " \t"  +  "error_max: " + error_max );
        }
        System.out.println(model_name + ":  " + "error_ave: " + ave_error_ave/Info.timesRepeat + " \t"  +  "error_max: " + ave_error_max/Info.timesRepeat);
        System.out.println("Shap_PGB_0 time : " + (ave_runtime * 0.001)/ Info.timesRepeat );  //+ "S"
    }

    //TODO 根据第一阶段计算得到的sv[]分配样本
    public void model_game_nog_scale(boolean gene_weight, String model_name) {

        //【警告】：这四句话不要调换顺序！
        //1、初始化：根据game初始化 1)num_features； 2）given_weights； 3）exact； 4）num_samples；5）halfSum
        GameClass game = new GameClass();
        game.gameInit(gene_weight, model_name);
        ArrayList<Integer>[][] coaSet = new ArrayList[game.num_features][game.num_features];
        initialization(game, gene_weight, model_name, coaSet);
        Comparer comp = new Comparer();

        //3、开始计算shapley value：
        long ave_runtime = 0;
        double ave_error_max = 0; //计算最大误差
        double ave_error_ave = 0;
        double[][] sv_values = new double[Info.timesRepeat][this.num_features];  //[runs][feature]
        for(int i=0; i< Info.timesRepeat; i++) {
            this.num_samples = Info.total_samples_num;
            Random random = new Random(game.seedSet[i]); //this.localSet[i]
            Allocation allo = new Allocation(game);
            ShapMatrixEntry[][] shap_matrix = new ShapMatrixEntry[this.num_features][]; //shap_matrix[len][feature]
            for (int ii = 0; ii < this.num_features; ii++) {   // 初始化二维数组的每个元素
                shap_matrix[ii] = new ShapMatrixEntry[this.num_features];
                for (int j = 0; j < this.num_features; j++) {
                    shap_matrix[ii][j] = new ShapMatrixEntry();
                }
            }

            long time_1 = System.currentTimeMillis();
            double[] sv = ShapleyApproximateNog_0(game, shap_matrix, coaSet, allo, random);
            long time_2 = System.currentTimeMillis();
            sv_values[i] = sv;

            // --------------------------------------------------计算误差------------------------------------------------
            double error_max = comp.computeMaxError(sv, this.exact); //计算最大误差
            double error_ave = comp.computeAverageError(sv, this.exact, this.num_features);  //计算平均误差
//            double mse = comp.computeMSE(shap_matrix, this.exact, this.num_features);
            //-----------------------------------------------------------------------------------------------------------
            ave_runtime += time_2 - time_1;
            ave_error_max += error_max;
            ave_error_ave += error_ave;
//            ave_mse += mse;
//            System.out.println("run: " + i + " " + "error_ave: " + error_ave + " \t"  +  "error_max: " + error_max + " \t"  +  "mse: " + mse);
            System.out.println("run: " + i + " " + "error_ave: " + error_ave + " \t"  +  "error_max: " + error_max );
        }
        //--------------------------------------补充内容：测评ACV-------------------------------------
//        double acv = comp.computeACV(sv_values, this.num_features);
        System.out.println(model_name + ":  " + "error_ave: " + ave_error_ave/Info.timesRepeat + " \t"  +  "error_max: " + ave_error_max/Info.timesRepeat);
//        System.out.println("average cv :" + acv);
        System.out.println("Shap_PGB_0 time : " + (ave_runtime * 0.001)/ Info.timesRepeat );  //+ "S"
    }

    //TODO 给每层分配样本
    private void sampleAllocation(ShapMatrixEntry[][] shap_matrix, ArrayList<Integer>[][] coaSet, Allocation allo, GameClass game, Random random) {
        int start_level = 2;
        if(game.isRealData){
            this.num_samples = this.num_samples - this.num_features*(this.num_features/2 + this.num_features-4);  //减去LM + EM
//            System.out.println("sampleNum - LM - EM: " + this.num_samples);
            //【方案1】：利用矩阵分配
            allo.sampleAllocation_9(this.num_features, this.newLevelMatrix, start_level, this.num_samples);
           //【方案2】：估计方差
//            int end_level = checkLevel_end(this.newLevelMatrix);  //end_level = 90,到第90层时，所有utility值都相等
//            start_level = Math.max(2, checkLevel_start(this.newLevelMatrix) + 1);
//            this.check_level_index = allo.sampleAllocationReal(this.num_features, newLevelMatrix, start_level, end_level, (int) ((1-game.check_weight) * this.num_samples), game, random, shap_matrix); //按随机采样的方差分配 （非确定性）
        }
        else{
            int end_level = checkLevel_end(this.newLevelMatrix);  //end_level = 90,到第90层时，所有utility值都相等
            start_level = Math.max(2, checkLevel_start(this.newLevelMatrix) + 1);
            // Voting: start_level =10, 长度为11的发生了变化      //Airport: start_level = 2
            this.check_level_index = allo.sampleAllocationVot_4(this.num_features, this.newLevelMatrix, start_level, end_level, (int) ((1-game.check_weight) * this.num_samples), game, random, shap_matrix); //按随机采样的方差分配 （非确定性）

            //-----------------------------------------来自voting的错误尝试-----------------------------------------------
//            allo.sampleAllocationVot_2(this.num_features, newLevelMatrix, level_index, this.num_samples, game); //num_samples进去一个用两次
//            this.check_level_index = level_index;  //这是神的指示！  level_index = 10
//            this.check_level_index = allo.sampleAllocationVot_2(this.num_features, newLevelMatrix, start_level, this.num_samples, game, random); //按随机采样的方差分配 （非确定性）
//            allo.sampleAllocationVot_3(this.num_features, newLevelMatrix, start_level, this.num_samples); //直接用LM计算层方差分配（确定性）
//            allo.sampleAllocation_uni_2(this.num_features, this.num_samples, 16, 33); //按照起始层的范围 + 均匀分配  //16-33（用2024L调出来的）
//            System.out.println("this.check_level_index: " + this.check_level_index);
        }

        for(int level=start_level; level<this.num_features-1; level++){
            this.allCoalitions[level] = new FeatureSubset[allo.num_sample[level] + this.num_features];  //初始化：start_level
            for(int i=0; i<allo.num_sample[level]; i++){
                this.allCoalitions[level][i] = new FeatureSubset(new ArrayList<Integer>(), 0.0);  //初始化这一层的每一个element
            }
            for(int ind=0; ind< this.num_features; ind++){  //用LM给当前层赋值
                this.allCoalitions[level][allo.num_sample[level] + ind] = new FeatureSubset(coaSet[level][ind], this.newLevelMatrix[level][ind]);
            }
        }
        this.allCoalitions[this.num_features-1] = new FeatureSubset[1]; //初始化最后一层
        this.allCoalitions[this.num_features-1][0] = new FeatureSubset(coaSet[this.num_features-1][0], this.newLevelMatrix[this.num_features-1][0]); //赋值最后一层

        //[2024.8.19]:再添加初始化
        for(int ind=2; ind<start_level;ind++){
            for(int i=0; i<this.num_features; i++) {
                shap_matrix[ind][i].sum += 0.0;
                shap_matrix[ind][i].count++;
                shap_matrix[ind][i].record.add(0.0);
            }
        }
        for(int ind=this.num_features-start_level; ind<this.num_features; ind++){   //this.num_features-start_level-1
            for(int i=0; i<this.num_features; i++) {
                shap_matrix[ind][i].sum += 0.0;
                shap_matrix[ind][i].count++;
                shap_matrix[ind][i].record.add(0.0);
            }
        }
    }

    //TODO 给每层分配样本
    private void sampleAllocation_0(ShapMatrixEntry[][] shap_matrix, ArrayList<Integer>[][] coaSet, Allocation allo, GameClass game, Random random) {
        if(game.isRealData){
//            this.num_samples = this.num_samples - this.num_features*(this.num_features/2 + this.num_features-4);  //减去LM + EM
//            System.out.println("sampleNum - LM - EM: " + this.num_samples);
            //【方案1】：利用矩阵分配
//            allo.sampleAllocation_9(this.num_features, this.newLevelMatrix, start_level, this.num_samples);
            //【方案2】：估计方差
            game.end_level = checkLevel_end(this.newLevelMatrix);  //end_level = 90,到第90层时，所有utility值都相等
            game.start_level = Math.max(game.start_level, checkLevel_start(this.newLevelMatrix) + 1);
            this.check_level_index = allo.sampleAllocationReal(this.num_features, newLevelMatrix, this.num_samples, game, random, shap_matrix); //按随机采样的方差分配 （非确定性）
        }
        else{
            game.end_level = checkLevel_end(this.newLevelMatrix);  //end_level = 90,到第90层时，所有utility值都相等
            game.start_level = Math.max(game.start_level, checkLevel_start(this.newLevelMatrix) + 1);
            // Voting: start_level =10, 长度为11的发生了变化      //Airport: start_level = 2
            this.check_level_index = allo.sampleAllocationVot_4(this.num_features, this.num_samples, game, random, shap_matrix); //按随机采样的方差分配 （非确定性）

            //-----------------------------------------来自voting的错误尝试-----------------------------------------------
//            allo.sampleAllocationVot_2(this.num_features, newLevelMatrix, level_index, this.num_samples, game); //num_samples进去一个用两次
//            this.check_level_index = level_index;  //这是神的指示！  level_index = 10
//            this.check_level_index = allo.sampleAllocationVot_2(this.num_features, newLevelMatrix, start_level, this.num_samples, game, random); //按随机采样的方差分配 （非确定性）
//            allo.sampleAllocationVot_3(this.num_features, newLevelMatrix, start_level, this.num_samples); //直接用LM计算层方差分配（确定性）
//            allo.sampleAllocation_uni_2(this.num_features, this.num_samples, 16, 33); //按照起始层的范围 + 均匀分配  //16-33（用2024L调出来的）
//            System.out.println("this.check_level_index: " + this.check_level_index);
        }

        for(int level= game.start_level; level<this.num_features-1; level++){
            this.allCoalitions[level] = new FeatureSubset[allo.num_sample[level] + this.num_features];  //初始化：start_level
            for(int i=0; i<allo.num_sample[level]; i++){
                this.allCoalitions[level][i] = new FeatureSubset(new ArrayList<Integer>(), 0.0);  //初始化这一层的每一个element
            }
            for(int ind=0; ind< this.num_features; ind++){  //用LM给当前层赋值
                this.allCoalitions[level][allo.num_sample[level] + ind] = new FeatureSubset(coaSet[level][ind], this.newLevelMatrix[level][ind]);
            }
        }
        this.allCoalitions[this.num_features-1] = new FeatureSubset[1]; //初始化最后一层
        this.allCoalitions[this.num_features-1][0] = new FeatureSubset(coaSet[this.num_features-1][0], this.newLevelMatrix[this.num_features-1][0]); //赋值最后一层

        //[2024.8.19]:再添加初始化
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

    //TODO 给每层分配样本
    private void sampleAllocation_1(ShapMatrixEntry[][] shap_matrix, ArrayList<Integer>[][] coaSet, Allocation allo, GameClass game, Random random) {
        game.end_level = checkLevel_end(this.newLevelMatrix);  //end_level = 90,到第90层时，所有utility值都相等
        game.start_level = Math.max(game.start_level, checkLevel_start(this.newLevelMatrix) + 1);
        int limit = Math.max(2, this.num_samples / (this.num_features * this.num_features * 2)); // 需要评估几个
        // Voting: start_level =10, 长度为11的发生了变化      //Airport: start_level = 2
        int allSamples = this.num_samples;

        switch (this.model) {
            case "airport":    //feature = 100
//                System.out.println("allSamples: " + allSamples);
                allSamples -= Info.num_of_features_airport*Info.num_of_features_airport/2 - Info.num_of_features_airport/2;
//                System.out.println("EM + LM samples: " + Info.num_of_features_airport*(Info.num_of_features_airport/2 + Info.num_of_features_airport-4));
//                System.out.println("EM samples: " + (Info.num_of_features_airport*Info.num_of_features_airport/2 + Info.num_of_features_airport/2));
//                System.out.println("allSamples: " + allSamples);
//                System.out.println("Evaluate samples: " + limit * Info.num_of_features_airport * (game.end_level - game.start_level));
                if(Info.is_gene_weight){
                    allSamples = (int) Math.ceil((1.0 - game.check_weight) * (allSamples - limit * (game.end_level - game.start_level) * this.num_features));
                }
                else{
                    allSamples = (int) Math.ceil((1.0 - game.check_weight) * (allSamples - limit * (game.end_level - game.start_level) * Info.num_of_features_airport));
                }

                break;
            case "bank":   //feature = 16
                allSamples = (int) Math.ceil((1.0 - game.check_weight) * (allSamples - limit * (game.end_level - game.start_level) * Info.num_of_features_bank));
                break;
            case "health":   //feature = 39
                allSamples = (int) Math.ceil((1.0 - game.check_weight) * (allSamples - limit * (game.end_level - game.start_level) * Info.num_of_features_health));
                break;
            case "voting":    //feature = 51
                if (Info.setting <= 100) {
                    allSamples = (int) Math.ceil((1.0 - game.check_weight) * allSamples);
                    limit++;
                } else {
                    allSamples = (int) Math.ceil((1.0 - game.check_weight) * (allSamples - limit * (game.end_level - game.start_level) * Info.num_of_features_voting));
                }
                break;

            default:
                allSamples = (int) Math.ceil((1.0 - game.check_weight) * (allSamples - limit * (game.end_level - game.start_level) * Info.num_of_features));
                break;
        }
        allSamples = Math.max(1,  allSamples);
//        System.out.println("limit: " + limit + ";    allSamples: " + allSamples);
        this.check_level_index = allo.sampleAllocation(this.num_features, this.newLevelMatrix, limit, allSamples, game, random, shap_matrix); //按随机采样的方差分配 （非确定性）

        //-------------------------------------------------------------------------------------------------------------
//        if(game.isRealData){
////            this.num_samples = this.num_samples - this.num_features*(this.num_features/2 + this.num_features-4);  //减去LM + EM
////            System.out.println("sampleNum - LM - EM: " + this.num_samples);
//            //【方案1】：利用矩阵分配
////            allo.sampleAllocation_9(this.num_features, this.newLevelMatrix, start_level, this.num_samples);
//            this.check_level_index = allo.sampleAllocationReal(this.num_features, this.newLevelMatrix, this.num_samples, game, random, shap_matrix); //按随机采样的方差分配 （非确定性）
//        }
//        else{
//            this.check_level_index = allo.sampleAllocationVot_4(this.num_features, this.num_samples, game, random, shap_matrix); //按随机采样的方差分配 （非确定性）
//        }
        //--------------------------------------------------------------------------------------------------------------

        for(int level= game.start_level; level<this.num_features-1; level++){
            this.allCoalitions[level] = new FeatureSubset[allo.num_sample[level] + this.num_features];  //初始化：start_level
            for(int i=0; i<allo.num_sample[level]; i++){
                this.allCoalitions[level][i] = new FeatureSubset(new ArrayList<Integer>(), 0.0);  //初始化这一层的每一个element
            }
            for(int ind=0; ind< this.num_features; ind++){  //用LM给当前层赋值
                this.allCoalitions[level][allo.num_sample[level] + ind] = new FeatureSubset(coaSet[level][ind], this.newLevelMatrix[level][ind]);
            }
        }
        this.allCoalitions[this.num_features-1] = new FeatureSubset[1]; //初始化最后一层
        this.allCoalitions[this.num_features-1][0] = new FeatureSubset(coaSet[this.num_features-1][0], this.newLevelMatrix[this.num_features-1][0]); //赋值最后一层

        //[2024.8.19]:再添加初始化
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

    private double calculateVariance(ArrayList<Double> list) {
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
        for (double num : list) {
            variance += Math.pow(num - mean, 2);
        }
        variance /= n;
        return variance;
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
        this.exact = game.exact;   // the exact shapley valu
        this.num_samples = Info.total_samples_num;
        this.model = modelName;
        this.evaluateMatrix = constructEvaluateMatrix(game); //第0层和第1层全算（层数= 减去的特征子集长度）
        this.newLevelMatrix = constructLevelMatrix_2(this.evaluateMatrix, game, coaSet);  //[len][fea], 从L2开始记录
        this.allCoalitions = new FeatureSubset[this.num_features][];  //存储每一层生成的coalition
    }

    /*TODO [版本12]: 原ShapleyApproximateNog_10: 跳过的层也需要被记录sv, sv=0 */
    private double[] ShapleyApproximateNog_0(GameClass game, ShapMatrixEntry[][] shap_matrix, ArrayList<Integer>[][] coaSet, Allocation allo, Random random) {

        sampleAllocation_1(shap_matrix, coaSet, allo, game, random);

        //1）（利用levelMatrix剪枝）扫描Matrix，计算1-联盟 & 2-联盟的shapley值  //第0层和第1层全算（层数= 减去的特征子集长度）
        computeMatrix_8(game, shap_matrix, this.evaluateMatrix);    //computeMatrix_3() 使用两个矩阵的优化

        // 3)计算第2层（level = 被减去的特征的长度） level_index在这里被初始化
        FeatureSubset[] coalitionSet = initialLevel_noSort_seed(game, this.evaluateMatrix, this.check_level_index, allo, random);  //level =2 （存储长度为3的特征子集）
        this.allCoalitions[this.check_level_index-1] = coalitionSet;  //长度为2的coalitions
        FeatureSubset[] coalitionSet_start = new FeatureSubset[0];

        //4)依次生成联盟 (ind / level 等于被减去的联盟的长度)
        for(int ind= game.start_level; ind <this.check_level_index; ind++){
            computeNextLevelNOG_15(game, ind, coalitionSet_start, shap_matrix, allo, random);   //纯填充
        }
        for(int ind= this.check_level_index; ind <this.num_features; ind++){
            if(coalitionSet.length == 0){
                coalitionSet = initialLevel_noSort_seed(game, this.evaluateMatrix, ind, allo, random);  //第一层的值构建出来
            }
            coalitionSet = computeNextLevelNOG_15(game, ind, coalitionSet, shap_matrix, allo, random);  //计算
        }
        // --------------------------------------------------计算误差------------------------------------------------
//        ShapMatrixEntry[] temp = new ShapMatrixEntry[this.num_features];
//        for(int i=0; i<this.num_features; i++){
//            temp[i] = new ShapMatrixEntry();
//            if( shap_matrix[i].count != 0) {
//                temp[i].sum = shap_matrix[i].sum / shap_matrix[i].count;
//            }
//        }
//        Comparer comp = new Comparer();
//        double error_max = comp.computeMaxError(temp, this.exact, this.num_features); //计算最大误差
//        double error_ave = comp.computeAverageError(temp, this.exact, this.num_features);  //计算平均误差
//        System.out.println("error_ave: " + error_ave + " \t"  +  "error_max: " + error_max);
        //----------------------------------------------------------------------------

        //第二轮计算
//        long time_1 = System.currentTimeMillis();
        double[] sv = randomRound_8(shap_matrix, game, random); //两个ShapMatrixEntry[]:方差记录每层 + layer_variance 选 key_fea
//        long time_2 = System.currentTimeMillis();
//        System.out.println("randomRound: " + (time_2-time_1)*0.001 + " S");
        return sv;
    }

    //TODO 两个ShapMatrixEntry[] + layer_variance 选 key_fea
    private double[] randomRound_8(ShapMatrixEntry[][] shap_matrix, GameClass game, Random random) {

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
//                if(variance_count[len][i] != 0){
                layer_variance_sum[i] += variance_level_fea[len][i];
                count ++;
//                }
            }
            layer_variance[i] = layer_variance_sum[i] / count;  //i在不同层的方差
        }

        //2）按照方差大小，挑选出一些需要被检查的feature
        int check_num_features = (int) Math.max(1, Math.ceil(game.key_features_weight * this.num_features));  //确定检查的个数// 最小堆，用于存储最大的 m 个数
        int check_num_samples = (int) Math.ceil(this.num_samples * game.check_weight / check_num_features);  //剩余的样本量 == 每个feature要计算几次
//        System.out.println("check_num_samples: " + (check_num_samples * check_num_features));
        PriorityQueue<Integer> checkFeaSet = findLargestFeatures(check_num_features, layer_variance);  //layer_variance选key_value
//        System.out.println(checkFeaSet.size() +  ": " + checkFeaSet.toString());
        for(Integer fea : checkFeaSet){
            int[] alloSamples = new int[this.num_features];
            if(layer_variance[fea] == 0 ){
                //方差等于0，但又需要被计算：则给每层均匀分配样本
                int temp = (int) Math.ceil((double) check_num_samples / (game.end_level - game.start_level));
                for(int ind=game.start_level; ind< game.end_level; ind++){
                    alloSamples[ind]  = temp;
                }
            }
            else {
               alloSamples = allocationByVar(fea, layer_variance_sum, check_num_samples, variance_level_fea);  //1)给每层划分样本
            }
                //2）计算边际贡献
                for(int level=game.start_level; level<game.end_level; level++){   //遍历每层
                    int count = alloSamples[level];  //这一层计算的个数
                    while(count > 0){
                            FeatureSubset[] set = this.allCoalitions[level];
                            int index = random.nextInt(set.length);
                            ArrayList<Integer> subset_1 = set[index].name;
                            double value_1 = set[index].value_fun;
                            double value_2 = 0;
                            if(subset_1.contains(fea)){  //包含了所需要的fea, 直接remove,求值
                                ArrayList<Integer> subset_2 = new ArrayList<>(subset_1);
                                subset_2.remove(subset_1.indexOf(fea));
                                value_2 = game.gameValue(this.model, subset_2);
                                shap_matrix[level][fea].sum += value_1 - value_2;
                                shap_matrix[level][fea].count ++;
                            }
                            else{   //如果不包含，remove + add 再求值
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
//                            temp += (value_1 - value_2) / alloSamples[level];
                            count --;
                        }
//                    if(alloSamples[level] !=0){
//                        shap_matrix[fea].sum += temp;
//                        shap_matrix[fea].count ++;
//                    }
//                    else{
//                        System.out.println(level + ": " + fea + ": " +  this.variance_level_fea[level][fea]);
//                    }
                }
        }
        return meanShapleyValue(shap_matrix);
    }

    //TODO 给每层分配样本
    private int[] allocationByVar(Integer fea, double[] layer_variance_sum, int num_samples, double[][] variance_level_fea) {
        double totalVar = layer_variance_sum[fea];
        int[] alloSamples = new int[this.num_features];  //给每一层分配样本
        for(int ind = 1; ind < this.num_features-1; ind++) {
            alloSamples[ind] = (int) Math.ceil(num_samples * variance_level_fea[ind][fea] / totalVar);
        }
        return alloSamples;
    }

    //TODO 找到方差最大的前k几个key features
    private PriorityQueue<Integer> findLargestFeatures(int k, double[] array) {
        // 最小堆，用于存储最大的 k 个数的索引，比较时使用数组中的值进行比较
        PriorityQueue<Integer> minHeap = new PriorityQueue<>(k, Comparator.comparingDouble(i -> array[i]));
        // 遍历数组
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
    public double[][] constructEvaluateMatrix(GameClass game) {
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

    public double[][] constructLevelMatrix(GameClass game){
        double[][] matrix = new double[this.num_features][this.num_features];
        for(int i=0; i<this.num_features; i++) {   //i: 横坐标 (也是第1个item)
            ArrayList<Integer> subset = new ArrayList<>();
            subset.add(i);  //单个特征联盟，1-联盟
            matrix[i][i] = game.gameValue(this.model, subset);
            for(int j=i+1; j<this.num_features; j++) {   //j: 纵坐标(也是第2个item)
                subset.add(j);
                matrix[i][j] = matrix[j][i] = game.gameValue(this.model, subset);  //复制两份
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

    //TODO computeMatrix_6 + 计算方差
    private void computeMatrix_8(GameClass game, ShapMatrixEntry[][] shap_matrix, double[][] evaluateMatrix){
        //要从第二层开始算
        // 0) 初始化
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
    private FeatureSubset[] initialLevel_noSort_seed(GameClass game, double[][] evaluateMatrix, int level_index, Allocation allo, Random random) {
        //TODO 性质：矩阵是一个对称的矩阵，所以2-联盟只需要看一半，存一半就可以

        // 1）初始化网格结构 （存3-联盟）
        FeatureSubset[] twoCoalition_set;  //用list存储所有的2-项集

        //情况1： 前面几层可以被剪枝
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

    //TODO 随机取出m个长度为len的元素
    public ArrayList<FeatureSubset> randomSubsets(GameClass game, int m, int len, Random random){
        ArrayList<FeatureSubset> subsets = new ArrayList<>();  //所有生成的subset的集合
        for (int i = 0; i < m; i++) {
            Set<Integer> subset = new HashSet<>();  //一个生成的subset(其中元素不重复)
            while (subset.size() < len) {
                subset.add(random.nextInt(this.num_features));
            }
            ArrayList<Integer> name = new ArrayList<>(subset);
            double value = game.gameValue(this.model, name);
            FeatureSubset ele = new FeatureSubset(name, value);
            subsets.add(ele);
        }
        return subsets;
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

    //TODO 原版computeNextLevelNOG_10：跳过的层也需要被记录sv, sv=0
    private FeatureSubset[] computeNextLevelNOG_15(GameClass game, int level, FeatureSubset[] coalitionSet, ShapMatrixEntry[][] shap_matrix, Allocation all, Random random) {
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

    //TODO [版本3]从一个网格中，随机选出一个样本 (添加了不为空集的判断)
    public FeatureSubset randomGet(FeatureSubset[] aGrid, Random random, int range){
        int index;
        // 从数组中随机选择一个非空的特征子集
        do {
            index = random.nextInt(range);
        }
        while (aGrid[index] == null);
        return aGrid[index];
    }

    private void meanShapleyValue(ShapMatrixEntry[] shap_matrix, ShapMatrixEntry[] shap_matrix_2, GameClass game) {
        for(int fea=0; fea<this.num_features; fea++){
            ShapMatrixEntry entry = shap_matrix[fea];
            if(entry.count != 0){
                if(shap_matrix_2[fea].count!=0){
                    entry.sum = entry.sum / entry.count * (1-game.check_weight) + shap_matrix_2[fea].sum /shap_matrix_2[fea].count  * game.check_weight;
//                    entry.sum = entry.sum / entry.count * (1-Info.check_weight) + shap_matrix_2[fea].sum * Info.check_weight;
//                    System.out.println(fea + ": " + entry.count + ": " + entry.sum);
                }
                else{
                    entry.sum = entry.sum / entry.count;
//                    System.out.println(fea + ": " + entry.count + ": " + entry.sum);
                }
            }
        }
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