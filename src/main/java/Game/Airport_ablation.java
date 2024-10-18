package Game;

import Global.*;
import structure.*;
import java.util.*;

//TODO 测试每个部分的改进
public class Airport_ablation {

    int num_of_features = Info.num_of_features_airport;

    /*TODO [Airport_game]  复制MiningShap_6 + all_3()
     * given_weights: 给定的weights[] 用于计算value_function*/
    public void airport_evaluate_1(double[] given_weights) {
        /*我觉得经过改造后，这俩函数等价的: ShapleyApproximate_airport_1() == ShapleyApproximate_airport_2() */  // ==>版本1准确率比较高
        long time_1 = System.currentTimeMillis();

        ShapMatrixEntry[] shap_matrix = ShapleyApproximate_airport_1(given_weights); //使用矩阵计算(复制MiningSHAP,改造)  ==>

//        ShapMatrixEntry[] shap_matrix = ShapleyApproximate_airport_2(given_weights); //使用矩阵计算(复制voting Game,改成airport)

        long time_2 = System.currentTimeMillis();

        double error_max = computeMaxError(shap_matrix, Info.airport_exact); //计算最大误差

        double error_ave = computeAverageError(shap_matrix, Info.airport_exact);  //计算平均误差

        System.out.println("ShapleyApproximate_airport_1 (mining_6 + alloc_3 + Evaluate Matrix):  " + "error_ave: " + error_ave + " \t"  +  "error_max: " + error_max );
        System.out.println("time : " + (time_2 - time_1) * 0.001 );  //+ "S"


    }

    /*TODO [Airport_game]  复制voting + all_3()
     * given_weights: 给定的weights[] 用于计算value_function*/
    public void airport_evaluate_2(double[] given_weights) {
        /*我觉得经过改造后，这俩函数等价的: ShapleyApproximate_airport_1() == ShapleyApproximate_airport_2() */  // ==>版本1准确率比较高
        long time_1 = System.currentTimeMillis();

//        ShapMatrixEntry[] shap_matrix = ShapleyApproximate_airport_1(given_weights); //使用矩阵计算(复制MiningSHAP,改造)  ==>

        ShapMatrixEntry[] shap_matrix = ShapleyApproximate_airport_2(given_weights); //使用矩阵计算(复制voting Game,改成airport)

        long time_2 = System.currentTimeMillis();

        double error_max = computeMaxError(shap_matrix, Info.airport_exact); //计算最大误差

        double error_ave = computeAverageError(shap_matrix, Info.airport_exact);  //计算平均误差

        System.out.println("ShapleyApproximate_airport_2 (voting + alloc_3 + Evaluate Matrix)):  " + "error_ave: " + error_ave + " \t"  +  "error_max: " + error_max );
        System.out.println("time : " + (time_2 - time_1) * 0.001 );  //+ "S"

    }

    /*TODO [Airport_game]  复制MiningShap_6 + all_2()
     * given_weights: 给定的weights[] 用于计算value_function*/
    public void airport_evaluate_3(double[] given_weights) {
        /*我觉得经过改造后，这俩函数等价的: ShapleyApproximate_airport_1() == ShapleyApproximate_airport_2() */  // ==>版本1准确率比较高
        long time_1 = System.currentTimeMillis();

        ShapMatrixEntry[] shap_matrix = ShapleyApproximate_airport_3(given_weights); //使用矩阵计算(复制voting Game,改成airport)

        long time_2 = System.currentTimeMillis();

        double error_max = computeMaxError(shap_matrix, Info.airport_exact); //计算最大误差

        double error_ave = computeAverageError(shap_matrix, Info.airport_exact);  //计算平均误差

        System.out.println("ShapleyApproximate_airport_3 (mining_6 + alloc_2):  " + "error_ave: " + error_ave + " \t"  +  "error_max: " + error_max );
        System.out.println("time : " + (time_2 - time_1) * 0.001 );  //+ "S"

    }

    /*TODO [Airport_game]  复制voting + all_2()
     * given_weights: 给定的weights[] 用于计算value_function*/
    public void airport_evaluate_4(double[] given_weights) {
        /*我觉得经过改造后，这俩函数等价的: ShapleyApproximate_airport_1() == ShapleyApproximate_airport_2() */  // ==>版本1准确率比较高
        long time_1 = System.currentTimeMillis();

        ShapMatrixEntry[] shap_matrix = ShapleyApproximate_airport_4(given_weights); //使用矩阵计算(复制voting Game,改成airport)

        long time_2 = System.currentTimeMillis();

        double error_max = computeMaxError(shap_matrix, Info.airport_exact); //计算最大误差

        double error_ave = computeAverageError(shap_matrix, Info.airport_exact);  //计算平均误差

        System.out.println("ShapleyApproximate_airport_4 (voting + alloc_2):  " + "error_ave: " + error_ave + " \t"  +  "error_max: " + error_max );
        System.out.println("time : " + (time_2 - time_1) * 0.001 );  //+ "S"

    }

    /*TODO [Airport_game] 复制Voting Game -- 非均匀网格划分
    *  MiningShap_4（）是原版*/
    public void airport_evaluate_5(double[] given_weights) {
        /*我觉得经过改造后，这俩函数等价的: ShapleyApproximate_airport_1() == ShapleyApproximate_airport_2() */  // ==>版本1准确率比较高
        long time_1 = System.currentTimeMillis();

        ShapMatrixEntry[] shap_matrix = ShapleyApproximate_airport_5(given_weights); //使用矩阵计算(复制voting Game,改成airport)

        long time_2 = System.currentTimeMillis();

        double error_max = computeMaxError(shap_matrix, Info.airport_exact); //计算最大误差

        double error_ave = computeAverageError(shap_matrix, Info.airport_exact);  //计算平均误差

        System.out.println("ShapleyApproximate_airport_5 (voting + grid 非均匀):  " + "error_ave: " + error_ave + " \t"  +  "error_max: " + error_max );
        System.out.println("time : " + (time_2 - time_1) * 0.001 );  //+ "S"
    }

    /*TODO [Airport_game] 复制MiningShap_6 -- 非均匀网格划分
     *  MiningShap_4（）是原版*/
    public void airport_evaluate_6(double[] given_weights) {
        /*我觉得经过改造后，这俩函数等价的: ShapleyApproximate_airport_1() == ShapleyApproximate_airport_2() */  // ==>版本1准确率比较高
        long time_1 = System.currentTimeMillis();

        ShapMatrixEntry[] shap_matrix = ShapleyApproximate_airport_6(given_weights); //使用矩阵计算(复制voting Game,改成airport)

        long time_2 = System.currentTimeMillis();

        double error_max = computeMaxError(shap_matrix, Info.airport_exact); //计算最大误差

        double error_ave = computeAverageError(shap_matrix, Info.airport_exact);  //计算平均误差

        System.out.println("ShapleyApproximate_airport_6 (mining_6 + grid 非均匀):  " + "error_ave: " + error_ave + " \t"  +  "error_max: " + error_max );
        System.out.println("time : " + (time_2 - time_1) * 0.001 );  //+ "S"
    }

    //TODO [版本1]复制MiningShap_6
    private ShapMatrixEntry[] ShapleyApproximate_airport_1(double[] given_weights) {
        //1) 用一个大数组记录特征对应的shapley value之和
        ShapMatrixEntry[] shap_matrix = new ShapMatrixEntry[this.num_of_features];
        for(int i=0; i<shap_matrix.length; i++){
            shap_matrix[i] = new ShapMatrixEntry();
        }

        //2）给每层分配样本
        Allocation allo = new Allocation();
//        allo.sampleAllocation_2(this.num_of_features);
        allo.sampleAllocation_3(this.num_of_features); //sample_num_level 每一层采样几个

        //3) 计算出一个大矩阵，存储1-联盟 & 2-联盟的值
        int[][] evaluateMatrix = constructEvaluateMatrix(given_weights); //第0层和第1层全算（层数= 减去的特征子集长度）

        //4）扫描evaluateMatrix，计算1-联盟 & 2-联盟的shapley值  //第0层和第1层全算（层数= 减去的特征子集长度）
        computeMatrix_airport(shap_matrix, given_weights, evaluateMatrix);

        // 5)计算第2层（level = 被减去的特征的长度）
        Grid grid_level = initialLevelCompute_airport(shap_matrix, given_weights, evaluateMatrix);  //level =2 （存储长度为2的特征子集）

        //4)依次生成联盟 (ind / level 等于被减去的联盟的长度)
        for(int ind=2; ind <this.num_of_features; ind++){
            grid_level = computeNextLevel_airport(ind, grid_level, given_weights, shap_matrix, allo);  //去除list中重复的元素！
        }

        printShapleyValue(shap_matrix);
        return shap_matrix;
    }

    /*TODO [版本2]：复制Voting Game */
    private ShapMatrixEntry[] ShapleyApproximate_airport_2(double[] given_weights) {

        //1) 用一个大数组记录特征对应的shapley value之和
        ShapMatrixEntry[] shap_matrix = new ShapMatrixEntry[this.num_of_features];
        for (int i = 0; i < shap_matrix.length; i++) {
            shap_matrix[i] = new ShapMatrixEntry();
        }

        //2）给每层分配样本
        Allocation allo = new Allocation();
//        allo.sampleAllocation_2(Info.num_of_features_voting-1); //sample_num_level 每一层采样几个 【逻辑错误】
        allo.sampleAllocation_3(this.num_of_features); //sample_num_level 每一层采样几个

        //3) 计算出一个大矩阵，存储1-联盟 & 2-联盟的值
        int[][] evaluateMatrix = constructEvaluateMatrix(given_weights); //第0层和第1层全算（层数= 减去的特征子集长度）

        //4）扫描evaluateMatrix，计算1-联盟 & 2-联盟的shapley值  //第0层和第1层全算（层数= 减去的特征子集长度）
        computeMatrix_airport(shap_matrix, given_weights, evaluateMatrix);

        // 5)计算第2层（level = 被减去的特征的长度）
        Grid grid_level = initialLevelCompute_airport(shap_matrix, given_weights, evaluateMatrix);

        //6)依次生成联盟 (ind / level 等于被减去的联盟的长度)
        for(int ind=2; ind <this.num_of_features; ind++){
            //计算某一层，存储到网格中
            grid_level = computeNextLevel_airport(ind, grid_level, given_weights, shap_matrix, allo);  //去除list中重复的元素！

        }

        printShapleyValue(shap_matrix);
        return shap_matrix;

    }

    //TODO [版本1]复制MiningShap_6 -- 分配方式2
    private ShapMatrixEntry[] ShapleyApproximate_airport_3(double[] given_weights) {
        //1) 用一个大数组记录特征对应的shapley value之和
        ShapMatrixEntry[] shap_matrix = new ShapMatrixEntry[this.num_of_features];
        for(int i=0; i<shap_matrix.length; i++){
            shap_matrix[i] = new ShapMatrixEntry();
        }

        //2）给每层分配样本
        Allocation allo = new Allocation();
        allo.sampleAllocation_2(this.num_of_features);
//        allo.sampleAllocation_3(this.num_of_features); //sample_num_level 每一层采样几个

        //3) 计算出一个大矩阵，存储1-联盟 & 2-联盟的值
        Grid grid_level = computeSingleFeature_3(shap_matrix, given_weights);    //grid_level: 每层的循环变量

        //4)依次生成联盟 (ind / level 等于被减去的联盟的长度)
        for(int ind=1; ind <this.num_of_features; ind++){
            grid_level = computeNextLevel_airport(ind, grid_level, given_weights, shap_matrix, allo);  //去除list中重复的元素！
        }

        printShapleyValue(shap_matrix);
        return shap_matrix;
    }

    /*TODO [版本2]：复制Voting Game -- 分配方式2 */
    private ShapMatrixEntry[] ShapleyApproximate_airport_4(double[] given_weights) {

        //1) 用一个大数组记录特征对应的shapley value之和
        ShapMatrixEntry[] shap_matrix = new ShapMatrixEntry[this.num_of_features];
        for (int i = 0; i < shap_matrix.length; i++) {
            shap_matrix[i] = new ShapMatrixEntry();
        }

        //2）给每层分配样本
        Allocation allo = new Allocation();
        allo.sampleAllocation_2(this.num_of_features); //sample_num_level 每一层采样几个 【逻辑错误】
//        allo.sampleAllocation_3(this.num_of_features); //sample_num_level 每一层采样几个

        // 3）为单个特征计算边际贡献（所有单特征） computeSingleFeature_3()均匀的网格
        Grid grid_level = computeSingleFeature_3(shap_matrix, given_weights);    //grid_level: 每层的循环变量

        //6)依次生成联盟 (ind / level 等于被减去的联盟的长度)
        for(int ind=1; ind <this.num_of_features; ind++){
            //计算某一层，存储到网格中
            grid_level = computeNextLevel_airport(ind, grid_level, given_weights, shap_matrix, allo);  //去除list中重复的元素！
        }

        printShapleyValue(shap_matrix);
        return shap_matrix;

    }

    /*TODO [版本2]：复制Voting Game -- 非均匀网格划分
    *  computeNextLevel_3（）的list 是 ArrayList<> */
    private ShapMatrixEntry[] ShapleyApproximate_airport_5(double[] given_weights) {

        //1) 用一个大数组记录特征对应的shapley value之和
        ShapMatrixEntry[] shap_matrix = new ShapMatrixEntry[this.num_of_features];
        for (int i = 0; i < shap_matrix.length; i++) {
            shap_matrix[i] = new ShapMatrixEntry();
        }

        //2）给每层分配样本
        Allocation allo = new Allocation();
        allo.sampleAllocation_2(this.num_of_features); //sample_num_level 每一层采样几个 【逻辑错误】
//        allo.sampleAllocation_3(this.num_of_features); //sample_num_level 每一层采样几个

        // 3）为单个特征计算边际贡献（所有单特征） computeSingleFeature_3()均匀的网格
        Grid grid_level = computeSingleFeature_3(shap_matrix, given_weights);    //grid_level: 每层的循环变量

        //6)依次生成联盟 (ind / level 等于被减去的联盟的长度)
        for(int ind=1; ind <this.num_of_features; ind++){
            //计算某一层，存储到网格中
            grid_level = computeNextLevel_3(ind, grid_level, given_weights, shap_matrix);  //去除list中重复的元素！
        }

        printShapleyValue(shap_matrix);
        return shap_matrix;

    }

    /*TODO [版本2]：复制MiningShap_6 -- 非均匀网格划分 */
    private ShapMatrixEntry[] ShapleyApproximate_airport_6(double[] given_weights) {
        //1) 用一个大数组记录特征对应的shapley value之和
        ShapMatrixEntry[] shap_matrix = new ShapMatrixEntry[this.num_of_features];
        for(int i=0; i<shap_matrix.length; i++){
            shap_matrix[i] = new ShapMatrixEntry();
        }

        //2）给每层分配样本
        Allocation allo = new Allocation();
        allo.sampleAllocation_2(this.num_of_features);
//        allo.sampleAllocation_3(this.num_of_features); //sample_num_level 每一层采样几个

        //3) 计算出一个大矩阵，存储1-联盟 & 2-联盟的值
        Grid grid_level = computeSingleFeature_3(shap_matrix, given_weights);    //grid_level: 每层的循环变量

        //4)依次生成联盟 (ind / level 等于被减去的联盟的长度)
        for(int ind=1; ind <this.num_of_features; ind++){
            grid_level = computeNextLevel_3(ind, grid_level, given_weights, shap_matrix);  //去除list中重复的元素！
        }

        printShapleyValue(shap_matrix);
        return shap_matrix;
    }

    //TODO 构建evaluateMatrix, 记录1-联盟 & 2-联盟的值
    private int[][] constructEvaluateMatrix(double[] given_weights) {
        int[][] matrix = new int[this.num_of_features][this.num_of_features];
        for(int i=0; i<this.num_of_features; i++){   //i: 横坐标 (也是第1个item)
            ArrayList<Integer> subset = new ArrayList<>();
            subset.add(i);  //单个特征联盟，1-联盟
            matrix[i][i] = value_airport(subset, given_weights);
            for(int j=i+1; j<this.num_of_features; j++){   //j: 纵坐标(也是第2个item)
                ArrayList<Integer> twoCoalition = new ArrayList<>(subset);
                twoCoalition.add(j);
                matrix[i][j] = matrix[j][i] = value_airport(twoCoalition, given_weights);  //复制两份
            }
        }
        return matrix;
    }

    //TODO 计算1-联盟 & 2-联盟的shapley值 => 利用矩阵EvaluateMatrix
    // 【版本2】均匀划分：为单个特征计算边际贡献（所有单特征）复制版本3，用于voting_game
    // 原版是 computeSingleFeature_voting()
    private void computeMatrix_airport(ShapMatrixEntry[] shap_matrix, double[] given_weights, int[][] evaluateMatrix) {

        // 0) 初始化
        long one_feature_sum = 0; //记录对角线元素的值

        // 1）读取1-联盟的shapley value
        for(int i=0; i<this.num_of_features; i++){
            shap_matrix[i].sum += evaluateMatrix[i][i];  //对角线上的元素依次填入shapley value[]的矩阵中
            shap_matrix[i].count ++;
            one_feature_sum += evaluateMatrix[i][i];  //对角线求和
        }

        // 2) 读取和计算2-联盟的shapley value
        for(int i=0; i<this.num_of_features; i++){  //i 是横坐标  一行就对应一个特征
            long line_sum = 0;
            for(int j=0; j<this.num_of_features; j++){  //j是纵坐标
                line_sum += evaluateMatrix[i][j];
            }
            shap_matrix[i].sum += 1.0 * (line_sum - one_feature_sum) / (this.num_of_features - 1);  //第二层的shapley value
            shap_matrix[i].count ++;
        }
    }

    //TODO 计算初始层，并存储到网格中(3-联盟-2联盟的值) => 利用矩阵EvaluateMatrix
    // 原版是 computeSingleFeature_voting()
    private Grid initialLevelCompute_airport(ShapMatrixEntry[] shap_matrix, double[] given_weights, int[][] evaluateMatrix) {

        //性质：矩阵是一个对称的矩阵，所以2-联盟只需要看一半，存一半就可以

        // 1）初始化网格结构
        Grid grids = new Grid();  //存储2-联盟的网格
        ArrayList<FeatureSubset> twoCoalition_set = new ArrayList<>();  //用list存储所有的2-项集
        //数组：虽然定长，但删除不方便

        // 1) 读取2-联盟的shapley value
        for(int i=0; i<this.num_of_features; i++){  //i 是横坐标  一行就对应一个特征
            List<Integer> subSet = new ArrayList<>();
            subSet.add(i);  //第1个特征
            for(int j=i+1; j<this.num_of_features; j++){  //j是纵坐标
                ArrayList<Integer> newSubset = new ArrayList<>(subSet);
                newSubset.add(j);  //第2个特征
                FeatureSubset ele = new FeatureSubset(newSubset, evaluateMatrix[i][j]);
                twoCoalition_set.add(ele);
            }
        }

        // 2）所有2-联盟排序
//        twoCoalition_set.sort(new Comparator<FeatureSubset>() {
//            @Override
//            public int compare(FeatureSubset ele1, FeatureSubset ele2) {
//                return Double.compare(ele1.value_fun, ele2.value_fun);
////                    return ele1.value_fun.compareTo(ele2.value_fun);
//            }
//        });

        //3）构建网格：这一层完成后，存储到网格中
        grids.ConstructGrid_2(twoCoalition_set);  //这是均匀划分的方式

        return grids;
    }

    //TODO 【版本2】
    // 添加功能：去除list中重复元素
    //原版是computeNextLevel_5()
    // weight[] 计算每层的采样权重； checkAndAllocate_transfer_4（）：按照每组个数等比例抽样；当前层生成的元素 list ：ArrayList();
    private Grid computeNextLevel_airport(int level, Grid grid, double[] given_weights, ShapMatrixEntry[] shap_matrix, Allocation all) {

        //**********************
//        for(ArrayList<FeatureSubset> ele : grid.grids_row){
//            System.out.print(ele.size() + "\t");
//        }

        // 1) 初始化
        Grid gridStructure = new Grid();  //用网格记录这一层
        // 记录这一层的shapley 值计算情况（因为每个特征对应的采样数可能不一致，需要结构体记录采样数量）
        ShapMatrixEntry[] temp = new ShapMatrixEntry[this.num_of_features];
        for(int i=0; i<this.num_of_features; i++){
            temp[i] = new ShapMatrixEntry();
        }

        // 记录当前层生成的特征子集，为下一层做好准备
//        ArrayList<FeatureSubset> list = new ArrayList<>();

        //【添加步骤】list中去除重复元素
        HashSet<ArrayList<Integer>> nameSet = new HashSet<>();  //HashSet去除重复
        ArrayList<FeatureSubset> setWithoutDuplicates = new ArrayList<>();

        //2）按照每组个数等比例抽样
        all.levelAllocate(level, grid, this.num_of_features, all.allocations);

        //**********************
//        for(int num : all.allocations){
//            System.out.print(num + "\t");
//        }
//        System.out.println();

        //3）按照样本分配开始计算
        for(int ind=0; ind<all.allocations.length; ind++){  //分配方案[]：给每个网格分配的个数
            int sample_num = all.allocations[ind];  //sample_num：当前网格采样的个数
            ArrayList<FeatureSubset> oneGrid = grid.grids_row[ind]; //grid：取出来的网格
            for(int counter=0; counter < sample_num; counter ++){  //count是采样的计数器
                FeatureSubset random_sample = grid.randomGet(oneGrid);  //random_sample: 从oneGrid中选出来的一个
                //2)选出来的这个FeatureSubset，与每个feature(不包含自己)构成一个新的FeatureSubset
                for(int i=0; i<this.num_of_features; i++){
                    if( !random_sample.name.contains(i)){   //挑选出的random_sample不包含当前的特征，就可以构成新的
                        ArrayList<Integer> name = new ArrayList<>(random_sample.name);
                        name.add(i);
                        double value = value_airport(name, given_weights);
                        FeatureSubset newFeaSub = new FeatureSubset(name, value);

//                        gridStructure.max = Math.max(gridStructure.max, value);
//                        gridStructure.min = Math.min(gridStructure.min, value);

                        //3）计算shapley value 并填写到矩阵中
                        temp[i].sum += value - random_sample.value_fun; //V(S U i) - V(S), random_sample就是S
                        temp[i].count ++;

                        //4）存储新生成特征子集S，为下一层选取做好准备 (只存储不重复的元素)
                        Collections.sort(newFeaSub.name);  //把元素的名字按顺序整理排好
                        if(!nameSet.contains(newFeaSub.name)){
                            setWithoutDuplicates.add(newFeaSub);
                            nameSet.add(newFeaSub.name);
//                        list[index] = newFeaSub;
//                        index++;
                        }

                    }
                }
            }
        }
        //4）这一层计算完了，计算均值
        for(int i=0; i<temp.length; i++){
            ShapMatrixEntry entry = temp[i];

            //***********************************
            if(entry.count != 0){
                shap_matrix[i].sum += 1.0f * entry.sum / entry.count;
                shap_matrix[i].count ++;
            }
        }

        //4)这一层完成后，存储到结构体中
        //***********************************
        gridStructure.ConstructGrid_2(setWithoutDuplicates);  //这是均匀划分的方式

        return gridStructure;
    }

    //TODO 输出和打印shapley value
    private void printShapleyValue(ShapMatrixEntry[] shap_matrix) {
//        int i=0;
        for(ShapMatrixEntry entry : shap_matrix){
            entry.sum = entry.sum / entry.count;
//            System.out.println(i + " : " + entry.sum + "\t  ");
//            i++;
        }
    }

    //TODO 计算最大误差 (Voting game & Airport game)
    private double computeMaxError(ShapMatrixEntry[] shap_matrix, double[] exact) {
        double error_max = 0;  //误差总量，最后除以特征数
        for(int i=0; i<Info.num_of_features_airport; i++){
            error_max = Math.max(Math.abs((shap_matrix[i].sum - exact[i]) / exact[i]), error_max);
        }
        return error_max;
    }

    //TODO 计算平均误差 (Voting game & Airport game)
    private double computeAverageError(ShapMatrixEntry[] shap_matrix, double[] exact) {
        double error_ave = 0;  //误差总量，最后除以特征数
        for(int i=0; i<Info.num_of_features_airport; i++){
            error_ave += Math.abs((shap_matrix[i].sum - exact[i]) / exact[i]);
        }
        error_ave = error_ave / Info.num_of_features_airport;
        return error_ave;
    }

    //TODO 【版本3】均匀划分：为单个特征计算边际贡献（所有单特征）
    // shap_matrix：存储所有特征SV.的大矩阵
    private Grid computeSingleFeature_3(ShapMatrixEntry[] shap_matrix, double[] given_weights) {
        // 0）初始化网格结构 （每层存成一个网格，用完后丢弃）
        Grid gridStructure = new Grid();  //表示这是存储1-联盟的网格
        FeatureSubset[] grid = new FeatureSubset[this.num_of_features];  //生成的所有特征子集，需要把他们都放入网格中（每层都用）

        //1）对每个1-联盟特征，生成
        for(int i=0; i<this.num_of_features; i++){ // Zi
            ArrayList<Integer> subset = new ArrayList<>();
            subset.add(i);

            // 2）计算联盟对应的值  value_voting(subset, sum)
            int value = value_airport(subset, given_weights);

            // 2) 计算这一层的权重  weight(0, Info.num_of_features)
            // 3) 乘上权重后，直接记录到矩阵中
            shap_matrix[i].sum += value;  //因为 subset=空集， value =0
            shap_matrix[i].count ++;
//            System.out.println(i + ": " + weight(0, Info.num_of_features) + " * " + value);

            FeatureSubset ele = new FeatureSubset(subset, value); //（名字，对应的值）
            grid[i] = ele;
        }

        //4)这一层完成后，存储到结构体中
        gridStructure.ConstructGrid_2(grid);  //这是均匀划分的方式

        return gridStructure;
    }

    //TODO 非均匀划分的方式 (List 是ArrayList)
    private Grid computeNextLevel_3(int level, Grid grid_1, double[] given_weights, ShapMatrixEntry[] shap_matrix) {

        //**********************
//        for(ArrayList<FeatureSubset> ele : grid_1.grids_row){
//            System.out.print(ele.size() + "\t");
//        }

        Grid gridStructure = new Grid();  //用网格记录这一层

        // 记录这一层的情况（因为每个特征对应的采样数可能不一致，需要结构体记录采样数量）
        ShapMatrixEntry[] temp = new ShapMatrixEntry[this.num_of_features];
        for(int i=0; i<this.num_of_features; i++){
            temp[i] = new ShapMatrixEntry();
        }

        //1）从每个网格grid中随机选一个
        //[问题] 无法保证每个网格中都有值，如果网格中没有值，就把名额让个其他的
        //[修改] 添加安全检查，再根据网格分布情况分布采样的个数
        // 安全检查 + 重新分配
//        int[] allocations = checkAndAllocate_transfer_1(grid_1);  //[版本1]简单分配，每个网格取1个
//        int[] allocations = checkAndAllocate_transfer_2(level, grid_1, Info.num_of_samples);  //[版本1]简单分配，每个网格取1个
        Allocation all = new Allocation();
//        all.checkAndAllocate_transfer_2(level, grid_1, Info.num_of_samples);
        all.checkAndAllocate_transfer_3(level, grid_1, Info.num_of_samples);  //按照每组个数等比例抽样

        //**********************
//        for(int num : all.allocations){
//            System.out.print(num + "\t");
//        }
//        System.out.println();

        //[第一版]每层抽样一个，lis大小就是 采样数 * 特征数量
        //记录当前层生成的特征子集，为下一层做好准备
//        ArrayList<FeatureSubset> list = new ArrayList<>();
//        FeatureSubset[] list = new FeatureSubset[all.level_samples * (Info.num_of_features-1)];  //【改】 换成了结构体中的 设定样本 + 补偿
        ArrayList<FeatureSubset> list = new ArrayList<>();
        //因为一个采样，就会生成最多（n-j）种组合, j是联盟的特征个数   // 这个list做了最大估计，会有空的,并且空的都在后面
        int index = 0;

        //[第1版]: 下面两行是第一版的函数
//        for(ArrayList<FeatureSubset> grid : grid_1.grids_row){
//          if(grid.size() >0){  //必须让网格中有值才能抽
        //[第2版]: 添加安全检查后
        for(int ind=0; ind< all.allocations.length; ind++){
            int sample_num = all.allocations[ind];  //sample_num：当前网格采样的个数
            ArrayList<FeatureSubset> grid = grid_1.grids_row[ind]; //grid：取出来的网格
            for(int counter=0; counter < sample_num; counter ++){  //count是采样的计数器
                FeatureSubset random_sample = grid_1.randomGet(grid);  //random_sample: 从grid中选出来的一个
                //2)选出来的这个FeatureSubset，与每个feature(不包含自己)构成一个新的FeatureSubset
                for(int i=0; i<Info.num_of_features; i++){
                    if( !random_sample.name.contains(i)){   //挑选出的random_sample不包含当前的特征，就可以构成新的
                        ArrayList<Integer> name = new ArrayList<>(random_sample.name);
                        name.add(i);
                        int value = value_airport(name, given_weights);
                        FeatureSubset newFeaSub = new FeatureSubset(name, value);

                        gridStructure.max = Math.max(gridStructure.max, value);
                        gridStructure.min = Math.min(gridStructure.min, value);

                        //3）计算shapley value 并填写到矩阵中
                        temp[i].sum += value - random_sample.value_fun; //V(S U i) - V(S), random_sample就是S
                        temp[i].count ++;

                        //4）存储新生成特征子集S，为下一层选取做好准备
//                        list.add(newFeaSub);
                        list.add(newFeaSub);
                        index++;
                    }
                }
            }
        }
        //4）这一层计算完了，计算均值
        for(int i=0; i<temp.length; i++){
            ShapMatrixEntry entry = temp[i];

            //***********************************
            if(entry.count != 0){
                shap_matrix[i].sum += 1.0f * entry.sum / entry.count;
                shap_matrix[i].count ++;
            }
        }

        //4)这一层完成后，存储到结构体中
        //***********************************
        gridStructure.ConstructGrid(list);  //这是非均匀划分的方式

        return gridStructure;
    }

    private int value_airport(ArrayList<Integer> subset, double[] given_weights) {
        int subset_max_ele = -1;
        if(subset.size() == 1){
            return (int) given_weights[subset.get(0)];  //返回值是最大下标对应的weights
        }
        else{
            for(int ele : subset){
                if(subset_max_ele < ele){
                    subset_max_ele = ele;
                }
            }
            return (int) given_weights[subset_max_ele];  //返回值是最大下标对应的weights
        }
    }

}
