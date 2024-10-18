package Game_2;

import Global.*;
import structure.*;
import java.util.*;

public class VotingGame_several {

    int num_of_features = Info.num_of_features_voting;

    /*TODO [Voting_game]
     * given_weights: 给定的weights[] 用于计算value_function
     *(ind / level 等于被减去的联盟的长度)
     * 修改1：去除重复元素，list[]中会生成重复的元素，需要去除 ConstructGrid_3(),在构建之前会去除重复的元素
     * */
    public void voting_game(double[] given_weights) {
        //1）初始化给定的函数
        double halfSum = Arrays.stream(given_weights).sum() / 2;  //对given_weights中的数据求和
//        System.out.println(sum);
//        ArrayList<Integer> all_weights = new ArrayList<>();
//        for(int ele : given_weights){
//            all_weights.add(ele);
//        }
//        int sum_weights = sum(all_weights);
//        System.out.println(sum_weights);

        ShapMatrixEntry[] shap_matrix = ShapleyApproximate_voting(given_weights, halfSum);  //均匀网格 + 比例分配每层样本

        double error_max = computeMaxError(shap_matrix, Info.voting_exact); //计算最大误差

        double error_ave = computeAverageError(shap_matrix, Info.voting_exact);  //计算平均误差
        System.out.println("voting Game:  " + "error_max: " + error_max + " \t" + "error_ave: " + error_ave );
    }

    /*TODO [版本2] Voting_game
     * 使用矩阵计算 */
    public void voting_game_2(double[] given_weights) {
        //1）初始化给定的函数
        double halfSum = Arrays.stream(given_weights).sum() / 2;  //对given_weights中的数据求和

        long ave_runtime = 0;
        double ave_error_max = 0; //计算最大误差
        double ave_error_ave = 0;
        for(int i=0; i< Info.timesRepeat; i++){
            long time_1 = System.currentTimeMillis();

            ShapMatrixEntry[] shap_matrix = ShapleyApproximate_voting_2(given_weights, halfSum);  //使用矩阵计算

            long time_2 = System.currentTimeMillis();
            double error_max = computeMaxError(shap_matrix, Info.voting_exact); //计算最大误差
            double error_ave = computeAverageError(shap_matrix, Info.voting_exact);  //计算平均误差

            ave_runtime += time_2 - time_1;
            ave_error_max += error_max;
            ave_error_ave += error_ave;
        }

        System.out.println("My algo_2 voting Game:  " + "error_ave: " + ave_error_ave/Info.timesRepeat + " \t" + "error_max: " + ave_error_max/Info.timesRepeat );
        System.out.println("SortGrid time : " + (ave_runtime * 0.001)/ Info.timesRepeat );  //+ "S"
    }

    /*TODO [版本6+3] 不排序 + 子集名字不排序
     * 使用矩阵计算 */
    public void voting_game_3(double[] given_weights) {
        //1）初始化给定的函数
        double halfSum = Arrays.stream(given_weights).sum() / 2;  //对given_weights中的数据求和

        long ave_runtime = 0;
        double ave_error_max = 0; //计算最大误差
        double ave_error_ave = 0;
        for(int i=0; i< Info.timesRepeat; i++){
            long time_1 = System.currentTimeMillis();

            ShapMatrixEntry[] shap_matrix = ShapleyApproximate_voting_3(given_weights, halfSum);  //不排序直接放入网格

            long time_2 = System.currentTimeMillis();
            double error_max = computeMaxError(shap_matrix, Info.voting_exact); //计算最大误差
            double error_ave = computeAverageError(shap_matrix, Info.voting_exact);  //计算平均误差

            ave_runtime += time_2 - time_1;
            ave_error_max += error_max;
            ave_error_ave += error_ave;
        }

        System.out.println("My algorithm_3 voting Game:  " + "error_ave: " + ave_error_ave/Info.timesRepeat + " \t" + "error_max: " + ave_error_max/Info.timesRepeat );
        System.out.println("Two NoSort time : " + (ave_runtime * 0.001)/ Info.timesRepeat );  //+ "S"

    }


    /*TODO  网格均匀划分 + 每层样本按比例分配
     *  原版是 ShapleyApproximate_airport_4() */
    private ShapMatrixEntry[] ShapleyApproximate_voting(double[] given_weights, double halfSum) {

        //1) 用一个大数组记录特征对应的shapley value之和
        ShapMatrixEntry[] shap_matrix = new ShapMatrixEntry[Info.num_of_features_voting];
        for(int i=0; i<shap_matrix.length; i++){
            shap_matrix[i] = new ShapMatrixEntry();
        }

        //2）给每层分配样本
        Allocation allo = new Allocation();
        allo.sampleAllocation_2(Info.num_of_features_voting-1); //sample_num_level 每一层采样几个

        // 3）为单个特征计算边际贡献（所有单特征） computeSingleFeature_3()均匀的网格
        Grid grid_level = computeSingleFeature_voting(shap_matrix, given_weights, halfSum);    //grid_level: 每层的循环变量

        // 5)生成2-联盟
//        Grid grid_level = new Grid();
//        grid_level = computeNextLevel(grid_1, sum, given_weights, shap_matrix);
//        System.out.println("level initiall: " + grid_level.grids_row[0].get(0).name.size());

        //4)依次生成联盟 (ind / level 等于被减去的联盟的长度)
        for(int ind=1; ind <Info.num_of_features_voting; ind++){
            //【备注】ind 就是用来代替grid_level.grids_row[0].get(0).name.size()，表示这个网格中存储的subset中包含几个特征
            // 因为网格中不一定都有值的，可能网格中不存在元素： grid_level.grids_row[0].get(0).name.size() 可能空指针异常
            // 5) 计算某一层，存储到网格中
//            grid_level = computeNextLevel_3(ind, grid_level, sum, given_weights, shap_matrix);  //这是均匀划分的方式
            grid_level = computeNextLevel_voting(ind, grid_level, given_weights, shap_matrix, allo, halfSum);  //这是均匀划分 + 比例抽样(log降维)
            //computeNextLevel_2() 换成了结构体allocation[]！  某层中结构体不存在，就设置为0（因为值过高，用调节）
//            System.out.println("level : " + grid_level.grids_row[0].get(0).name.size());
//            System.out.print("level: " + ind + "\t");
//            printShapleyValue_test(shap_matrix);
        }

        //5）最后一层（不需要！因为最后1次已经算到最后一层）
//        computeLastLevel(grid_level, sum, given_weights, shap_matrix);

        printShapleyValue(shap_matrix);
        return shap_matrix;
    }

    /*TODO [版本2]：利用矩阵简化计算
     * 网格均匀划分 + 每层样本按比例分配 */
    private ShapMatrixEntry[] ShapleyApproximate_voting_2(double[] given_weights, double halfSum) {

        //1) 用一个大数组记录特征对应的shapley value之和
        ShapMatrixEntry[] shap_matrix = new ShapMatrixEntry[Info.num_of_features_voting];
        for(int i=0; i<shap_matrix.length; i++){
            shap_matrix[i] = new ShapMatrixEntry();
        }

        //2）给每层分配样本
        Allocation allo = new Allocation();
//        allo.sampleAllocation_2(Info.num_of_features_voting-1); //sample_num_level 每一层采样几个 【逻辑错误】
        allo.sampleAllocation_3(Info.num_of_features_voting); //sample_num_level 每一层采样几个

        //3) 计算出一个大矩阵，存储1-联盟 & 2-联盟的值
        int[][] evaluateMatrix = constructEvaluateMatrix(given_weights, halfSum); //第0层和第1层全算（层数= 减去的特征子集长度）

        //4）扫描evaluateMatrix，计算1-联盟 & 2-联盟的shapley值  //第0层和第1层全算（层数= 减去的特征子集长度）
        computeMatrix_voting(shap_matrix, given_weights, halfSum, evaluateMatrix);    //grid_level: 每层的循环变量

        // 5)计算第2层（level = 被减去的特征的长度）
        Grid grid_level = initialLevelCompute_voting(shap_matrix, given_weights, halfSum, evaluateMatrix);  //level =2 （存储长度为2的特征子集）
//        Grid grid_level = new Grid();
//        grid_level = computeNextLevel(grid_1, sum, given_weights, shap_matrix);
//        System.out.println("level initiall: " + grid_level.grids_row[0].get(0).name.size());

        //4)依次生成联盟 (ind / level 等于被减去的联盟的长度)
        for(int ind=2; ind <Info.num_of_features_voting; ind++){
            //【备注】ind 就是用来代替grid_level.grids_row[0].get(0).name.size()，表示这个网格中存储的subset中包含几个特征
            // 因为网格中不一定都有值的，可能网格中不存在元素： grid_level.grids_row[0].get(0).name.size() 可能空指针异常
            // 5) 计算某一层，存储到网格中
//            grid_level = computeNextLevel_3(ind, grid_level, sum, given_weights, shap_matrix);  //这是均匀划分的方式
//            grid_level = computeNextLevel_voting(ind, grid_level, given_weights, shap_matrix, allo, halfSum);  //这是均匀划分 + 比例抽样(log降维)
            grid_level = computeNextLevel_voting_2(ind, grid_level, given_weights, shap_matrix, allo, halfSum);  //去除list中重复的元素！

            //computeNextLevel_2() 换成了结构体allocation[]！  某层中结构体不存在，就设置为0（因为值过高，用调节）
//            System.out.println("level : " + grid_level.grids_row[0].get(0).name.size());
//            System.out.print("level: " + ind + "\t");
//            printShapleyValue_test(shap_matrix);
        }

        //5）最后一层（不需要！因为最后1次已经算到最后一层）
//        computeLastLevel(grid_level, sum, given_weights, shap_matrix);

        printShapleyValue(shap_matrix);
        return shap_matrix;
    }


    /*TODO [版本6+3]：不排序直接放网格 + 子集不排序
     * 网格均匀划分 + 每层样本按比例分配 */
    private ShapMatrixEntry[] ShapleyApproximate_voting_3(double[] given_weights, double halfSum) {

        //1) 用一个大数组记录特征对应的shapley value之和
        ShapMatrixEntry[] shap_matrix = new ShapMatrixEntry[Info.num_of_features_voting];
        for(int i=0; i<shap_matrix.length; i++){
            shap_matrix[i] = new ShapMatrixEntry();
        }

        //2) 计算出一个大矩阵，存储1-联盟 & 2-联盟的值
        int[][] evaluateMatrix = constructEvaluateMatrix(given_weights, halfSum); //第0层和第1层全算（层数= 减去的特征子集长度）
        //************************************ 添加一个矩阵 ************************************
        int[][] levelMatrix = constructLevelMatrix(given_weights, halfSum);
        //判断并记录从哪一层开始计算
        int level_index = checkLevel(levelMatrix);  //level_index：表示这层subset的length -1 (因为从0开始)

        /*到第level——index 有值，方差不为0，因此就需要构造上一层的特征数量（level-1）,然后从当前层 - 上层开始计算*/

        //3)然后给每层分配样本
        Allocation allo = new Allocation();
//        allo.sampleAllocation_2(Info.num_of_features_voting-1); //sample_num_level 每一层采样几个 【逻辑错误】
        allo.sampleAllocation_4(this.num_of_features, level_index); //使用两个矩阵

        //4）（利用levelMatrix剪枝）扫描Matrix，计算1-联盟 & 2-联盟的shapley值  //第0层和第1层全算（层数= 减去的特征子集长度）
        computeMatrix_voting_3(shap_matrix, evaluateMatrix, levelMatrix, level_index);    //computeMatrix_voting_3() 使用两个矩阵的优化

        // 5)计算第2层（level = 被减去的特征的长度）
        Grid grid_level = initialLevelCompute_voting_noSort(given_weights, halfSum, evaluateMatrix, level_index, allo);  //level =2 （存储长度为2的特征子集）
     /* 构造上层的元素的网格，虽然检查了对角线上的元素，但生成的时候使用valueFunction会更保险，因为随机生成的subset可能是不连续的，
     不能保证一定等于对角线上的统一值，相当于一次安全检查 */

        //4)依次生成联盟 (ind / level 等于被减去的联盟的长度)
        for(int ind=level_index; ind <this.num_of_features; ind++){
            //【备注】ind 就是用来代替grid_level.grids_row[0].get(0).name.size()，表示这个网格中存储的subset中包含几个特征
            // 因为网格中不一定都有值的，可能网格中不存在元素： grid_level.grids_row[0].get(0).name.size() 可能空指针异常
            // 5) 计算某一层，存储到网格中
//            grid_level = computeNextLevel_3(ind, grid_level, sum, given_weights, shap_matrix);  //这是均匀划分的方式
//            grid_level = computeNextLevel_voting(ind, grid_level, given_weights, shap_matrix, allo, halfSum);  //这是均匀划分 + 比例抽样(log降维)
            grid_level = computeNextLevel_voting_4(ind, grid_level, given_weights, shap_matrix, allo, halfSum);  //去除list中重复的元素！

            //computeNextLevel_2() 换成了结构体allocation[]！  某层中结构体不存在，就设置为0（因为值过高，用调节）
//            System.out.println("level : " + grid_level.grids_row[0].get(0).name.size());
//            System.out.print("level: " + ind + "\t");
//            printShapleyValue_test(shap_matrix);
        }

        //5）最后一层（不需要！因为最后1次已经算到最后一层）
//        computeLastLevel(grid_level, sum, given_weights, shap_matrix);

        printShapleyValue(shap_matrix);
        return shap_matrix;
    }



    /*TODO [版本6] 网格不排序 + 均匀划分 + 使用2个矩阵计算 （版本2的升级版）*/
    public void voting_evaluate_6(double[] given_weights) {
        //1）初始化给定的函数
        double halfSum = Arrays.stream(given_weights).sum() / 2;  //对given_weights中的数据求和

        long ave_runtime = 0;
        double ave_error_max = 0; //计算最大误差
        double ave_error_ave = 0;
        for(int i=0; i< Info.timesRepeat; i++){
            long time_1 = System.currentTimeMillis();

            ShapMatrixEntry[] shap_matrix = ShapleyApproximate_voting_6(given_weights, halfSum);  //不排序+2个矩阵计算

            long time_2 = System.currentTimeMillis();
            double error_max = computeMaxError(shap_matrix, Info.voting_exact); //计算最大误差
            double error_ave = computeAverageError(shap_matrix, Info.voting_exact);  //计算平均误差

            ave_runtime += time_2 - time_1;
            ave_error_max += error_max;
            ave_error_ave += error_ave;
        }

        System.out.println("My alg_6 improve voting Game : " + "error_ave: " + ave_error_ave/Info.timesRepeat + " \t" + "error_max: " + ave_error_max/Info.timesRepeat );
        System.out.println("TwoMat + noSort time : " + (ave_runtime * 0.001)/ Info.timesRepeat );  //+ "S"
    }

    /*TODO [版本6]：利用矩阵简化计算 (使用两个矩阵) --> 网格不排序
     * 网格均匀划分 + 每层样本按比例分配 */
    private ShapMatrixEntry[] ShapleyApproximate_voting_6(double[] given_weights, double halfSum) {

        //1) 用一个大数组记录特征对应的shapley value之和
        ShapMatrixEntry[] shap_matrix = new ShapMatrixEntry[Info.num_of_features_voting];
        for(int i=0; i<shap_matrix.length; i++){
            shap_matrix[i] = new ShapMatrixEntry();
        }

        //2) 计算出一个大矩阵，存储1-联盟 & 2-联盟的值
        int[][] evaluateMatrix = constructEvaluateMatrix(given_weights, halfSum); //第0层和第1层全算（层数= 减去的特征子集长度）
        //************************************ 添加一个矩阵 ************************************
        int[][] levelMatrix = constructLevelMatrix(given_weights, halfSum);
        //判断并记录从哪一层开始计算
        int level_index = checkLevel(levelMatrix);  //level_index：表示这层subset的length -1 (因为从0开始)

        /*到第level——index 有值，方差不为0，因此就需要构造上一层的特征数量（level-1）,然后从当前层 - 上层开始计算*/

        //3)然后给每层分配样本
        Allocation allo = new Allocation();
//        allo.sampleAllocation_2(Info.num_of_features_voting-1); //sample_num_level 每一层采样几个 【逻辑错误】
        allo.sampleAllocation_4(this.num_of_features, level_index); //使用两个矩阵

        //4）（利用levelMatrix剪枝）扫描Matrix，计算1-联盟 & 2-联盟的shapley值  //第0层和第1层全算（层数= 减去的特征子集长度）
        computeMatrix_voting_3(shap_matrix, evaluateMatrix, levelMatrix, level_index);    //computeMatrix_voting_3() 使用两个矩阵的优化

        // 5)计算第2层（level = 被减去的特征的长度）
        Grid grid_level = initialLevelCompute_voting_noSort(given_weights, halfSum, evaluateMatrix, level_index, allo);  //level =2 （存储长度为2的特征子集）
     /* 构造上层的元素的网格，虽然检查了对角线上的元素，但生成的时候使用valueFunction会更保险，因为随机生成的subset可能是不连续的，
     不能保证一定等于对角线上的统一值，相当于一次安全检查 */

        //4)依次生成联盟 (ind / level 等于被减去的联盟的长度)
        for(int ind=level_index; ind <this.num_of_features; ind++){
            //【备注】ind 就是用来代替grid_level.grids_row[0].get(0).name.size()，表示这个网格中存储的subset中包含几个特征
            // 因为网格中不一定都有值的，可能网格中不存在元素： grid_level.grids_row[0].get(0).name.size() 可能空指针异常
            // 5) 计算某一层，存储到网格中
//            grid_level = computeNextLevel_3(ind, grid_level, sum, given_weights, shap_matrix);  //这是均匀划分的方式
//            grid_level = computeNextLevel_voting(ind, grid_level, given_weights, shap_matrix, allo, halfSum);  //这是均匀划分 + 比例抽样(log降维)
            grid_level = computeNextLevel_voting_3(ind, grid_level, given_weights, shap_matrix, allo, halfSum);  //去除list中重复的元素！

            //computeNextLevel_2() 换成了结构体allocation[]！  某层中结构体不存在，就设置为0（因为值过高，用调节）
//            System.out.println("level : " + grid_level.grids_row[0].get(0).name.size());
//            System.out.print("level: " + ind + "\t");
//            printShapleyValue_test(shap_matrix);
        }

        //5）最后一层（不需要！因为最后1次已经算到最后一层）
//        computeLastLevel(grid_level, sum, given_weights, shap_matrix);

        printShapleyValue(shap_matrix);
        return shap_matrix;
    }


    /*TODO [版本2]：利用矩阵简化计算 (没写完的，不要用)
     * 记录网格的 [min, max]
     * 1) 若矩阵全0，按照weight 分；
     * 2）无weight, 按照项的顺序分；
     * */
    private ShapMatrixEntry[] ShapleyApproximate_voting_x(double[] given_weights, double halfSum) {

        //1) 用一个大数组记录特征对应的shapley value之和
        ShapMatrixEntry[] shap_matrix = new ShapMatrixEntry[Info.num_of_features_voting];
        for(int i=0; i<shap_matrix.length; i++){
            shap_matrix[i] = new ShapMatrixEntry();
        }

        //2）给每层分配样本
        Allocation allo = new Allocation();
        allo.sampleAllocation_2(Info.num_of_features_voting-1); //sample_num_level 每一层采样几个

        //3) 计算出一个大矩阵，存储1-联盟 & 2-联盟的值
        int[][] evaluateMatrix = constructEvaluateMatrix(given_weights, halfSum); //第0层和第1层全算（层数= 减去的特征子集长度）

        //4）扫描evaluateMatrix，计算1-联盟 & 2-联盟的shapley值  //第0层和第1层全算（层数= 减去的特征子集长度）
        computeMatrix_voting(shap_matrix, given_weights, halfSum, evaluateMatrix);    //grid_level: 每层的循环变量

        // 5)计算第2层（level = 被减去的特征的长度）
        Grid grid_level = initialLevelCompute_voting(shap_matrix, given_weights, halfSum, evaluateMatrix);  //level =2 （存储长度为2的特征子集）
//        Grid grid_level = new Grid();
//        grid_level = computeNextLevel(grid_1, sum, given_weights, shap_matrix);
//        System.out.println("level initiall: " + grid_level.grids_row[0].get(0).name.size());

        //4)依次生成联盟 (ind / level 等于被减去的联盟的长度)
        for(int ind=2; ind <Info.num_of_features_voting; ind++){
            //【备注】ind 就是用来代替grid_level.grids_row[0].get(0).name.size()，表示这个网格中存储的subset中包含几个特征
            // 因为网格中不一定都有值的，可能网格中不存在元素： grid_level.grids_row[0].get(0).name.size() 可能空指针异常
            // 5) 计算某一层，存储到网格中
//            grid_level = computeNextLevel_3(ind, grid_level, sum, given_weights, shap_matrix);  //这是均匀划分的方式
            grid_level = computeNextLevel_voting(ind, grid_level, given_weights, shap_matrix, allo, halfSum);  //这是均匀划分 + 比例抽样(log降维)
            //computeNextLevel_2() 换成了结构体allocation[]！  某层中结构体不存在，就设置为0（因为值过高，用调节）
//            System.out.println("level : " + grid_level.grids_row[0].get(0).name.size());
//            System.out.print("level: " + ind + "\t");
//            printShapleyValue_test(shap_matrix);
        }

        //5）最后一层（不需要！因为最后1次已经算到最后一层）
//        computeLastLevel(grid_level, sum, given_weights, shap_matrix);

        printShapleyValue(shap_matrix);
        return shap_matrix;
    }

    //TODO 构建evaluateMatrix, 记录1-联盟 & 2-联盟的值
    private int[][] constructEvaluateMatrix(double[] given_weights, double halfSum) {
        int[][] matrix = new int[Info.num_of_features_voting][Info.num_of_features_voting];
        for(int i=0; i<Info.num_of_features_voting; i++){   //i: 横坐标 (也是第1个item)
            ArrayList<Integer> subset = new ArrayList<>();
            subset.add(i);  //单个特征联盟，1-联盟
            matrix[i][i] = value_voting(subset, given_weights, halfSum);
            for(int j=i+1; j<Info.num_of_features_voting; j++){   //j: 纵坐标(也是第2个item)
                ArrayList<Integer> twoCoalition = new ArrayList<>(subset);
                twoCoalition.add(j);
                matrix[i][j] = matrix[j][i] = value_voting(twoCoalition, given_weights, halfSum);  //复制两份
            }
        }
        return matrix;
    }

    //TODO【版本4】均匀划分：为单个特征计算边际贡献（所有单特征）复制版本3，用于voting_game
    // 原版是 computeSingleFeature_3()
    // shap_matrix：存储所有特征SV.的大矩阵
    private Grid computeSingleFeature_voting(ShapMatrixEntry[] shap_matrix, double[] given_weights, double halfSum) {
        // 0）初始化网格结构 （每层存成一个网格，用完后丢弃）
        Grid gridStructure = new Grid();  //表示这是存储1-联盟的网格
        FeatureSubset[] grid = new FeatureSubset[Info.num_of_features_voting];  //生成的所有特征子集，需要把他们都放入网格中（每层都用）

        //1）对每个1-联盟特征，生成
        for(int i=0; i<Info.num_of_features_voting; i++){ // Zi
            ArrayList<Integer> subset = new ArrayList<>();
            subset.add(i);

            // 2）计算联盟对应的值  value_voting(subset, sum)
            int value = value_voting(subset, given_weights, halfSum);

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


    //TODO 计算1-联盟 & 2-联盟的shapley值 => 利用矩阵EvaluateMatrix
    // 【版本2】均匀划分：为单个特征计算边际贡献（所有单特征）复制版本3，用于voting_game
    // 原版是 computeSingleFeature_voting()
    private void computeMatrix_voting(ShapMatrixEntry[] shap_matrix, double[] given_weights, double halfSum, int[][] evaluateMatrix) {

        // 0) 初始化
        long one_feature_sum = 0; //记录对角线元素的值

        // 1）读取1-联盟的shapley value
       for(int i=0; i<Info.num_of_features_voting; i++){
           shap_matrix[i].sum += evaluateMatrix[i][i];  //对角线上的元素依次填入shapley value[]的矩阵中
           shap_matrix[i].count ++;
           one_feature_sum += evaluateMatrix[i][i];  //对角线求和
       }

       // 2) 读取和计算2-联盟的shapley value
        for(int i=0; i<Info.num_of_features_voting; i++){  //i 是横坐标  一行就对应一个特征
            long line_sum = 0;
            for(int j=0; j<Info.num_of_features_voting; j++){  //j是纵坐标
                    line_sum += evaluateMatrix[i][j];
            }
            shap_matrix[i].sum += 1.0 * (line_sum - one_feature_sum) / (Info.num_of_features_voting - 1);  //第二层的shapley value
            shap_matrix[i].count ++;
        }
    }

    //TODO 计算初始层，并存储到网格中(3-联盟-2联盟的值) => 利用矩阵EvaluateMatrix
    // 原版是 computeSingleFeature_voting()
    private Grid initialLevelCompute_voting(ShapMatrixEntry[] shap_matrix, double[] given_weights, double halfSum, int[][] evaluateMatrix) {

        //性质：矩阵是一个对称的矩阵，所以2-联盟只需要看一半，存一半就可以

        // 1）初始化网格结构 （存3-联盟）
        Grid grids = new Grid();  //表示这是存储1-联盟的网格
        ArrayList<FeatureSubset> twoCoalition_set = new ArrayList<>();  //用list存储所有的2-项集
        //数组：虽然定长，但删除不方便

        // 1) 读取2-联盟的shapley value
        for(int i=0; i<Info.num_of_features_voting; i++){  //i 是横坐标  一行就对应一个特征
            List<Integer> subSet = new ArrayList<>();
            subSet.add(i);  //第1个特征
            for(int j=i+1; j<Info.num_of_features_voting; j++){  //j是纵坐标
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

        //4)计算shapely value
//        computeValue(grids);

        //3）为每个特征都单独算(有多数重复的值)
//        computeLevelTwo(twoCoalition_set);  //计算第二层（被减去的特征子集的长度为2）

        return grids;
    }


    //TODO [不排序放入网格] 计算初始层，并存储到网格中(3-联盟-2联盟的值) => 利用矩阵EvaluateMatrix
    // 原版是 computeSingleFeature_voting()
    private Grid initialLevelCompute_voting_noSort(double[] given_weights, double halfSum, int[][] evaluateMatrix, int level_index, Allocation allo) {
        //TODO 性质：矩阵是一个对称的矩阵，所以2-联盟只需要看一半，存一半就可以

        // 1）初始化网格结构 （存3-联盟）
        Grid grids = new Grid();  //表示这是存储1-联盟的网格

        //情况1： 前面几层可以被剪枝
        if(level_index > 2){
            //从某一层开始计算，在这层中随机生成若干个组合（按照分配方式）
            ArrayList<FeatureSubset> result = randomSubsets(allo.num_sample[level_index], level_index, given_weights, halfSum);  //n个元素中随机取出m个长度为k的元素

            //放入网格中
            grids.ConstructGrid_3(result);  //这是均匀划分的方式
        }

        // 情况2：正常的计算过程，没有可以被剪枝
        else{
            ArrayList<FeatureSubset> twoCoalition_set = new ArrayList<>();  //用list存储所有的2-项集
            //数组：虽然定长，但删除不方便

            // 1) 读取2-联盟的shapley value
            for (int i = 0; i < Info.num_of_features_voting; i++) {  //i 是横坐标  一行就对应一个特征
                List<Integer> subSet = new ArrayList<>();
                subSet.add(i);  //第1个特征
                for (int j = i + 1; j < Info.num_of_features_voting; j++) {  //j是纵坐标
                    ArrayList<Integer> newSubset = new ArrayList<>(subSet);
                    newSubset.add(j);  //第2个特征
                    FeatureSubset ele = new FeatureSubset(newSubset, evaluateMatrix[i][j]);
                    twoCoalition_set.add(ele);
                }
            }
            //3）构建网格：这一层完成后，存储到网格中
            grids.ConstructGrid_3(twoCoalition_set);  //这是均匀划分的方式
        }

        return grids;
    }

    //TODO 随机取出m个长度为len的元素
    public ArrayList<FeatureSubset> randomSubsets(int m, int len, double[] given_weights, double halfSum){
        ArrayList<FeatureSubset> subsets = new ArrayList<>();  //所有生成的subset的集合
        Random random = new Random();
        for (int i = 0; i < m; i++) {
            Set<Integer> subset = new HashSet<>();  //一个生成的subset(其中元素不重复)
            while (subset.size() < len) {
                subset.add(random.nextInt(this.num_of_features));
            }
            ArrayList<Integer> name = new ArrayList<>(subset);
            FeatureSubset ele = new FeatureSubset(name, value_voting(name, given_weights, halfSum));
            subsets.add(ele);
        }
        return subsets;
    }

    //TODO 计算第二层（被减去的特征子集的长度为2）
    private void computeLevelTwo(ArrayList<FeatureSubset> twoCoalition_set) {

        //1）对每个特征，取出对应的序列，并且存储到网格中
        for(int i=0; i<Info.num_of_features_voting; i++){ // Zi

            ArrayList<FeatureSubset> list = new ArrayList<>(twoCoalition_set);  //先把所有2-联盟都复制
           //去掉一行 + 一列
            for(FeatureSubset ele : twoCoalition_set){
                if(ele.name.contains(i)){
                    list.remove(ele);
                }
            }

            //2)存入网格中
            Grid gridStructure = new Grid();
            gridStructure.ConstructGrid_2(list);  //这是均匀划分的方式

            //3）计算

        }


    }

    //TODO 计算下一层的shapley value (网格中数量基本一致)
    // 原版是computeNextLevel_5()
    // weight[] 计算每层的采样权重
    // checkAndAllocate_transfer_4（）：按照每组个数等比例抽样
    // ConstructGrid_2（）可传入Arraylist  (定长数组不方便操作)
    // 当前层生成的元素 list ：ArrayList();
    private Grid computeNextLevel_voting(int level, Grid grid, double[] given_weights, ShapMatrixEntry[] shap_matrix, Allocation all, double halfSum) {

        //************ 打印 **********
//        for(ArrayList<FeatureSubset> ele : grid.grids_row){
//            System.out.print(ele.size() + "\t");
//        }

        // 1) 初始化
        Grid gridStructure = new Grid();  //用网格记录这一层


        // 记录这一层的shapley 值计算情况（因为每个特征对应的采样数可能不一致，需要结构体记录采样数量）
        ShapMatrixEntry[] temp = new ShapMatrixEntry[Info.num_of_features_voting];
        for(int i=0; i<Info.num_of_features_voting; i++){
            temp[i] = new ShapMatrixEntry();
        }
        // 记录当前层生成的特征子集，为下一层做好准备
        ArrayList<FeatureSubset> list = new ArrayList<>();
        //数组的大小一直都是问题，不如改成Arraylist
//        int arrLength = Math.max(all.level_samples * (Info.num_of_features-1), Info.num_of_features *(Info.num_of_features-1));
//        FeatureSubset[] list = new FeatureSubset[arrLength];  //【改】 换成了结构体中的 设定样本 + 补偿
        //因为一个采样，就会生成最多（n-j）种组合, j是联盟的特征个数   // 这个list做了最大估计，会有空的,并且空的都在后面

        //2）按照每组个数等比例抽样 double halfSum
        all.levelAllocate(level, grid, Info.num_of_samples, all.allocations);

        //************打印**********
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
                for(int i=0; i<Info.num_of_features_voting; i++){
                    if( !random_sample.name.contains(i)){   //挑选出的random_sample不包含当前的特征，就可以构成新的
                        ArrayList<Integer> name = new ArrayList<>(random_sample.name);
                        name.add(i);
                        double value = value_voting(name, given_weights, halfSum);
                        FeatureSubset newFeaSub = new FeatureSubset(name, value);

//                        gridStructure.max = Math.max(gridStructure.max, value);
//                        gridStructure.min = Math.min(gridStructure.min, value);

                        //3）计算shapley value 并填写到矩阵中
                        temp[i].sum += value - random_sample.value_fun; //V(S U i) - V(S), random_sample就是S
                        temp[i].count ++;

                        //4）存储新生成特征子集S，为下一层选取做好准备
                        list.add(newFeaSub);
//                        list[index] = newFeaSub;
//                        index++;
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
        gridStructure.ConstructGrid_2(list);  //这是均匀划分的方式

        return gridStructure;
    }

    //TODO 【版本2】
    // 添加功能：去除list中重复元素
    //原版是computeNextLevel_5()
    // weight[] 计算每层的采样权重； checkAndAllocate_transfer_4（）：按照每组个数等比例抽样；当前层生成的元素 list ：ArrayList();
    private Grid computeNextLevel_voting_2(int level, Grid grid, double[] given_weights, ShapMatrixEntry[] shap_matrix, Allocation all, double halfSum) {

        //************ 打印 print **********
//        for(ArrayList<FeatureSubset> ele : grid.grids_row){
//            System.out.print(ele.size() + "\t");
//        }

        // 1) 初始化
        Grid gridStructure = new Grid();  //用网格记录这一层
        // 记录这一层的shapley 值计算情况（因为每个特征对应的采样数可能不一致，需要结构体记录采样数量）
        ShapMatrixEntry[] temp = new ShapMatrixEntry[Info.num_of_features_voting];
        for(int i=0; i<Info.num_of_features_voting; i++){
            temp[i] = new ShapMatrixEntry();
        }

        // 记录当前层生成的特征子集，为下一层做好准备
//        ArrayList<FeatureSubset> list = new ArrayList<>();

        //【添加步骤】list中去除重复元素
        HashSet<ArrayList<Integer>> nameSet = new HashSet<>();  //HashSet去除重复
        ArrayList<FeatureSubset> setWithoutDuplicates = new ArrayList<>();

        //2）按照每组个数等比例抽样 double halfSum
        all.levelAllocate(level, grid, Info.num_of_samples, all.allocations);

        //************ 打印 print **********
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
                for(int i=0; i<Info.num_of_features_voting; i++){
                    if( !random_sample.name.contains(i)){   //挑选出的random_sample不包含当前的特征，就可以构成新的
                        ArrayList<Integer> name = new ArrayList<>(random_sample.name);
                        name.add(i);
                        double value = value_voting(name, given_weights, halfSum);
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

    //TODO 【版本3】 网格不排序存入
    //原版是computeNextLevel_5()
    // weight[] 计算每层的采样权重； checkAndAllocate_transfer_4（）：按照每组个数等比例抽样；当前层生成的元素 list ：ArrayList();
    private Grid computeNextLevel_voting_3(int level, Grid grid, double[] given_weights, ShapMatrixEntry[] shap_matrix, Allocation all, double halfSum) {

        //************ 打印 print **********
//        for(ArrayList<FeatureSubset> ele : grid.grids_row){
//            System.out.print(ele.size() + "\t");
//        }

        // 1) 初始化
        Grid gridStructure = new Grid();  //用网格记录这一层
        // 记录这一层的shapley 值计算情况（因为每个特征对应的采样数可能不一致，需要结构体记录采样数量）
        ShapMatrixEntry[] temp = new ShapMatrixEntry[Info.num_of_features_voting];
        for(int i=0; i<Info.num_of_features_voting; i++){
            temp[i] = new ShapMatrixEntry();
        }

        // 记录当前层生成的特征子集，为下一层做好准备
//        ArrayList<FeatureSubset> list = new ArrayList<>();

        //【添加步骤】list中去除重复元素
        HashSet<ArrayList<Integer>> nameSet = new HashSet<>();  //HashSet去除重复
        ArrayList<FeatureSubset> setWithoutDuplicates = new ArrayList<>();

        //2）按照每组个数等比例抽样 double halfSum
        all.levelAllocate(level, grid, Info.num_of_samples, all.allocations);

        //************ 打印 print **********
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
                for(int i=0; i<Info.num_of_features_voting; i++){
                    if( !random_sample.name.contains(i)){   //挑选出的random_sample不包含当前的特征，就可以构成新的
                        ArrayList<Integer> name = new ArrayList<>(random_sample.name);
                        name.add(i);
                        double value = value_voting(name, given_weights, halfSum);
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
        gridStructure.ConstructGrid_3(setWithoutDuplicates);  //这是均匀划分的方式

        return gridStructure;
    }

    //TODO 【版本4】 不排序 + 子集名字不排序
    //原版是computeNextLevel_5()
    // weight[] 计算每层的采样权重； checkAndAllocate_transfer_4（）：按照每组个数等比例抽样；当前层生成的元素 list ：ArrayList();
    private Grid computeNextLevel_voting_4(int level, Grid grid, double[] given_weights, ShapMatrixEntry[] shap_matrix, Allocation all, double halfSum) {

        //************ 打印 print **********
//        for(ArrayList<FeatureSubset> ele : grid.grids_row){
//            System.out.print(ele.size() + "\t");
//        }

        // 1) 初始化
        Grid gridStructure = new Grid();  //用网格记录这一层
        // 记录这一层的shapley 值计算情况（因为每个特征对应的采样数可能不一致，需要结构体记录采样数量）
        ShapMatrixEntry[] temp = new ShapMatrixEntry[Info.num_of_features_voting];
        for(int i=0; i<Info.num_of_features_voting; i++){
            temp[i] = new ShapMatrixEntry();
        }

        // 记录当前层生成的特征子集，为下一层做好准备
//        ArrayList<FeatureSubset> list = new ArrayList<>();

        //【添加步骤】list中去除重复元素
        HashSet<ArrayList<Integer>> nameSet = new HashSet<>();  //HashSet去除重复
        ArrayList<FeatureSubset> setWithoutDuplicates = new ArrayList<>();

        //2）按照每组个数等比例抽样 double halfSum
        all.levelAllocate(level, grid, Info.num_of_samples, all.allocations);

        //************ 打印 print **********
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
                for(int i=0; i<Info.num_of_features_voting; i++){
                    if( !random_sample.name.contains(i)){   //挑选出的random_sample不包含当前的特征，就可以构成新的
                        ArrayList<Integer> name = new ArrayList<>(random_sample.name);
                        name.add(i);
                        double value = value_voting(name, given_weights, halfSum);
                        FeatureSubset newFeaSub = new FeatureSubset(name, value);

//                        gridStructure.max = Math.max(gridStructure.max, value);
//                        gridStructure.min = Math.min(gridStructure.min, value);

                        //3）计算shapley value 并填写到矩阵中
                        temp[i].sum += value - random_sample.value_fun; //V(S U i) - V(S), random_sample就是S
                        temp[i].count ++;

                        //4）存储新生成特征子集S，为下一层选取做好准备 (只存储不重复的元素)
//                        Collections.sort(newFeaSub.name);  //把元素的名字按顺序整理排好
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
        gridStructure.ConstructGrid_3(setWithoutDuplicates);  //这是均匀划分的方式

        return gridStructure;
    }


    //TODO 利用两个矩阵(evaluateMatrix + levelMatrix)优化性能-voting_game
    // 原版是 computeMatrix_voting()
    private void computeMatrix_voting_3(ShapMatrixEntry[] shap_matrix, int[][] evaluateMatrix, int[][] levelMatrix, int level_index){
        // 有一些层可以被pruning
        if(level_index > 2){
            //第一层是单独特征
            for(int fea = 0; fea< this.num_of_features; fea ++){
                shap_matrix[fea].sum += levelMatrix[0][0];
                shap_matrix[fea].count++;   //前面的几层都是0；值(sum)不用变，数量++；证明经过了计算
            }
            //前面几层的值都是0，可以被直接跳过
            for(int i =0 ; i<level_index-1; i++){
                int value = levelMatrix[i+1][0] - levelMatrix[i][0];
                for(int fea = 0; fea< this.num_of_features; fea ++){
                    shap_matrix[fea].sum += value;
                    shap_matrix[fea].count++;   //前面的几层都是0；值(sum)不用变，数量++；证明经过了计算
                }
            }
        }
        //要从第二层开始算
        else {
            // 0) 初始化
            long one_feature_sum = 0; //记录对角线元素的值

            // 1）读取1-联盟的shapley value
            for (int i = 0; i < this.num_of_features; i++) {
                shap_matrix[i].sum += evaluateMatrix[i][i];  //对角线上的元素依次填入shapley value[]的矩阵中
                shap_matrix[i].count++;
                one_feature_sum += evaluateMatrix[i][i];  //对角线求和
            }

            // 2) 读取和计算2-联盟的shapley value
            for (int i = 0; i < this.num_of_features; i++) {  //i 是横坐标  一行就对应一个特征
                long line_sum = 0;
                for (int j = 0; j < this.num_of_features; j++) {  //j是纵坐标
                    line_sum += evaluateMatrix[i][j];
                }
                shap_matrix[i].sum += 1.0 * (line_sum - one_feature_sum) / (this.num_of_features - 1);  //第二层的shapley value
                shap_matrix[i].count++;
            }
        }
    }

    //TODO 构建矩阵存储连续的联盟值
    private int[][] constructLevelMatrix(double[] given_weights, double halfSum){
        int[][] matrix = new int[Info.num_of_features_voting][Info.num_of_features_voting];
        for(int i=0; i<Info.num_of_features_voting; i++) {   //i: 横坐标 (也是第1个item)
            ArrayList<Integer> subset = new ArrayList<>();
            subset.add(i);  //单个特征联盟，1-联盟
            matrix[i][i] = value_voting(subset, given_weights, halfSum);
            for(int j=i+1; j<Info.num_of_features_voting; j++) {   //j: 纵坐标(也是第2个item)
                subset.add(j);
                matrix[i][j] = matrix[j][i] = value_voting(subset, given_weights, halfSum);  //复制两份
            }
        }
        return matrix;
    }

    //TODO 判断从哪层开始计算
    private int checkLevel(int[][] levelMatrix) {
        /* 记录开始计算的层数 */
        int line_ind = 0;
        //在每个对角线上检查
        int limit_step = levelMatrix.length * Info.allConf;
        for(int step = 0; step < limit_step; step ++){  //line_ind 是对角线的标记
            int line_max = levelMatrix[step][0];   // 这条对角线的最小值
            int line_min = levelMatrix[step][0];   // 这条对角线的最大值
            for(int i=0; (i + step) < levelMatrix.length; i++){
                if(line_max < levelMatrix[i][i+step]){
                    line_max = levelMatrix[i][i+step];
                }
                else if(line_min > levelMatrix[i][i+step]){
                    line_min = levelMatrix[i][i+step];
                }
            }
            if(line_max != line_min ){
                line_ind = step;
                break;  //只是为了检测是否为0
            }
        }
        return line_ind;
    }

    //TODO 输出和打印shapley value
    private void printShapleyValue(ShapMatrixEntry[] shap_matrix) {
        for(ShapMatrixEntry entry : shap_matrix){
            entry.sum = entry.sum / entry.count;
//            System.out.println(i + " : " + entry.sum + "\t  ");
        }
    }

    //TODO 计算最大误差 (Voting game & Airport game)
    private double computeMaxError(ShapMatrixEntry[] shap_matrix, double[] exact) {
        double error_max = 0;  //误差总量，最后除以特征数
        for(int i=0; i<Info.num_of_features_voting; i++){
            error_max = Math.max(Math.abs((shap_matrix[i].sum - exact[i]) / exact[i]), error_max);
        }
        return error_max;
    }

    //TODO 计算平均误差 (Voting game & Airport game)
    private double computeAverageError(ShapMatrixEntry[] shap_matrix, double[] exact) {
        double error_ave = 0;  //误差总量，最后除以特征数
        for(int i=0; i<Info.num_of_features_voting; i++){
            error_ave += Math.abs((shap_matrix[i].sum - exact[i]) / exact[i]);
        }
        error_ave = error_ave / Info.num_of_features_voting;
        return error_ave;
    }

    //TODO 定义 Voting game函数v(S)
    private int value_voting(ArrayList<Integer> subset, double[] given_weights, double halfSum) {
        int weights_sum = 0;
        //求weights_sum
        for(int ele : subset){
            weights_sum += given_weights[ele];
        }
        if(weights_sum > halfSum){
            return 1;
        }
        else return 0;
    }


}
