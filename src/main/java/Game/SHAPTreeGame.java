package Game;

import Global.*;
import structure.*;
import java.util.*;

//希望通过mining的方式获得shapley value
/* [改进] 原版是 MiningShap_6
* 1） 采样方式的改进：log 降维
* 5)
* */
public class SHAPTreeGame {

    /*TODO [SpanningTree_game]
     * spanningTree 不需要传参，没有weights[]*/
    public void spanningTree_game() {

        int[][] matrix = generateMatrix();
        ShapMatrixEntry[] shap_matrix = ShapleyApproximate_tree(matrix);  //均匀网格 + 比例分配每层样本

        double error_max = computeMaxError(shap_matrix, Info.tree_exact); //计算最大误差

        double error_ave = computeAverageError(shap_matrix, Info.tree_exact);  //计算平均误差
        System.out.println("Tree Game: " + "error_max: " + error_max + " \t" + "error_ave: " + error_ave );

    }

    /*TODO [SpanningTree_game]
     * 版本2：使用评估矩阵 */
    public void spanningTree_game_2() {

        int[][] matrix = generateMatrix();
        ShapMatrixEntry[] shap_matrix = ShapleyApproximate_tree_2(matrix);  //[版本2：使用矩阵] 均匀网格 + 比例分配每层样本

        double error_max = computeMaxError(shap_matrix, Info.tree_exact); //计算最大误差

        double error_ave = computeAverageError(shap_matrix, Info.tree_exact);  //计算平均误差
        System.out.println("Tree Game: " + "error_max: " + error_max + " \t" + "error_ave: " + error_ave );

    }

    //TODO 生成大矩阵 (对齐特征, N={0, 2, ... , 99}))
    private int[][] generateMatrix() {
        //预设的特征取值为[0, 99], 函数中的值为[1, 100];  //因此需要转换
        int[][] matrix = new int[Info.num_of_features_tree] [Info.num_of_features_tree]; //
        for(int i=0; i<Info.num_of_features_tree; i++){
            for(int j=0; j<Info.num_of_features_tree; j++){
                if((i==j+1) || (i==j-1) ||(i==0 && j==Info.num_of_features_tree-1) || (i==Info.num_of_features_tree-1 && j==0)){
                    matrix[i][j] = 1;
                }
//                else if(i==0 || j==0){
//                    matrix[i][j] = 101;
//                }
                else{
                    matrix[i][j] = (int) Float.POSITIVE_INFINITY; //表示正无穷
//                    matrix[i][j] = 999999;
                }
            }
        }
        return matrix;
    }

    //TODO 生成大矩阵 (按照原版, N={1, 2, ... , 100})
    private int[][] generateMatrix_0() {
        //预设的特征取值为[0, 99], 函数中的值为[1, 100];  //因此需要转换
        int[][] matrix = new int[Info.num_of_features_tree +1] [Info.num_of_features_tree +1]; //
        for(int i=0; i<Info.num_of_features_tree; i++){
            for(int j=0; j<Info.num_of_features_tree; j++){
                if((i==j+1) || (i==j-1) ||(i==1 && j==Info.num_of_features_tree) || (i==Info.num_of_features_tree && j==1)){
                    matrix[i][j] = 1;
                }
                else if(i==0 || j==0){
                    matrix[i][j] = 101;
                }
                else{
                    matrix[i][j] = (int) Float.POSITIVE_INFINITY; //表示正无穷
//                    matrix[i][j] = 999999;
                }
            }
        }
        return matrix;
    }

    //找到序列中被省略的那个元素
    private int eleSearch(ArrayList<Integer> eleSet) {
        int ele = -1;  //这是不被包含的特征
        for(int i=0; i<Info.num_of_features_tree; i++){
            if( !eleSet.contains(i)){
                ele = i;
            }
        }
        return ele;
    }

    //TODO 输出和打印shapley value
    private void printShapleyValue(double[] shap_matrix) {
        int i=0;
        for(double value : shap_matrix){
            value = value / Info.num_of_features_tree;
            System.out.println(i + " : " + value + "\t  ");
            i++;
        }
    }

    //TODO 输出和打印shapley value
    private void printShapleyValue(ShapMatrixEntry[] shap_matrix) {
        int i=0;
        for(ShapMatrixEntry entry : shap_matrix){
            entry.sum = entry.sum / entry.count;
            System.out.println(i + " : " + entry.sum + "\t  ");
            i++;
        }
    }

    //TODO 输出和打印shapley value
    private void printShapleyValue_test(double[] shap_matrix) {
        int i=0;
        for(double value : shap_matrix){
            value = value / Info.num_of_features_tree;
            System.out.print(i + ": " + value + "\t");
            i++;
        }
//        System.out.println();
    }

    //TODO 计算下一层的shapley value (网格中数量基本一致)
    // 原版：computeNextLevel_5()
    private Grid computeNextLevel_tree(int level, Grid grid_1, ShapMatrixEntry[] shap_matrix, Allocation all, int[][] matrix) {

        //**********************
        for(ArrayList<FeatureSubset> ele : grid_1.grids_row){
            System.out.print(ele.size() + "\t");
        }

        Grid gridStructure = new Grid();  //用网格记录这一层

        // 记录这一层的情况（因为每个特征对应的采样数可能不一致，需要结构体记录采样数量）
        ShapMatrixEntry[] temp = new ShapMatrixEntry[Info.num_of_features_tree];
        for(int i=0; i<Info.num_of_features_tree; i++){
            temp[i] = new ShapMatrixEntry();
        }

        //1）从每个网格grid中随机选一个
        //[问题] 无法保证每个网格中都有值，如果网格中没有值，就把名额让个其他的
        //[修改] 添加安全检查，再根据网格分布情况分布采样的个数
        // 安全检查 + 重新分配
//        int[] allocations = checkAndAllocate_transfer_1(grid_1);  //[版本1]简单分配，每个网格取1个
//        int[] allocations = checkAndAllocate_transfer_2(level, grid_1, Info.num_of_samples);  //[版本1]简单分配，每个网格取1个
//        all.checkAndAllocate_transfer_2(level, grid_1, Info.num_of_samples);
        all.levelAllocate(level, grid_1, Info.num_of_samples, all.allocations);  //按照每组个数等比例抽样

        //**********************
        for(int num : all.allocations){
            System.out.print(num + "\t");
        }
        System.out.println();

        //[第一版]每层抽样一个，lis大小就是 采样数 * 特征数量
        //记录当前层生成的特征子集，为下一层做好准备
        ArrayList<FeatureSubset> list = new ArrayList<>();
        //数组的大小一直都是问题，不如改成Arraylist
//        int arrLength = Math.max(all.level_samples * (Info.num_of_features-1), Info.num_of_features *(Info.num_of_features-1));
//        FeatureSubset[] list = new FeatureSubset[arrLength];  //【改】 换成了结构体中的 设定样本 + 补偿
        //因为一个采样，就会生成最多（n-j）种组合, j是联盟的特征个数   // 这个list做了最大估计，会有空的,并且空的都在后面
//        int index = 0;

        //[第1版]: 下面两行是第一版的函数
//        for(ArrayList<FeatureSubset> grid : grid_1.grids_row){
//          if(grid.size() >0){  //必须让网格中有值才能抽
        //[第2版]: 添加安全检查后
        for(int ind=0; ind<all.allocations.length; ind++){  //分配方案[]：给每个网格分配的个数
            int sample_num = all.allocations[ind];  //sample_num：当前网格采样的个数
            ArrayList<FeatureSubset> grid = grid_1.grids_row[ind]; //grid：取出来的网格
            for(int counter=0; counter < sample_num; counter ++){  //count是采样的计数器
                FeatureSubset random_sample = grid_1.randomGet(grid);  //random_sample: 从grid中选出来的一个
                //2)选出来的这个FeatureSubset，与每个feature(不包含自己)构成一个新的FeatureSubset
                for(int i=0; i<Info.num_of_features_tree; i++){
                    if( !random_sample.name.contains(i)){   //挑选出的random_sample不包含当前的特征，就可以构成新的
                        ArrayList<Integer> name = new ArrayList<>(random_sample.name);
                        name.add(i);
                        int value = value_tree(name, matrix);
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

    //TODO 安全检查：是否需要重新分配
    private boolean checkAllocation(Grid grid_1) {
        boolean allocate_flag = false;
        //1)检查是否需要重新分配
        for (ArrayList<FeatureSubset> grid : grid_1.grids_row) {
            //2)检查是否有网格为空
            if (grid.size() == 0) {
                allocate_flag = true;
                break;
            }
        }
        return allocate_flag;
    }

    //TODO 分配样本的方法：根据网格分布的情况分配，解决数据倾斜的问题
    // grid_1 ：为这个网格分配样本
    /*假设[0, 24, 30, 74, 72, 94, 140, 142, 162, 182]   total_weihts=920
    * 刚开始 [0,0,0,0,0,1,1,1,1,1] 前面分不到的永远都分不到，这个程序会死循环
    * */
    private int[] allocateSamples(Grid grid_1, int total_samples, int[] allocations, int total_weihts) {
//        int[] allocations = new int[Info.num_of_grids]; // 表示为每个网格分配的采样数，故数组长度为网格个数

        // 2）按照 [比例 * 采样总数] 分配, 比例 = 网格元素 / 总元素total
        int remaining_samples = total_samples;
        for(int ind=0; ind < grid_1.grids_row.length; ind++){
            ArrayList<FeatureSubset> grid = grid_1.grids_row[ind];
            if(grid.size() == 0){
                allocations[ind] = 0;
            }
            else{
                allocations[ind] += total_samples * grid.size() / total_weihts;
                remaining_samples -= total_samples * grid.size() / total_weihts;
            }
        }
            //5) 若样本没分配完，要继续分配
            if(remaining_samples != 0){
                allocateSamples (grid_1, remaining_samples, allocations, total_weihts);
            }
        return allocations;
    }


    //TODO 计算最大误差 (Voting game & Airport game)
    private double computeMaxError(ShapMatrixEntry[] shap_matrix, double[] exact) {
        double error_max = 0;  //误差总量，最后除以特征数
        for(int i=0; i<Info.num_of_features_tree; i++){
            error_max = Math.max(Math.abs((shap_matrix[i].sum - exact[i]) / exact[i]), error_max);
        }
        return error_max;
    }

    //TODO 计算最大误差 (Shoes game & Tree game)
    private double computeMaxError(ShapMatrixEntry[] shap_matrix, double exact) {
        double error_max = 0;  //误差总量，最后除以特征数
        for(int i=0; i<Info.num_of_features_tree; i++){
            error_max = Math.max(Math.abs((shap_matrix[i].sum - exact) / exact), error_max);
        }
        return error_max;
    }

    //TODO 计算平均误差 (Voting game & Airport game)
    private double computeAverageError(ShapMatrixEntry[] shap_matrix, double[] exact) {
        double error_ave = 0;  //误差总量，最后除以特征数
        for(int i=0; i<Info.num_of_features_tree; i++){
            error_ave += Math.abs((shap_matrix[i].sum - exact[i]) / exact[i]);
        }
        error_ave = error_ave / Info.num_of_features_tree;
        return error_ave;
    }

    //TODO 计算平均误差 (Shoes game & Tree game)
    private double computeAverageError(ShapMatrixEntry[] shap_matrix, double exact) {
        double error_ave = 0;  //误差总量，最后除以特征数
        for(int i=0; i<Info.num_of_features_tree; i++){
            error_ave += Math.abs((shap_matrix[i].sum - exact) / exact);
        }
        error_ave = error_ave / Info.num_of_features_tree;
        return error_ave;
    }

    /*TODO [Shoes Game]：网格均匀划分 + 每层样本按比例分配
    *  原版是 ShapleyApproximate_airport_4() */
    public ShapMatrixEntry[] ShapleyApproximate_tree(int[][] matrix) {

//        int[][] evaluateMatrix = constructEvaluateMatrix(matrix);

        //1) 用一个大数组记录特征对应的shapley value之和
        ShapMatrixEntry[] shap_matrix = new ShapMatrixEntry[Info.num_of_features_tree];
        for(int i=0; i<shap_matrix.length; i++){
            shap_matrix[i] = new ShapMatrixEntry();
        }

        //2）给每层分配样本
        Allocation allo = new Allocation();
        allo.sampleAllocation_2(Info.num_of_features_tree-1); //sample_num_level 每一层采样几个 【逻辑错误：最后一层没有】
//        allo.sampleAllocation_3(Info.num_of_features_tree); //sample_num_level 每一层采样几个 【需要配合评估矩阵使用，前两层都是0】

        // 3）为单个特征计算边际贡献（所有单特征） computeSingleFeature_3()均匀的网格
        Grid grid_level = computeSingleFeature_tree(shap_matrix, matrix);    //grid_level: 每层的循环变量

        // 5)生成2-联盟
//        Grid grid_level = new Grid();
//        grid_level = computeNextLevel(grid_1, sum, given_weights, shap_matrix);
//        System.out.println("level initiall: " + grid_level.grids_row[0].get(0).name.size());

        //4)依次生成联盟
        for(int ind=1; ind <Info.num_of_features_tree; ind++){
            //【备注】ind 就是用来代替grid_level.grids_row[0].get(0).name.size()，表示这个网格中存储的subset中包含几个特征
            // 因为网格中不一定都有值的，可能网格中不存在元素： grid_level.grids_row[0].get(0).name.size() 可能空指针异常
            // 5) 计算某一层，存储到网格中
//            grid_level = computeNextLevel_3(ind, grid_level, sum, given_weights, shap_matrix);  //这是均匀划分的方式
            grid_level = computeNextLevel_tree(ind, grid_level, shap_matrix, allo, matrix);  //这是均匀划分 + 比例抽样(log降维)
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

    /*TODO [Shoes Game]：网格均匀划分 + 每层样本按比例分配
     *  原版是 ShapleyApproximate_airport_4() */
    public ShapMatrixEntry[] ShapleyApproximate_tree_2(int[][] matrix) {

//        int[][] evaluateMatrix = constructEvaluateMatrix(matrix);

        //1) 用一个大数组记录特征对应的shapley value之和
        ShapMatrixEntry[] shap_matrix = new ShapMatrixEntry[Info.num_of_features_tree];
        for(int i=0; i<shap_matrix.length; i++){
            shap_matrix[i] = new ShapMatrixEntry();
        }

        //2）给每层分配样本
        Allocation allo = new Allocation();
//        allo.sampleAllocation_2(Info.num_of_features_tree-1); //sample_num_level 每一层采样几个 【逻辑错误】最后一层是0
        allo.sampleAllocation_3(Info.num_of_features_tree); //sample_num_level 每一层采样几个

        //3) 计算出一个大矩阵，存储1-联盟 & 2-联盟的值
        int[][] evaluateMatrix = constructEvaluateMatrix(matrix); //第0层和第1层全算（层数= 减去的特征子集长度）

        //4）扫描evaluateMatrix，计算1-联盟 & 2-联盟的shapley值  //第0层和第1层全算（层数= 减去的特征子集长度）
        computeMatrix_tree(shap_matrix, evaluateMatrix);    //grid_level: 每层的循环变量

        // 5)计算第2层（level = 被减去的特征的长度）
        Grid grid_level = initialLevelCompute_tree(shap_matrix, evaluateMatrix);  //level =2 （存储长度为2的特征子集）
        //grid_level: 每层的循环变量

        // 5)生成2-联盟
//        Grid grid_level = new Grid();
//        grid_level = computeNextLevel(grid_1, sum, given_weights, shap_matrix);
//        System.out.println("level initiall: " + grid_level.grids_row[0].get(0).name.size());

        //4)依次生成联盟
        for(int ind=2; ind <Info.num_of_features_tree; ind++){
            //【备注】ind 就是用来代替grid_level.grids_row[0].get(0).name.size()，表示这个网格中存储的subset中包含几个特征
            // 因为网格中不一定都有值的，可能网格中不存在元素： grid_level.grids_row[0].get(0).name.size() 可能空指针异常
            // 5) 计算某一层，存储到网格中
//            grid_level = computeNextLevel_3(ind, grid_level, sum, given_weights, shap_matrix);  //这是均匀划分的方式
            grid_level = computeNextLevel_tree(ind, grid_level, shap_matrix, allo, matrix);  //这是均匀划分 + 比例抽样(log降维)
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


    //TODO 【版本3】均匀划分：为单个特征计算边际贡献（所有单特征）
    // 原版：computeSingleFeature_3()
    // shap_matrix：存储所有特征SV.的大矩阵
    private Grid computeSingleFeature_tree(ShapMatrixEntry[] shap_matrix, int[][] matrix) {
        // 0）初始化网格结构 （每层存成一个网格，用完后丢弃）
        Grid gridStructure = new Grid();  //表示这是存储1-联盟的网格
        FeatureSubset[] grid = new FeatureSubset[Info.num_of_features_tree];  //生成的所有特征子集，需要把他们都放入网格中（每层都用）

        //1）对每个1-联盟特征，生成
        for(int i=0; i<Info.num_of_features_tree; i++){ // Zi
            ArrayList<Integer> subset = new ArrayList<>();
            subset.add(i);

            // 2）计算联盟对应的值  value_voting(subset, sum)
            int value = 101;  //Tree Game 的单项集是 101  S = {1 U 0}

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

    //TODO [自己简化版] Spanning Tree Game
    private int value_tree(ArrayList<Integer> subset, int[][] matrix) {
        int sum = 0;
        Collections.sort(subset);
        for(int ind=0; ind < subset.size()-1; ind++){
            int ele_1 = subset.get(ind);  //第一个项
            int ele_2 = subset.get(ind+1); //第二个项
            if(ele_1 == ele_2 - 1){
                sum ++;  //表示联通
            }
            else{
//                return (int) Float.POSITIVE_INFINITY;
                return 0;
            }
        }
        return sum;  //返回值
    }

    //TODO [标准版]Spanning Tree Game
    private long value_tree_0(ArrayList<Integer> subset, int[][] matrix) {
        long sum = 0;
        Collections.sort(subset);
        for(int ind_1=0; ind_1 < subset.size(); ind_1++){
            int ele_1 = subset.get(ind_1);  //第一个项
            for(int ind_2=ind_1 +1; ind_2 < subset.size(); ind_2++){
                int ele_2 = subset.get(ind_2); //第二个项
                sum +=  matrix[ele_1+1][ele_2+1];  //项的取值范围[0, 99] => 映射成数组[1, 100]
            }
        }
        return sum;  //返回值
    }

    //TODO 构建evaluateMatrix, 记录1-联盟 & 2-联盟的值
    private int[][] constructEvaluateMatrix(int[][] edge_matrix) {
        int[][] matrix = new int[Info.num_of_features_tree][Info.num_of_features_tree];
        for(int i=0; i<Info.num_of_features_tree; i++){   //i: 横坐标 (也是第1个item)
            ArrayList<Integer> subset = new ArrayList<>();
            subset.add(i);  //单个特征联盟，1-联盟
            matrix[i][i] = value_tree(subset, edge_matrix);
            for(int j=i+1; j<Info.num_of_features_tree; j++){   //j: 纵坐标(也是第2个item)
                ArrayList<Integer> twoCoalition = new ArrayList<>(subset);
                twoCoalition.add(j);
                matrix[i][j] = matrix[j][i] = value_tree(twoCoalition, edge_matrix);  //复制两份
            }
        }
        return matrix;
    }

    //TODO 计算1-联盟 & 2-联盟的shapley值 => 利用矩阵EvaluateMatrix
    // 【版本2】均匀划分：为单个特征计算边际贡献（所有单特征）复制版本3，用于shoes_game
    // 原版是 computeSingleFeature_tree()
    private void computeMatrix_tree(ShapMatrixEntry[] shap_matrix, int[][] evaluateMatrix) {

        // 0) 初始化
        long one_feature_sum = 0; //记录对角线元素的值

        // 1）读取1-联盟的shapley value
        for(int i=0; i<Info.num_of_features_tree; i++){
            shap_matrix[i].sum += evaluateMatrix[i][i];  //对角线上的元素依次填入shapley value[]的矩阵中
            shap_matrix[i].count ++;
            one_feature_sum += one_feature_sum;  //对角线求和
        }

        // 2) 读取和计算2-联盟的shapley value
        for(int i=0; i<Info.num_of_features_tree; i++){  //i 是横坐标  一行就对应一个特征
            long line_sum = 0;
            for(int j=0; j<Info.num_of_features_tree; j++){  //j是纵坐标
                line_sum += evaluateMatrix[i][j];
            }
            shap_matrix[i].sum += 1.0 * (line_sum - one_feature_sum) / (Info.num_of_features_tree-1);  //第二层的shapley value
            shap_matrix[i].count ++;
        }
    }

    //TODO 计算初始层，并存储到网格中(3-联盟-2联盟的值) => 利用矩阵EvaluateMatrix
    // 原版是 computeSingleFeature_voting()
    private Grid initialLevelCompute_tree(ShapMatrixEntry[] shap_matrix, int[][] evaluateMatrix) {

        //性质：矩阵是一个对称的矩阵，所以2-联盟只需要看一半，存一半就可以

        // 1）初始化网格结构 （存3-联盟）
        Grid grids = new Grid();  //表示这是存储1-联盟的网格
        ArrayList<FeatureSubset> twoCoalition_set = new ArrayList<>();  //用list存储所有的2-项集
        //数组：虽然定长，但删除不方便

        // 1) 读取2-联盟的shapley value
        for(int i=0; i<Info.num_of_features_tree; i++){  //i 是横坐标  一行就对应一个特征
            List<Integer> subSet = new ArrayList<>();
            subSet.add(i);  //第1个特征
            for(int j=i+1; j<Info.num_of_features_tree; j++){  //j是纵坐标
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



}
