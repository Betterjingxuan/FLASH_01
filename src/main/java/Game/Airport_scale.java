package Game;

import structure.*;
import Global.*;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

//TODO 改变feature的数量，测试运行时间

public class Airport_scale {

    int num_of_features = Info.num_of_features;   //改成了从原始路径设置的feature数量

    double[] given_weights = new double[Info.num_of_features];  //这是新生成的airport_weight

    double[] benchmark_sv = new double[Info.num_of_features];

    //TODO 随机生成n个特征和对应的权重 （n = Info.num_of_features）
    public void featureGenerate(){

        // 创建一个 Random 对象
        Random random = new Random();

        //原始数组
        double[] originalArray = Info.given_weights_airport;
        //原始数组长度
        int leng = Info.given_weights_airport.length;
        //连续从数组中挑选出来，存入数组arr
        for (int i = 0; i < this.num_of_features; i++) {
            int randomIndex = random.nextInt(leng); // 随机选择原数组中的索引
            this.given_weights[i] = originalArray[randomIndex]; // 将选择的元素添加到新数组中
        }

        //把数组arr[] 存入文件中

        //把它改成直接读入npy文件

    }

    /*TODO [Airport_game]
     * given_weights: 给定的weights[] 用于计算value_function*/
    public void airport_game() {

        /*【改】：改用given_weight*/
        String path = Info.ROOT + "airport_" + this.num_of_features + ".npy";
        DataInputStream dataInputStream = null;

        try {
            dataInputStream = new DataInputStream(Files.newInputStream(Paths.get(path)));
            NumpyReader reader = new NumpyReader();
            this.given_weights = reader.readIntArray(dataInputStream);  //读入.npy文件中存储的数组
        }
        catch (Exception e) {
            throw new RuntimeException(e);
        }

        /*我觉得经过改造后，这俩函数等价的: ShapleyApproximate_airport_1() == ShapleyApproximate_airport_2() */  // ==>版本1准确率比较高
        long time_1 = System.currentTimeMillis();

        ShapMatrixEntry[] shap_matrix = ShapleyApproximate_airport_1(this.given_weights); //使用矩阵计算(复制MiningSHAP,改造)  ==>

//        ShapMatrixEntry[] shap_matrix = ShapleyApproximate_airport_2(given_weights); //使用矩阵计算(复制voting Game,改成airport)

        long time_2 = System.currentTimeMillis();

        /*【改】：读入benchmark : airport_exact*/
        String benchmark_path = Info.benchmark_path;
        FileOpera opera = new FileOpera();
        this.benchmark_sv = opera.file_read(benchmark_path, this.num_of_features);  //读入benchmark

        double error_max = computeMaxError(shap_matrix, this.benchmark_sv); //计算最大误差

        double error_ave = computeAverageError(shap_matrix, this.benchmark_sv);  //计算平均误差

        System.out.println("Airport Game:  " + "error_ave: " + error_ave + " \t"  +  "error_max: " + error_max );
        System.out.println("airport scale time : " + (time_2 - time_1) * 0.001 );  //+ "S"
    }


    /*TODO [版本2] 不排序放入网格
     * given_weights: 给定的weights[] 用于计算value_function*/
    public void airport_game_2() {

        /*【改】：改用given_weight*/
        String path = Info.ROOT + "airport_" + this.num_of_features + ".npy";
        DataInputStream dataInputStream = null;

        try {
            dataInputStream = new DataInputStream(Files.newInputStream(Paths.get(path)));
            NumpyReader reader = new NumpyReader();
            this.given_weights = reader.readIntArray(dataInputStream);  //读入.npy文件中存储的数组
        }
        catch (Exception e) {
            throw new RuntimeException(e);
        }

        //1) 用一个大数组记录特征对应的shapley value之和
        ShapMatrixEntry[] shap_matrix = new ShapMatrixEntry[this.num_of_features];
        for(int i=0; i<shap_matrix.length; i++){
            shap_matrix[i] = new ShapMatrixEntry();
        }

        /*我觉得经过改造后，这俩函数等价的: ShapleyApproximate_airport_1() == ShapleyApproximate_airport_2() */  // ==>版本1准确率比较高
        long time_1 = System.currentTimeMillis();

        shap_matrix = ShapleyApproximate_airport_3(shap_matrix, this.given_weights); //使用矩阵计算(复制MiningSHAP,改造)  ==>

//        ShapMatrixEntry[] shap_matrix = ShapleyApproximate_airport_2(given_weights); //使用矩阵计算(复制voting Game,改成airport)

        long time_2 = System.currentTimeMillis();

        printShapleyValue(shap_matrix);

        /*【改】：读入benchmark : airport_exact*/
        String benchmark_path = Info.benchmark_path;
        FileOpera opera = new FileOpera();
        this.benchmark_sv = opera.file_read(benchmark_path, this.num_of_features);  //读入benchmark

        double error_max = computeMaxError(shap_matrix, this.benchmark_sv); //计算最大误差

        double error_ave = computeAverageError(shap_matrix, this.benchmark_sv);  //计算平均误差

        System.out.println("Airport Game:  " + "error_ave: " + error_ave + " \t"  +  "error_max: " + error_max );
        System.out.println("airport scale time (NO sort): " + (time_2 - time_1) * 0.001 );  //+ "S"
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
        int[][] evaluateMatrix = constructEvaluateMatrix(this.given_weights); //第0层和第1层全算（层数= 减去的特征子集长度）

        //4）扫描evaluateMatrix，计算1-联盟 & 2-联盟的shapley值  //第0层和第1层全算（层数= 减去的特征子集长度）
        computeMatrix_airport(shap_matrix, this.given_weights, evaluateMatrix);

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

    //TODO [版本1]复制MiningShap_6
    private ShapMatrixEntry[] ShapleyApproximate_airport_3(ShapMatrixEntry[] shap_matrix, double[] given_weights) {

        //2）给每层分配样本
        Allocation allo = new Allocation();
//        allo.sampleAllocation_2(this.num_of_features);
        allo.sampleAllocation_3(this.num_of_features); //sample_num_level 每一层采样几个

        //3) 计算出一个大矩阵，存储1-联盟 & 2-联盟的值
        int[][] evaluateMatrix = constructEvaluateMatrix(this.given_weights); //第0层和第1层全算（层数= 减去的特征子集长度）

        //4）扫描evaluateMatrix，计算1-联盟 & 2-联盟的shapley值  //第0层和第1层全算（层数= 减去的特征子集长度）
        computeMatrix_airport(shap_matrix, this.given_weights, evaluateMatrix);

        // 5)计算第2层（level = 被减去的特征的长度）
        Grid grid_level = initialLevelCompute_airport_2(shap_matrix, given_weights, evaluateMatrix);  //level =2 （存储长度为2的特征子集）

        //4)依次生成联盟 (ind / level 等于被减去的联盟的长度)
        for(int ind=2; ind <this.num_of_features; ind++){
            grid_level = computeNextLevel_airport_3(ind, grid_level, given_weights, shap_matrix, allo);  //不排序直接放网格
        }

        return shap_matrix;
    }

    //TODO 构建evaluateMatrix, 记录1-联盟 & 2-联盟的值
    private int[][] constructEvaluateMatrix(double[] given_weights) {
        int[][] matrix = new int[this.num_of_features][this.num_of_features];
        for(int i=0; i<this.num_of_features; i++){   //i: 横坐标 (也是第1个item)
            ArrayList<Integer> subset = new ArrayList<>();
            subset.add(i);  //单个特征联盟，1-联盟
            matrix[i][i] = value_airport_2(subset, given_weights);
            for(int j=i+1; j<this.num_of_features; j++){   //j: 纵坐标(也是第2个item)
                ArrayList<Integer> twoCoalition = new ArrayList<>(subset);
                twoCoalition.add(j);
                matrix[i][j] = matrix[j][i] = value_airport_2(twoCoalition, given_weights);  //复制两份
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

    //TODO 计算初始层，并存储到网格中(3-联盟-2联盟的值) => 利用矩阵EvaluateMatrix
    // 原版是 computeSingleFeature_voting()
    private Grid initialLevelCompute_airport_2(ShapMatrixEntry[] shap_matrix, double[] given_weights, int[][] evaluateMatrix) {

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
        grids.ConstructGrid_3(twoCoalition_set);  //这是均匀划分的方式

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

        //2）按照每组个数等比例抽样 double halfSum
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
                        double value = value_airport_2(name, given_weights);
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

    //TODO 【版本3】 不排序直接放入网格
    //computeNextLevel_airport()
    // weight[] 计算每层的采样权重； checkAndAllocate_transfer_4（）：按照每组个数等比例抽样；当前层生成的元素 list ：ArrayList();
    private Grid computeNextLevel_airport_3(int level, Grid grid, double[] given_weights, ShapMatrixEntry[] shap_matrix, Allocation all) {

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

        //2）按照每组个数等比例抽样 double halfSum
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
                        double value = value_airport_2(name, given_weights);
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
        gridStructure.ConstructGrid_3(setWithoutDuplicates);  //不排序

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
        for(int i=0; i<this.num_of_features; i++){
            error_max = Math.max(Math.abs((shap_matrix[i].sum - exact[i]) / exact[i]), error_max);
        }
        return error_max;
    }

    //TODO 计算平均误差 (Voting game & Airport game)
    private double computeAverageError(ShapMatrixEntry[] shap_matrix, double[] exact) {
        double error_ave = 0;  //误差总量，最后除以特征数
        for(int i=0; i<this.num_of_features; i++){
            error_ave += Math.abs((shap_matrix[i].sum - exact[i]) / exact[i]);
        }
        error_ave = error_ave / this.num_of_features;
        return error_ave;
    }

    //TODO 因为scale时生成的weight无序，所以不能直接用下标判断对应的返回值
    private int value_airport_2(ArrayList<Integer> subset, double[] given_weights) {
        int maxValue = -1;
        for(int ele : subset){
            maxValue = Math.max((int)given_weights[ele], maxValue);
        }
        return  maxValue;  //返回值是最大值
    }

}
