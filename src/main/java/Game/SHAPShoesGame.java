package Game;

import Global.*;
import structure.*;
import java.util.*;

//希望通过mining的方式获得shapley value
/* [改进] 原版是 MiningShap_6
* 1） 采样方式的改进：log 降维
* 5)
* */

public class SHAPShoesGame {

    int num_of_features = Info.num_of_features_shoes;

    /*TODO [Shoes_game]
     * shoes 不需要传参，没有weights[]*/
    public void shoes_game() {

        ShapMatrixEntry[] shap_matrix = ShapleyApproximate_shoes();  //均匀网格 + 比例分配每层样本

        double error_max = computeMaxError(shap_matrix, Info.shoes_exact); //计算最大误差

        double error_ave = computeAverageError(shap_matrix, Info.shoes_exact);  //计算平均误差
        System.out.println("Shoes Game: " + "error_max: " + error_max + " \t" + "error_ave: " + error_ave );

    }

    //TODO 利用两个矩阵计算
    public void shoes_game_2() {
        long ave_runtime = 0;
        double ave_error_max = 0; //计算最大误差
        double ave_error_ave = 0;
        for(int i=0; i< Info.timesRepeat; i++) {
            long time_1 = System.currentTimeMillis();
            ShapMatrixEntry[] shap_matrix = ShapleyApproximate_shoes_2();  //均匀网格 + 比例分配每层样本
            long time_2 = System.currentTimeMillis();
            double error_max = computeMaxError(shap_matrix, Info.shoes_exact); //计算最大误差
            double error_ave = computeAverageError(shap_matrix, Info.shoes_exact);  //计算平均误差

            ave_runtime += time_2 - time_1;
            ave_error_max += error_max;
            ave_error_ave += error_ave;
        }
            System.out.println("Shoes Game_2: " + "error_ave: " + ave_error_ave/Info.timesRepeat + " \t" + "error_max: " + ave_error_max/Info.timesRepeat );
            System.out.println("TwoMat time : " + (ave_runtime * 0.001)/ Info.timesRepeat );  //+ "S"

    }

    //TODO[版本3] 不排序网格 + 不排序自己子集名字
    public void shoes_game_3() {

        long ave_runtime = 0;
        double ave_error_max = 0; //计算最大误差
        double ave_error_ave = 0;
        for(int i=0; i< Info.timesRepeat; i++) {
            long time_1 = System.currentTimeMillis();
            ShapMatrixEntry[] shap_matrix = ShapleyApproximate_shoes_3();  //均匀网格 + 比例分配每层样本
            double error_max = computeMaxError(shap_matrix, Info.shoes_exact); //计算最大误差
            long time_2 = System.currentTimeMillis();
            double error_ave = computeAverageError(shap_matrix, Info.shoes_exact);  //计算平均误差

            ave_runtime += time_2 - time_1;
            ave_error_max += error_max;
            ave_error_ave += error_ave;
        }
        System.out.println("Shoes Game_3: " + "error_ave: " + ave_error_ave/Info.timesRepeat + " \t" + "error_max: " + ave_error_max/Info.timesRepeat );
        System.out.println("TwoMat + TwoNoSort time : " +  (ave_runtime * 0.001)/ Info.timesRepeat );  //+ "S"

    }

    //TODO 计算最后一层的shapley value
    private void computeLastLevel(Grid grid_level, int sum, int[] given_weights, double[] shap_matrix) {
        for(ArrayList<FeatureSubset> oneGrid : grid_level.grids_row){
            //1）对每个1-联盟特征，都加入
            ArrayList<Integer> subset = new ArrayList<>();
            for(int i=0; i<Info.num_of_features_shoes; i++){  // Zi
                subset.add(i);
            }
            // 2）计算联盟对应的值  value_voting(subset, sum)
            int value = value_airport(subset, given_weights);

            //3）方案1：遍历上层生成的每个元素，
            for(FeatureSubset set : oneGrid){
                //4）找到那个不被包含的元素
                int fea = eleSearch(set.name);
            }
            //方案2：直接构造然后求值
            for(int fea=0; fea<Info.num_of_features_shoes; fea++){
                ArrayList<Integer> otherSet = new ArrayList<>(subset);
                otherSet.remove(fea);
                int otherValue = value_airport(otherSet, given_weights);
                shap_matrix[fea] += value - otherValue;
            }
        }
    }

    //找到序列中被省略的那个元素
    private int eleSearch(ArrayList<Integer> eleSet) {
        int ele = -1;  //这是不被包含的特征
        for(int i=0; i<Info.num_of_features_shoes; i++){
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
            value = value / Info.num_of_features_shoes;
            System.out.println(i + " : " + value + "\t  ");
            i++;
        }
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

    //TODO 输出和打印shapley value
    private void printShapleyValue_test(double[] shap_matrix) {
//        int i=0;
        for(double value : shap_matrix){
            value = value / Info.num_of_features_shoes;
//            System.out.print(i + ": " + value + "\t");
//            i++;
        }
//        System.out.println();
    }


    //TODO 【版本3】均匀划分：为单个特征计算边际贡献（所有单特征）
    // 原版：computeSingleFeature_3()
    // shap_matrix：存储所有特征SV.的大矩阵
    private Grid computeSingleFeature_shoes(ShapMatrixEntry[] shap_matrix) {
        // 0）初始化网格结构 （每层存成一个网格，用完后丢弃）
        Grid gridStructure = new Grid();  //表示这是存储1-联盟的网格
        FeatureSubset[] grid = new FeatureSubset[Info.num_of_features_shoes];  //生成的所有特征子集，需要把他们都放入网格中（每层都用）

        //1）对每个1-联盟特征，生成
        for(int i=0; i<Info.num_of_features_shoes; i++){ // Zi
            ArrayList<Integer> subset = new ArrayList<>();
            subset.add(i);

            // 2）计算联盟对应的值  value_voting(subset, sum)
            int value = value_shoes(subset);

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

    //TODO 计算下一层的shapley value
    // checkAndAllocate_transfer_1(): 每个网格只取一个值
    // checkAndAllocate_transfer_2(): 每个网格可取多个值
    // allocations[] 装进了结构体
    private Grid computeNextLevel_2(int level, Grid grid_1, int sum, int[] given_weights, double[] shap_matrix) {

        //**********************
        for(ArrayList<FeatureSubset> ele : grid_1.grids_row){
            System.out.print(ele.size() + "\t");
        }

        Grid gridStructure = new Grid();  //用网格记录这一层

        // 记录这一层的情况（因为每个特征对应的采样数可能不一致，需要结构体记录采样数量）
        ShapMatrixEntry[] temp = new ShapMatrixEntry[Info.num_of_features_shoes];
        for(int i=0; i<Info.num_of_features_shoes; i++){
            temp[i] = new ShapMatrixEntry();
        }

        //1）从每个网格grid中随机选一个
        //[问题] 无法保证每个网格中都有值，如果网格中没有值，就把名额让个其他的
        //[修改] 添加安全检查，再根据网格分布情况分布采样的个数
        // 安全检查 + 重新分配
//        int[] allocations = checkAndAllocate_transfer_1(grid_1);  //[版本1]简单分配，每个网格取1个
//        int[] allocations = checkAndAllocate_transfer_2(level, grid_1, Info.num_of_samples);  //[版本1]简单分配，每个网格取1个
        Allocation all = new Allocation();
        all.checkAndAllocate_transfer_2(level, grid_1, Info.num_of_samples);

        //**********************
       for(int num : all.allocations){
           System.out.print(num + "\t");
       }
        System.out.println();


        //[第一版]每层抽样一个，lis大小就是 采样数 * 特征数量
        //记录当前层生成的特征子集，为下一层做好准备
//        ArrayList<FeatureSubset> list = new ArrayList<>();
        FeatureSubset[] list = new FeatureSubset[(int) all.total_samples * (Info.num_of_features_shoes-1)];  //【改】 换成了结构体中的 设定样本 + 补偿
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
                for(int i=0; i<Info.num_of_features_shoes; i++){
                    if( !random_sample.name.contains(i)){   //挑选出的random_sample不包含当前的特征，就可以构成新的
                        ArrayList<Integer> name = new ArrayList<>(random_sample.name);
                        name.add(i);
                        double value = value_airport(name, given_weights);
                        FeatureSubset newFeaSub = new FeatureSubset(name, value);

                        gridStructure.max = Math.max(gridStructure.max, value);
                        gridStructure.min = Math.min(gridStructure.min, value);

                        //3）计算shapley value 并填写到矩阵中
                        temp[i].sum += value - random_sample.value_fun; //V(S U i) - V(S), random_sample就是S
                        temp[i].count ++;

                        //4）存储新生成特征子集S，为下一层选取做好准备
//                        list.add(newFeaSub);
                        list[index] = newFeaSub;
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
                shap_matrix[i] += 1.0f * entry.sum / entry.count;
            }
            else{
                shap_matrix[i] += 0;  //抽样不存在时，就不计算
//                System.out.println("L" + level + ": " + i);
            }
        }

        //4)这一层完成后，存储到网格结构中
        gridStructure.ConstructGrid(list);  //存储到一个网格结构中

        return gridStructure;
    }

    //TODO 计算下一层的shapley value (网格中数量基本一致)
    // allocations[] 装进了结构体
    private Grid computeNextLevel_3(int level, Grid grid_1, int sum, int[] given_weights, ShapMatrixEntry[] shap_matrix) {

        //**********************
        for(ArrayList<FeatureSubset> ele : grid_1.grids_row){
            System.out.print(ele.size() + "\t");
        }

        Grid gridStructure = new Grid();  //用网格记录这一层

        // 记录这一层的情况（因为每个特征对应的采样数可能不一致，需要结构体记录采样数量）
        ShapMatrixEntry[] temp = new ShapMatrixEntry[Info.num_of_features_shoes];
        for(int i=0; i<Info.num_of_features_shoes; i++){
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
        for(int num : all.allocations){
            System.out.print(num + "\t");
        }
        System.out.println();

        //[第一版]每层抽样一个，lis大小就是 采样数 * 特征数量
        //记录当前层生成的特征子集，为下一层做好准备
//        ArrayList<FeatureSubset> list = new ArrayList<>();
        FeatureSubset[] list = new FeatureSubset[all.level_samples * (Info.num_of_features_shoes-1)];  //【改】 换成了结构体中的 设定样本 + 补偿
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
                for(int i=0; i<Info.num_of_features_shoes; i++){
                    if( !random_sample.name.contains(i)){   //挑选出的random_sample不包含当前的特征，就可以构成新的
                        ArrayList<Integer> name = new ArrayList<>(random_sample.name);
                        name.add(i);
                        double value = value_airport(name, given_weights);
                        FeatureSubset newFeaSub = new FeatureSubset(name, value);

                        gridStructure.max = Math.max(gridStructure.max, value);
                        gridStructure.min = Math.min(gridStructure.min, value);

                        //3）计算shapley value 并填写到矩阵中
                        temp[i].sum += value - random_sample.value_fun; //V(S U i) - V(S), random_sample就是S
                        temp[i].count ++;

                        //4）存储新生成特征子集S，为下一层选取做好准备
//                        list.add(newFeaSub);
                        list[index] = newFeaSub;
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
        gridStructure.ConstructGrid_2(list);  //这是均匀划分的方式

        return gridStructure;
    }

    //TODO 计算下一层的shapley value (网格中数量基本一致)
    // 原版：computeNextLevel_5()
    private Grid computeNextLevel_shoes(int level, Grid grid_1, ShapMatrixEntry[] shap_matrix, Allocation all) {

        //**********************
//        for(ArrayList<FeatureSubset> ele : grid_1.grids_row){
//            System.out.print(ele.size() + "\t");
//        }

        Grid gridStructure = new Grid();  //用网格记录这一层

        // 记录这一层的情况（因为每个特征对应的采样数可能不一致，需要结构体记录采样数量）
        ShapMatrixEntry[] temp = new ShapMatrixEntry[Info.num_of_features_shoes];
        for(int i=0; i<Info.num_of_features_shoes; i++){
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
//        for(int num : all.allocations){
//            System.out.print(num + "\t");
//        }
//        System.out.println();

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
                for(int i=0; i<Info.num_of_features_shoes; i++){
                    if( !random_sample.name.contains(i)){   //挑选出的random_sample不包含当前的特征，就可以构成新的
                        ArrayList<Integer> name = new ArrayList<>(random_sample.name);
                        name.add(i);
                        double value = value_shoes(name);
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


    //修改1：每个网格可以取多个，因为特征子集S越多，省略的计算越多，可能会出现空值
    //修改2：给定补偿个数，按照补偿个数按倍数来扩充
    //修改3：安全检查，看看网格中的数量够不够，若不够了就转移给其他网格

    //TODO 【版本2】简单分配（每个网格取多个）
    //基本原则：每个网格采样多个，若出现网格数量为0时，把当前分配到的样本数，直接转嫁给最大数量的网格（网格数量按降序排列）
    // 有点笨的方法，现在不管网格中含有几个，都是抽样后放回。如果网格只有1个值，然后这个值就会被反复抽
    // level 表示传入当前层存储的元素，包含的特征数量
    // level_samples: 这一层采样的个数 (m)
    // total_level_samples ：level_samples +  补偿采样（m*level/(n-level)）,n是特征的数量  （这样补有点多）
    //
    private int[] checkAndAllocate_transfer_2(int level, Grid grid, int level_samples) {
        //1）初始化函数allocations[]函数，初始时每个网格取1个
        int[] allocations = new int[Info.num_of_grids];
        //compensation_samples 补偿的采样数量  // Math.round() 四舍五入地返回最接近的整数
        int compensation_samples = Math.round(1.0f * level_samples * level / (Info.num_of_features_shoes - level));  //补太多了
//        int total_level_samples = Math.round(1.0f * level_samples * level / (Info.num_of_features - level)) + level_samples;
//        int compensation_samples = level * m;   //第几层(隐藏了几个特征)就补几个
        //每个网格中分配的采样数
        int oneGrid_samples = Math.round( 1.0f * (level_samples + compensation_samples) / Info.num_of_grids) ;

        //1)检查是否需要重新分配
        int ZoneCount = 0;  //计数器，计数空网格的个数
        for (ArrayList<FeatureSubset> oneGrid : grid.grids_row) {
            //2)检查是否有网格为空,计数
            if (oneGrid.size() == 0) {
                ZoneCount ++;
            }
        }

        //2）如果真的有空网格，就需要把样本转移给其他的网格
        //情况1：无空网格，每个网格取相同的样本数量
        if(ZoneCount == 0){
            Arrays.fill(allocations, oneGrid_samples);
        }
        //情况2：有空网格，把样本转移给其他的网格 (ZoneCount > 0)
        else{
            //Step1 : grid_size 用于存储每个网格对应的数量
            List<Integer[]> grid_size_set = new ArrayList<>();
            for(int index_g = 0; index_g<grid.grids_row.length; index_g++){  //ind 是网格对应的下标
                ArrayList<FeatureSubset> oneGrid = grid.grids_row[index_g];
                if(oneGrid.size() >0){   //只有大于0的才可以加入候补集中，用于顶替采样
                    Integer[] temp = {index_g , oneGrid.size()};
                    grid_size_set.add(temp);  //(gridID, 存放元素的数量)
                }
            }
            //Step2 : 排序
//            Collections.sort(grid_size_set, new Comparator<Integer[]>() {
            grid_size_set.sort(new Comparator<Integer[]>() {
                @Override
                public int compare(Integer[] array1, Integer[] array2) {
                    // 以数组的第2个元素作为比较依据，降序排列 (按存放元素的数量)
                    return array2[1].compareTo(array1[1]);
                }
            });  // 排序后的list就是  (第1大的网格ID，数量），(第2大的网格的ID，数量)

            //3) 重新分配
            //【版本2】：遍历每一层，遇到了空网格的就找最大的出来顶替
            int index_transfer = 0;
            for(int ind = 0; ind<grid.grids_row.length; ind++){
                ArrayList<FeatureSubset> oneGrid = grid.grids_row[ind];
                if(oneGrid.size() != 0){  //有值的，可以直接取
                    allocations[ind] += oneGrid_samples;
                }
                else {
                    //找到替换的下标
                    int replace_index = grid_size_set.get(index_transfer)[0];
                    //把采样个数加上
                    allocations[replace_index] += oneGrid_samples;
                    //条件判断，对index_transfer赋值；
                    if(index_transfer < grid_size_set.size()-1){
                        index_transfer ++;
                    }
                    else{
                        index_transfer = 0;
                    }
                }
            }
        }
        return allocations;
    }

    //TODO 等比例采样
    //新版函数，需要排序
    private int[] checkAndAllocate_2(Grid grid) {
        //1）初始化函数allocations[]函数，初始时每个网格取1个
        int[] allocations = new int[Info.num_of_grids];
        for(int i=0; i<allocations.length; i++){
            assert Info.num_of_samples == Info.num_of_grids;
            allocations[i] = 1; // Info.num_of_samples / Info.num_of_grids
        }
        //2）安全检查，是否需要重新分配
        boolean flag = checkAllocation(grid);

        //3）若需要重新分，按照每个网格中含元素的数量重新分配
        if(flag){
            // 1) 先计算总共有几个样本 (避免后面的迭代重复计算)
            int total_weihts = 0;
            for(ArrayList<FeatureSubset> agrid : grid.grids_row){
                total_weihts += agrid.size();
            }
            //初始化一个新的allocations
            int[] newAllocations = new int[Info.num_of_grids];
            newAllocations = allocateSamples(grid, Info.num_of_samples, newAllocations, total_weihts);
            allocations = newAllocations;
        }
        //4）返回值
        return allocations;
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

    private int value_airport(ArrayList<Integer> subset, int[] given_weights) {
        int subset_max_ele = -1;
        if(subset.size() == 1){
            return given_weights[subset.get(0)];  //返回值是最大下标对应的weights
        }
        else{
            for(int ele : subset){
                if(subset_max_ele < ele){
                    subset_max_ele = ele;
                }
            }
            return given_weights[subset_max_ele];  //返回值是最大下标对应的weights
        }
    }

    //TODO 计算最大误差 (Voting game & Airport game)
    private double computeMaxError(ShapMatrixEntry[] shap_matrix, double[] exact) {
        double error_max = 0;  //误差总量，最后除以特征数
        for(int i=0; i<Info.num_of_features_shoes; i++){
            error_max = Math.max(Math.abs((shap_matrix[i].sum - exact[i]) / exact[i]), error_max);
        }
        return error_max;
    }

    //TODO 计算最大误差 (Shoes game & Tree game)
    private double computeMaxError(ShapMatrixEntry[] shap_matrix, double exact) {
        double error_max = 0;  //误差总量，最后除以特征数
        for(int i=0; i<Info.num_of_features_shoes; i++){
            error_max = Math.max(Math.abs((shap_matrix[i].sum - exact) / exact), error_max);
        }
        return error_max;
    }

    //TODO 计算平均误差 (Voting game & Airport game)
    private double computeAverageError(ShapMatrixEntry[] shap_matrix, double[] exact) {
        double error_ave = 0;  //误差总量，最后除以特征数
        for(int i=0; i<Info.num_of_features_shoes; i++){
            error_ave += Math.abs((shap_matrix[i].sum - exact[i]) / exact[i]);
        }
        error_ave = error_ave / Info.num_of_features_shoes;
        return error_ave;
    }

    //TODO 计算平均误差 (Shoes game & Tree game)
    private double computeAverageError(ShapMatrixEntry[] shap_matrix, double exact) {
        double error_ave = 0;  //误差总量，最后除以特征数
        for(int i=0; i<Info.num_of_features_shoes; i++){
            error_ave += Math.abs((shap_matrix[i].sum - exact) / exact);
        }
        error_ave = error_ave / Info.num_of_features_shoes;
        return error_ave;
    }

    /*TODO [Shoes Game]：网格均匀划分 + 每层样本按比例分配
    *  原版是 ShapleyApproximate_airport_4() */
    public ShapMatrixEntry[] ShapleyApproximate_shoes() {

        //1) 用一个大数组记录特征对应的shapley value之和
        ShapMatrixEntry[] shap_matrix = new ShapMatrixEntry[Info.num_of_features_shoes];
        for(int i=0; i<shap_matrix.length; i++){
            shap_matrix[i] = new ShapMatrixEntry();
        }

        //2）给每层分配样本
        Allocation allo = new Allocation();
        allo.sampleAllocation_2(Info.num_of_features_shoes-1); //sample_num_level 每一层采样几个

        // 3）为单个特征计算边际贡献（所有单特征） computeSingleFeature_3()均匀的网格
        Grid grid_level = computeSingleFeature_shoes(shap_matrix);    //grid_level: 每层的循环变量

        // 5)生成2-联盟
//        Grid grid_level = new Grid();
//        grid_level = computeNextLevel(grid_1, sum, given_weights, shap_matrix);
//        System.out.println("level initiall: " + grid_level.grids_row[0].get(0).name.size());

        //4)依次生成联盟
        for(int ind=1; ind <Info.num_of_features_shoes; ind++){
            //【备注】ind 就是用来代替grid_level.grids_row[0].get(0).name.size()，表示这个网格中存储的subset中包含几个特征
            // 因为网格中不一定都有值的，可能网格中不存在元素： grid_level.grids_row[0].get(0).name.size() 可能空指针异常
            // 5) 计算某一层，存储到网格中
//            grid_level = computeNextLevel_3(ind, grid_level, sum, given_weights, shap_matrix);  //这是均匀划分的方式
            grid_level = computeNextLevel_shoes(ind, grid_level, shap_matrix, allo);  //这是均匀划分 + 比例抽样(log降维)
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

    //TODO 利用两个矩阵计算
    public ShapMatrixEntry[] ShapleyApproximate_shoes_2() {

        //1) 用一个大数组记录特征对应的shapley value之和
        ShapMatrixEntry[] shap_matrix = new ShapMatrixEntry[this.num_of_features];
        for(int i=0; i<shap_matrix.length; i++){
            shap_matrix[i] = new ShapMatrixEntry();
        }

        //3) 计算出一个大矩阵，存储1-联盟 & 2-联盟的值
        int[][] evaluateMatrix = constructEvaluateMatrix(); //第0层和第1层全算（层数= 减去的特征子集长度）
        int[][] levelMatrix = constructLevelMatrix();
        //判断并记录从哪一层开始计算
        int level_index = checkLevel(levelMatrix);  //level_index：表示这层subset的length -1 (因为从0开始)

        //2）给每层分配样本
        Allocation allo = new Allocation();
//        allo.sampleAllocation_2(Info.num_of_features_shoes-1); //sample_num_level 每一层采样几个
        allo.sampleAllocation_4(this.num_of_features, level_index); //sample_num_level 每一层采样几个

        //4）扫描evaluateMatrix，计算1-联盟 & 2-联盟的shapley值  //第0层和第1层全算（层数= 减去的特征子集长度）
        computeMatrix_3(shap_matrix, evaluateMatrix, levelMatrix, level_index);    //grid_level: 每层的循环变量

        // 5)计算第2层（level = 被减去的特征的长度）
        Grid grid_level = initialLevelCompute_shoes(shap_matrix, evaluateMatrix);  //level =2 （存储长度为2的特征子集）

        // 5)生成2-联盟
//        Grid grid_level = new Grid();
//        grid_level = computeNextLevel(grid_1, sum, given_weights, shap_matrix);
//        System.out.println("level initiall: " + grid_level.grids_row[0].get(0).name.size());

        //4)依次生成联盟
        for(int ind=2; ind <Info.num_of_features_shoes; ind++){
            //【备注】ind 就是用来代替grid_level.grids_row[0].get(0).name.size()，表示这个网格中存储的subset中包含几个特征
            // 因为网格中不一定都有值的，可能网格中不存在元素： grid_level.grids_row[0].get(0).name.size() 可能空指针异常
            // 5) 计算某一层，存储到网格中
//            grid_level = computeNextLevel_3(ind, grid_level, sum, given_weights, shap_matrix);  //这是均匀划分的方式
            grid_level = computeNextLevel_shoes(ind, grid_level, shap_matrix, allo);  //这是均匀划分 + 比例抽样(log降维)
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


    //TODO 利用矩阵计算1-联盟 & 2-联盟
    public ShapMatrixEntry[] ShapleyApproximate_shoes_3() {

        //1) 用一个大数组记录特征对应的shapley value之和
        ShapMatrixEntry[] shap_matrix = new ShapMatrixEntry[this.num_of_features];
        for(int i=0; i<shap_matrix.length; i++){
            shap_matrix[i] = new ShapMatrixEntry();
        }

        //2) 计算出一个大矩阵，存储1-联盟 & 2-联盟的值
        int[][] evaluateMatrix = constructEvaluateMatrix(); //第0层和第1层全算（层数= 减去的特征子集长度）
        //************************************ 添加一个矩阵 ************************************
        int[][] levelMatrix = constructLevelMatrix();
        //判断并记录从哪一层开始计算
        int level_index = checkLevel(levelMatrix);  //level_index：表示这层subset的length -1 (因为从0开始)

        /*到第level——index 有值，方差不为0，因此就需要构造上一层的特征数量（level-1）,然后从当前层 - 上层开始计算*/

        //3)然后给每层分配样本
        Allocation allo = new Allocation();
//        allo.sampleAllocation_2(Info.num_of_features_voting-1); //sample_num_level 每一层采样几个 【逻辑错误】
        allo.sampleAllocation_4(this.num_of_features, level_index); //使用两个矩阵

        //4）（利用levelMatrix剪枝）扫描Matrix，计算1-联盟 & 2-联盟的shapley值  //第0层和第1层全算（层数= 减去的特征子集长度）
        computeMatrix_3(shap_matrix, evaluateMatrix, levelMatrix, level_index);    //computeMatrix_voting_3() 使用两个矩阵的优化

        // 5)计算第2层（level = 被减去的特征的长度）
        Grid grid_level = initialLevelCompute_voting_noSort(evaluateMatrix, level_index, allo);  //level =2 （存储长度为2的特征子集）
     /* 构造上层的元素的网格，虽然检查了对角线上的元素，但生成的时候使用valueFunction会更保险，因为随机生成的subset可能是不连续的，
     不能保证一定等于对角线上的统一值，相当于一次安全检查 */

        //4)依次生成联盟 (ind / level 等于被减去的联盟的长度)
        for(int ind=level_index; ind <this.num_of_features; ind++){
            //【备注】ind 就是用来代替grid_level.grids_row[0].get(0).name.size()，表示这个网格中存储的subset中包含几个特征
            // 因为网格中不一定都有值的，可能网格中不存在元素： grid_level.grids_row[0].get(0).name.size() 可能空指针异常
            // 5) 计算某一层，存储到网格中
//            grid_level = computeNextLevel_3(ind, grid_level, sum, given_weights, shap_matrix);  //这是均匀划分的方式
//            grid_level = computeNextLevel_voting(ind, grid_level, given_weights, shap_matrix, allo, halfSum);  //这是均匀划分 + 比例抽样(log降维)
            grid_level = computeNextLevel_voting_4(ind, grid_level, shap_matrix, allo);  //去除list中重复的元素！

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


    //TODO [Shoes Game]
    private int value_shoes(ArrayList<Integer> subset) {
        int left_shoes = 0;    //left shoes 的个数
        int right_shoes = 0;  //right shoes 的个数
        for(int ele : subset){
            if(ele < 50){
                left_shoes ++;
            }
            else{
                right_shoes ++;
            }
        }
        return Math.min(left_shoes, right_shoes);  //返回值
    }


    //TODO 构建evaluateMatrix, 记录1-联盟 & 2-联盟的值
    private int[][] constructEvaluateMatrix() {
        int[][] matrix = new int[this.num_of_features][this.num_of_features];
        for(int i=0; i<this.num_of_features; i++){   //i: 横坐标 (也是第1个item)
            ArrayList<Integer> subset = new ArrayList<>();
            subset.add(i);  //单个特征联盟，1-联盟
            matrix[i][i] = value_shoes(subset);
            for(int j=i+1; j<this.num_of_features; j++){   //j: 纵坐标(也是第2个item)
                ArrayList<Integer> twoCoalition = new ArrayList<>(subset);
                twoCoalition.add(j);
                matrix[i][j] = matrix[j][i] = value_shoes(twoCoalition);  //复制两份
            }
        }
        return matrix;
    }


    //TODO 计算1-联盟 & 2-联盟的shapley值 => 利用矩阵EvaluateMatrix
    // 【版本2】均匀划分：为单个特征计算边际贡献（所有单特征）复制版本3，用于shoes_game
    // 原版是 computeSingleFeature_shoes()
    private void computeMatrix_shoes(ShapMatrixEntry[] shap_matrix, int[][] evaluateMatrix) {

        // 0) 初始化
        long one_feature_sum = 0; //记录对角线元素的值

        // 1）读取1-联盟的shapley value
        for(int i=0; i<Info.num_of_features_shoes; i++){
            shap_matrix[i].sum += evaluateMatrix[i][i];  //对角线上的元素依次填入shapley value[]的矩阵中
            shap_matrix[i].count ++;
            one_feature_sum += one_feature_sum;  //对角线求和
        }

        // 2) 读取和计算2-联盟的shapley value
        for(int i=0; i<Info.num_of_features_shoes; i++){  //i 是横坐标  一行就对应一个特征
            long line_sum = 0;
            for(int j=0; j<Info.num_of_features_shoes; j++){  //j是纵坐标
                line_sum += evaluateMatrix[i][j];
            }
            shap_matrix[i].sum += 1.0 * (line_sum - one_feature_sum) / (Info.num_of_features_shoes-1);  //第二层的shapley value
            shap_matrix[i].count ++;
        }
    }

    //TODO 计算初始层，并存储到网格中(3-联盟-2联盟的值) => 利用矩阵EvaluateMatrix
    // 原版是 computeSingleFeature_voting()
    private Grid initialLevelCompute_shoes(ShapMatrixEntry[] shap_matrix, int[][] evaluateMatrix) {

        //性质：矩阵是一个对称的矩阵，所以2-联盟只需要看一半，存一半就可以

        // 1）初始化网格结构 （存3-联盟）
        Grid grids = new Grid();  //表示这是存储1-联盟的网格
        ArrayList<FeatureSubset> twoCoalition_set = new ArrayList<>();  //用list存储所有的2-项集
        //数组：虽然定长，但删除不方便

        // 1) 读取2-联盟的shapley value
        for(int i=0; i<Info.num_of_features_shoes; i++){  //i 是横坐标  一行就对应一个特征
            List<Integer> subSet = new ArrayList<>();
            subSet.add(i);  //第1个特征
            for(int j=i+1; j<Info.num_of_features_shoes; j++){  //j是纵坐标
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

    //TODO 构建矩阵存储连续的联盟值
    private int[][] constructLevelMatrix(){
        int[][] matrix = new int[this.num_of_features][this.num_of_features];
        for(int i=0; i<this.num_of_features; i++) {   //i: 横坐标 (也是第1个item)
            ArrayList<Integer> subset = new ArrayList<>();
            subset.add(i);  //单个特征联盟，1-联盟
            matrix[i][i] = value_shoes(subset);
            for(int j=i+1; j<Info.num_of_features_voting; j++) {   //j: 纵坐标(也是第2个item)
                subset.add(j);
                matrix[i][j] = matrix[j][i] = value_shoes(subset);  //复制两份
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

    //TODO 利用两个矩阵(evaluateMatrix + levelMatrix)优化性能-voting_game
    // 原版是 computeMatrix_voting()
    private void computeMatrix_3(ShapMatrixEntry[] shap_matrix, int[][] evaluateMatrix, int[][] levelMatrix, int level_index){
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

    //TODO [不排序放入网格] 计算初始层，并存储到网格中(3-联盟-2联盟的值) => 利用矩阵EvaluateMatrix
    // 原版是 computeSingleFeature_voting()
    private Grid initialLevelCompute_voting_noSort(int[][] evaluateMatrix, int level_index, Allocation allo) {
        //TODO 性质：矩阵是一个对称的矩阵，所以2-联盟只需要看一半，存一半就可以

        // 1）初始化网格结构 （存3-联盟）
        Grid grids = new Grid();  //表示这是存储1-联盟的网格

        //情况1： 前面几层可以被剪枝
        if(level_index > 2){
            //从某一层开始计算，在这层中随机生成若干个组合（按照分配方式）
            ArrayList<FeatureSubset> result = randomSubsets(allo.num_sample[level_index], level_index);  //n个元素中随机取出m个长度为k的元素

            //放入网格中
            grids.ConstructGrid_3(result);  //这是均匀划分的方式
        }

        // 情况2：正常的计算过程，没有可以被剪枝
        else{
            ArrayList<FeatureSubset> twoCoalition_set = new ArrayList<>();  //用list存储所有的2-项集
            //数组：虽然定长，但删除不方便

            // 1) 读取2-联盟的shapley value
            for (int i = 0; i < this.num_of_features; i++) {  //i 是横坐标  一行就对应一个特征
                List<Integer> subSet = new ArrayList<>();
                subSet.add(i);  //第1个特征
                for (int j = i + 1; j < this.num_of_features; j++) {  //j是纵坐标
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
    public ArrayList<FeatureSubset> randomSubsets(int m, int len){
        ArrayList<FeatureSubset> subsets = new ArrayList<>();  //所有生成的subset的集合
        Random random = new Random();
        for (int i = 0; i < m; i++) {
            Set<Integer> subset = new HashSet<>();  //一个生成的subset(其中元素不重复)
            while (subset.size() < len) {
                subset.add(random.nextInt(this.num_of_features));
            }
            ArrayList<Integer> name = new ArrayList<>(subset);
            FeatureSubset ele = new FeatureSubset(name, value_shoes(name));
            subsets.add(ele);
        }
        return subsets;
    }

    //TODO 【版本4】 不排序 + 子集名字不排序
    //原版是computeNextLevel_5()
    // weight[] 计算每层的采样权重； checkAndAllocate_transfer_4（）：按照每组个数等比例抽样；当前层生成的元素 list ：ArrayList();
    private Grid computeNextLevel_voting_4(int level, Grid grid, ShapMatrixEntry[] shap_matrix, Allocation all) {

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
                for(int i=0; i<this.num_of_features; i++){
                    if( !random_sample.name.contains(i)){   //挑选出的random_sample不包含当前的特征，就可以构成新的
                        ArrayList<Integer> name = new ArrayList<>(random_sample.name);
                        name.add(i);
                        double value = value_shoes(name);
                        FeatureSubset newFeaSub = new FeatureSubset(name, value);

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


}
