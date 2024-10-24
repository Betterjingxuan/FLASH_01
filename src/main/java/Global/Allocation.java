package Global;

import Game.GameClass;
import structure.ShapMatrixEntry;
import java.math.BigInteger;
import java.util.*;

public class Allocation {

    //每层应该分配的权重（整体）
    public double[] weigh_for_sample; //每层采样的权重（个数等于层数）

    public int[] num_sample; //每层采样的个数（个数等于层数）

    // 记录权重之和
    public double total_weights;
    public int total_samples; //采样的总个数(maybe与设定的不相等)

    public double check_weight;   //最后阶段方差计算的比例
    public double number_weight;  //在样本分配时，数量特征所占的比例
    public double variance_weight; //在样本分配时，方差特征所占的比例


    public Allocation(GameClass game){
        this.check_weight = game.check_weight;
        this.number_weight = game.number_weight;
        this.variance_weight = game.variance_weight;
    }

    //TODO 从第2层开始分配：传入的参数是sample,sample * 2 = evaluation number
    public void sampleAllocation_unif(int num_features) {
        this.weigh_for_sample = new double[num_features];  //记录每层采样的按权重
        this.num_sample = new int[num_features];  // 记录每层采样几个（按权重）
        this.total_weights = 0.0; //total_weight 权重总数

        // 1）依次遍历每层，计算权重存入数组中weigh_for_sample[] 中
        for(int i=2; i<num_features; i++){
            double log = num_features * Math.log(num_features)- i * Math.log(i) - (num_features-i) * Math.log(num_features-i);  //最大值70
            // C(n,k)= n!/ k!(n-k)!  // log(m*k) = log m + log k; log(m/k) = log m - log k;  // log(n!) 约等于 nlog(n)
            this.weigh_for_sample[i] = log;
            this.total_weights += log;
//            System.out.print(log + "\t");
        }

        // 2）按照1）中设置的权重 + 设定的采样数，实际分配采样个数
        //计算总共采样个数
        int sample = (int) Math.ceil(2.0f * Info.total_samples_num / (num_features-2));  //每层分配的采样数

        for(int ind = 2; ind <num_features; ind ++){
            this.num_sample[ind] = sample;
            this.total_samples += sample;  // 采样的总个数
//            System.out.print( this.num_sample[ind] + "\t");
        }
//        System.out.println("this.total_samples: " + this.total_samples);
    }

    //TODO 从第2层开始分配（大数组解决了前两层的问题,所以level从2开始）
    public void sampleAllocation_unif(int num_features, int num_samples) {
        this.weigh_for_sample = new double[num_features];  //记录每层采样的按权重
        this.num_sample = new int[num_features];  // 记录每层采样几个（按权重）
        this.total_weights = 0.0; //total_weight 权重总数

        // 1）依次遍历每层，计算权重存入数组中weigh_for_sample[] 中
        for(int i=2; i<num_features; i++){
            double log = num_features * Math.log(num_features)- i * Math.log(i) - (num_features-i) * Math.log(num_features-i);  //最大值70
            // C(n,k)= n!/ k!(n-k)!  // log(m*k) = log m + log k; log(m/k) = log m - log k;  // log(n!) 约等于 nlog(n)
            this.weigh_for_sample[i] = log;
            this.total_weights += log;
//            System.out.print(log + "\t");
        }

        // 2）按照1）中设置的权重 + 设定的采样数，实际分配采样个数
        //计算总共采样个数
        int sample = (int) Math.ceil(1.0f * num_samples / (num_features-2));  //每层分配的采样数

        for(int ind = 2; ind <num_features; ind ++){
            this.num_sample[ind] = sample;
            this.total_samples += sample;  // 采样的总个数
//            System.out.print( this.num_sample[ind] + "\t");
        }
        System.out.println("this.total_samples: " + this.total_samples);
    }

    //TODO 均匀分 + 确定范围
    public void sampleAllocation_uni_2(int num_features, int num_samples, int startLev, int endLev) {
//        this.weigh_for_sample = new double[num_features];  //记录每层采样的按权重
        this.num_sample = new int[num_features];  // 记录每层采样几个（按权重）
//        this.total_weights = 0.0; //total_weight 权重总数

        // 2）按照1）中设置的权重 + 设定的采样数，实际分配采样个数  // num_samples: 计算总共采样个数
        int sample = (int) Math.ceil(1.0f * num_samples / (endLev-startLev));  //每层分配的采样数

        for(int ind = startLev; ind <endLev; ind ++){
            this.num_sample[ind] += sample;
            this.total_samples += sample;  // 采样的总个数
//            System.out.print( this.num_sample[ind] + "\t");
        }
        System.out.println("this.total_samples: " + this.total_samples);
    }

    //TODO 按每层utility方差，比例分配：效果不好
    public void sampleAllocation_air(int level, int num_features, int num_samples, double[][]newLevelMatrix) {
        this.weigh_for_sample = new double[num_features];  //记录每层采样的按权重
        this.num_sample = new int[num_features];  // 记录每层采样几个（按权重）
        this.total_weights = 0.0; //total_weight 权重总数

        //计算每一层的方差
        double[] varianceArray = new double[num_features+1];  //每一层的方差
        double allVariance = 0;
        for(int len = level; len < num_features; len ++) {  //line_ind 是对角线的标记
            double step_mean = 0;  //这条对对角线上的方差
            for(int ind=0; ind < num_features; ind++){
                step_mean += newLevelMatrix[len][ind];
            }
           step_mean = step_mean / num_features; //计算当前层的均值
            //计算方差
            double variance = 0.0;
            for(int ind=0; ind < num_features; ind++){
                variance += Math.pow(Math.abs(newLevelMatrix[len][ind] - step_mean), 3);
            }
            double oneVariance = variance / num_features;
            varianceArray[len] = oneVariance;
            allVariance += oneVariance;
        }

        //按照方差大小，比例分配样本
        for(int ind = level; ind <num_features; ind ++){
            int sample = (int) (Math.ceil(num_samples * varianceArray[ind] / allVariance));  //每层分配的采样数
            this.num_sample[ind] = sample;
            this.total_samples += sample;  // 采样的总个数
        }
        System.out.println("this.total_samples: " + this.total_samples);

        // 2）按照1）中设置的权重 + 设定的采样数，实际分配采样个数
//        int sample = (int) Math.ceil(1.0f * num_samples / (num_features-12));  //每层分配的采样数
//        for(int ind = level; ind <num_features; ind ++){
//            this.num_sample[ind] = sample;
//            this.total_samples += sample;  // 采样的总个数
//        }
//        System.out.println("this.total_samples: " + this.total_samples);
    }

    /* TODO 给每层分配样本：stratified sampling
     *   num_features ： 特征的数量 -1
     *   [特别说明]
     * 1） level 从0开始，第0层存放所有的1-项集; (单独写特殊情况 )
     * 2） 从n个中选出i个的组合种类 ： c(n,i)
     *  // weigh_for_sample[] 每层采样的比例
     * */
    public void sampleAllocation_2(int num_features) {
        this.weigh_for_sample = new double[num_features+1];  //记录每层采样的按权重
        this.num_sample = new int[num_features+1];  // 记录每层采样几个（按权重）
        this.total_weights = 0.0; //total_weight 权重总数

        // 1）依次遍历每层，计算权重存入数组中weigh_for_sample[] 中
        for(int i=1; i<num_features; i++){
            double log = num_features * Math.log(num_features)- i * Math.log(i) - (num_features-i) * Math.log(num_features-i);  //最大值70
            // C(n,k)= n!/ k!(n-k)!  // log(m*k) = log m + log k; log(m/k) = log m - log k;  // log(n!) 约等于 nlog(n)
            this.weigh_for_sample[i] = log;
            this.total_weights += log;
//            System.out.print(log + "\t");
        }

        // 2）按照1）中设置的权重 + 设定的采样数，实际分配采样个数
        //计算总共采样个数
        double radio = Info.total_samples_num / this.total_weights;  //定义扩充比例 (total_samples_num 总体采样数)

        for(int ind = 1; ind <this.weigh_for_sample.length; ind ++){
            int sample = (int) Math.round(radio * this.weigh_for_sample[ind]);  //每层分配的采样数
            this.num_sample[ind] = sample;
            this.total_samples += sample;  // 采样的总个数
        }
    }

    //TODO 从第2层开始分配（大数组解决了前两层的问题,所以level从2开始）
    public void sampleAllocation_3(int num_features) {
        this.weigh_for_sample = new double[num_features];  //记录每层采样的按权重
        this.num_sample = new int[num_features];  // 记录每层采样几个（按权重）
        this.total_weights = 0.0; //total_weight 权重总数

        // 1）依次遍历每层，计算权重存入数组中weigh_for_sample[] 中
        for(int i=2; i<num_features; i++){
            double log = num_features * Math.log(num_features)- i * Math.log(i) - (num_features-i) * Math.log(num_features-i);  //最大值70
            // C(n,k)= n!/ k!(n-k)!  // log(m*k) = log m + log k; log(m/k) = log m - log k;  // log(n!) 约等于 nlog(n)
            this.weigh_for_sample[i] = log;
            this.total_weights += log;
//            System.out.print(log + "\t");
        }

        // 2）按照1）中设置的权重 + 设定的采样数，实际分配采样个数
        //计算总共采样个数
        double radio = Info.total_samples_num / this.total_weights;  //定义扩充比例 (total_samples_num 总体采样数)

        for(int ind = 2; ind <num_features; ind ++){
            int sample = (int) Math.round(radio * this.weigh_for_sample[ind]);  //每层分配的采样数
            this.num_sample[ind] = sample;
            this.total_samples += sample;  // 采样的总个数
//            System.out.print( this.num_sample[ind] + "\t");
        }
//        System.out.println("this.total_samples: " + this.total_samples);
    }

    //TODO 从第k层开始分配（大数组解决了前两层的问题,所以level从2开始）
    public void sampleAllocation_4(int num_features, int k) {
        this.weigh_for_sample = new double[num_features];  //记录每层采样的按权重
        this.num_sample = new int[num_features];  // 记录每层采样几个（按权重）
        this.total_weights = 0.0; //total_weight 权重总数

        // 1）依次遍历每层，计算权重存入数组中weigh_for_sample[] 中
        int star_level = Math.max(k,2);
        for(int i=star_level; i<num_features; i++){
            double log = num_features * Math.log(num_features)- i * Math.log(i) - (num_features-i) * Math.log(num_features-i);  //最大值70
            // C(n,k)= n!/ k!(n-k)!  // log(m*k) = log m + log k; log(m/k) = log m - log k;  // log(n!) 约等于 nlog(n)
            this.weigh_for_sample[i] = log;
            this.total_weights += log;
//            System.out.print(log + "\t");
        }

        // 2）按照1）中设置的权重 + 设定的采样数，实际分配采样个数
        //计算总共采样个数
        double radio = Info.total_samples_num / this.total_weights;  //定义扩充比例 (total_samples_num 总体采样数)

        for(int ind = star_level; ind <num_features; ind ++){
            int sample = (int) Math.round(radio * this.weigh_for_sample[ind]);  //每层分配的采样数
            this.num_sample[ind] = sample;
            this.total_samples += sample;  // 采样的总个数
//            System.out.print( this.num_sample[ind] + "\t");
        }
        System.out.println("this.total_samples: " + this.total_samples);
    }

    //TODO 从第k层开始分配（大数组解决了前两层的问题,所以level从2开始）
    //[改进] 最后1层 =1， 倒数第二层=n; 把最后两层脱离计算
    public void sampleAllocation_5(int num_features, int k) {
        this.weigh_for_sample = new double[num_features];  //记录每层采样的按权重
        this.num_sample = new int[num_features];  // 记录每层采样几个（按权重）
        this.total_weights = 0.0; //total_weight 权重总数

        // 1）依次遍历每层，计算权重存入数组中weigh_for_sample[] 中
        int star_level = Math.max(k,2);
        for(int i=star_level; i<num_features-1; i++){
            double log = num_features * Math.log(num_features)- i * Math.log(i) - (num_features-i) * Math.log(num_features-i);  //最大值70
            // C(n,k)= n!/ k!(n-k)!  // log(m*k) = log m + log k; log(m/k) = log m - log k;  // log(n!) 约等于 nlog(n)
            this.weigh_for_sample[i] = log;
            this.total_weights += log;
//            System.out.print(log + "\t");
        }

        // 2）按照1）中设置的权重 + 设定的采样数，实际分配采样个数
        //计算总共采样个数
        double radio = Info.total_samples_num / this.total_weights;  //定义扩充比例 (total_samples_num 总体采样数)

        for(int ind = star_level; ind <num_features-1; ind ++){
            int sample = (int) Math.round(radio * this.weigh_for_sample[ind]);  //每层分配的采样数
            this.num_sample[ind] = sample;
            this.total_samples += sample;  // 采样的总个数
//            System.out.print( this.num_sample[ind] + "\t");
        }

        //3）给最后1层赋值
        this.num_sample[num_features-1] = num_features;  //最后第1层 存储S的长度为 (n-1)
        this.total_samples += num_features;  // 采样的总个数
//        System.out.println("this.total_samples: " + this.total_samples);
    }


    //TODO 二阶段分配样本：数量 + 均衡的方差  最后一层正常算
    public void sampleAllocation_9(int num_features, double[][]newLevelMatrix, int k) {
        this.weigh_for_sample = new double[num_features];  //记录每层采样的按权重
        this.num_sample = new int[num_features];  // 记录每层采样几个（按权重）
        this.total_weights = 0.0; //total_weight 权重总数

        // 1）依次遍历每层，计算权重存入数组中weigh_for_sample[] 中 (最后一层不参与计算)
        int level = Math.max(2,k);
        for(int i=level; i<num_features; i++){
            double log = num_features * Math.log(num_features)- i * Math.log(i) - (num_features-i) * Math.log(num_features-i);  //最大值70
            // C(n,k)= n!/ k!(n-k)!  // log(m*k) = log m + log k; log(m/k) = log m - log k;  // log(n!) 约等于 nlog(n)
            this.weigh_for_sample[i] = log;
            this.total_weights += log;
//            System.out.print(log + "\t");
        }

        //【添加第二阶段】 计算每一层的方差
        double[] varianceArray = new double[num_features+1];  //每一层的方差
        double allVariance = 0;
        for(int len = 2; len < num_features; len ++) {  //line_ind 是对角线的标记
            double step_sum = 0;  //这条对对角线上的方差
            for(int ind=0; ind < num_features; ind++){
                step_sum += newLevelMatrix[len][ind];
            }
            double step_mean = step_sum / num_features; //计算当前层的均值
            //计算方差
            double variance = 0.0;
            for(int ind=0; ind < num_features; ind++){
                variance += Math.pow(newLevelMatrix[len][ind] - step_mean, 2);
            }
            double oneVariance = variance / num_features;
            varianceArray[len] = oneVariance;
            allVariance += oneVariance;
//            System.out.print(oneVariance + "\t");
        }
//        System.out.println();

        //给出方差权重的比例
        double[] varianceRatio = new double[Info.num_of_features];
        for(int i=level; i<num_features; i++){
            varianceRatio[i] = varianceArray[i] / allVariance;
        }

        // 2）按照1）中设置的权重 + 设定的采样数，实际分配采样个数
        //计算总共采样个数
//        double radio = Info.number_weight * Info.total_samples_num / this.total_weights;  //定义扩充比例 (total_samples_num 总体采样数)
        double radio = this.number_weight * Info.total_samples_num / this.total_weights;  //定义扩充比例 (total_samples_num 总体采样数)
        for(int ind = level; ind <num_features; ind ++){
//            int sample = (int) (Math.ceil(radio * this.weigh_for_sample[ind] + Info.number_variance * Info.total_samples_num * varianceRatio[ind]));  //每层分配的采样数
//            int sample = 1;
//            if(varianceRatio[ind] != 0){
                int sample = (int) (Math.ceil(radio * this.weigh_for_sample[ind] + this.variance_weight * Info.total_samples_num * varianceRatio[ind]));  //每层分配的采样数
//            }
            this.num_sample[ind] = sample;
            this.total_samples += sample;  // 采样的总个数
//            System.out.print( this.num_sample[ind] + " \t");
        }
        System.out.println("this.total_samples: " + this.total_samples);
    }

    //TODO 二阶段分配样本：数量 + 均衡的方差  最后一层正常算
    public void sampleAllocation_9(int num_features, double[][]newLevelMatrix, int k, int total_samples_num) {
        this.weigh_for_sample = new double[num_features];  //记录每层采样的按权重
        this.num_sample = new int[num_features];  // 记录每层采样几个（按权重）
        this.total_weights = 0.0; //total_weight 权重总数

        // 1）依次遍历每层，计算权重存入数组中weigh_for_sample[] 中 (最后一层不参与计算)
        int level = Math.max(2,k);
        for(int i=level; i<num_features; i++){
            double log = num_features * Math.log(num_features)- i * Math.log(i) - (num_features-i) * Math.log(num_features-i);  //最大值70
            // C(n,k)= n!/ k!(n-k)!  // log(m*k) = log m + log k; log(m/k) = log m - log k;  // log(n!) 约等于 nlog(n)
            this.weigh_for_sample[i] = log;
            this.total_weights += log;
//            System.out.print(log + "\t");
        }

        //【添加第二阶段】 计算每一层的方差
        double[] varianceArray = new double[num_features+1];  //每一层的方差
        double allVariance = 0;
        for(int len = 2; len < num_features; len ++) {  //line_ind 是对角线的标记
            double step_sum = 0;  //这条对对角线上的方差
            for(int ind=0; ind < num_features; ind++){
                step_sum += newLevelMatrix[len][ind];
            }
            double step_mean = step_sum / num_features; //计算当前层的均值
            //计算方差
            double variance = 0.0;
            for(int ind=0; ind < num_features; ind++){
                variance += Math.pow(newLevelMatrix[len][ind] - step_mean, 2);
            }
            double oneVariance = variance / num_features;
            varianceArray[len] = oneVariance;
            allVariance += oneVariance;
//            System.out.print(oneVariance + "\t");
        }
//        System.out.println();

        //给出方差权重的比例
        double[] varianceRatio = new double[Info.num_of_features];
        for(int i=level; i<num_features; i++){
            varianceRatio[i] = varianceArray[i] / allVariance;
        }

        // 2）按照1）中设置的权重 + 设定的采样数，实际分配采样个数
        //计算总共采样个数
        double radio = this.number_weight * total_samples_num / this.total_weights;  //定义扩充比例 (total_samples_num 总体采样数)
        for(int ind = level; ind <num_features; ind ++){
            int sample = (int) (Math.ceil(radio * this.weigh_for_sample[ind] + this.variance_weight * total_samples_num * varianceRatio[ind]));  //每层分配的采样数
            this.num_sample[ind] = sample;
            this.total_samples += sample;  // 采样的总个数
//            System.out.print( this.num_sample[ind] + " \t");
        }
        System.out.println("this.total_samples: " + this.total_samples);
    }

    public void sampleAllocationVot(int num_features, double[][] newLevelMatrix, int levelIndex, int allSamples, GameClass game) {
        this.weigh_for_sample = new double[num_features];  //记录每层采样的按权重
        this.num_sample = new int[num_features];  // 记录每层采样几个（按权重）
        this.total_weights = 0.0; //total_weight 权重总数
        int level = Math.max(2,levelIndex);
        int limit = Math.min(Math.max(2, Info.total_samples_num / (num_features * num_features * 2)), num_features); // 需要评估几个
        Random random = new Random(Info.seed);

        // 初始化：varianceArr & utility
        ShapMatrixEntry[][] utility = new ShapMatrixEntry[num_features][]; // 需要换成带计数器的大数组
        for(int i=0; i<utility.length; i++){   //外层循环：features对应每个长度的collations (n个feature，n+1层)
            utility[i] = new ShapMatrixEntry[num_features];
            for(int j=0; j<num_features; j++) {  //内层循坏：每个features
                utility[i][j] = new ShapMatrixEntry();
                utility[i][j].record = new ArrayList<>();  //记录每层的方差
            }
        }

        //传入的levelIndex = 9; 取出一个长度为10的，加上长度为11；
        for(int l = level; l<num_features; l++){  // LM[i][j] - LM[i-1][j-1]
            for(int fea=0; fea<num_features; fea++){ //对于每个feature
                //第一个值：直接减去斜对角的值
                double value = 0;
                if(fea == num_features - 1){
                    value = newLevelMatrix[l][fea] - newLevelMatrix[l-1][0];  //l=10，存储长度11
                }
                else{
                    value = newLevelMatrix[l][fea] - newLevelMatrix[l-1][fea+1];  //l=10，存储长度11
                }
                utility[l][fea].record.add(value);  // l=10; 储的sub长度: 11 - 10
                utility[l][fea].sum += value;
                utility[l][fea].count ++;

                //第二个值：选取LM对应的这层包含的值
                for(int count = 0; count <limit; count ++){
                    //再随机选1个长度相同的，构成新的subset
                    Set<Integer> hashSet = new HashSet<>();
                    while (hashSet.size() <= l) {  //level=9, l=10，存储对应长度为11的subset
                        int number = random.nextInt(num_features); // 从0到n（包含n）中随机生成一个数
                        hashSet.add(number); // 添加到集合中，如果已经存在则不会添加
                    }
                    ArrayList<Integer> subset_1 = new ArrayList<>(hashSet);
                    if(!subset_1.contains(fea)){
                        int removeInd = random.nextInt(subset_1.size());
                        subset_1.set(removeInd, fea);
                    }
                    ArrayList<Integer> subset_2 = new ArrayList<>(subset_1);
                    subset_2.remove(Integer.valueOf(fea));  //删除元素，而不是下标
                    double value_1 = game.gameValue(game.model.gameName, subset_1); //len=11
                    double value_2 = game.gameValue(game.model.gameName, subset_2); //len=10
                    utility[l][fea].record.add(value_1 - value_2);  // l=10; 储的sub长度: 11 - 10
                    utility[l][fea].sum += value;
                    utility[l][fea].count ++;
                }
            }
        }

        //计算对应的方差 & 权重之和
        double[] varianceArr = new double[num_features];
        double key_total_weights = 0;
        for(int lev = 0; lev<num_features; lev++){  //一层的方差
            ShapMatrixEntry[] level_u = utility[lev];
            double variance = 0;
            for(ShapMatrixEntry ele_u : level_u){
                variance += varComputation(ele_u.record);
            }
            varianceArr[lev] = variance;
            this.total_weights +=variance;
            if(variance > 2.0){
                key_total_weights += variance;
            }
        }

        //第一次:按照权重分配
        for(int lev = 0; lev<num_features; lev++){
            int sample = 0;
            if(varianceArr[lev] != 0){
                sample = (int) Math.ceil(1.0 * allSamples * varianceArr[lev]/ this.total_weights);
            }
            this.num_sample[lev] += sample;
            this.total_samples += sample;  // 采样的总个数
        }

        //第二次:再分给方差大的
//        for(int lev = 0; lev<num_features; lev++){
//            int sample = 0;
//            if(varianceArr[lev] > 2.0){
//                sample = (int) Math.ceil(0.1f * allSamples * varianceArr[lev] / key_total_weights );
//            }
//            this.num_sample[lev] += sample;
//            this.total_samples += sample;  // 采样的总个数
//        }
        System.out.println(" total_samples " + this.total_samples);
    }

    //TODO 不使用LM，在每层中随机选取limit个计算MC
    public int sampleAllocationVot_2(int num_features, double[][] newLevelMatrix, int levelIndex, int allSamples, GameClass game) {
        this.weigh_for_sample = new double[num_features];  //记录每层采样的按权重
        this.num_sample = new int[num_features];  // 记录每层采样几个（按权重）
        this.total_weights = 0.0; //total_weight 权重总数
        int level = Math.max(2,levelIndex);
        int limit = Math.min(Math.max(2, Info.total_samples_num / (num_features * num_features * 2)), num_features); // 需要评估几个
        Random random = new Random(Info.seed);
//        System.out.println("limit: " + limit);

        // 初始化：varianceArr & utility
        ShapMatrixEntry[][] utility = new ShapMatrixEntry[num_features][]; // 需要换成带计数器的大数组
        for(int i=0; i<utility.length; i++){   //外层循环：features对应每个长度的collations (n个feature，n+1层)
            utility[i] = new ShapMatrixEntry[num_features];
            for(int j=0; j<num_features; j++) {  //内层循坏：每个features
                utility[i][j] = new ShapMatrixEntry();
                utility[i][j].record = new ArrayList<>();  //记录每层的方差
            }
        }

        //传入的levelIndex = 9; 取出一个长度为10的，加上长度为11；
        for(int l = level; l<num_features; l++){  // LM[i][j] - LM[i-1][j-1]
            for(int fea=0; fea<num_features; fea++){ //对于每个feature
                //第一个值：直接减去斜对角的值
//                double value = 0;
//                if(fea == num_features - 1){
//                    value = newLevelMatrix[l][fea] - newLevelMatrix[l-1][0];  //l=10，存储长度11
//                }
//                else{
//                    value = newLevelMatrix[l][fea] - newLevelMatrix[l-1][fea+1];  //l=10，存储长度11
//                }
//                utility[l][fea].record.add(value);  // l=10; 储的sub长度: 11 - 10
//                utility[l][fea].sum += value;
//                utility[l][fea].count ++;

                //第二个值：选取LM对应的这层包含的值
                for(int count = 0; count <=limit; count ++){
                    //再随机选1个长度相同的，构成新的subset
                    Set<Integer> hashSet = new HashSet<>();
                    while (hashSet.size() <= l) {  //level=9, l=10，存储对应长度为11的subset
                        int number = random.nextInt(num_features); // 从0到n（包含n）中随机生成一个数
                        hashSet.add(number); // 添加到集合中，如果已经存在则不会添加
                    }
                    ArrayList<Integer> subset_1 = new ArrayList<>(hashSet);
                    if(!subset_1.contains(fea)){
                        int removeInd = random.nextInt(subset_1.size());
                        subset_1.set(removeInd, fea);
                    }
                    ArrayList<Integer> subset_2 = new ArrayList<>(subset_1);
                    subset_2.remove(Integer.valueOf(fea));  //删除元素，而不是下标
                    double value_1 = game.gameValue(game.model.gameName, subset_1); //len=11
                    double value_2 = game.gameValue(game.model.gameName, subset_2); //len=10
                    utility[l][fea].record.add(value_1 - value_2);  // l=10; 储的sub长度: 11 - 10
                    utility[l][fea].sum += value_1 - value_2;
                    utility[l][fea].count ++;
                }
            }
        }

        //计算对应的方差 & 权重之和
        double[] varianceArr = new double[num_features];
        for(int lev = 0; lev<num_features; lev++){  //一层的方差
            ShapMatrixEntry[] level_u = utility[lev];
            double variance = 0;
            for(ShapMatrixEntry ele_u : level_u){
                variance += varComputation(ele_u.record);
            }
//            System.out.print(lev + ":" + variance + "\t");
            varianceArr[lev] = variance;
            this.total_weights +=variance;
        }

        //--------------------------第二次:分给方差大的----------------------------
//        double check_weight = 0.2;
//        double present = 0.05;
//        double key_total_weights = 0;
//        for(int lev = 19; lev<=32; lev++) {
////            if (varianceArr[lev] > present * this.total_weights) {
//                key_total_weights += varianceArr[lev];
////            }
//        }
//        for(int lev = 19; lev<=32; lev++){
//            int sample = 0;
////            if(varianceArr[lev] > present * this.total_weights){
//                sample = (int) Math.ceil(check_weight * allSamples * varianceArr[lev] / key_total_weights );
////            }
//            this.num_sample[lev] += sample;
//            this.total_samples += sample;  // 采样的总个数
//        }
//        System.out.println(" total_samples " + this.total_samples);
        //--------------------------------------------第一次:按照权重分配--------------------------------------------------
       int apartSam = (int) Math.ceil((allSamples - this.total_samples) / this.total_weights);
        for(int lev = 0; lev<num_features; lev++){
            int sample = 0;
            if(varianceArr[lev] != 0){
                sample = (int) Math.ceil(varianceArr[lev] * apartSam);
            }
            this.num_sample[lev] += sample;
            this.total_samples += sample;  // 采样的总个数
//            System.out.print(lev + ":" + this.num_sample[lev] + " \t ");
        }
        System.out.println(" total_samples " + this.total_samples);

        int return_lev = 0;
        for(int lev = levelIndex; lev<num_features; lev++){
            if(this.num_sample[lev] != 0){
                return_lev = lev;
                break;
            }
        }
        return return_lev;
    }

    //TODO 按随机采样的方差分配
    public int sampleAllocationVot_2(int num_features, double[][] newLevelMatrix, int levelIndex, int allSamples, GameClass game, Random random) {
        this.weigh_for_sample = new double[num_features];  //记录每层采样的按权重
        this.num_sample = new int[num_features];  // 记录每层采样几个（按权重）
        this.total_weights = 0.0; //total_weight 权重总数
        int level = Math.max(2,levelIndex);
        int limit = Math.min(Math.max(2, allSamples / (num_features * num_features * 2)), num_features); // 需要评估几个
//        System.out.println("limit: " + limit);

        // 初始化：varianceArr & utility
        ShapMatrixEntry[][] utility = new ShapMatrixEntry[num_features][]; // 需要换成带计数器的大数组
        for(int i=0; i<utility.length; i++){   //外层循环：features对应每个长度的collations (n个feature，n+1层)
            utility[i] = new ShapMatrixEntry[num_features];
            for(int j=0; j<num_features; j++) {  //内层循坏：每个features
                utility[i][j] = new ShapMatrixEntry();
                utility[i][j].record = new ArrayList<>();  //记录每层的方差
            }
        }

        //传入的levelIndex = 9; 取出一个长度为10的，加上长度为11；
        for(int l = level; l<num_features; l++){  // LM[i][j] - LM[i-1][j-1]
            for(int fea=0; fea<num_features; fea++){ //对于每个feature
                //第一个值：直接减去斜对角的值
//                double value = 0;
//                if(fea == num_features - 1){
//                    value = newLevelMatrix[l][fea] - newLevelMatrix[l-1][0];  //l=10，存储长度11
//                }
//                else{
//                    value = newLevelMatrix[l][fea] - newLevelMatrix[l-1][fea+1];  //l=10，存储长度11
//                }
//                utility[l][fea].record.add(value);  // l=10; 储的sub长度: 11 - 10
//                utility[l][fea].sum += value;
//                utility[l][fea].count ++;

                //第二个值：选取LM对应的这层包含的值
                for(int count = 0; count <=limit; count ++){
                    //再随机选1个长度相同的，构成新的subset
                    Set<Integer> hashSet = new HashSet<>();
                    while (hashSet.size() <= l) {  //level=9, l=10，存储对应长度为11的subset
                        int number = random.nextInt(num_features); // 从0到n（包含n）中随机生成一个数
                        hashSet.add(number); // 添加到集合中，如果已经存在则不会添加
                    }
                    ArrayList<Integer> subset_1 = new ArrayList<>(hashSet);
                    if(!subset_1.contains(fea)){
                        int removeInd = random.nextInt(subset_1.size());
                        subset_1.set(removeInd, fea);
                    }
                    ArrayList<Integer> subset_2 = new ArrayList<>(subset_1);
                    subset_2.remove(Integer.valueOf(fea));  //删除元素，而不是下标
                    double value_1 = game.gameValue(game.model.gameName, subset_1); //len=11
                    double value_2 = game.gameValue(game.model.gameName, subset_2); //len=10
                    utility[l][fea].record.add(value_1 - value_2);  // l=10; 储的sub长度: 11 - 10
                    utility[l][fea].sum += value_1 - value_2;
                    utility[l][fea].count ++;
                }
            }
        }

        //计算对应的方差 & 权重之和
        double[] varianceArr = new double[num_features];
        for(int lev = 0; lev<num_features; lev++){  //一层的方差
            ShapMatrixEntry[] level_u = utility[lev];
            double variance = 0;
            for(ShapMatrixEntry ele_u : level_u){
                variance += varComputation(ele_u.record);
            }
//            System.out.print(lev + ":" + variance + "\t");
            varianceArr[lev] = variance;
            this.total_weights +=variance;
        }

        //--------------------------第3次:分给方差大的----------------------------
//        double check_weight = 0.1;  //控制该阶段的采样总数
//        double present = 0.05;   //某层的方差与总方差的占比
//        double key_total_weights = 0;
//        for(int lev = 16; lev<33; lev++) {   //for(int lev = 19; lev<32; lev++) {   // for(int lev = level; lev<num_features; lev++)
////            if (varianceArr[lev] > present * this.total_weights) {
//                key_total_weights += varianceArr[lev];
////            }
//        }
//        for(int lev = 16; lev<33; lev++){
//            int sample = 0;
////            if(varianceArr[lev] > present * this.total_weights){
//                sample = (int) Math.ceil(check_weight * allSamples * varianceArr[lev] / key_total_weights );
////            }
//            this.num_sample[lev] += sample;
//            this.total_samples += sample;  // 采样的总个数
//        }
//        System.out.println(" total_samples " + this.total_samples);
        //--------------------------------------------第1次:按照权重分配--------------------------------------------------
        int apartSam = (int) Math.ceil(1.0f * allSamples / this.total_weights);  //每一份是几个样本
        for(int lev = 0; lev<num_features; lev++){
            int sample = 0;
            if(varianceArr[lev] != 0){
                sample = (int) Math.ceil(varianceArr[lev] * apartSam);
            }
            this.num_sample[lev] += sample;
            this.total_samples += sample;  // 采样的总个数
//            System.out.print(lev + ":" + this.num_sample[lev] + " \t ");
        }
        System.out.println(" total_samples " + this.total_samples);
        //--------------------------------------------第2次:平均分给中间层-------------------------------------------------
        // 2）按照1）中设置的权重 + 设定的采样数，实际分配采样个数  // num_samples: 计算总共采样个数
//        int sample = (int) Math.ceil(0.0f * allSamples / (33-16));  //每层分配的采样数
//        for(int ind = 16; ind <33; ind ++){
//            this.num_sample[ind] += sample;
//            this.total_samples += sample;  // 采样的总个数
////            System.out.print( this.num_sample[ind] + "\t");
//        }
//        System.out.println("this.total_samples: " + this.total_samples);
       //---------------------------------------------------------------------------------------------------------------
        int return_lev = levelIndex;
        for(int lev = levelIndex; lev<num_features; lev++){
            if(this.num_sample[lev] != 0){
                return_lev = lev;
                break;
            }
        }
        return return_lev;
    }

    //TODO 直接用LM计算层方差分配：不是两层之间减去，而是直接计算这层的值
    public void sampleAllocationVot_3(int num_features, double[][] newLevelMatrix, int levelIndex, int allSamples) {
        this.weigh_for_sample = new double[num_features];  //记录每层采样的按权重
        this.num_sample = new int[num_features];  // 记录每层采样几个（按权重）
        this.total_weights = 0.0; //total_weight 权重总数
        int level = Math.max(2,levelIndex);
        int limit = Math.min(Math.max(2, Info.total_samples_num / (num_features * num_features * 2)), num_features); // 需要评估几个

        // 初始化：varianceArr & utility
        ShapMatrixEntry[] utility_level = new ShapMatrixEntry[num_features]; // 需要换成带计数器的大数组
            for(int l=0; l<utility_level.length; l++) {
                utility_level[l] = new ShapMatrixEntry();
                utility_level[l].record = new ArrayList<>();  //记录每层的方差
                for(int fea=0; fea<num_features; fea++){ //对于每个feature
                    utility_level[l].record.add(newLevelMatrix[l][fea]);  // l=10; 储的sub长度: 11 - 10
                    utility_level[l].sum += newLevelMatrix[l][fea];
                    utility_level[l].count ++;
                }
            }

        //计算对应的方差 & 权重之和
        double key_present = 0.03;
        double[] varianceArr = new double[num_features];
        double key_total_weights = 0;
        for(int lev = 0; lev<num_features; lev++){  //一层的方差
            ShapMatrixEntry level_u = utility_level[lev];
            varianceArr[lev] = varComputation(level_u.record);
            this.total_weights += varianceArr[lev];
//            System.out.print(varianceArr[lev] + " \t ");
        }
//        System.out.println();
        for(int lev = 0; lev<num_features; lev++) {  //一层的方差
//            if (varianceArr[lev] > this.total_weights * key_present) {
                key_total_weights += varianceArr[lev];
//                System.out.print(lev + " \t ");
//            }
        }
//        System.out.println();

        //----------------第二次:再分给方差大的------------------
//        for(int lev = 0; lev<num_features; lev++){
//            int sample = 0;
//            if(varianceArr[lev] > this.total_weights * key_present){
//                sample = (int) Math.ceil(0.5 * allSamples * varianceArr[lev] / key_total_weights);
////                System.out.print(lev + " \t ");
//            }
//            this.num_sample[lev] += sample;
////            System.out.print(this.num_sample[lev] + "\t");
//            this.total_samples += sample;  // 采样的总个数
//        }
//        System.out.println();
//        System.out.println(" total_samples " + this.total_samples);

        //第一次:按照权重分配
        int second_sam = allSamples - this.total_samples;
        for(int lev = 0; lev<num_features; lev++){
            int sample = 0;
            if(varianceArr[lev] != 0){
                sample = (int) Math.ceil(second_sam * varianceArr[lev]/ this.total_weights);
            }
//            System.out.print(sample + " \t ");
            this.num_sample[lev] += sample;
            this.total_samples += sample;  // 采样的总个数
        }
        System.out.println(" total_samples " + this.total_samples);
    }

    //TODO 按随机采样的方差分配
    //shap_matrix[len][feature]
    public int sampleAllocationVot_4(int num_features, double[][] newLevelMatrix, int start_level, int end_level, int allSamples, GameClass game, Random random, ShapMatrixEntry[][] shap_matrix) {
        this.weigh_for_sample = new double[num_features];  //记录每层采样的按权重
        this.num_sample = new int[num_features];  // 记录每层采样几个（按权重）
        this.total_weights = 0.0; //total_weight 权重总数
        int limit = Math.min(Math.max(2, allSamples / (num_features * num_features * 2)), num_features); // 需要评估几个
//        System.out.println("limit: " + limit);

        // 初始化：varianceArr & utility
        ShapMatrixEntry[][] utility = new ShapMatrixEntry[num_features][]; // 需要换成带计数器的大数组
        for(int j=0; j<utility.length; j++){   //外层循环：features对应每个长度的collations (n个feature，n+1层)
            utility[j] = new ShapMatrixEntry[num_features];
            for(int i=0; i<num_features; i++) {  //内层循坏：每个features
                utility[j][i] = new ShapMatrixEntry();
            }
        }

        //传入的levelIndex = 9; 取出一个长度为10的，加上长度为11；
        for(int l = start_level; l<=end_level; l++){  // LM[i][j] - LM[i-1][j-1]
            for(int fea=0; fea<num_features; fea++){ //对于每个feature
                //第一个值：直接减去斜对角的值
//                double value = 0;
//                if(fea == num_features - 1){
//                    value = newLevelMatrix[l][fea] - newLevelMatrix[l-1][0];  //l=10，存储长度11
//                }
//                else{
//                    value = newLevelMatrix[l][fea] - newLevelMatrix[l-1][fea+1];  //l=10，存储长度11
//                }
//                shap_matrix[l][fea].record.add(value);  // l=10; 储的sub长度: 11 - 10
//                shap_matrix[l][fea].sum += value;
//                shap_matrix[l][fea].count ++;

                //第二个值：选取LM对应的这层包含的值
                for(int count = 0; count <=limit; count ++){
                    //再随机选1个长度相同的，构成新的subset
                    Set<Integer> hashSet = new HashSet<>();
                    while (hashSet.size() <= l) {  //level=9, l=10，存储对应长度为11的subset
                        int number = random.nextInt(num_features); // 从0到n（包含n）中随机生成一个数
                        hashSet.add(number); // 添加到集合中，如果已经存在则不会添加
                    }
                    ArrayList<Integer> subset_1 = new ArrayList<>(hashSet);  //前项，包含fea
                    if(!subset_1.contains(fea)){
                        int removeInd = random.nextInt(subset_1.size());
                        subset_1.set(removeInd, fea);
                    }
                    ArrayList<Integer> subset_2 = new ArrayList<>(subset_1);  //后项，不含fea
                    subset_2.remove(Integer.valueOf(fea));  //删除元素，而不是下标
                    double value_1 = game.gameValue(game.model.gameName, subset_1); //len=11
                    double value_2 = game.gameValue(game.model.gameName, subset_2); //len=10
                    utility[l][fea].record.add(value_1 - value_2);  // l=10; 储的sub长度: 11 - 10
                    utility[l][fea].sum += value_1 - value_2;
                    utility[l][fea].count ++;
                    //20240821添加：把估计的值存入
                    shap_matrix[l][fea].record.add(value_1 - value_2);
                    shap_matrix[l][fea].sum += value_1 - value_2;
                    shap_matrix[l][fea].count ++;
                }
            }
        }

        //计算对应的方差 & 权重之和
        double[] varianceArr = new double[num_features];
        for(int lev = 0; lev<num_features; lev++){  //一层的方差
            ShapMatrixEntry[] level_u = utility[lev];
            double variance = 0;
            for(ShapMatrixEntry ele_u : level_u){
                variance += varComputation(ele_u.record);
            }
//            System.out.print(lev + ":" + variance + "\t");
            varianceArr[lev] = variance;
            this.total_weights +=variance;
        }

        //--------------------------------------------第1次:按照权重分配--------------------------------------------------
        int apartSam = (int) Math.ceil(1.0f * allSamples / this.total_weights);  //每一份是几个样本
        for(int lev = 0; lev<num_features; lev++){
            int sample = 0;
            if(varianceArr[lev] != 0){
                sample = (int) Math.ceil(varianceArr[lev] * apartSam);
            }
            this.num_sample[lev] += sample;
            this.total_samples += sample;  // 采样的总个数
//            System.out.print(lev + ":" + this.num_sample[lev] + " \t ");
        }
//        System.out.println(" total_samples " + this.total_samples);
        //--------------------------------------------第2次:平均分给中间层-------------------------------------------------
        // 2）按照1）中设置的权重 + 设定的采样数，实际分配采样个数  // num_samples: 计算总共采样个数
//        int sample = (int) Math.ceil(0.0f * allSamples / (33-16));  //每层分配的采样数
//        for(int ind = 16; ind <33; ind ++){
//            this.num_sample[ind] += sample;
//            this.total_samples += sample;  // 采样的总个数
////            System.out.print( this.num_sample[ind] + "\t");
//        }
//        System.out.println("this.total_samples: " + this.total_samples);
        //-----------------------------------------第3次:分给方差大的------------------------------------------------------
//        double check_weight = 0.1;  //控制该阶段的采样总数
//        double present = 0.05;   //某层的方差与总方差的占比
//        double key_total_weights = 0;
//        for(int lev = 16; lev<33; lev++) {   //for(int lev = 19; lev<32; lev++) {   // for(int lev = level; lev<num_features; lev++)
////            if (varianceArr[lev] > present * this.total_weights) {
//                key_total_weights += varianceArr[lev];
////            }
//        }
//        for(int lev = 16; lev<33; lev++){
//            int sample = 0;
////            if(varianceArr[lev] > present * this.total_weights){
//                sample = (int) Math.ceil(check_weight * allSamples * varianceArr[lev] / key_total_weights );
////            }
//            this.num_sample[lev] += sample;
//            this.total_samples += sample;  // 采样的总个数
//        }
//        System.out.println(" total_samples " + this.total_samples);
        //---------------------------------------------------------------------------------------------------------------
        int return_lev = start_level;
        for(int lev = start_level; lev<num_features; lev++){
            if(this.num_sample[lev] != 0){
                return_lev = lev;
                break;
            }
        }
        return return_lev;
    }

    //TODO 按随机采样的方差分配
    //shap_matrix[len][feature]
    public int sampleAllocationVot_4(int num_features, int allSamples, GameClass game, Random random, ShapMatrixEntry[][] shap_matrix) {
        this.weigh_for_sample = new double[num_features];  //记录每层采样的按权重
        this.num_sample = new int[num_features];  // 记录每层采样几个（按权重）
        this.total_weights = 0.0; //total_weight 权重总数
        int limit = Math.min(Math.max(2, allSamples / (num_features * num_features * 2)), num_features); // 需要评估几个
        System.out.println("limit: " + limit);

        // 初始化：varianceArr & utility
        ShapMatrixEntry[][] utility = new ShapMatrixEntry[num_features][]; // 需要换成带计数器的大数组
        for(int j=0; j<utility.length; j++){   //外层循环：features对应每个长度的collations (n个feature，n+1层)
            utility[j] = new ShapMatrixEntry[num_features];
            for(int i=0; i<num_features; i++) {  //内层循坏：每个features
                utility[j][i] = new ShapMatrixEntry();
            }
        }

        //传入的levelIndex = 9; 取出一个长度为10的，加上长度为11；
        for(int l = game.start_level; l<=game.end_level; l++){  // LM[i][j] - LM[i-1][j-1]
            for(int fea=0; fea<num_features; fea++){ //对于每个feature
                //第一个值：直接减去斜对角的值
//                double value = 0;
//                if(fea == num_features - 1){
//                    value = newLevelMatrix[l][fea] - newLevelMatrix[l-1][0];  //l=10，存储长度11
//                }
//                else{
//                    value = newLevelMatrix[l][fea] - newLevelMatrix[l-1][fea+1];  //l=10，存储长度11
//                }
//                shap_matrix[l][fea].record.add(value);  // l=10; 储的sub长度: 11 - 10
//                shap_matrix[l][fea].sum += value;
//                shap_matrix[l][fea].count ++;

                //第二个值：选取LM对应的这层包含的值
                for(int count = 0; count <=limit; count ++){
                    //再随机选1个长度相同的，构成新的subset
                    Set<Integer> hashSet = new HashSet<>();
                    while (hashSet.size() <= l) {  //level=9, l=10，存储对应长度为11的subset
                        int number = random.nextInt(num_features); // 从0到n（包含n）中随机生成一个数
                        hashSet.add(number); // 添加到集合中，如果已经存在则不会添加
                    }
                    ArrayList<Integer> subset_1 = new ArrayList<>(hashSet);  //前项，包含fea
                    if(!subset_1.contains(fea)){
                        int removeInd = random.nextInt(subset_1.size());
                        subset_1.set(removeInd, fea);
                    }
                    ArrayList<Integer> subset_2 = new ArrayList<>(subset_1);  //后项，不含fea
                    subset_2.remove(Integer.valueOf(fea));  //删除元素，而不是下标
                    double value_1 = game.gameValue(game.model.gameName, subset_1); //len=11
                    double value_2 = game.gameValue(game.model.gameName, subset_2); //len=10
                    utility[l][fea].record.add(value_1 - value_2);  // l=10; 储的sub长度: 11 - 10
                    utility[l][fea].sum += value_1 - value_2;
                    utility[l][fea].count ++;
                    //20240821添加：把估计的值存入
                    shap_matrix[l][fea].record.add(value_1 - value_2);
                    shap_matrix[l][fea].sum += value_1 - value_2;
                    shap_matrix[l][fea].count ++;
                }
            }
        }

        //计算对应的方差 & 权重之和
        double[] varianceArr = new double[num_features];
        for(int lev = 0; lev<num_features; lev++){  //一层的方差
            ShapMatrixEntry[] level_u = utility[lev];
            double variance = 0;
            for(ShapMatrixEntry ele_u : level_u){
                variance += varComputation(ele_u.record);
            }
//            System.out.print(lev + ":" + variance + "\t");
            varianceArr[lev] = variance;
            this.total_weights +=variance;
        }

        //--------------------------------------------第1次:按照权重分配--------------------------------------------------
        int apartSam = (int) Math.ceil((1.0 - game.check_weight) * allSamples / this.total_weights);  //每一份是几个样本
        System.out.println("apartSam: " + apartSam);
        for(int lev = 0; lev<num_features; lev++){
            int sample = 0;
            if(varianceArr[lev] != 0){
                sample = (int) Math.ceil(varianceArr[lev] * apartSam);
            }
            this.num_sample[lev] += sample;
            this.total_samples += sample;  // 采样的总个数
            System.out.print(lev + ":" + this.num_sample[lev] + " \t ");
        }
        System.out.println("total_samples " + this.total_samples);
        //--------------------------------------------第2次:平均分给中间层-------------------------------------------------
        // 2）按照1）中设置的权重 + 设定的采样数，实际分配采样个数  // num_samples: 计算总共采样个数
//        int sample = (int) Math.ceil(0.0f * allSamples / (33-16));  //每层分配的采样数
//        for(int ind = 16; ind <33; ind ++){
//            this.num_sample[ind] += sample;
//            this.total_samples += sample;  // 采样的总个数
////            System.out.print( this.num_sample[ind] + "\t");
//        }
//        System.out.println("this.total_samples: " + this.total_samples);
        //-----------------------------------------第3次:分给方差大的------------------------------------------------------
//        double check_weight = 0.1;  //控制该阶段的采样总数
//        double present = 0.05;   //某层的方差与总方差的占比
//        double key_total_weights = 0;
//        for(int lev = 16; lev<33; lev++) {   //for(int lev = 19; lev<32; lev++) {   // for(int lev = level; lev<num_features; lev++)
////            if (varianceArr[lev] > present * this.total_weights) {
//                key_total_weights += varianceArr[lev];
////            }
//        }
//        for(int lev = 16; lev<33; lev++){
//            int sample = 0;
////            if(varianceArr[lev] > present * this.total_weights){
//                sample = (int) Math.ceil(check_weight * allSamples * varianceArr[lev] / key_total_weights );
////            }
//            this.num_sample[lev] += sample;
//            this.total_samples += sample;  // 采样的总个数
//        }
//        System.out.println(" total_samples " + this.total_samples);
        //---------------------------------------------------------------------------------------------------------------
        int return_lev = game.start_level;
        for(int lev = game.start_level; lev<num_features; lev++){
            if(this.num_sample[lev] != 0){
                return_lev = lev;
                break;
            }
        }
        return return_lev;
    }


    //TODO 样本减法：按随机采样的方差分配
    //shap_matrix[len][feature]
    public int sampleAllocationReal(int num_features, double[][] newLevelMatrix, int start_level, int end_level, int allSamples, GameClass game, Random random, ShapMatrixEntry[][] shap_matrix) {
        this.weigh_for_sample = new double[num_features];  //记录每层采样的按权重
        this.num_sample = new int[num_features];  // 记录每层采样几个（按权重）
        this.total_weights = 0.0; //total_weight 权重总数
        int limit = Math.min(Math.max(2, allSamples / (num_features * num_features * 2)), num_features); // 需要评估几个
//        System.out.println("limit: " + limit);

        // 初始化：varianceArr & utility
        ShapMatrixEntry[][] utility = new ShapMatrixEntry[num_features][]; // 需要换成带计数器的大数组
        for(int j=0; j<utility.length; j++){   //外层循环：features对应每个长度的collations (n个feature，n+1层)
            utility[j] = new ShapMatrixEntry[num_features];
            for(int i=0; i<num_features; i++) {  //内层循坏：每个features
                utility[j][i] = new ShapMatrixEntry();
            }
        }

        //传入的levelIndex = 9; 取出一个长度为10的，加上长度为11；
        for(int l = start_level; l<=end_level; l++){  // LM[i][j] - LM[i-1][j-1]
            for(int fea=0; fea<num_features; fea++){ //对于每个feature
                //第一个值：直接减去斜对角的值
                double value = 0;
                if(fea == num_features - 1){
                    value = newLevelMatrix[l][fea] - newLevelMatrix[l-1][0];  //l=10，存储长度11
                }
                else{
                    value = newLevelMatrix[l][fea] - newLevelMatrix[l-1][fea+1];  //l=10，存储长度11
                }
                shap_matrix[l][fea].record.add(value);  // l=10; 储的sub长度: 11 - 10
                shap_matrix[l][fea].sum += value;
                shap_matrix[l][fea].count ++;

                //第二个值：选取LM对应的这层包含的值
                for(int count = 0; count <limit; count ++){
                    //再随机选1个长度相同的，构成新的subset
                    Set<Integer> hashSet = new HashSet<>();
                    while (hashSet.size() <= l) {  //level=9, l=10，存储对应长度为11的subset
                        int number = random.nextInt(num_features); // 从0到n（包含n）中随机生成一个数
                        hashSet.add(number); // 添加到集合中，如果已经存在则不会添加
                    }
                    ArrayList<Integer> subset_1 = new ArrayList<>(hashSet);  //前项，包含fea
                    if(!subset_1.contains(fea)){
                        int removeInd = random.nextInt(subset_1.size());
                        subset_1.set(removeInd, fea);
                    }
                    ArrayList<Integer> subset_2 = new ArrayList<>(subset_1);  //后项，不含fea
                    subset_2.remove(Integer.valueOf(fea));  //删除元素，而不是下标
                    double value_1 = game.gameValue(game.model.gameName, subset_1); //len=11
                    double value_2 = game.gameValue(game.model.gameName, subset_2); //len=10
                    utility[l][fea].record.add(value_1 - value_2);  // l=10; 储的sub长度: 11 - 10
                    utility[l][fea].sum += value_1 - value_2;
                    utility[l][fea].count ++;
                    //20240821添加：把估计的值存入
                    shap_matrix[l][fea].record.add(value_1 - value_2);
                    shap_matrix[l][fea].sum += value_1 - value_2;
                    shap_matrix[l][fea].count ++;
                }
            }
        }

        //计算对应的方差 & 权重之和
        double[] varianceArr = new double[num_features];
        for(int lev = 0; lev<num_features; lev++){  //一层的方差
            ShapMatrixEntry[] level_u = utility[lev];
            double variance = 0;
            for(ShapMatrixEntry ele_u : level_u){
                variance += varComputation(ele_u.record);
            }
//            System.out.print(lev + ":" + variance + "\t");
            varianceArr[lev] = variance;
            this.total_weights +=variance;
        }
        //--------------------------------------------第1次:按照权重分配--------------------------------------------------
        allSamples = Math.max(1, allSamples - limit * (end_level - start_level - 1) * num_features);
//        allSamples = allSamples - limit * (end_level - start_level - 1) * num_features;
//        if(allSamples == 0){
//            System.out.println(limit + " * " + " ( " + end_level + " - " + start_level + " ) * n = " + (allSamples - limit * (end_level - start_level - 1) * num_features));
//        }
//        allSamples = allSamples - limit * (end_level - start_level-1) * num_features;
//        allSamples = Math.max(allSamples, 1);
        System.out.println("allSamples: " + allSamples);
        int apartSam = (int) Math.ceil(1.0f * allSamples / this.total_weights);  //每一份是几个样本
        for(int lev = 0; lev<num_features; lev++){
            int sample = 0;
            if(varianceArr[lev] != 0){
                sample = (int) Math.ceil(varianceArr[lev] * apartSam);
            }
            this.num_sample[lev] += sample;
            this.total_samples += sample;  // 采样的总个数
//            System.out.print(lev + ":" + this.num_sample[lev] + " \t ");
        }
        System.out.println(" total_samples: " + this.total_samples);
        int return_lev = start_level;
        for(int lev = start_level; lev<num_features; lev++){
            if(this.num_sample[lev] != 0){
                return_lev = lev;
                break;
            }
        }
        return return_lev;
    }

    //TODO 样本减法：按随机采样的方差分配
    //shap_matrix[len][feature]
    public int sampleAllocationReal(int num_features, double[][] newLevelMatrix, int allSamples, GameClass game, Random random, ShapMatrixEntry[][] shap_matrix) {
        this.weigh_for_sample = new double[num_features];  //记录每层采样的按权重
        this.num_sample = new int[num_features];  // 记录每层采样几个（按权重）
        this.total_weights = 0.0; //total_weight 权重总数
        int limit = Math.min(Math.max(2, allSamples / (num_features * num_features * 2)), num_features); // 需要评估几个
        System.out.println("limit: " + limit);

        // 初始化：varianceArr & utility
        ShapMatrixEntry[][] utility = new ShapMatrixEntry[num_features][]; // 需要换成带计数器的大数组
        for(int j=0; j<utility.length; j++){   //外层循环：features对应每个长度的collations (n个feature，n+1层)
            utility[j] = new ShapMatrixEntry[num_features];
            for(int i=0; i<num_features; i++) {  //内层循坏：每个features
                utility[j][i] = new ShapMatrixEntry();
            }
        }

        //传入的levelIndex = 9; 取出一个长度为10的，加上长度为11；
        for(int l = game.start_level; l<=game.end_level; l++){  // LM[i][j] - LM[i-1][j-1]
            for(int fea=0; fea<num_features; fea++){ //对于每个feature
                //第一个值：直接减去斜对角的值
                double value = 0;
                if(fea == num_features - 1){
                    value = newLevelMatrix[l][fea] - newLevelMatrix[l-1][0];  //l=10，存储长度11
                }
                else{
                    value = newLevelMatrix[l][fea] - newLevelMatrix[l-1][fea+1];  //l=10，存储长度11
                }
                shap_matrix[l][fea].record.add(value);  // l=10; 储的sub长度: 11 - 10
                shap_matrix[l][fea].sum += value;
                shap_matrix[l][fea].count ++;

                //第二个值：选取LM对应的这层包含的值
                for(int count = 0; count <limit; count ++){
                    //再随机选1个长度相同的，构成新的subset
                    Set<Integer> hashSet = new HashSet<>();
                    while (hashSet.size() <= l) {  //level=9, l=10，存储对应长度为11的subset
                        int number = random.nextInt(num_features); // 从0到n（包含n）中随机生成一个数
                        hashSet.add(number); // 添加到集合中，如果已经存在则不会添加
                    }
                    ArrayList<Integer> subset_1 = new ArrayList<>(hashSet);  //前项，包含fea
                    if(!subset_1.contains(fea)){
                        int removeInd = random.nextInt(subset_1.size());
                        subset_1.set(removeInd, fea);
                    }
                    ArrayList<Integer> subset_2 = new ArrayList<>(subset_1);  //后项，不含fea
                    subset_2.remove(Integer.valueOf(fea));  //删除元素，而不是下标
                    double value_1 = game.gameValue(game.model.gameName, subset_1); //len=11
                    double value_2 = game.gameValue(game.model.gameName, subset_2); //len=10
                    utility[l][fea].record.add(value_1 - value_2);  // l=10; 储的sub长度: 11 - 10
                    utility[l][fea].sum += value_1 - value_2;
                    utility[l][fea].count ++;
                    //20240821添加：把估计的值存入
                    shap_matrix[l][fea].record.add(value_1 - value_2);
                    shap_matrix[l][fea].sum += value_1 - value_2;
                    shap_matrix[l][fea].count ++;
                }
            }
        }

        //计算对应的方差 & 权重之和
        double[] varianceArr = new double[num_features];
        for(int lev = 0; lev<num_features; lev++){  //一层的方差
            ShapMatrixEntry[] level_u = utility[lev];
            double variance = 0;
            for(ShapMatrixEntry ele_u : level_u){
                variance += varComputation(ele_u.record);
            }
//            System.out.print(lev + ":" + variance + "\t");
            varianceArr[lev] = variance;
            this.total_weights +=variance;
        }
        //--------------------------------------------第1次:按照权重分配--------------------------------------------------
        allSamples = (int) Math.ceil((1.0 - game.check_weight) * (allSamples - limit * (game.end_level - game.start_level - 1) * num_features));
        allSamples = Math.max(1,  allSamples);
//        allSamples = allSamples - limit * (end_level - start_level - 1) * num_features;
//        if(allSamples == 0){
//            System.out.println(limit + " * " + " ( " + end_level + " - " + start_level + " ) * n = " + (allSamples - limit * (end_level - start_level - 1) * num_features));
//        }
//        allSamples = allSamples - limit * (end_level - start_level-1) * num_features;
//        allSamples = Math.max(allSamples, 1);
        System.out.println("allSamples: " + allSamples);
        int apartSam = (int) Math.ceil(1.0f * allSamples / this.total_weights);  //每一份是几个样本
        for(int lev = 0; lev<num_features; lev++){
            int sample = 0;
            if(varianceArr[lev] != 0){
                sample = (int) Math.ceil(varianceArr[lev] * apartSam);
            }
            this.num_sample[lev] += sample;
            this.total_samples += sample;  // 采样的总个数
//            System.out.print(lev + ":" + this.num_sample[lev] + " \t ");
        }
        System.out.println(" total_samples: " + this.total_samples);
        int return_lev = game.start_level;
        for(int lev = game.start_level; lev<num_features; lev++){
            if(this.num_sample[lev] != 0){
                return_lev = lev;
                break;
            }
        }
        return return_lev;
    }

    //TODO: allocate the number of evaluations by the variance of each layer
    public int sampleAllocation(int num_features, double[][] newLevelMatrix, int limit, int allSamples, GameClass game, Random random, ShapMatrixEntry[][] shap_matrix) {
        this.weigh_for_sample = new double[num_features];  //记录每层采样的按权重
        this.num_sample = new int[num_features];  // 记录每层采样几个（按权重）
        this.total_weights = 0.0; //total_weight 权重总数

        // 初始化：varianceArr & utility
        ShapMatrixEntry[][] utility = new ShapMatrixEntry[num_features][];
        for(int j=0; j<utility.length; j++){   //外层循环：features对应每个长度的collations (n个feature，n+1层)
            utility[j] = new ShapMatrixEntry[num_features];
            for(int i=0; i<num_features; i++) {  //内层循坏：每个features
                utility[j][i] = new ShapMatrixEntry();
            }
        }

        for(int l = game.start_level; l<=game.end_level; l++){  // LM[i][j] - LM[i-1][j-1]
            for(int fea=0; fea<num_features; fea++){ //对于每个feature
               if(game.isRealData){
                   //第一个值：直接减去斜对角的值
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

                //第二个值：选取LM对应的这层包含的值
                for(int count = 0; count <limit; count ++){
                    //再随机选1个长度相同的，构成新的subset
                    Set<Integer> hashSet = new HashSet<>();
                    while (hashSet.size() <= l) {
                        int number = random.nextInt(num_features); // 从0到n（包含n）中随机生成一个数
                        hashSet.add(number); // 添加到集合中，如果已经存在则不会添加
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

        //计算对应的方差 & 权重之和
        double[] varianceArr = new double[num_features];
        for(int lev = 0; lev<num_features; lev++){  //一层的方差
            ShapMatrixEntry[] level_u = utility[lev];
            double variance = 0;
            for(ShapMatrixEntry ele_u : level_u){
                variance += varComputation(ele_u.record);
            }
//            System.out.print(lev + ":" + variance + "\t");
            varianceArr[lev] = variance;
            this.total_weights +=variance;
        }
        //--------------------------------------------第1次:按照权重分配--------------------------------------------------
        int apartSam = (int) Math.ceil(allSamples/this.total_weights);  //每一份是几个样本
//        System.out.println(apartSam);
        for(int lev = 0; lev<num_features; lev++){
            int sample = 0;
            if(varianceArr[lev] != 0){
                sample = (int) Math.ceil(varianceArr[lev] * apartSam);
            }
            this.num_sample[lev] += sample;
//            this.total_samples += sample;  // 采样的总个数
//            System.out.print(lev + ":" + this.num_sample[lev] + " \t ");
        }
//        System.out.println(" total_samples: " + this.total_samples);
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
        // Calculate the mean
        for (double value : record) {
            sum += value;
        }
        double mean = sum / record.size();
        // Calculate the variance
        double sumSquares = 0.0;
        for (double value : record) {
            sumSquares += Math.pow(value - mean, 2);
        }
        return sumSquares / record.size();  //特别指定ddof=1 分母是(n-1) 而不是1
    }
    private double varComputation_pow(ArrayList<Double> record) {
        if(record.size() <= 1){
            return 0;
        }
        double sum = 0.0;
        // Calculate the mean
        for (double value : record) {
            sum += value;
        }
        double mean = sum / record.size();
        // Calculate the variance
        double sumSquares = 0.0;
        for (double value : record) {
            sumSquares += Math.pow(value - mean, 4);
        }
        return sumSquares / record.size();  //特别指定ddof=1 分母是(n-1) 而不是1
    }
    public void sampleAllocation_11(int num_features, double[][]newLevelMatrix, int k) {
        this.weigh_for_sample = new double[num_features];  //记录每层采样的按权重
        this.num_sample = new int[num_features];  // 记录每层采样几个（按权重）
        this.total_weights = 0.0; //total_weight 权重总数

        // 1）依次遍历每层，计算权重存入数组中weigh_for_sample[] 中 (最后一层不参与计算)
        int level = Math.max(2,k);
        for(int i=level; i<num_features-1; i++){
            double log = num_features * Math.log(num_features)- i * Math.log(i) - (num_features-i) * Math.log(num_features-i);  //最大值70
            // C(n,k)= n!/ k!(n-k)!  // log(m*k) = log m + log k; log(m/k) = log m - log k;  // log(n!) 约等于 nlog(n)
            this.weigh_for_sample[i] = log;
            this.total_weights += log;
//            System.out.print(log + "\t");
        }

        //【添加第二阶段】 计算每一层的方差
        double[] varianceArray = new double[num_features+1];  //每一层的方差
        double allVariance = 0;
        for(int len = 2; len < num_features; len ++) {  //line_ind 是对角线的标记
            double step_sum = 0;  //这条对对角线上的方差
            for(int ind=0; ind < num_features; ind++){
                step_sum += newLevelMatrix[len][ind];
            }
            double step_mean = step_sum / num_features; //计算当前层的均值
            //计算方差
            double variance = 0.0;
            for(int ind=0; ind < num_features; ind++){
                variance += Math.pow(newLevelMatrix[len][ind] - step_mean, 2);
            }
            double oneVariance = variance / num_features;
            varianceArray[len] = oneVariance;
            allVariance += oneVariance;
//            System.out.print(oneVariance + "\t");
        }
//        System.out.println();

        //给出方差权重的比例
        double[] varianceRatio = new double[Info.num_of_features];
        for(int i=level; i<num_features; i++){
            varianceRatio[i] = varianceArray[i] / allVariance;
        }

        // 2）按照1）中设置的权重 + 设定的采样数，实际分配采样个数
        //计算总共采样个数
//        double radio = Info.number_weight * Info.total_samples_num / this.total_weights;  //定义扩充比例 (total_samples_num 总体采样数)
        double radio = this.number_weight * Info.total_samples_num / this.total_weights;  //定义扩充比例 (total_samples_num 总体采样数)
        while(this.total_samples < Info.total_samples_num){
            for(int ind = level; ind <num_features; ind ++){
//            int sample = (int) (Math.ceil(radio * this.weigh_for_sample[ind] + Info.number_variance * Info.total_samples_num * varianceRatio[ind]));  //每层分配的采样数
//            int sample = 1;
//            if(varianceRatio[ind] != 0){
                int sample = (int) (Math.ceil(radio * this.weigh_for_sample[ind] + this.variance_weight * Info.total_samples_num * varianceRatio[ind]));  //每层分配的采样数
//            }
                this.num_sample[ind] = sample;
                this.total_samples += sample;  // 采样的总个数
//            System.out.print( this.num_sample[ind] + " \t");
            }
        }

        System.out.println("this.total_samples: " + this.total_samples);
    }

    //TODO 二阶段分配样本：数量 + 均衡的方差
    //[改进] 最后1层 =1， 倒数第二层=n; 把最后两层脱离计算
    public void sampleAllocation_8(int num_features, double[][]newLevelMatrix, int k) {
        this.weigh_for_sample = new double[num_features];  //记录每层采样的按权重
        this.num_sample = new int[num_features];  // 记录每层采样几个（按权重）
        this.total_weights = 0.0; //total_weight 权重总数

        // 1）依次遍历每层，计算权重存入数组中weigh_for_sample[] 中 (最后一层不参与计算)
        int level = Math.max(2,k);
        for(int i=level; i<num_features-1; i++){
            double log = num_features * Math.log(num_features)- i * Math.log(i) - (num_features-i) * Math.log(num_features-i);  //最大值70
            // C(n,k)= n!/ k!(n-k)!  // log(m*k) = log m + log k; log(m/k) = log m - log k;  // log(n!) 约等于 nlog(n)
            this.weigh_for_sample[i] = log;
            this.total_weights += log;
//            System.out.print(log + "\t");
        }

        //【添加第二阶段】 计算每一层的方差
        double[] varianceArray = new double[num_features+1];  //每一层的方差
        double allVariance = 0;
        for(int len = 2; len < num_features; len ++) {  //line_ind 是对角线的标记
            double step_sum = 0;  //这条对对角线上的方差
            for(int ind=0; ind < num_features; ind++){
                step_sum += newLevelMatrix[len][ind];
            }
            double step_mean = step_sum / num_features; //计算当前层的均值
            //计算方差
            double variance = 0.0;
            for(int ind=0; ind < num_features; ind++){
                variance += Math.pow(newLevelMatrix[len][ind] - step_mean, 2);
            }
            double oneVariance = variance / num_features;
            varianceArray[len] = oneVariance;
            allVariance += oneVariance;
//            System.out.print(oneVariance + "\t");
        }
//        System.out.println();

        //给出方差权重的比例
        double[] varianceRatio = new double[Info.num_of_features];
        for(int i=level; i<num_features-1; i++){
            varianceRatio[i] = varianceArray[i] / allVariance;
        }

        // 2）按照1）中设置的权重 + 设定的采样数，实际分配采样个数
        //计算总共采样个数
        double total_samples_num = Math.max(Info.num_of_features, Info.total_samples_num*0.05);
        double radio = this.number_weight * total_samples_num / this.total_weights;  //定义扩充比例 (total_samples_num 总体采样数)
        for(int ind = level; ind <num_features-1; ind ++){
            int sample = (int) Math.ceil(radio * this.weigh_for_sample[ind] + this.variance_weight * total_samples_num * varianceRatio[ind]);  //每层分配的采样数
            this.num_sample[ind] = sample;
            this.total_samples += sample;  // 采样的总个数
//            System.out.print( this.num_sample[ind] + " \t");
        }
//        System.out.println();

        //3）给最后两层赋值
        this.num_sample[num_features-1] = num_features;   //最后1层
        this.total_samples += num_features;  // 采样的总个数
        System.out.println("this.total_samples: " + this.total_samples);
    }

    //TODO 二阶段分配样本：数量 + 均衡的方差  最后一层正常算
    //修改：当个数超过本层最大值时，选取本层的最大值
    public void sampleAllocation_10(int num_features, double[][]newLevelMatrix, int k) {
        this.weigh_for_sample = new double[num_features];  //记录每层采样的按权重
        this.num_sample = new int[num_features];  // 记录每层采样几个（按权重）
        this.total_weights = 0.0; //total_weight 权重总数
        int[] limits = combinations(num_features);

        // 1）依次遍历每层，计算权重存入数组中weigh_for_sample[] 中 (最后一层不参与计算)
        int level = Math.max(2,k);
        for(int i=level; i<num_features; i++){
            double log = num_features * Math.log(num_features)- i * Math.log(i) - (num_features-i) * Math.log(num_features-i);  //最大值70
            // C(n,k)= n!/ k!(n-k)!  // log(m*k) = log m + log k; log(m/k) = log m - log k;  // log(n!) 约等于 nlog(n)
            this.weigh_for_sample[i] = log;
            this.total_weights += log;
//            System.out.print(log + "\t");
        }

        //【添加第二阶段】 计算每一层的方差
        double[] varianceArray = new double[num_features];  //每一层的方差
        double allVariance = 0;
        for(int len = level; len < num_features; len ++) {  //line_ind 是对角线的标记
            double step_sum = 0;  //这条对对角线上的方差
            for(int ind=0; ind < num_features; ind++){
                step_sum += newLevelMatrix[len][ind]- newLevelMatrix[len-1][ind];  //LM[0] 存长度为1的coalitions, LM[N-1]存长度为n的全集
            }
            double step_mean = step_sum / num_features; //计算当前层的均值
            //计算方差
            double variance = 0.0;
            for(int ind=0; ind < num_features; ind++){
                variance += Math.abs((newLevelMatrix[len][ind] - newLevelMatrix[len-1][ind] - step_mean) / step_mean);
            }
            double oneVariance = variance / num_features;
            varianceArray[len] = oneVariance;
            allVariance += oneVariance;
//            System.out.print(oneVariance + "\t");
        }
//        System.out.println();

        //给出方差权重的比例
        double[] varianceRatio = new double[Info.num_of_features];
        for(int i=level; i<num_features; i++){
            varianceRatio[i] = varianceArray[i] / allVariance;
        }

        // 2）按照1）中设置的权重 + 设定的采样数，实际分配采样个数
        //计算总共采样个数
//        double radio = Info.number_weight * Info.total_samples_num / this.total_weights;  //定义扩充比例 (total_samples_num 总体采样数)
        double radio = this.number_weight * Info.total_samples_num / this.total_weights;  //定义扩充比例 (total_samples_num 总体采样数)
        for(int ind = level; ind <num_features; ind ++){
//            int sample = (int) (Math.ceil(radio * this.weigh_for_sample[ind] + Info.number_variance * Info.total_samples_num * varianceRatio[ind]));  //每层分配的采样数
            int sample = 1;
            if(varianceRatio[ind] != 0){
                sample = (int) (Math.ceil(radio * this.weigh_for_sample[ind] + this.variance_weight * Info.total_samples_num * varianceRatio[ind]));  //每层分配的采样数
                sample = Math.min(sample, limits[ind]);
            }
            this.num_sample[ind] = sample;
            this.total_samples += sample;  // 采样的总个数
//            System.out.print( this.num_sample[ind] + " \t");
        }

        while(this.total_samples < Info.total_samples_num){
            int restSamples = Info.total_samples_num - this.total_samples;
            if(num_features % 2 ==0){
                int sample = (int) Math.ceil(restSamples * varianceRatio[num_features/2]);
                sample = Math.min(sample, limits[num_features/2] - this.num_sample[num_features/2]);
                this.num_sample[num_features/2] += sample;
                this.total_samples += sample;  // 采样的总个数
                for(int step = 1; step <num_features/2; step ++){
                    int subSample_1 = 0; int subSample_2 = 0;
                    if(varianceRatio[num_features/2 - step] != 0) {
                        subSample_1 = (int) (Math.ceil(restSamples * varianceRatio[num_features / 2 - step]));
                        subSample_1 = Math.min(subSample_1, limits[num_features/2- step] - this.num_sample[num_features / 2 - step]);
                        this.num_sample[num_features / 2 - step] += subSample_1;
                    }
                    if(varianceRatio[num_features/2 + step] != 0){
                        subSample_2 = (int) (Math.ceil(restSamples * varianceRatio[num_features/2 + step]));
                        subSample_2 = Math.min(subSample_2, limits[num_features/2 + step] - this.num_sample[num_features / 2 + step]);
                        this.num_sample[num_features/2 + step] += subSample_2;
                    }
                    this.total_samples += subSample_1 + subSample_2;  // 采样的总个数
                }
            }
            else{
                int sample = (int) (Math.ceil(restSamples * varianceRatio[(num_features-1)/2]));
                sample = Math.min(sample, limits[(num_features-1)/2] - this.num_sample[(num_features-1)/2]);
                this.num_sample[(num_features-1)/2] += sample;
                this.total_samples += sample;  // 采样的总个数
                for(int step = 1; step <(num_features-1)/2; step ++){
                    int subSample_1 = 0;  int subSample_2 = 0;
                    if(varianceRatio[(num_features-1)/2 - step] != 0) {
                        subSample_1 = (int) (Math.ceil(restSamples * varianceRatio[(num_features-1) / 2 - step]));
                        subSample_1 = Math.min(subSample_1, limits[(num_features-1)/2- step] - this.num_sample[(num_features-1)/ 2 - step]);
                        this.num_sample[(num_features-1) / 2 - step] += subSample_1;
                    }
                    if(varianceRatio[(num_features-1)/2 + step] != 0){
                        subSample_2 = (int) (Math.ceil(restSamples * varianceRatio[(num_features-1)/2 + step]));
                        subSample_2 = Math.min(subSample_2, limits[(num_features-1)/2 + step] - this.num_sample[(num_features-1)/ 2 + step]);
                        this.num_sample[num_features/2 + step] += subSample_2;
                    }
                this.total_samples += subSample_1 + subSample_2;  // 采样的总个数
                }
            }
        }
        System.out.println("this.total_samples: " + this.total_samples);
    }


    private int[] combinations(int num_features) {
        int[] coef = new int[num_features];
        for (int i = 0; i < num_features; i++) {
            BigInteger bigInteger = CombinationCalculator(num_features, i);
            if (bigInteger.compareTo(BigInteger.valueOf(Integer.MAX_VALUE)) > 0) {
                coef[i] = Integer.MAX_VALUE;
            }
            else {
                coef[i] = bigInteger.intValue();
            }
        }
        return coef;
    }

    private BigInteger CombinationCalculator (int num_features, int s) {
        BigInteger numerator = BigInteger.ONE;
        BigInteger denominator = BigInteger.ONE;
        for (int i = 0; i < s; i++) {
            numerator = numerator.multiply(BigInteger.valueOf(num_features - i));
            denominator = denominator.multiply(BigInteger.valueOf(i + 1));
        }
        return numerator.divide(denominator);
    }

}
