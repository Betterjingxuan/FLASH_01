package AlgoVersion;

import Game.GameClass;
import Global.Comparer;
import Global.Info;
import structure.ShapMatrixEntry;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

public class CC_algorithm {

    int num_features;  // the number of features
    double[] given_weights;
    double[] exact;  // the exact shapley value
    double halfSum;  //for Voting game
    int num_samples;

    /* TODO [CC Algorithm]: 通过complementary contributions 计算 shapley value
       num_sample: 采样的数量
       model : 当前进行的game */
    public void CC_Shap(boolean gene_weight, String model){
        GameClass game = new GameClass();
        initialization(game, gene_weight, model);

        long ave_runtime = 0;
        double ave_error_max = 0; //计算最大误差
        double ave_error_ave = 0;
//        double ave_mse = 0;
        for(int i=0; i< Info.timesRepeat; i++) {
            Random PermutationGene = new Random(game.seedSet[i]);
            //3）计算shapley value
            long time_1 = System.currentTimeMillis();
            ShapMatrixEntry[] shap_matrix = computeShapBySampling(game, this.num_samples, model, PermutationGene);
            long time_2 = System.currentTimeMillis();

            //4）计算误差
            Comparer comparator = new Comparer();
            double error_ave = comparator.computeAverageError(shap_matrix, this.exact, this.num_features);  //计算平均误差
            double error_max = comparator.computeMaxError(shap_matrix, this.exact, this.num_features); //计算最大误差

            ave_runtime += time_2 - time_1;
            ave_error_max += error_max;
            ave_error_ave += error_ave;
            System.out.println("run: " + i + " " + "error_ave: " + error_ave + " \t"  +  "error_max: " + error_max);
//            for(ShapMatrixEntry ele: shap_matrix){
//                System.out.print(ele.sum + "\t");
//            }
//            System.out.println();
        }
        System.out.println(model + " Game:  " + "error_ave: " + ave_error_ave/Info.timesRepeat + " \t"  +  "error_max: " + ave_error_max/Info.timesRepeat);
        // 5）输出时间
        System.out.println("CC time : " + (ave_runtime * 0.001)/ Info.timesRepeat );  //+ "S"
//        HashMap<Integer, double[]> s = new HashMap<>();
//        for(int i=0; i<num_features; i++){
//            s.put(i, new double[2]);
//        }

    }

    public void CC_scale(boolean gene_weight, String model){
        ShapMatrixEntry[][] sv_values = new ShapMatrixEntry[Info.timesRepeat][];  //[runs][feature]
        GameClass game = new GameClass();
        initialization_eva(game, gene_weight, model, sv_values);
        Comparer comparator = new Comparer();

        long ave_runtime = 0;
        double ave_error_max = 0; //计算最大误差
        double ave_error_ave = 0;
        for(int i=0; i< Info.timesRepeat; i++) {
            Random PermutationGene = new Random(game.seedSet[i]);
            //3）计算shapley value
            long time_1 = System.currentTimeMillis();
            ShapMatrixEntry[] shap_matrix = computeShapBySampling(game, this.num_samples, model, PermutationGene);
            long time_2 = System.currentTimeMillis();
            sv_values[i] = shap_matrix;

            //4）计算误差
            double error_ave = comparator.computeAverageError(shap_matrix, this.exact, this.num_features);  //计算平均误差
            double error_max = comparator.computeMaxError(shap_matrix, this.exact, this.num_features); //计算最大误差

            ave_runtime += time_2 - time_1;
            ave_error_max += error_max;
            ave_error_ave += error_ave;
            System.out.println("run: " + i + " " + "error_ave: " + error_ave + " \t"  +  "error_max: " + error_max);
        }
        double acv = comparator.computeACV(sv_values, this.num_features);
        System.out.println(model + " Game:  " + "error_ave: " + ave_error_ave/Info.timesRepeat + " \t"  +  "error_max: " + ave_error_max/Info.timesRepeat + " \t");
        System.out.println("average cv :" + acv);
        // 5）输出时间
        System.out.println("CC time : " + (ave_runtime * 0.001)/ Info.timesRepeat );  //+ "S"
//        HashMap<Integer, double[]> s = new HashMap<>();
//        for(int i=0; i<num_features; i++){
//            s.put(i, new double[2]);
//        }

    }

    //TODO 通过CC algorithm 计算 shapley value
    private ShapMatrixEntry[] computeShapBySampling(GameClass game, int num_sample, String model, Random random) {

        ShapMatrixEntry[][] shap_matrix = new ShapMatrixEntry[this.num_features+1][]; // 需要换成带计数器的大数组
        for(int i=0; i<shap_matrix.length; i++){   //外层循环：features对应每个长度的collations (n个feature，n+1层)
            shap_matrix[i] = new ShapMatrixEntry[this.num_features];
            for(int j=0; j<this.num_features; j++) {  //内层循坏：每个features
                shap_matrix[i][j] = new ShapMatrixEntry();
            }
        }

        //进行若干次采样（num_sample：采样的次数）
        for(int r=0; r<num_sample; r+=2){
            //1）生成一个打乱的序列 & 一个随机数
            ArrayList<Integer> p = permutation(this.num_features, random);
            int random_i = random.nextInt(this.num_features);  //随机生成一个数,表示feature subset的长度
            //2）利用序列p, 求每个feature 的 marginal contribution
            ArrayList<Integer> subset_1 = new ArrayList<>();   //*subset_1 第一个特征子集
            double value_1 = 0;
            ArrayList<Integer> subset_2 = new ArrayList<>();   //*subset_2 第二个特征子集
            double value_2 = 0;

            /* S = [ABC];   P = N-S = [DEF] */
            for(int ind = 0; ind <random_i; ind ++) { // ele 对应一个特征
                subset_1.add(p.get(ind));
            }
            for(int ind = random_i; ind <p.size(); ind ++) { // ele 对应一个特征
                subset_2.add(p.get(ind));
            }

            //*将两个list重新排列
//            Collections.sort(subset_1);
//            Collections.sort(subset_2);

            //4)分别求函数值
            value_1 = game.gameValue(model, subset_1);
            value_2 = game.gameValue(model, subset_2);
            //5）求complementary contribution
            double comp_contrib = value_1 - value_2;

            //6)存入对应的集合中
            for(Integer ele : subset_1){
                shap_matrix[random_i][ele].sum += comp_contrib;
                shap_matrix[random_i][ele].count ++;
            }
            for(Integer ele : subset_2){
                shap_matrix[this.num_features - random_i][ele].sum -= comp_contrib;
//                shap_matrix[num_features - random_i][ele].sum += value_2 - value_1;
                shap_matrix[this.num_features - random_i][ele].count ++;
            }

        }
        // 5) 求shapley value的平均值，对于每个特征&每个长度求均值
        ShapMatrixEntry[] resultShap = new ShapMatrixEntry[this.num_features];  //这是最后返回的大数组
        for(int i=0; i<this.num_features; i++) {  //内层循环：features对应每个长度的collations
            resultShap[i] = new ShapMatrixEntry();
        }

        for(ShapMatrixEntry[] featureSet : shap_matrix){   //外层循环：features对应每个长度的collations
//            ShapMatrixEntry[] featureSet = [fea];
            for(int fea=0; fea<featureSet.length ; fea++){  //内层循坏：每个features
                ShapMatrixEntry entry = featureSet[fea];   //相同长度的一层中，每个特征对应的sv.
                if(entry.count != 0){
                    entry.sum = entry.sum / entry.count;
                    resultShap[fea].sum += entry.sum;  //对于每个长度求和
                    resultShap[fea].count ++;
                }
            }
        }

        //6）求一个整体的均值
        for(ShapMatrixEntry entry : resultShap){
            if(entry.count != 0){
                entry.sum = entry.sum / entry.count;
            }
            else{
                entry.sum =0;
            }
        }

        return resultShap;

    }

    //TODO 通过Monte Carlo random sampling 计算 shapley value
    private double[] computeShapBySampling_scale(GameClass game, int num_features, int num_sample, String model, double[] given_weights) {
        double[] shap_matrix = new double[num_features];  //大数组

        //进行若干次采样（num_sample：采样的次数）
        for(int r=0; r<num_sample; r++){
            //1）生成一个打乱的序列
            int[] p = permutation(num_features);
//            int[] tmpExample = new int[num_features]; //构建一个tmpExample
//            tmpExample = Arrays.copyOf(example, num_features);

            //2）利用序列p, 求每个feature 的 marginal contribution
            /* P = [ABC]: A - 0; AB -A; ABC-AB
             *  P = [BAC]: B - 0; AB -B; ABC-AB */
            for(int ind = 0; ind <p.length; ind ++){ // ele 对应一个特征
                int ele = p[ind];  //ele就是对应的特征i
                //*subset_1 第一个特征子集
                ArrayList<Integer> subset_1 = new ArrayList<>();
                for(int i =0; i<=ind; i++){
                    subset_1.add(p[i]);
                }
                double value_1 = 0;
                //*subset_2 第二个特征子集
                ArrayList<Integer> subset_2 = new ArrayList<>(subset_1);
                subset_2.remove(ind);
                double value_2 = 0;

                //*将两个list重新排列
                Collections.sort(subset_1);
                Collections.sort(subset_2);

                //4)分别求函数值
                value_1 = game.gameValue(model, subset_1);
                value_2 = game.gameValue(model, subset_2);
                shap_matrix[ele] += value_1 - value_2;
            }
//            System.out.println("finish: " + r + "  /  " + num_sample);
        }
        // 5) 求r次采样的平均值
        for(int i=0; i < shap_matrix.length; i++){
            shap_matrix[i] = shap_matrix[i] / num_sample;
        }
        return shap_matrix;
    }

    //TODO 通过Monte Carlo random sampling 计算 shapley value
    // benchmark == scale 一模一样，只是benchmark() 版本会的打印进度
    private double[] computeShapBySampling_benchmark(GameClass game, int num_features, int num_sample, String model, double[] given_weights) {
        double[] shap_matrix = new double[num_features];  //大数组

        //进行若干次采样（num_sample：采样的次数）
        for(int r=0; r<num_sample; r++){
            //1）生成一个打乱的序列
            int[] p = permutation(num_features);
//            int[] tmpExample = new int[num_features]; //构建一个tmpExample
//            tmpExample = Arrays.copyOf(example, num_features);

            //2）利用序列p, 求每个feature 的 marginal contribution
            /* P = [ABC]: A - 0; AB -A; ABC-AB
             *  P = [BAC]: B - 0; AB -B; ABC-AB */
            for(int ind = 0; ind <p.length; ind ++){ // ele 对应一个特征
                int ele = p[ind];  //ele就是对应的特征i
                //*subset_1 第一个特征子集
                ArrayList<Integer> subset_1 = new ArrayList<>();
                for(int i =0; i<=ind; i++){
                    subset_1.add(p[i]);
                }
                double value_1 = 0;
                //*subset_2 第二个特征子集
                ArrayList<Integer> subset_2 = new ArrayList<>(subset_1);
                subset_2.remove(ind);
                double value_2 = 0;

                //*将两个list重新排列
                Collections.sort(subset_1);
                Collections.sort(subset_2);

                //4)分别求函数值
                value_1 = game.gameValue(model, subset_1);
                value_2 = game.gameValue(model, subset_2);
                shap_matrix[ele] += value_1 - value_2;
            }
            System.out.println("finish: " + r + "  /  " + num_sample);
        }
        // 5) 求r次采样的平均值
        for(int i=0; i < shap_matrix.length; i++){
            shap_matrix[i] = shap_matrix[i] / num_sample;
        }
        return shap_matrix;
    }

    //TODO 生成一个长度为n随机序列，序列中的值是[0， n-1]
    private int[] permutation(int numFeatures) {
        //1) 初始化perm序列（initial 对应的位置是对应的值）
        int[] perm = new int[numFeatures];
        for(int i=0; i<numFeatures; i++){
            perm[i] = i;
        }

        //2）打乱perm序列
        Random rand = new Random();  //System.currentTimeMillis()
        /* Fisher-Yates洗牌算法（Knuth洗牌算法）来对数组进行打乱顺序。
        该算法的思想是从数组末尾开始，依次将当前位置的元素与前面随机位置的元素交换，直到数组的第一个位置。
        这样可以保证每个元素被随机置换的概率相同。*/
        for(int i=numFeatures-1; i>1; i--){
            int j = rand.nextInt(i+1); //从[0, i+1)中随机选取一个int
            int temp = perm[i];  // 交换位置
            perm[i] = perm[j];
            perm[j] = temp;
        }
        return perm;
    }

    private ArrayList<Integer> permutation(int numFeatures, Random PermutationGene) {
        //1) 初始化perm序列（initial 对应的位置是对应的值）
        ArrayList<Integer> perm = new ArrayList<>();
        for(int i=0; i<numFeatures; i++){
            perm.add(i);
        }

        // 生成一个新的种子
        int sequenceSeed = PermutationGene.nextInt();
//        int sequenceSeed = PermutationGene.nextInt();
        Random random = new Random(sequenceSeed);

        // 使用 Collections.shuffle 打乱序列
        Collections.shuffle(perm, random);

        return perm;
    }

    private ArrayList<Integer> permutation_2(int numFeatures, Random PermutationGene) {
        //1) 初始化perm序列（initial 对应的位置是对应的值）
        ArrayList<Integer> perm = new ArrayList<>();
        for(int i=0; i<numFeatures; i++){
            perm.add(i);
        }

        // 生成一个新的种子
        int sequenceSeed = PermutationGene.nextInt();
        Random random = new Random(sequenceSeed);

        // 随机打乱序列
        for(int i=0; i<numFeatures; i++){  // (int i=numFeatures/2; i>1; i--)  (int i=0; i<numFeatures/2; i++)
            int j = random.nextInt(numFeatures); //从[0, i+1)中随机选取一个int
            int temp = perm.get(j);  // 交换位置
            perm.set(j, perm.get(i));
            perm.set(i, temp);
        }

        return perm;
    }

    private void initialization(GameClass game, boolean gene_weight, String modelName) {
        game.gameInit(gene_weight, modelName);
        this.num_features = game.num_features; //the number of features
        this.exact = game.exact;   // the exact shapley value
        this.given_weights = game.given_weights;
        this.halfSum = game.halfSum;  //for Voting game
        this.num_samples = Info.total_samples_num;
    }

    //TODO game.seedSet();
    private void initialization_eva(GameClass game, boolean gene_weight, String modelName, ShapMatrixEntry[][] sv_values) {
        game.gameInit(gene_weight, modelName);
        this.num_features = game.num_features; //the number of features
        this.exact = game.exact;   // the exact shapley value
        this.given_weights = game.given_weights;
        this.halfSum = game.halfSum;  //for Voting game
        this.num_samples = Info.total_samples_num;
        for(ShapMatrixEntry[] ele : sv_values){
            ele = new ShapMatrixEntry[this.num_features];
        }
    }

}
