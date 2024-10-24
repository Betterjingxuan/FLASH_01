package AlgoVersion;

import Game.GameClass;
import Global.Comparer;
import Global.Info;

import java.util.ArrayList;
import java.util.Random;

/*TODO subset不排序的airport*/
public class MC_algorithm {

    int num_features;  //the number of features
    double[] exact;   // the exact shapley value
    int num_samples;

    /* TODO [MC Algorithm]: 通过Monte Carlo sampling 计算 shapley value
       num_sample: 采样的数量; model : 当前进行的game */
    public void MC_Shap(boolean gene_weight, String model){

        //Initialization
        GameClass game = new GameClass();
        initialization(game, gene_weight, model);

        long ave_runtime = 0;
        double ave_error_max = 0;
        double ave_error_ave = 0;
        for(int i=0; i< Info.timesRepeat; i++) {
            Random PermutationGene = new Random(game.seedSet[i]);
            long time_1 = System.currentTimeMillis();
            double[] shap_matrix = computeShapBySampling_2(game, this.num_features, this.num_samples, model, PermutationGene);   //实验版
            long time_2 = System.currentTimeMillis();

            //4）estimate error
            Comparer comparator = new Comparer();
            double error_max = comparator.computeMaxError(shap_matrix, this.exact); //计算最大误差
            double error_ave = comparator.computeAverageError(shap_matrix, this.exact, this.num_features);  //计算平均误差

            ave_runtime += time_2 - time_1;
            ave_error_max += error_max;
            ave_error_ave += error_ave;
            System.out.println("run: " + i + " " + "error_ave: " + error_ave + " \t"  +  "error_max: " + error_max);
        }

        System.out.println(model + " Game:  " + "error_ave: " + ave_error_ave/Info.timesRepeat + " \t"  + "error_max: " + ave_error_max/Info.timesRepeat);
        System.out.println("MC_V4 time : " + (ave_runtime * 0.001)/ Info.timesRepeat);
    }

    public void MCShap_scale(boolean gene_weight, String model){

        //1)初始化
        GameClass game = new GameClass();
        initialization(game, gene_weight, model);
        Comparer comparator = new Comparer();

        long ave_runtime = 0;
        double ave_error_max = 0; //计算最大误差
        double ave_error_ave = 0;
        double[][] sv_values = new double[Info.timesRepeat][this.num_features];  //[runs][feature]
//        double ave_mse = 0;
        for(int i=0; i< Info.timesRepeat; i++) {
            Random PermutationGene = new Random(game.seedSet[i]);
            long time_1 = System.currentTimeMillis();
            double[] shap_matrix = computeShapBySampling_2(game, this.num_features, this.num_samples, model, PermutationGene);   //实验版
            long time_2 = System.currentTimeMillis();

            //4）计算误差
            double error_max = comparator.computeMaxError(shap_matrix, this.exact); //计算最大误差
            double error_ave = comparator.computeAverageError(shap_matrix, this.exact, this.num_features);  //计算平均误差
//            double mse = comparator.computeMSE(shap_matrix, this.exact, this.num_features);

            sv_values[i] = shap_matrix;
            ave_runtime += time_2 - time_1;
            ave_error_max += error_max;
            ave_error_ave += error_ave;
//            ave_mse += mse;
//            System.out.println("run: " + i + " " + "error_ave: " + error_ave + " \t"  +  "error_max: " + error_max + " \t"  +  "mse: " + mse);
            System.out.println("run: " + i + " " + "error_ave: " + error_ave + " \t"  +  "error_max: " + error_max);
        }
        double acv = comparator.computeACV(sv_values, this.num_features);

        // 5）输出时间
//        System.out.println(model + " Game:  " + "error_ave: " + ave_error_ave/Info.timesRepeat + " \t"  + "error_max: " + ave_error_max/Info.timesRepeat + " \t"  +  "mse: " + ave_mse/Info.timesRepeat);
        System.out.println(model + " Game:  " + "error_ave: " + ave_error_ave/Info.timesRepeat + " \t"  + "error_max: " + ave_error_max/Info.timesRepeat);
        System.out.println("average cv :" + acv);
        System.out.println("MC_V4 time : " + (ave_runtime * 0.001)/ Info.timesRepeat);  //+ "S"
    }

    //TODO 通过Monte Carlo random sampling 计算 shapley value
    private double[] computeShapBySampling_2(GameClass game, int num_features, int num_sample, String model, Random PermutationGene) {
        double[] shap_matrix = new double[num_features];  //大数组

        //进行若干次采样（num_sample：采样的次数）
        for(int r=0; r<num_sample; r++){
            //1）生成一个打乱的序列
            ArrayList<Integer> p = permutation_3(PermutationGene);  //20240731
//            ArrayList<Integer> p = permutation(num_features, PermutationGene);

            //2）利用序列p, 求每个feature 的 marginal contribution
            /* P = [ABC]: A - 0; AB -A; ABC-AB
             *  P = [BAC]: B - 0; AB -B; ABC-AB */
            for(int ind = 0; ind <p.size(); ind ++){ // ele 对应一个特征
                int ele = p.get(ind);  //ele就是对应的特征i
                //*subset_1 第1个特征子集
                ArrayList<Integer> subset_1 = new ArrayList<>();
                double value_1 = 0;
                ArrayList<Integer> subset_2 = new ArrayList<>();
                double value_2 = 0;
                for(int i =0; i<ind; i++){
                    subset_1.add(p.get(i));
                }
                //*subset_2 第二个特征子集
                for(int i =0; i<ind; i++){
                    subset_2.add(p.get(i));
                }
                subset_2.add(ele);
                //4)分别求函数值
                value_1 = game.gameValue(model, subset_1);
                value_2 = game.gameValue(model, subset_2);
                shap_matrix[ele] += value_2 - value_1;
            }
        }
        // 5) 求r次采样的平均值
        for(int i=0; i < shap_matrix.length; i++){
            shap_matrix[i] = shap_matrix[i] / num_sample;
        }
        return shap_matrix;
    }

    //TODO 通过Monte Carlo random sampling 计算 shapley value
    private double[] computeShapBySampling_scale(GameClass game, int num_features, int num_sample, String model) {
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
//                Collections.sort(subset_1);
//                Collections.sort(subset_2);

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
        Random rand = new Random();
        /* Fisher-Yates洗牌算法（Knuth洗牌算法）来对数组进行打乱顺序。
        该算法的思想是从数组末尾开始，依次将当前位置的元素与前面随机位置的元素交换，直到数组的第一个位置。
        这样可以保证每个元素被随机置换的概率相同。*/
        for(int i=numFeatures-1; i>0; i--){
            int j = rand.nextInt(i+1); //从[0, i+1)中随机选取一个int
            int temp = perm[i];  // 交换位置
            perm[i] = perm[j];
            perm[j] = temp;
        }
        return perm;
    }

    private ArrayList<Integer> permutation_3(Random PermutationGene) {
        //1) 初始化perm序列（initial 对应的位置是对应的值）
        ArrayList<Integer> perm = new ArrayList<>();
        for(int i=0; i<this.num_features; i++){
            perm.add(i);
        }

        // 生成一个新的种子
        int sequenceSeed = PermutationGene.nextInt();
        Random random = new Random(sequenceSeed);
        double step = Math.min(Math.max(1.0, 4.6 - Math.log10(Info.setting)), 4);
//        System.out.println("step:" + step);

        // 随机打乱序列
        for(double i = 0; i<this.num_features; i += step){  // (int i=numFeatures/2; i>1; i--)  (int i=0; i<numFeatures/2; i++)
//            int index = (int) Math.round(i);
            int index = (int) Math.min(this.num_features-1, Math.round(i));
//            System.out.println(index);
            int j = random.nextInt(this.num_features); //从[0, i+1)中随机选取一个int
            int temp = perm.get(j);  // 交换位置
            perm.set(j, perm.get(index));
            perm.set(index, temp);
        }
//        System.out.println(perm.toString());
        return perm;
    }

    private void initialization(GameClass game, boolean gene_weight, String modelName) {
        game.gameInit(gene_weight, modelName);
        this.num_features = game.num_features; //the number of features
        this.exact = game.exact;   // the exact shapley value
//        this.given_weights = game.given_weights;
//        this.halfSum = game.halfSum;  //for Voting game
        this.num_samples = Info.total_samples_num / (game.num_features + 1);  //一条permutation 要predict (n+1)次
    }
}
