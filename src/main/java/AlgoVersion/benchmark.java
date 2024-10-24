package AlgoVersion;

import Game.GameClass;
import Global.Info;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

public class benchmark {

    int num_features;  //the number of features
    double[] exact;   // the exact shapley value
    int num_samples;

    /* TODO [MC Algorithm]: 通过Monte Carlo sampling 计算 shapley value
       num_sample: 采样的数量; model : 当前进行的game */
    public void MCShap_benchmark(boolean gene_weight, String model){

        //1)初始化
        GameClass game = new GameClass();
        initialization(game, gene_weight, model);
        Random PermutationGene = new Random(Info.seed);
        long time_1 = System.currentTimeMillis();
        computeShapBySampling(game, this.num_features, this.num_samples, model, PermutationGene);   //实验版
        long time_2 = System.currentTimeMillis();

        // 5）输出时间
        System.out.println("MC_V4 time : " + (time_2 - time_1) * 0.001);  //+ "S"
    }

    //TODO 通过Monte Carlo random sampling 计算 shapley value
    // benchmark == scale 一模一样，只是benchmark() 版本会的打印进度
    private double[] computeShapBySampling(GameClass game, int num_features, int num_sample, String model, Random PermutationGene) {
        double[] shap_matrix = new double[num_features];  //大数组

        //进行若干次采样（num_sample：采样的次数）
        for(int r=0; r<num_sample; r++){
            //1）生成一个打乱的序列
            ArrayList<Integer> p = permutation(num_features, PermutationGene);

            //2）利用序列p, 求每个feature 的 marginal contribution
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
            System.out.println("finish: " + r + "  /  " + num_sample);
        }
        // 5) 求r次采样的平均值
        for(int i=0; i < shap_matrix.length; i++){
            shap_matrix[i] = shap_matrix[i] / num_sample;
            System.out.print(shap_matrix[i] + ",");
        }
        return shap_matrix;
    }

    private ArrayList<Integer> permutation(int numFeatures, Random PermutationGene) {
        //1) 初始化perm序列（initial 对应的位置是对应的值）
        ArrayList<Integer> perm = new ArrayList<>();
        for(int i=0; i<numFeatures; i++){
            perm.add(i);
        }

        // 生成一个新的种子
        int sequenceSeed = PermutationGene.nextInt();
        Random random = new Random(sequenceSeed);

        // 随机打乱序列
        Collections.shuffle(perm, random);
        return perm;
    }

    private void initialization(GameClass game, boolean gene_weight, String modelName) {
        game.gameInit(gene_weight, modelName);
        this.num_features = game.num_features; //the number of features
        this.exact = game.exact;   // the exact shapley value
//        this.given_weights = game.given_weights;
//        this.halfSum = game.halfSum;  //for Voting game
        this.num_samples = Info.total_samples_num;  //一条permutation 要predict (n+1)次
    }
}
