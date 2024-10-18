package SeedVersion;

import Game.Airport_scale;
import Game.VotingGame_scale;
import Global.Info;

public class main_5_copy
{
    public static void main( String[] args ) {

        //1）【evaluate test】性能测试-
        main_5_copy alg = new main_5_copy();

        //3）【scale test】 只用于测试时间
        alg.scaleTest();
    }

    //TODO 【scale test】 变换feature的个数，测试时间
    private void scaleTest() {

        //1）随机生成n个特征和对应权重
//        Airport_scale alg = new Airport_scale();
//        alg.featureGenerate();  //

        //2）直接从.npy文件读入
        MonteCarlo_double mc_alg = new MonteCarlo_double(); //子集不需要排序
//        mc_alg.MCShap_scale(Info.is_gene_weight,Info.model_name);

        MCN mcn = new MCN();
//        mcn.MCN_scale(Info.is_gene_weight, Info.model_name);

        CC_algorithm_double cc = new CC_algorithm_double();
//        cc.CC_scale(Info.is_gene_weight, Info.model_name);  //false: 表示使用默认的weight

        CCN_algorithm_double_3 ccn_3 = new CCN_algorithm_double_3();  //这是标准的版本【20240808】
        ccn_3.CCN_scale(Info.is_gene_weight, Info.model_name);   //每轮count重新赋值，去掉向上取整 arr_m[]

        S_SVARM alo_svarm = new S_SVARM();
//        alo_svarm.SSVARM_algorithm(Info.is_gene_weight, Info.model_name);

        PSA_time PSA = new PSA_time();
        PSA.model_game_nog_scale(Info.is_gene_weight, Info.model_name);

    }

}

