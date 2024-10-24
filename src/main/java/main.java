import Global.Info;
import AlgoVersion.*;

public class main
{
    public static void main( String[] args )
    {
        main alg = new main();
        alg.evaluationTest();
//        alg.otherBaseline();
    }
    private void evaluationTest() {

        MC_algorithm mc = new MC_algorithm();
        mc.MC_Shap(Info.is_gene_weight,Info.model_name);   // Benchmark

        CC_algorithm cc = new CC_algorithm();
        cc.CC_Shap(Info.is_gene_weight, Info.model_name);  //CC-SIGMOD-2023

        CCN_algorithm ccn = new CCN_algorithm();
        ccn.CCN_Shap(Info.is_gene_weight, Info.model_name);   //CCN-SIGMOD-2023

        S_SVARM ssvarm = new S_SVARM();
        ssvarm.SSVARM_Shap(Info.is_gene_weight, Info.model_name); //S_SVARM-AAAI-2024
//
        FLASH_algorithm flash = new FLASH_algorithm();
        flash.FLASH_Shap(Info.is_gene_weight, Info.model_name);
    }

    private void otherBaseline(){
        MCN_algorithm mcnAlgorithm = new MCN_algorithm();
        mcnAlgorithm.MCN_scale(Info.is_gene_weight, Info.model_name);

    }

}

