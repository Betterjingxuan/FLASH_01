import config.Info;
import AlgoVersion.*;

public class main
{
    public static void main( String[] args )
    {
        main alg = new main();
        alg.evaluationTest();
    }
    private void evaluationTest() {

        FLASH_algorithm alg_1 = new FLASH_algorithm();
        alg_1.FLASH(Info.is_gene_weight, Info.model_name);

        CC_algorithm alg_2 = new CC_algorithm();
        alg_2.CC(Info.is_gene_weight, Info.model_name);  //CC-SIGMOD-2023

        CCN_algorithm alg_3 = new CCN_algorithm();
        alg_3.CCN(Info.is_gene_weight, Info.model_name);   //CCN-SIGMOD-2023

        S_SVARM alg_4 = new S_SVARM();
        alg_4.SSVARM(Info.is_gene_weight, Info.model_name); //S_SVARM-AAAI-2024

        MC_algorithm alg_5 = new MC_algorithm();
        alg_5.MC(Info.is_gene_weight,Info.model_name);   // Benchmark
    }

}

