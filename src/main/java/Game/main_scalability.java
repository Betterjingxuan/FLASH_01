package Game;

import AlgoVersion.*;
import Global.Info;

public class main_scalability
{
    public static void main( String[] args )
    {
        main_scalability alg = new main_scalability();
        alg.scaleTest();   //【Scalability test】
    }

       //TODO 【Scalability test】 可以变换feature的个数，
    private void scaleTest() {

        MC_algorithm mc = new MC_algorithm();
        mc.MCShap_scale(Info.is_gene_weight,Info.model_name);

        CC_algorithm cc = new CC_algorithm();
        cc.CC_scale(Info.is_gene_weight, Info.model_name);

        CCN_algorithm ccn = new CCN_algorithm();
        ccn.CCN_scale(Info.is_gene_weight, Info.model_name);

        S_SVARM ssvarm = new S_SVARM();
        ssvarm.SSVARM_scale(Info.is_gene_weight, Info.model_name);

        FLASH_algorithm flash = new FLASH_algorithm();
        flash.FLASH_scale(Info.is_gene_weight, Info.model_name);
    }

    private void otherBaseline(){
        MCN_algorithm mcnAlgorithm = new MCN_algorithm();
        mcnAlgorithm.MCN_scale(Info.is_gene_weight, Info.model_name);
    }

}

