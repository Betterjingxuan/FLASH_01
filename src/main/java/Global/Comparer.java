package Global;
import structure.*;

// 比较器
public class Comparer {


    //TODO 计算最大误差 (Voting game & Airport game)
    public double computeMaxError(ShapMatrixEntry[] shap_matrix, double[] exact, int num_of_features) {
        double error_max = 0;  //误差总量，最后除以特征数
        for(int i=0; i<num_of_features; i++){
            if(exact[i] != 0){
                error_max = Math.max(Math.abs((shap_matrix[i].sum - exact[i]) /exact[i]), error_max);
//                System.out.print(i + "-" + Math.abs((shap_matrix[i].sum - exact[i]) /exact[i]) + "\t");
            }
        }
//        System.out.println();
        return error_max;
    }

    //TODO 计算平均误差 (Voting game & Airport game)
    public double computeAverageError(ShapMatrixEntry[] shap_matrix, double[] exact, int num_of_features) {
        double error_ave = 0;  //误差总量，最后除以特征数
        int total_features = num_of_features;
        for(int i=0; i<num_of_features; i++){
            if(exact[i] !=0) {
                error_ave += Math.abs((shap_matrix[i].sum - exact[i]) / exact[i]);
            }
            else{
                total_features --;
            }
        }
        error_ave = error_ave / total_features;
        return error_ave;
    }

    public double computeAverageError(double[] shap_matrix, double[] exact, int num_of_features) {
        double error_ave = 0;  //误差总量，最后除以特征数
        int total_features = num_of_features;
        for(int i=0; i<exact.length; i++){
            if(exact[i] !=0){
                error_ave += Math.abs((shap_matrix[i] - exact[i]) / exact[i]);
            }
            else{
                total_features --;
            }
        }
        error_ave = error_ave / total_features;
        return error_ave;
    }


    //TODO 跳过第二个feature
    public double computeMaxError_2(double[] shap_matrix, double[] exact) {
        double error_max = 0;  //误差总量，最后除以特征数
        for(int i=0; i<2; i++){
            if(exact[i] != 0){
                error_max = Math.max(Math.abs((shap_matrix[i] - exact[i]) / exact[i]), error_max);
//                System.out.println(i + "-" + shap_matrix[i] + "\t" + exact[i] + "\t" + Math.abs((shap_matrix[i] - exact[i]) / exact[i]) + "\t");
            }
        }
        for(int i=3; i<exact.length; i++){
            if(exact[i] != 0){
                error_max = Math.max(Math.abs((shap_matrix[i] - exact[i]) / exact[i]), error_max);
//                System.out.println(i + "-" + shap_matrix[i] + "\t" + exact[i] + "\t" + Math.abs((shap_matrix[i] - exact[i]) / exact[i]) + "\t");
            }
        }
        return error_max;
    }

    //TODO 计算最大误差 (Voting game & Airport game)
    public double computeMaxError(double[] shap_matrix, double[] exact) {
        double error_max = 0;  //误差总量，最后除以特征数
        for(int i=0; i<exact.length; i++){
            if(exact[i] != 0){
                error_max = Math.max(Math.abs((shap_matrix[i] - exact[i]) / exact[i]), error_max);
//                System.out.println(i + "-" + shap_matrix[i] + "\t" + exact[i] + "\t" + Math.abs((shap_matrix[i] - exact[i]) / exact[i]) + "\t");
            }
        }
        return error_max;
    }

    public double computeMSE(ShapMatrixEntry[] shap_matrix, double[] exact, int num_of_features) {
        double mse_ave = 0;  //误差总量，最后除以特征数
        int total_features = num_of_features;
        for(int i=0; i<exact.length; i++){
            if(exact[i] !=0){
                mse_ave += Math.pow(shap_matrix[i].sum  - exact[i], 2);
            }
            else{
                total_features --;
            }
        }
        mse_ave = mse_ave / total_features;
        return mse_ave;
    }

    // TODO: compute the average coefficient of variation   /* svValues[runs][feature] */
    public double computeACV(double[][] svValues, int featureNum) {

        //1) 先计算均值
        double[] average = new double[featureNum];
        for(int run=0; run<svValues.length; run++){  //每轮计算得到的该feature的sv.
            for(int fea=0; fea<featureNum; fea++){  //遍历每个feature
                average[fea] += svValues[run][fea];
            }
        }

        //2) 数组中每个sv.都算一次均值
        for(int fea=0; fea<featureNum; fea++){
            average[fea] = average[fea] / svValues.length;
        }

        //3)计算ACV
        double acv_count = 0;
        for(int fea=0; fea<featureNum; fea++){
            double acv = 0;
            for(int run=0; run<svValues.length; run++){  //在每次计算中
                acv += Math.pow(svValues[run][fea] - average[fea], 2);  //每个标准差
            }
            acv = Math.sqrt(acv / svValues.length) / Math.abs(average[fea]);
            acv_count += acv;
        }
        return acv_count / featureNum;
    }

    public double computeACV(ShapMatrixEntry[][] svValues, int featureNum) {

        //1) 先计算均值
        double[] average = new double[featureNum];
        for(int run=0; run<svValues.length; run++){  //每轮计算得到的该feature的sv.
            for(int fea=0; fea<featureNum; fea++){  //遍历每个feature
                average[fea] += svValues[run][fea].sum;
            }
        }

        //2) 数组中每个sv.都算一次均值
        for(int fea=0; fea<featureNum; fea++){
            average[fea] = average[fea] / svValues.length;
        }

        //3)计算ACV
        double acv_count = 0;
        for(int fea=0; fea<featureNum; fea++){
            double acv = 0;
            for(int run=0; run<svValues.length; run++){  //在每次计算中
                acv += Math.pow(svValues[run][fea].sum - average[fea], 2);  //每个标准差
            }
            acv = Math.sqrt(acv / svValues.length) / Math.abs(average[fea]);
            acv_count += acv;
        }
        return acv_count / featureNum;
    }


}
