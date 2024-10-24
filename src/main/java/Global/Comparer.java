package Global;
import structure.*;

public class Comparer {

    public double computeMaxError(ShapMatrixEntry[] shap_matrix, double[] exact, int num_of_features) {
        double error_max = 0;
        for(int i=0; i<num_of_features; i++){
            if(exact[i] != 0){
                error_max = Math.max(Math.abs((shap_matrix[i].sum - exact[i]) /exact[i]), error_max);
            }
        }
        return error_max;
    }

    public double computeAverageError(ShapMatrixEntry[] shap_matrix, double[] exact, int num_of_features) {
        double error_ave = 0;
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
        double error_ave = 0;
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

    public double computeMaxError(double[] shap_matrix, double[] exact) {
        double error_max = 0;
        for(int i=0; i<exact.length; i++){
            if(exact[i] != 0){
                error_max = Math.max(Math.abs((shap_matrix[i] - exact[i]) / exact[i]), error_max);
            }
        }
        return error_max;
    }


    // TODO: compute the average coefficient of variation
    public double computeACV(double[][] svValues, int featureNum) {

        double[] average = new double[featureNum];
        for(int run=0; run<svValues.length; run++){
            for(int fea=0; fea<featureNum; fea++){
                average[fea] += svValues[run][fea];
            }
        }

        for(int fea=0; fea<featureNum; fea++){
            average[fea] = average[fea] / svValues.length;
        }

        double acv_count = 0;
        for(int fea=0; fea<featureNum; fea++){
            double acv = 0;
            for(int run=0; run<svValues.length; run++){
                acv += Math.pow(svValues[run][fea] - average[fea], 2);
            }
            acv = Math.sqrt(acv / svValues.length) / Math.abs(average[fea]);
            acv_count += acv;
        }
        return acv_count / featureNum;
    }

    public double computeACV(ShapMatrixEntry[][] svValues, int featureNum) {

        double[] average = new double[featureNum];
        for(int run=0; run<svValues.length; run++){
            for(int fea=0; fea<featureNum; fea++){
                average[fea] += svValues[run][fea].sum;
            }
        }

        for(int fea=0; fea<featureNum; fea++){
            average[fea] = average[fea] / svValues.length;
        }

        double acv_count = 0;
        for(int fea=0; fea<featureNum; fea++){
            double acv = 0;
            for(int run=0; run<svValues.length; run++){
                acv += Math.pow(svValues[run][fea].sum - average[fea], 2);
            }
            acv = Math.sqrt(acv / svValues.length) / Math.abs(average[fea]);
            acv_count += acv;
        }
        return acv_count / featureNum;
    }


}
