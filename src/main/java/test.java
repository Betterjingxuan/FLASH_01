import Game.Health;
import Game.ModelGame;
import Global.Info;
import com.google.gson.internal.bind.util.ISO8601Utils;
import okhttp3.OkHttpClient;
import structure.ShapMatrixEntry;

import java.util.ArrayList;
import java.util.Random;

public class test {
    public static void main( String[] args ){
//        test_1();

        int n = 39;
        int feature = 33;
        int runs = 10;
        int sampleSetting= 100;  //每一层来50个
        Random random = new Random();
        test_2(n, feature, runs, sampleSetting, random);  //分层采样
//        test_3(n, feature, runs, sampleSetting, random);  //全部一起采样，随机长度
//        test_4(n, feature, runs, sampleSetting, random);  //从对应的集合中取值，计算模拟分布
    }

    //TODO 全部一起采样，随机长度
    private static void test_3(int n, int feature, int runs, int limit, Random random) {
        ArrayList<Integer> subSet = new ArrayList<>();
        subSet.add(feature);
        OkHttpClient client = new OkHttpClient();
        Health health = new Health();
        double value = health.Health_value(Info.instance_health, subSet, client) - health.Health_value(Info.instance_health, new ArrayList< Integer>(), client);
//        System.out.println(subSet + " " + value);
        //---------------------------------------
        double total = value;
        double error = 0;
            for(int r = 0; r < runs; r++) {
                int count_1=0; int count_2 = 0;
                for (int count = 0; count < limit * n; count++) {
                    int length = random.nextInt(n);
                    ArrayList<Integer> subSet_1 = new ArrayList<>();
                    while (subSet_1.size() < length) {
                        int fea = random.nextInt(n);
                        if (fea != feature) {
                            subSet_1.add(fea);
                        }
                    }
                    ArrayList<Integer> subSet_2 = new ArrayList<>(subSet_1);
                    subSet_2.add(feature);
                    double value_2 = health.Health_value(Info.instance_health, subSet_2, client);
                    double value_1 = health.Health_value(Info.instance_health, subSet_1, client);
//                System.out.println(subSet_1.toString() + ": "+  value_2 + " - " +  value_1 + " = " + (value_2 - value_1));
                    total += value_2 - value_1;
                    if(value_2 - value_1 > 0.005){
//                        System.out.println(subSet_1.toString() + ": "+  value_2 + " - " +  value_1 + " = " + (value_2 - value_1));
                        count_1 ++;
                    }
                    else if(value_2 - value_1 < -0.005){
                        count_2 ++;
                    }
                }
                error += Math.abs((total/(limit*n+1) + 4.997727298366311E-4)/ -4.997727298366311E-4);
                System.out.println("total" + ": " + (total / (limit * n + 1)) + ";  err: " + Math.abs((total/(limit*n+1) + 4.997727298366311E-4)/ -4.997727298366311E-4));
                System.out.println(" > 0.005: " + count_1 + ";  <-0.005 :" + count_2);
            }
//            total += sum/limit;

        System.out.println("error " + ": " + error / runs);
    }

    //TODO 分层采样 每层均匀计算
    private static void test_2(int n, int feature, int runs, int limit, Random random) {
        ArrayList<Integer> subSet = new ArrayList<>();
        subSet.add(feature);
        OkHttpClient client = new OkHttpClient();
        Health health = new Health();
        double error = 0;
        double value = health.Health_value(Info.instance_health, subSet, client) - health.Health_value(Info.instance_health, new ArrayList< Integer>(), client);
//        System.out.println(subSet + " " + value);
        //---------------------------------------
        for(int r = 0; r<runs; r++) {
            double total = value;
            int count_1 = 0;  int count_2 = 0;
            for (int length = 1; length < n; length++) {  //每一层
                double sum = 0;
                for (int count = 0; count < limit; count++) {
                    ArrayList<Integer> subSet_1 = new ArrayList<>();
                    while (subSet_1.size() < length) {
                        int fea = random.nextInt(n);
                        if (fea != feature) {
                            subSet_1.add(fea);
                        }
                    }
                    ArrayList<Integer> subSet_2 = new ArrayList<>(subSet_1);
                    subSet_2.add(feature);
                    double value_2 = health.Health_value(Info.instance_health, subSet_2, client);
                    double value_1 = health.Health_value(Info.instance_health, subSet_1, client);
//                System.out.println(subSet_1.toString() + ": "+  value_2 + " - " +  value_1 + " = " + (value_2 - value_1));
//                if(value_2 - value_1 > 0.005){
////                    System.out.println(subSet_1.toString() + ": "+  value_2 + " - " +  value_1 + " = " + (value_2 - value_1));
//                    count_1 ++;
//                }
//                else if(value_2 - value_1 < -0.005){
//                    count_2 ++;
//                }
                    sum += value_2 - value_1;
                }
//            System.out.println("sum" + length + ": " + sum/limit + ",  count: " + count_small);
                total += sum / limit;
            }
//            System.out.println(" > 0.005: " + count_1 + ";  <-0.005 :" + count_2);
            error += Math.abs((total / n +4.997727298366311E-4) / -4.997727298366311E-4);
            System.out.println("total " + ": " + total / n  +"; err: " + Math.abs((total / n + 4.997727298366311E-4) / -4.997727298366311E-4));
        }
        System.out.println("error: " + error/runs);
    }

    //TODO 从对应的集合中取值，计算模拟分布
    private static void test_4(int n, int feature, int runs, int limit, Random random) {
        ArrayList<Integer> subSet = new ArrayList<>();
        subSet.add(feature);
        ArrayList<Double> set_1 = new ArrayList<>();
        ArrayList<Double> set_2 = new ArrayList<>();
        OkHttpClient client = new OkHttpClient();
        Health health = new Health();
        double value = health.Health_value(Info.instance_health, subSet, client) - health.Health_value(Info.instance_health, new ArrayList< Integer>(), client);
//        System.out.println(subSet + " " + value);
        //---------------------------------------
        double total = value;
        double sum = 0;
        int count_small = 0;
        double error = 0;
        for(int r = 0; r < runs; r++) {
            int count_1=0; int count_2 = 0;
            for (int count = 0; count < limit * n; count++) {
                int length = random.nextInt(n);
                ArrayList<Integer> subSet_1 = new ArrayList<>();
                while (subSet_1.size() < length) {
                    int fea = random.nextInt(n);
                    if (fea != feature) {
                        subSet_1.add(fea);
                    }
                }
                ArrayList<Integer> subSet_2 = new ArrayList<>(subSet_1);
                subSet_2.add(feature);
                double value_2 = health.Health_value(Info.instance_health, subSet_2, client);
                double value_1 = health.Health_value(Info.instance_health, subSet_1, client);
//                System.out.println(subSet_1.toString() + ": "+  value_2 + " - " +  value_1 + " = " + (value_2 - value_1));
//                if(value_2 - value_1 > 0){
//                    System.out.println(subSet_1.toString() + ": "+  value_2 + " - " +  value_1 + " = " + (value_2 - value_1));
//                    count_small ++;
//                }
                total += value_2 - value_1;
                if(value_2 - value_1 > 0.005){
//                        System.out.println(subSet_1.toString() + ": "+  value_2 + " - " +  value_1 + " = " + (value_2 - value_1));
                    count_1 ++;
                    set_1.add(value_2 - value_1);
                }
                else if(value_2 - value_1 < -0.005){
                    count_2 ++;
                    set_2.add(value_2 - value_1);
                }
            }
            error += Math.abs((total/(limit*n+1) + 4.997727298366311E-4)/ -4.997727298366311E-4);
            System.out.println("total" + ": " + (total / (limit * n + 1)) + ";  err: " + Math.abs((total/(limit*n+1) + 4.997727298366311E-4)/ -4.997727298366311E-4));
            System.out.println(" > 0.005: " + count_1 + ";  <-0.005 :" + count_2);


            double w1 = 1.0 * set_1.size() / (set_1.size() + set_2.size());
            double w2 = 1.0 * set_2.size()  / (set_1.size() + set_2.size());
            double samples = 0.1 * limit * n;
            double ave_1 = 0;  double ave_2=0;
            for(int c1 =0; c1<w1*samples;c1++){
                int index = random.nextInt(set_1.size());
                ave_1 += set_1.get(index);
            }
            ave_1 = ave_1 / (w1*samples);
            for(int c2 =0; c2<w2*samples;c2++){
                int index = random.nextInt(set_2.size());
                ave_2 += set_2.get(index);
            }
            ave_2 = ave_2 / (w2*samples);
            System.out.println("ave_1: " + ave_1 + "; ave_2: "+ ave_2);
            double sam = ave_1 * w1 + ave_2 * w2;
            System.out.println("sum: " + sam);

        }
//            total += sum/limit;

//        System.out.println("error " + ": " + error / runs);
    }

    private static void test_1() {
        double[] shap_matrix = {-0.014611071787583522,-0.08328619518150122,-0.003246676797668139,0.04919729792536833,
                -0.018453077389261663,-0.01091678268634356,-0.008123940191207787,-0.0022037232246918557,3.039035468529432E-4,
                0.4904677525926859,0.1338849453589855,-0.0188254102682456,0.007959137360254923,3.5008864525036933E-4,
                0.007935855633173233,0.011014129870977158,-0.005540817975997925,0.04351088786736513,-0.002380846402583978,
                -0.005646795034408569,0.02627546664996025,0.026018875531661205,-0.001682653641089415,0.007997037508548835,
                0.027741461992263794,0.0028560069891122673,-0.05238335369489132,-0.01070033854398972,-0.04225082771900373,
                0.005611715790552971,-0.0030545775706951436,-7.690848448337653E-4,-0.016837351979353488,-0.01231508711591745,
                0.01829457502716627,-0.02999442357283372,0.007400869845579832,-0.029458919492287513,0.008145102896751502};

        double[] exact = {-0.014732393569556567, -0.07629048324072113, -0.006387789447242633, 0.022390938173640424,
                -0.003129081310943151, -0.008296038449121019, 0.01793108050259642, 0.0012370176121664163, -0.002186118692575166,
                0.28644243636272176, 0.17130706371632523, -0.016320565294760923, 0.02051089037121154, 0.018682278897892684,
                0.008038969158123318, 0.014523624373542574, 0.008564082347355688, 0.046203438399457494, 0.030484623263102886,
                0.0024481860800168644, 0.0026920723725564013, 0.02666117363820951, -0.006981838861017082, 0.013990571785137917,
                0.06623300700910771, -0.005588510203165695, -0.06552317377009358, 0.008439235016899423, -0.04978723311018533,
                0.007551880376967482, -0.0036597300886319806, 0.004694384932076224, -0.011130217132736476, -2.1837694233713242E-4,
                0.009561208863128931, -0.012432925909490158, -0.00445350377666167, 0.0032777903923239464, -0.0024648179271998696};
        double err_max = computeMaxError(shap_matrix, exact, 39);
        double err_ave = computeAverageError(shap_matrix, exact, 39);
        System.out.println(err_max + "    " + err_ave);
    }


    public static double computeMaxError(double[] shap_matrix, double[] exact, int num_of_features) {
        double error_max = 0;  //误差总量，最后除以特征数
        for(int i=0; i<num_of_features; i++){
            if(exact[i] != 0){
                error_max = Math.max(Math.abs((shap_matrix[i] - exact[i]) /exact[i]), error_max);
                System.out.print(i + "-" + Math.abs((shap_matrix[i] - exact[i]) /exact[i]) + "\t");
            }
        }
        return error_max;
    }

    //TODO 计算平均误差 (Voting game & Airport game)
    public static double computeAverageError(double[] shap_matrix, double[] exact, int num_of_features) {
        double error_ave = 0;  //误差总量，最后除以特征数
        int total_features = num_of_features;
        for(int i=0; i<num_of_features; i++){
            if(exact[i] !=0) {
                error_ave += Math.abs((shap_matrix[i] - exact[i]) / exact[i]);
            }
            else{
                total_features --;
            }
        }
        error_ave = error_ave / total_features;
        return error_ave;
    }
}
