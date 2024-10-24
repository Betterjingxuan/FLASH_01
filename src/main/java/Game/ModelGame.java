package Game;

import Global.Info;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import okhttp3.*;

import java.io.IOException;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class ModelGame {
    public String gameName;

    public ModelGame(String game_name){
        this.gameName = game_name;
    }

    //TODO 定义 Health函数v(S)
    public double Health_value(ArrayList<Integer> subset, OkHttpClient client) {
        double[] input = new double[Info.num_of_features_health];  //作为输入的instance
        System.arraycopy(Info.instance_health_avg,0, input, 0, input.length);
        double[] sample=Info.instance_health;

        for (Integer ele : subset) {
//            input[ele] = Info.instance_iot_2[ele];
            input[ele] = sample[ele];
        }
        Gson gson = new Gson();
        Map<String, double[]> payload = new HashMap<>();
        payload.put("inputs", input);
        String json_data = gson.toJson(payload);

        // Define MediaType for JSON
        MediaType JSON = MediaType.get("application/json; charset=utf-8");

        // Create RequestBody
        RequestBody body = RequestBody.create(json_data, JSON);

        /**************** 9091-9095: probability, 9096-9105:class *****************/
        // Create POST request
        Request request = new Request.Builder()
                .url("http://pandax1.d2.comp.nus.edu.sg:9091/predict/")  // 9091-9094   //9092不用   //9106-probability
                .post(body)
                .build();
        Map<String, Object> result = new HashMap<>();
        try (Response response = client.newCall(request).execute()) {
            if (!response.isSuccessful())
                throw new IOException("Unexpected code " + response);

            // Handle the response
            String jsonResponse = response.body().string();
            Type type = new TypeToken<Map<String, Object>>() {
            }.getType();
            Map<String, Object> resultMap = gson.fromJson(jsonResponse, type);
            result=resultMap;
            response.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
//    if((double)result.get("prediction") == 0){
//      System.out.println(subset);
//    }
        return (double)result.get("prediction");
    }

    //TODO 定义 BankMarketing函数v(S)
    public double BankMarketing_value(ArrayList<Integer> subset, OkHttpClient client) {
        double[] input = new double[Info.num_of_features_bank];  //作为输入的instance
        System.arraycopy(Info.instance_bank_avg,0, input, 0, input.length);
        double[] sample=Info.instance_bank;

        for (Integer ele : subset) {
            input[ele] = sample[ele];
        }
        Gson gson = new Gson();
        Map<String, double[]> payload = new HashMap<>();
        payload.put("inputs", input);
        String json_data = gson.toJson(payload);

        // Define MediaType for JSON
        MediaType JSON = MediaType.get("application/json; charset=utf-8");

        // Create RequestBody
        RequestBody body = RequestBody.create(json_data, JSON);

        /**************** http://localhost:9109/predict/*****************/
        // Create POST request
        Request request = new Request.Builder()
                .url("http://pandax1.d2.comp.nus.edu.sg:9109/predict/")  // 9091-9094   //9092不用   //9106-probability
                .post(body)
                .build();
        Map<String, Object> result = new HashMap<>();
        try (Response response = client.newCall(request).execute()) {
            if (!response.isSuccessful())
                throw new IOException("Unexpected code " + response);

            // Handle the response
            String jsonResponse = response.body().string();
            Type type = new TypeToken<Map<String, Object>>() {
            }.getType();
            Map<String, Object> resultMap = gson.fromJson(jsonResponse, type);
            result=resultMap;
            response.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return (double)result.get("prediction");
    }

    //TODO 定义 Voting game函数v(S)
    public int value_voting(ArrayList<Integer> subset, double[] given_weights, double halfSum) {
        int weights_sum = 0;
        //求weights_sum
        for(int ele : subset){
            weights_sum += given_weights[ele];
        }
        if(weights_sum > halfSum){
            return 1;
        }
        else return 0;
    }

    //TODO 定义 Airport game函数v(S)
    public double value_airport(ArrayList<Integer> subset, double[] given_weights) {
        double maxValue = 0;
        for(int ele : subset){
            maxValue = Math.max(given_weights[ele], maxValue);
        }
        return maxValue;
    }

    //TODO 定义 shoes game函数v(S)
    public int value_shoes(ArrayList<Integer> subset){
        int left_shoes = 0;    //left shoes 的个数
        int right_shoes = 0;  //right shoes 的个数
        for(int ele : subset){
            if(ele < Info.num_of_features / 2){
                left_shoes ++;
            }
            else{
                right_shoes ++;
            }
        }
        return Math.min(left_shoes, right_shoes);  //返回值
    }
    public double value_modelPrediction(double[] model_instance_ave, ArrayList<Integer> subset) {
        double[] input = new double[Info.num_of_features];  //作为输入的instance
        //复制model_instance_ave 到input[]
        System.arraycopy(model_instance_ave, 0, input, 0, model_instance_ave.length);  //需要逐个复制，才不会改变值
        //替换成存在的model_instance
        for(Integer ele : subset){
            input[ele] = Info.model_instance[ele];
        }
        return 0.00019049644470214844 + Math.exp(-0.00000009069984307813187 * (Math.pow(46.0 - input[0], 2.0) + Math.pow(2.0 - input[1], 2.0) + Math.pow(3.0 - input[2], 2.0) + Math.pow(2.0 - input[3], 2.0) + Math.pow(2.0 - input[4], 2.0) + Math.pow(668.0 - input[5], 2.0) + Math.pow(1.0 - input[6], 2.0) + Math.pow(2.0 - input[7], 2.0) + Math.pow(3.0 - input[8], 2.0) + Math.pow(15.0 - input[9], 2.0) + Math.pow(5.0 - input[10], 2.0) + Math.pow(1263.0 - input[11], 2.0) + Math.pow(2.0 - input[12], 2.0) + Math.pow(-1.0 - input[13], 2.0) + Math.pow(0.0 - input[14], 2.0) + Math.pow(4.0 - input[15], 2.0))) + Math.exp(-0.00000009069984307813187 * (Math.pow(48.0 - input[0], 2.0) + Math.pow(2.0 - input[1], 2.0) + Math.pow(3.0 - input[2], 2.0) + Math.pow(1.0 - input[3], 2.0) + Math.pow(2.0 - input[4], 2.0) + Math.pow(559.0 - input[5], 2.0) + Math.pow(1.0 - input[6], 2.0) + Math.pow(2.0 - input[7], 2.0) + Math.pow(3.0 - input[8], 2.0) + Math.pow(15.0 - input[9], 2.0) + Math.pow(5.0 - input[10], 2.0) + Math.pow(1231.0 - input[11], 2.0) + Math.pow(2.0 - input[12], 2.0) + Math.pow(-1.0 - input[13], 2.0) + Math.pow(0.0 - input[14], 2.0) + Math.pow(4.0 - input[15], 2.0)));
    }

}



