package Game;

import config.Info;
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

    public double Health_value(ArrayList<Integer> subset, OkHttpClient client) {
        double[] input = new double[Info.num_of_features_health];  //input instance
        System.arraycopy(Info.instance_health_avg,0, input, 0, input.length);
        double[] sample=Info.instance_health;

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

        // Create POST request
        Request request = new Request.Builder()
                .url("http://pandax1.d2.comp.nus.edu.sg:9091/predict/")
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

    public double BankMarketing_value(ArrayList<Integer> subset, OkHttpClient client) {
        double[] input = new double[Info.num_of_features_bank];  //input instance
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

        // Create POST request
        Request request = new Request.Builder()
                .url("http://pandax1.d2.comp.nus.edu.sg:9109/predict/")
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

    public int value_voting(ArrayList<Integer> subset, double[] given_weights, double halfSum) {
        int weights_sum = 0;
        for(int ele : subset){
            weights_sum += given_weights[ele];
        }
        if(weights_sum > halfSum){
            return 1;
        }
        else return 0;
    }

    public double value_airport(ArrayList<Integer> subset, double[] given_weights) {
        double maxValue = 0;
        for(int ele : subset){
            maxValue = Math.max(given_weights[ele], maxValue);
        }
        return maxValue;
    }


}



