package Game;

import Global.Info;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;
import java.lang.reflect.Type;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.ArrayList;


public class Health {

  public double Health_value(double[] instance_ave, ArrayList<Integer> subset, OkHttpClient client) {
    double[] input = new double[Info.num_of_features];  //作为输入的instance
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


}