package Global;

import config.Info;
import structure.UtilityPair;
import java.io.*;

public class FileOpera {

    public double[] file_read(String file_path, int num_features) {

        double[] benchmark = new double[num_features];
        BufferedReader buffer_reader = null;
        File file = new File(file_path);

        try {
            if(file.isFile() && file.exists()) {
                InputStreamReader instream_reader = new InputStreamReader(new FileInputStream(file));
                buffer_reader = new BufferedReader(instream_reader);
                String lineTex = null;
                while ((lineTex = buffer_reader.readLine()) != null) {
                    String[] entrySet = lineTex.split(",");
                    for(int i=0; i < entrySet.length; i++){
                        benchmark[i] = Double.parseDouble(String.valueOf(entrySet[i]));
                    }
                }
            }
            else {
                System.out.println("The file does not exist");
            }
        }
        catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        catch (IOException e) {
            throw new RuntimeException(e);
        }
        return benchmark;
    }

    public void writeToCSV(UtilityPair[][] utility_matrix){
        // If the file doesn't exist, it will be created in the specified directory; if it exists, it will be overwritten.
        String FilePath = Info.generate_data_output;
        File file = new File(FilePath);
        if (file.exists()) {
            try {
                BufferedWriter writeText = new BufferedWriter(new FileWriter(FilePath, false));
                for (UtilityPair[] itemRecord : utility_matrix) {
                    for(UtilityPair ele : itemRecord){
                        String text = "[" + ele.present + "("+ ele.present_num + ")" + "," + ele.absent + "("+ ele.absent_num + ")" +"]";
//                        writeText.append(text);
                        writeText.write(text);
                    }
                    writeText.newLine();
                    writeText.write("------------------------");
                    writeText.newLine();
                }
                writeText.flush();
                writeText.close();
            } catch (FileNotFoundException e) {
                System.out.println("File not found");
            } catch (IOException e) {
                System.out.println("File read/write error");
            }
        }
        else{
            System.out.println("File not found");
        }
    }

}

