package Global;

import Global.Info;
import structure.ShapMatrixEntry;
import structure.UtilityPair;

import java.io.*;

//TODO 负责读入和写文件
public class FileOpera {

    /* TODO 写文件的函数 */
    public void outputExUtilTable(String file_path, double[] benchmark){
        //要写入文件的信息
        File file = new File(file_path);
        //判断之前是否已存在这个数据集文件
        if(!file.exists()) {
            try {
                file.createNewFile();
//                OutputStreamWriter outputStream_Writer = new OutputStreamWriter(new FileOutputStream(file, true));
                BufferedWriter buf_write = new BufferedWriter(new FileWriter(file));

                //依次遍历数组中的元素，逗号分开
                for(double key : benchmark){
                    buf_write.write(key + ", ");
                }
                buf_write.flush();
                buf_write.close();
            }
            catch (IOException e) {
                e.printStackTrace();
            }
        }
        else{
            System.out.println("The file is exists, please check! ");
        }
    }

    /* TODO 写文件的函数 */
    public void outputExUtilTable_2(String file_path, ShapMatrixEntry[] benchmark){
        //要写入文件的信息
        File file = new File(file_path);
        //判断之前是否已存在这个数据集文件
        if(!file.exists()) {
            try {
                file.createNewFile();
//                OutputStreamWriter outputStream_Writer = new OutputStreamWriter(new FileOutputStream(file, true));
                BufferedWriter buf_write = new BufferedWriter(new FileWriter(file));

                //依次遍历数组中的元素，逗号分开
                for(ShapMatrixEntry key : benchmark){
                    buf_write.write(key.sum + ", ");
                }
                buf_write.flush();
                buf_write.close();
            }
            catch (IOException e) {
                e.printStackTrace();
            }
        }
        else{
            System.out.println("The file is exists, please check! ");
        }
    }

    /* TODO 读入文件的函数
     * @input file_path 文件存放的路径
     * @return buffer_reader 缓冲区中的内容
     * */
    public double[] file_read(String file_path, int num_features) {

        //返回一个double的数组
        double[] benchmark = new double[num_features];

        //定义一块缓冲区buffer_reader
        BufferedReader buffer_reader = null;
        File file = new File(file_path);

        try {
            //如果文件合理且存在，开始读入数据
            if(file.isFile() && file.exists()) {
                InputStreamReader instream_reader = new InputStreamReader(new FileInputStream(file));
                buffer_reader = new BufferedReader(instream_reader);
                String lineTex = null;  //当前读取到的内容 lineTex
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


    /*TODO 写文件到 .csv 文件中*/
    public void writeToCSV(UtilityPair[][] utility_matrix){
        // 如果该目录下不存在该文件，则文件会被创建到指定目录下。如果该目录有同名文件，那么该文件将被覆盖。
        String FilePath = "D:/DiskOfJingxuan/Code/PythonCode/ShapleyValueByDataMining/" + "utility_matrix.txt";
        File file = new File(FilePath);
        if (file.exists()) {
            try {
                //通过BufferedReader类创建一个使用默认大小输出缓冲区的缓冲字符输出流
                BufferedWriter writeText = new BufferedWriter(new FileWriter(FilePath, false));
                //调用write的方法将字符串写到流中
                for (UtilityPair[] itemRecord : utility_matrix) {  //外层循环：取出一层长度相等的coalitions 的记录 itemRecord
                    for(UtilityPair ele : itemRecord){  //内层循环：取出某层内的每个feature对应的记录
                        String text = "[" + ele.present + "("+ ele.present_num + ")" + "," + ele.absent + "("+ ele.absent_num + ")" +"]";
//                        writeText.append(text);
                        writeText.write(text);
                    }
                    writeText.newLine();    //换行
                    writeText.write("------------------------");
                    writeText.newLine();    //换行
                }
                writeText.flush();
                writeText.close();
            } catch (FileNotFoundException e) {
                System.out.println("没有找到指定文件");
            } catch (IOException e) {
                System.out.println("文件读写出错");
            }
        }
        else{
            System.out.println("没有找到指定文件");
        }
    }

}

