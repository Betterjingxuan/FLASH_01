package Global;

import config.Info;
import java.io.InputStream;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class NumpyReader {
    private static final int MAGIC_LEN = 8;
    private static final Map<String, String> VERSION2HTYPE = new HashMap<String, String>(){
        {
            // short type
            put("10", "<H");
            // int type
            put("20", "<I");
        }
    };

    public double[] readIntArray(InputStream inputStream) throws Exception {
        String version = NumpyReader.readVersion(inputStream);
        String header = NumpyReader.readArrayHeader(inputStream, version);
        int itemLength = NumpyReader.itemLength(header);
        String shape = NumpyReader.arrayShape(header);
        double[] array = new double[Info.num_of_features];
        byte[] item = new byte[itemLength];
        for (int i = 0; i < Info.num_of_features; i++) {
            inputStream.read(item);
            array[i] = NumpyReader.bytes2Int(item);
        }
        return array;
    }


    public double[][] readDoubleArray(InputStream inputStream) throws Exception {
        String version = NumpyReader.readVersion(inputStream);
        String header = NumpyReader.readArrayHeader(inputStream, version);
        int itemLength = NumpyReader.itemLength(header);
        String shape = NumpyReader.arrayShape(header);
        int row = Integer.valueOf(shape.split(",")[0]);
        int col = Integer.valueOf(shape.split(",")[1]);
        double[][] array = new double[row][col];
        byte[] item = new byte[itemLength];
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                inputStream.read(item);
                array[i][j] = NumpyReader.bytes2Double(item);
            }
        }
        return array;
    }


    public static String readVersion(InputStream inputStream) throws Exception {
        byte[] magicStr = new byte[MAGIC_LEN];
        inputStream.read(magicStr);
        return magicStr[MAGIC_LEN-2] + "" + magicStr[MAGIC_LEN-1];
    }


    public static String readArrayHeader(InputStream inputStream, String version) throws Exception {
        String headerLengthType = VERSION2HTYPE.get(version);
        if (headerLengthType == null) {
            throw new Exception("Invalid version " + version);
        }
        byte[] headerLengthInfo = new byte[headerLengthType.getBytes().length];
        inputStream.read(headerLengthInfo);
        int headerLength = getHeaderLength(headerLengthInfo, headerLengthType);
        byte[] headerBytes = new byte[headerLength];
        inputStream.read(headerBytes);
        String headers = new String(headerBytes);
        return headers;
    }

    public static int getHeaderLength(byte[] headerLengthInfo, String headerLengthType) throws Exception {
        int hlen = 0;
        if ("<H".equals(headerLengthType)) {
            hlen = 8;   // for short type
        } else if ("<I".equals(headerLengthType)){
            hlen = 16; // for int type
        } else {
            throw new Exception("Unsupported type.");
        }
        return bytes2Int(Arrays.copyOf(headerLengthInfo, hlen));
    }


    public static int itemLength(String header) throws Exception {
        String[] headerArray = header.split(",");
        String dataType = headerArray[0];
        if (dataType.contains("'<f")) {
            int index = dataType.indexOf("<f");
            return Integer.valueOf(dataType.substring(index+2, index+3));
        } else if (dataType.contains("'<i")) {
            int index = dataType.indexOf("<i");
            return Integer.valueOf(dataType.substring(index+2, index+3));
        }
        throw new Exception("Unsupported type.");
    }


    public static String arrayShape(String header) {
        String[] headerArray = header.split(",", 3);
        String shape = headerArray[2];
        return shape.substring(shape.indexOf("(")+1, shape.indexOf(")")).replace(" ", "");
    }

    public static double bytes2Double(byte[] arr) {
        long value = 0;
        for (int i = 0; i < 8; i++) {
            value |= ((long) (arr[i] & 0xff)) << (8 * i);
        }
        return Double.longBitsToDouble(value);
    }

    public static int bytes2Int(byte[] src) {
        int value;
        value = (int) ((src[0] & 0xFF)
                | ((src[1] & 0xFF)<<8)
                | ((src[2] & 0xFF)<<16)
                | ((src[3] & 0xFF)<<24));
        return value;
    }



}

