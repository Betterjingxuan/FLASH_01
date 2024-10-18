package Global;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Objects;

public class FeatureSubset {
    public ArrayList<Integer> name;
    public double value_fun;

    public HashSet<Integer> hashName;

    //[版本1]：ArrayList
    public FeatureSubset(ArrayList<Integer> subset, double value){
        this.name = new ArrayList<>(subset);
        this.value_fun = value;
    }

    //[版本2]：HashSet 可去重
    public FeatureSubset(HashSet<Integer> hashName, double value){
        this.hashName = new HashSet<>(hashName);
        this.value_fun = value;
    }

    /*Name是HashSet 可行*/
    @Override
    public int hashCode() {
        return Objects.hash(hashName);
    }

    @Override
    public boolean equals(Object obj){
        if (this == obj)
            return true;
        if (obj == null || getClass() != obj.getClass())
            return false;
        FeatureSubset other = (FeatureSubset) obj;
        return Objects.equals(hashName, other.hashName);
    }


    /*Name是ArrayList 不可行*/
//    @Override
//    public int hashCode() {
//        HashSet<Integer> hashName = new HashSet<>(this.name);
//        return Objects.hash(hashName);
//    }
//
//    @Override
//    public boolean equals(Object obj){
//        if (this == obj)
//            return true;
//        if (obj == null || getClass() != obj.getClass())
//            return false;
//        FeatureSubset other = (FeatureSubset) obj;
//        return Objects.equals(name, other.name);
//    }

}
