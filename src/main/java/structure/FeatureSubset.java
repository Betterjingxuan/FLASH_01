package structure;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Objects;

public class FeatureSubset {
    public ArrayList<Integer> name;
    public double value_fun;

    public HashSet<Integer> hashName;

    public FeatureSubset(ArrayList<Integer> subset, double value){
        this.name = new ArrayList<>(subset);
        this.value_fun = value;
    }

    public FeatureSubset(HashSet<Integer> hashName, double value){
        this.hashName = new HashSet<>(hashName);
        this.value_fun = value;
    }

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

}
