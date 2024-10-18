package Global;

import java.util.*;

public class Grid {
    public double min;  //网格的最大值
    public double max; //网格的最小值
    public int count;  //sampling 计数

    public ArrayList<FeatureSubset>[] grids_row;  //表示这是一行的网格

    //TODO 【版本1】构建这一层的网格（按坐标存储，非均匀）
    // [注意] ：
    // 传入的参数 elementSet ：存储这一层生成的所有element, 数量是 element的数量
    // 返回值 grid ：将element整理到网格中
    //number ： 记录的数据的数量
    public void ConstructGrid (FeatureSubset[] elementSet){
        //1)初始化网格
//        @SuppressWarnings("unchecked")  //通过 @SuppressWarnings("unchecked") 注解，告诉编译器你知道这里可能涉及类型不安全的操作
        this.grids_row = new ArrayList[Info.num_of_grids];
        for(int i = 0; i<this.grids_row.length; i++){
            this.grids_row[i] = new ArrayList<>();
        }
        //2) 计算最大差距，计算每个网格的间隔
        double global_gap = max - min;
        assert Info.num_of_grids > 1;  //网格数量需要大于1
        double grid_gap = global_gap / (Info.num_of_grids - 1);

        //3）遍历每个元素，放入grid中 (改成迭代器遍历)
        // 因为elementSet设置的最大最大估计，会有空的,且空值都在后面 （所以改成迭代器迭代会更好）
        // 但是没有数组类型的迭代器，需要把数组转为list，再用 list.iterator()  （这样不够方便）
        //        List<FeatureSubset> list = Arrays.asList(elementSet);
        //        Iterator<FeatureSubset> iterator = list.iterator();
        // 所以还是用for-each循环 + 条件判断来做
        for(FeatureSubset ele : elementSet){
            if(ele == null){
                break;  // 如果为空直接跳出循环
            }
            int index = (int) (ele.value_fun / grid_gap);
            this.grids_row[index].add(ele);
        }
    }

    public void ConstructGrid (ArrayList<FeatureSubset> elementSet){
        //1)初始化网格
//        @SuppressWarnings("unchecked")  //通过 @SuppressWarnings("unchecked") 注解，告诉编译器你知道这里可能涉及类型不安全的操作
        this.grids_row = new ArrayList[Info.num_of_grids];
        for(int i = 0; i<this.grids_row.length; i++){
            this.grids_row[i] = new ArrayList<>();
        }
        //2) 计算最大差距，计算每个网格的间隔
        double global_gap = max - min;
        assert Info.num_of_grids > 1;  //网格数量需要大于1
        double grid_gap = global_gap / (Info.num_of_grids - 1);

        //3）遍历每个元素，放入grid中 (改成迭代器遍历)
        // 因为elementSet设置的最大最大估计，会有空的,且空值都在后面 （所以改成迭代器迭代会更好）
        // 但是没有数组类型的迭代器，需要把数组转为list，再用 list.iterator()  （这样不够方便）
        //        List<FeatureSubset> list = Arrays.asList(elementSet);
        //        Iterator<FeatureSubset> iterator = list.iterator();
        // 所以还是用for-each循环 + 条件判断来做
        for(FeatureSubset ele : elementSet){
            if(ele == null){
                break;  // 如果为空直接跳出循环
            }
            int index = (int) (ele.value_fun / grid_gap);
            this.grids_row[index].add(ele);
        }
    }

    //TODO 【版本2】把元素均匀划分到网格中(传入FeatureSubset[])
    // 问题：return Double.compare(a.value_fun, b.value_fun); 空指针异常
    // 本来elementSet 设计时就按照最大数组容量设置，所以会有空值
    public void ConstructGrid_2 (FeatureSubset[] elementSet){
        //1)初始化网格
//        @SuppressWarnings("unchecked")  //通过 @SuppressWarnings("unchecked") 注解，告诉编译器你知道这里可能涉及类型不安全的操作
        this.grids_row = new ArrayList[Info.num_of_grids];
        for(int i = 0; i<this.grids_row.length; i++){
            this.grids_row[i] = new ArrayList<>();
        }
        //2) 计算每个网格存储元素的个数
        int eleOneGrid = Math.round((float) elementSet.length / Info.num_of_grids);

        //3) 对元素进行排序
        // 使用Arrays.sort()方法对数组进行排序,
        Arrays.sort(elementSet, new Comparator<FeatureSubset>() {
            @Override  //使用自定义排序器对数组进行排序
            public int compare(FeatureSubset a, FeatureSubset b) {
                // 按照 value 属性降序排序
//                return a.value_fun- b.value_fun;  //不接受返回类型为 double
//                return Double.compare(a.value_fun, b.value_fun);  //【空指针异常】
                //【解决】处理空指针异常
                if (a == null && b == null) {
                    return 0;
                }
                else if (a == null) {
                    return 1;  // 如果a为null，认为b较大
                }
                else if (b == null) {
                    return -1;  // 如果b为null，认为a较大
                }
                else {
                    // 按照 value_fun 属性升序排序
                    return Double.compare(a.value_fun, b.value_fun);
                }
            }
        });

        //4) 遍历数组，依次放入网格
        /*处理流程：1、依次向网格中填入特征子集；*/
        int index = 0;  //所有元素的坐标
        //Step1: 遍历网格，依次放入
        for(int gridInd=0; gridInd<Info.num_of_grids; gridInd++){
            //程序的终止条件： ① 结合遍历结束； ② 集合中遇到了空值
            if(index >= elementSet.length || elementSet[index] == null){
                break;  // 直接跳出循环
            }
            else{
                //Step2: 在每个网格中放入若干个
                for(int count =0; count < eleOneGrid; count ++){  // eleOneGrid 单个网格的计数
                    if(index >=  elementSet.length){
                        break;
                    }
                    else{
                        this.grids_row[gridInd].add(elementSet[index]);
                        index ++;
                    }
                }
            }
        }

        //5）特殊情况：若网格数量大于list的大小，四舍五入的值为0；就每个网格放1一个，什么时候用完了就结束
        if(eleOneGrid == 0){
            int ind = 0;
            for(FeatureSubset ele : elementSet){
                if(ele == null){
                    break;  // 如果为空直接跳出循环
                }
                assert ind < Info.num_of_grids;
                this.grids_row[ind].add(ele); //网格多，元素少
                ind ++;
            }
        }
    }

    //TODO 【版本2】把元素均匀划分到网格中(传入ArrayList)
    // 问题：return Double.compare(a.value_fun, b.value_fun); 空指针异常
    // 本来elementSet 设计时就按照最大数组容量设置，所以会有空值
    public void ConstructGrid_2 (ArrayList<FeatureSubset> elementSet){
        //1)初始化网格
//        @SuppressWarnings("unchecked")  //通过 @SuppressWarnings("unchecked") 注解，告诉编译器你知道这里可能涉及类型不安全的操作
        this.grids_row = new ArrayList[Info.num_of_grids];
        for(int i = 0; i<this.grids_row.length; i++){
            this.grids_row[i] = new ArrayList<>();
        }
        //2) 计算每个网格存储元素的个数
        int eleOneGrid = Math.round((float) elementSet.size() / Info.num_of_grids);

        //3) 对元素进行排序
        // 使用Arrays.sort()方法对数组进行排序,
        elementSet.sort(new Comparator<FeatureSubset>() {
            @Override  //使用自定义排序器对数组进行排序
            public int compare(FeatureSubset a, FeatureSubset b) {
                // 按照 value 属性降序排序
//                return a.value_fun- b.value_fun;  //不接受返回类型为 double
//                return Double.compare(a.value_fun, b.value_fun);  //【空指针异常】
                //【解决】处理空指针异常
                if (a == null && b == null) {
                    return 0;
                } else if (a == null) {
                    return 1;  // 如果a为null，认为b较大
                } else if (b == null) {
                    return -1;  // 如果b为null，认为a较大
                } else {
                    // 按照 value_fun 属性升序排序
                    return Double.compare(a.value_fun, b.value_fun);
                }
            }
        });

        //4) 遍历数组，依次放入网格
        /*处理流程：1、依次向网格中填入特征子集；*/
        int index = 0;  //所有元素的坐标
        //Step1: 遍历网格，依次放入
        for(int gridInd=0; gridInd<Info.num_of_grids; gridInd++){
            //程序的终止条件： ① 结合遍历结束； ② 集合中遇到了空值
            if(index >= elementSet.size() || elementSet.get(index) == null){
                break;  // 直接跳出循环
            }
            else{
                //Step2: 在每个网格中放入若干个
                for(int count =0; count < eleOneGrid; count ++){  // eleOneGrid 单个网格的计数
                    if(index >=  elementSet.size()){
                        break;
                    }
                    else{
                        this.grids_row[gridInd].add(elementSet.get(index));
                        index ++;
                    }
                }
            }
        }

        //5）特殊情况：若网格数量大于list的大小，四舍五入的值为0；就每个网格放1一个，什么时候用完了就结束
        if(eleOneGrid == 0){
            int ind = 0;
            for(FeatureSubset ele : elementSet){
                if(ele == null){
                    break;  // 如果为空直接跳出循环
                }
                assert ind < Info.num_of_grids;
                this.grids_row[ind].add(ele); //网格多，元素少
                ind ++;
            }
        }
    }

    //TODO 【版本3】网格不排序，直接存储
    public void ConstructGrid_3 (ArrayList<FeatureSubset> elementSet){
        //1)初始化网格
//        @SuppressWarnings("unchecked")  //通过 @SuppressWarnings("unchecked") 注解，告诉编译器你知道这里可能涉及类型不安全的操作
        this.grids_row = new ArrayList[Info.num_of_grids];
        for(int i = 0; i<this.grids_row.length; i++){
            this.grids_row[i] = new ArrayList<>();
        }
        //2) 计算每个网格存储元素的个数
        int eleOneGrid = Math.round((float) elementSet.size() / Info.num_of_grids);

        //3) 对元素进行排序
        // 使用Arrays.sort()方法对数组进行排序,
//        elementSet.sort(new Comparator<FeatureSubset>() {
//            @Override  //使用自定义排序器对数组进行排序
//            public int compare(FeatureSubset a, FeatureSubset b) {
//                // 按照 value 属性降序排序
////                return a.value_fun- b.value_fun;  //不接受返回类型为 double
////                return Double.compare(a.value_fun, b.value_fun);  //【空指针异常】
//                //【解决】处理空指针异常
//                if (a == null && b == null) {
//                    return 0;
//                } else if (a == null) {
//                    return 1;  // 如果a为null，认为b较大
//                } else if (b == null) {
//                    return -1;  // 如果b为null，认为a较大
//                } else {
//                    // 按照 value_fun 属性升序排序
//                    return Double.compare(a.value_fun, b.value_fun);
//                }
//            }
//        });

        //4) 遍历数组，依次放入网格
        /*处理流程：1、依次向网格中填入特征子集；*/
        int index = 0;  //所有元素的坐标
        //Step1: 遍历网格，依次放入
        for(int gridInd=0; gridInd<Info.num_of_grids; gridInd++){
            //程序的终止条件： ① 结合遍历结束； ② 集合中遇到了空值
            if(index >= elementSet.size()){
                break;  // 直接跳出循环
            }
            else{
                //Step2: 在每个网格中放入若干个
                for(int count =0; count < eleOneGrid; count ++){  // eleOneGrid 单个网格的计数
                    if(index >=  elementSet.size()){
                        break;
                    }
                    else{
                        this.grids_row[gridInd].add(elementSet.get(index));
                        index ++;
                    }
                }
            }
        }

        //5）特殊情况：若网格数量大于list的大小，四舍五入的值为0；就每个网格放1一个，什么时候用完了就结束
        if(eleOneGrid == 0){
            int ind = 0;
            for(FeatureSubset ele : elementSet){
                if(ele == null){
                    break;  // 如果为空直接跳出循环
                }
                assert ind < Info.num_of_grids;
                this.grids_row[ind].add(ele); //网格多，元素少
                ind ++;
            }
        }
    }


    //TODO 【版本4】HashSet + 不排序直接存储 (慢笨笨版)
    public void ConstructGrid_4(HashSet<FeatureSubset> elementSet){
        //1)初始化网格
//        @SuppressWarnings("unchecked")  //通过 @SuppressWarnings("unchecked") 注解，告诉编译器你知道这里可能涉及类型不安全的操作
        this.grids_row = new ArrayList[Info.num_of_grids];
        for(int i = 0; i<this.grids_row.length; i++){
            this.grids_row[i] = new ArrayList<>();
        }
        //2) 计算每个网格存储元素的个数
        int eleOneGrid = Math.round((float) elementSet.size() / Info.num_of_grids);

        //3) 对元素进行排序(省略不排序了)

        //4) 遍历数组，依次放入网格
        /*处理流程：1、依次向网格中填入特征子集；*/
        int gridInd = 0; //网格的坐标
        int count_ele = 0; //网格内存储元素的计数器
        for(FeatureSubset ele : elementSet){
            if(count_ele < eleOneGrid){
                this.grids_row[gridInd].add(ele);
                count_ele ++;
            }
            else{
                count_ele = 0;
                gridInd ++;
            }
        }
        // [方法2] 使用迭代器遍历 HashSet，将元素分配到数组数组中 （也很慢）
//        Iterator<FeatureSubset> iterator = elementSet.iterator();
//        while (iterator.hasNext()) {
//            FeatureSubset element = iterator.next();
//            this.grids_row[gridInd].add(element);
//            count_ele++;
//            if(count_ele > eleOneGrid ){
//                count_ele = 0;
//                gridInd ++;
//            }
//        }

        //5）特殊情况：若网格数量大于list的大小，四舍五入的值为0；就每个网格放1一个，什么时候用完了就结束
        if(eleOneGrid == 0){
            int ind = 0;
            for(FeatureSubset ele : elementSet){
                if(ele == null){
                    break;  // 如果为空直接跳出循环
                }
                assert ind < Info.num_of_grids;
                this.grids_row[ind].add(ele); //网格多，元素少
                ind ++;
            }
        }

    }

    //TODO 【版本3】网格不排序，直接存储 (笨笨版+1)
    public void ConstructGrid_5 (HashSet<FeatureSubset> elementSet){
        //1)初始化网格
//        @SuppressWarnings("unchecked")  //通过 @SuppressWarnings("unchecked") 注解，告诉编译器你知道这里可能涉及类型不安全的操作
        this.grids_row = new ArrayList[Info.num_of_grids];
        Iterator<FeatureSubset> iterator = elementSet.iterator();
        for(int i = 0; i<this.grids_row.length; i++){
            this.grids_row[i] = new ArrayList<>();
        }
        //2) 计算每个网格存储元素的个数
        int eleOneGrid = Math.round((float) elementSet.size() / Info.num_of_grids);

        //4) 遍历数组，依次放入网格
        /*处理流程：1、依次向网格中填入特征子集；*/
        int index = 0;  //所有元素的坐标
        //Step1: 遍历网格，依次放入
        for(int gridInd=0; gridInd<Info.num_of_grids; gridInd++){
            //程序的终止条件： ① 结合遍历结束； ② 集合中遇到了空值
            if(index >= elementSet.size()){
                break;  // 直接跳出循环
            }
            else{
                //Step2: 在每个网格中放入若干个
                for(int count =0; count < eleOneGrid; count ++){  // eleOneGrid 单个网格的计数
                    if(index >=  elementSet.size()){
                        break;
                    }
                    else{
                        this.grids_row[gridInd].add(iterator.next());
                        index ++;
                    }
                }
            }
        }

        //5）特殊情况：若网格数量大于list的大小，四舍五入的值为0；就每个网格放1一个，什么时候用完了就结束
        if(eleOneGrid == 0){
            int ind = 0;
            for(FeatureSubset ele : elementSet){
                if(ele == null){
                    break;  // 如果为空直接跳出循环
                }
                assert ind < Info.num_of_grids;
                this.grids_row[ind].add(ele); //网格多，元素少
                ind ++;
            }
        }
    }

    //TODO 【版本3】去除list中重复的元素
    // 问题：list中会有重复的元素，影响采样结果
    // 所以在存入网格之前先进行去重
//    public void ConstructGrid_3 (ArrayList<FeatureSubset> elementSet_1){
//
//        //1)初始化网格
////        @SuppressWarnings("unchecked")  //通过 @SuppressWarnings("unchecked") 注解，告诉编译器你知道这里可能涉及类型不安全的操作
//        this.grids_row = new ArrayList[Info.num_of_grids];
//        for(int i = 0; i<this.grids_row.length; i++){
//            this.grids_row[i] = new ArrayList<>();
//        }
//        //2) 计算每个网格存储元素的个数
//        int eleOneGrid = Math.round((float) elementSet.size() / Info.num_of_grids);
//
//        //3) 对元素进行排序
//        // 使用Arrays.sort()方法对数组进行排序,
//        elementSet.sort(new Comparator<FeatureSubset>() {
//            @Override  //使用自定义排序器对数组进行排序
//            public int compare(FeatureSubset a, FeatureSubset b) {
//                // 按照 value 属性降序排序
////                return a.value_fun- b.value_fun;  //不接受返回类型为 double
////                return Double.compare(a.value_fun, b.value_fun);  //【空指针异常】
//                //【解决】处理空指针异常
//                if (a == null && b == null) {
//                    return 0;
//                } else if (a == null) {
//                    return 1;  // 如果a为null，认为b较大
//                } else if (b == null) {
//                    return -1;  // 如果b为null，认为a较大
//                } else {
//                    // 按照 value_fun 属性升序排序
//                    return Double.compare(a.value_fun, b.value_fun);
//                }
//            }
//        });
//
//        //4) 遍历数组，依次放入网格
//        /*处理流程：1、依次向网格中填入特征子集；*/
//        int index = 0;  //所有元素的坐标
//        //Step1: 遍历网格，依次放入
//        for(int gridInd=0; gridInd<Info.num_of_grids; gridInd++){
//            //程序的终止条件： ① 结合遍历结束； ② 集合中遇到了空值
//            if(index >= elementSet.size() || elementSet.get(index) == null){
//                break;  // 直接跳出循环
//            }
//            else{
//                //Step2: 在每个网格中放入若干个
//                for(int count =0; count < eleOneGrid; count ++){  // eleOneGrid 单个网格的计数
//                    if(index >=  elementSet.size()){
//                        break;
//                    }
//                    else{
//                        this.grids_row[gridInd].add(elementSet.get(index));
//                        index ++;
//                    }
//                }
//            }
//        }
//
//        //5）特殊情况：若网格数量大于list的大小，四舍五入的值为0；就每个网格放1一个，什么时候用完了就结束
//        if(eleOneGrid == 0){
//            int ind = 0;
//            for(FeatureSubset ele : elementSet){
//                if(ele == null){
//                    break;  // 如果为空直接跳出循环
//                }
//                assert ind < Info.num_of_grids;
//                this.grids_row[ind].add(ele); //网格多，元素少
//                ind ++;
//            }
//        }
//    }

    //TODO 从一个网格中，随机选出一个样本
    public FeatureSubset randomGet_0(ArrayList<FeatureSubset> aGrid){
        Random random = new Random();
        int index = random.nextInt(aGrid.size());
        return aGrid.get(index);
    }

    //TODO [版本2]从一个网格中，随机选出一个样本 (添加了不为空集的判断)
    public FeatureSubset randomGet(ArrayList<FeatureSubset> aGrid){
        Random random = new Random();
        int index;
        // 从数组中随机选择一个非空的特征子集
        do {
            index = random.nextInt(aGrid.size());
        }
        while (aGrid.get(index) == null);
        return aGrid.get(index);
    }
}
