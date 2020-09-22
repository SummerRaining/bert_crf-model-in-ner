在CCKS数据集上做NER任务，使用各种模型做对比：

bert：  test:  f1: 0.76524, precision: 0.75362, recall: 0.77722

bert-crf: 

在天池数据集上使用bert训练测试。

1、基于原文件，修改加载数据的函数，即可运行。目标的数据输入格式为：['word',label]

2、尝试模型的步骤，参考CCKS的论文。





提交数据的位置有错误：

我的结果：

T1	DRUG_EFFICACY 138 142	清热解毒
T2	DRUG_EFFICACY 143 147	化湿除带
T3	DRUG_EFFICACY 148 152	祛瘀止痛
T4	DRUG_EFFICACY 153 157	散结消癥
T5	SYMPTOM 167 171	小腹疼痛
T6	SYMPTOM 172 176	腰骶酸痛
T7	SYMPTOM 177 181	带下量多
T8	SYMPTOM 184 186	发热

论坛上的结果：

T1 117 122 DRUG 红虎灌肠液
T2 138 142 DRUG_EFFICACY 清热解毒
T3 143 147 DRUG_EFFICACY 化湿除带
T4 148 152 DRUG_EFFICACY 祛瘀止痛
T5 153 157 DRUG_EFFICACY 散结消癥
T6 160 165 DISEASE 慢性盆腔炎
T7 167 171 SYMPTOM 小腹疼痛
T8 172 176 SYMPTOM 腰骶酸痛
T9 177 181 SYMPTOM 带下量多
T10 184 186 SYMPTOM 发热



解决方案：

1、查看原代码中的逻辑。

2、用正则匹配，找到出现的位置。

3、验收标准：和论坛上的结果一致