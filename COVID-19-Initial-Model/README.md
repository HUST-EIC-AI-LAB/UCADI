# COVID-19
This repo is for the experiment codes.     

We regard COVID-19 diagnosis as a multi-classification task and classified to four classes including `healthy`, `COVID-19`, `other viral pneumonia`, `bacterial pneumonia`.  As for our datasets, we use multi-center datasets consisting of data from Main Campus hospital, Optical Valley hospital, Sino-French hospital and 18 hospitals in NCCID.

**Train and validation split:**

| Patients / CTs | N/A (healthy) | COVID-19  | Other viral | Bacterial | Total patients / CTs |
| -------------- | :-----------: | :-------: | :---------: | :-------: | :------------------: |
| Main Campus    |   224 / 727   | 135 / 922 |  56 / 250   | 254 / 934 |      669 / 2833      |
| Optical Valley |   75 / 278    | 112 / 425 |    0 / 0    |  13 / 47  |      200 / 750       |
| Sino-French    |   43 / 131    | 158 / 853 |    0 / 0    |  25 / 97  |      226 / 1081      |
| NCCID          |  392 / 1491   | 199 / 654 |    0 / 0    |   0 / 0   |      591 / 2145      |

**Test split:** 

| Patients / CTs | N/A (healthy) | COVID-19 | Other viral | Bacterial | Total patients / CTs |
| -------------- | :-----------: | :------: | :---------: | :-------: | :------------------: |
| Main Campus    |   58 / 191    | 34 / 191 |   19 / 72   | 50 / 170  |      103 / 624       |
| Optical Valley |    12 / 44    | 23 / 88  |    0 / 0    |   2 / 8   |       37 / 140       |
| Sino-French    |    10 / 27    | 37 / 244 |   1 / 12    |  8 / 27   |       56 / 310       |
| NCCID          |   235 / 362   | 90 / 175 |    0 / 0    |   0 / 0   |      345 / 537       |

**Corresponding label to class (four- class classification):**

| label | name                  |
| ----- | --------------------- |
| 0     | healthy               |
| 1     | COVID-19              |
| 2     | other viral pneumonia |
| 3     | bacterial pneumonia   |

### Data Preprocess

Codes for data preprocessing are in:  `./utils`.    

The raw CT images we get from hospitals are not exactly what we can feed to network directly. So there are some cleaning operations to be conducted to the initial datasets. The operations are built based on our careful check with the CT images. **We find that when the slice numbers of CT are less than 15 and the width or height of CT images are not equal to 512 pixels, the images are usually useless.** So that we clip all images' pixels to [-1200, 600], which is a ordinay operation in medical image. Finally, we calculate the mean and std of the whole datasets and then normalize each image.    


### Model  
We ultilize 3D-DenseNet as our baseline model. Before we feed images into network, we find that if we can cover the whole lung in temporal direction, the model behaves much better. Besides, we confim that there is a linear relation between slice thickness and slice numbers. As a result, a sample strategy is proposed as  the following pseudo codes said:  
```python
if slice z_len <= 80:
    random start index;
    choose image every 1 interval; # if start=0, choose [0,1,2,...,13,14,15]
elif slice z_len <= 160:
    random start index from [10, z_len - 60];
    choose image every 2 interval; # if start=10, choose [10,12,14,...,36,38,40]
else:
    start=random.randrange(20, z_len - 130)
    random start index;
    choose image every 5 interval; # if start=0, choose [20,25,30,...,85,90,95] 
```
- Resize sequence images to [16,128,128].
- Without augmentation.
- Regulization --- linear scheduler of dropblock (block=5) from prob=0.0 to prob=0.5.
- Optimizer --- torch.optim.SGD(params, lr=0.01, momentum=0.9).
- No bias decay --- weight decay = 4e-5.
- Lr_scheduler ---Warmup and CosineAnnealing.
- Output layer --- FC(features, 4) -> weighted cross entropy of [0.2, 0.2, 0.4, 0.2]
- batch size --- 70.
- Machine Resource --- 2 Tesla V100.

### Federated Learning

Due to levels of incompleteness, isolation, and the heterogeneity in the different data resources, the locally trained models exhibited less-than-ideal test performances on other CT sources. To overcome this hurdle, We proposed a federated learning framework to facilitate UCADI, intergrating ethnically diverse cohorts as part of global joint effort on developing a precise and generalized AI diagnostic model. The concrete introduction of federated learning locates at the repo: https://github.com/HUST-EIC-AI-LAB/COVID-19-Fedrated-Learning-Framework.


### Results
We use the hospital's name indicates the model trained on the hospital's data resources in the following tables, and 'Federated' means trained with four clients separately having train data from Main Campus, Optical Valley, Sino-French hospital and NCCID based on federated learning framework.

**COVID-19 peneumonia identification performance of CNN models** on **China data**(China data means the merged version of test dataset including Main Campus hospital, Optical Valley hospital, Sino-French hospital) and **UK data**(UK data means the data from 18 hospitals in NCCID) as following:

**China data:**

|             | Main Campus | Optical Valley | Sino-French | NCCID | Federated |
| :---------: | :---------: | :------------: | :---------: | :---: | :-------: |
| Sensitivity |    0.538    |     0.973      |    0.900    | 0.313 |   0.973   |
| Specificity |    0.926    |     0.444      |    0.759    | 0.907 |   0.951   |
|     AUC     |    0.840    |     0.884      |    0.922    | 0.745 |   0.980   |

**UK data:**

|             | Main Campus | Optical Valley | Sino-French | NCCID | Federated |
| :---------: | :---------: | :------------: | :---------: | :---: | :-------: |
| Sensitivity |    0.054    |     0.541      |    1,999    | 0.703 |   0.730   |
| Specificity |    0.835    |     0.626      |    0.160    | 0.961 |   0.942   |
|     AUC     |    0.487    |     0.647      |    0.613    | 0.882 |   0.894   |

**Furthermore**
We refined the CNN by introducing three severities of COVID-19 pneumonia (Figure 3 a, b, and c) and then validated and tested the performance of three-severity classification task. Specifically, this task classifies COVID-19 cases into three severities corresponding to three radiological degrees: I, II, and III representing low, moderate, and high impact on prognosis (this severity standard is internally proposed by department of radiology at Tongji Hospital which examined approximately 5000 COVID-19 patients during COVID-19 outbreak). We validated the CNN in this task which achieved overall 50.7% sensitivity and 93.2% specificity. We conducted the test using 80 COVID-19 cases (28 COVID-19-1, 36 COVID-19-II, 16 COVID-19-III) to compare the performance between the CNN and six radiologists. The CNN achieved overall sensitivity of 61.3% and specificity of 93.6% while six radiologists obtained overall 52.9% in sensitivity and 93.1% in specificity. The result demonstrated that the CNN performed comparable competence to radiologists in assessing the severity of confirmed COVID-19 patients.






