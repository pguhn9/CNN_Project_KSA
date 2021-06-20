## Object Detection 팀 프로젝트

#### 날짜: 2021.06.20



## 1. 프로젝트 목적

* SSD 모델을 이용해 객체 탐지 모델 만들기.
* 객체 탐지 모델 평가



## 프로젝트 내용

1. 데이터 설명

   * 11종의 앵무새 객체 검출을 위한 jpg 이미지 데이터.
   * 각 이미지 경로와 Ground Truth 정보를 담은 CSV 파일.
   * 데이터비율: 약 train : validation: test = 6:1:1 

2. 프로젝트 환경설정

   * OS: Window10
   * GPU: RTX 2060 Super
   * CUDA: 10.1
   * CUDNN: 7.6.0
   * IDE: Pycharm

   * 파이썬 환경설정
     * python= 3.7
     * tensorflow-gpu= 2.1
     * keras= 2.2.4
     * numpy= 1.19.5
     * matplotlib
     * opencv-python
     * tqdm 등

3. 데이터 전처리

   * **DataAugmentation** 

     * 회전
     
       <img src="https://github.com/pguhn9/CNN_Project_KSA/blob/main/Project_ObjectDetection/images/%EC%95%B5%EB%AC%B4_90%EB%8F%84_%ED%9A%8C%EC%A0%84.PNG" width="400px" height="400px" title="앵무_회전" alt="앵무_회전"></img><br/>
     
     * 이동

       <img src="https://github.com/pguhn9/CNN_Project_KSA/blob/main/Project_ObjectDetection/images/%EC%95%B5%EB%AC%B4_%EC%9D%B4%EB%8F%991.PNG" width="400px" height="400px" title="앵무_이동" alt="앵무_이동"></img><br/>

     * 뒤집기

       <img src="https://github.com/pguhn9/CNN_Project_KSA/blob/main/Project_ObjectDetection/images/%EC%95%B5%EB%AC%B4_%ED%94%8C%EB%A6%BD.PNG" width="400px" height="400px" title="앵무_뒤집기" alt="앵무_뒤집기"></img><br/>

     * 확대 및 축소

       <img src="https://github.com/pguhn9/CNN_Project_KSA/blob/main/Project_ObjectDetection/images/%EC%95%B5%EB%AC%B4_%EB%A6%AC%EC%82%AC%EC%9D%B4%EC%A6%88.PNG" width="400px" height="400px" title="앵무_리사이즈" alt="앵무_리사이즈"></img><br/>

4. 모델 선정 및 학습

   * SSD - Backbone Network(VGG-16)

     <img src="https://github.com/pguhn9/CNN_Project_KSA/blob/main/Project_ObjectDetection/images/SSD_backbone(VGG16).PNG" width="800px" height="300px" title="ssd_vgg16" alt="ssd_vgg16"></img><br/>

   * 전이학습

     기존의 모델 형태로 새로운 데이터를 처음부터 학습한 결과 loss값이 발산하는 문제 가 발생 

     => 비슷한 문제로 사전에 학습된 가중치 사용하여 해결

      가중치 모델명 : VGG_ILSVRC_16_layers_fc_reduced.h5

   * Callback 함수

     * ModelCheckpoint : validation_loss 값을 기준으로 최적의 모델 저장
     * CSVLogger : 각 epoch 별 loss값 csv로 저장
     * LearningRateScheduler : 과적합 방지를 위한 learning_rate 조정
     * TerminateOnNaN : loss의 발산을 막기 위한 함수

5.  모델 평가

   * **loss** 그래프

     <img src="https://github.com/pguhn9/CNN_Project_KSA/blob/main/Project_ObjectDetection/images/SSD_Loss_graph74.PNG" width="400px" height="400px" title="SSD_Loss_graph" alt="SSD_Loss_graph"></img><br/>

     epoch이 진행될수록 train과 validation의 성능 차이가 커지는 것으로 보아 더 큰  epoch으로 성능 향상을 기대하기는 힘들 것으로 보임.

   *  최종 성능

     Train_loss : 1.89771  Validation_loss : 2.9813

6. 테스트 결과

   * Test image result

     <img src="https://github.com/pguhn9/CNN_Project_KSA/blob/main/Project_ObjectDetection/images/%EC%95%B5%EB%AC%B4%EC%95%BC3.PNG" width="400px" height="400px" title="앵무_결과1" alt="앵무_결과1"></img><br/>

     <img src="https://github.com/pguhn9/CNN_Project_KSA/blob/main/Project_ObjectDetection/images/%EC%95%B5%EB%AC%B4%EC%95%BC2.PNG" width="400px" height="400px" title="앵무_결과2" alt="앵무_결과2"></img><br/>

     <img src="https://github.com/pguhn9/CNN_Project_KSA/blob/main/Project_ObjectDetection/images/%EC%95%B5%EB%AC%B4%EC%95%BC4.PNG" width="400px" height="400px" title="앵무_결과3" alt="앵무_결과3"></img><br/>

   * Parrot AP Graph

     <img src="https://github.com/pguhn9/CNN_Project_KSA/blob/main/Project_ObjectDetection/images/parrot_AP_graph1.png" width="400px" height="800px" title="parrot_AP_graph1" alt="parrot_AP_graph1"></img><br/>

   * Parrot mAP(mean Average Precision)

     <img src="https://github.com/pguhn9/CNN_Project_KSA/blob/main/Project_ObjectDetection/images/parrot_AP_mAP1.PNG" width="400px" height="500px" title="parrot_AP_mAP1" alt="parrot_AP_mAP1"></img><br/>

     Gpffin과 sulphur_crested cockatoo에 대해서는 100의 평균 정밀도를 보임.

      모델 전체 클래스에 대해 0.937의 mAP 값을 가짐.









## 참고

* SSD tutorial : https://github.com/pierluigiferrari/ssd_keras#readme
* 데이터 비공개(비공개 요청) 이후 관련 데이터, 코드 삭제할 수 있음.

