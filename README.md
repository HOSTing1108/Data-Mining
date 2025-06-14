# Fake News Classification Using Logistic Regression
**Topic**: Proposing a machine learning model for fake news classification

## 실험 환경 정보
- **Python 버전**: 3.10.16
- **PyTorch 버전**: 2.7.0+cpu
- **Scikit-learn 버전**: 1.6.1
- **Pandas 버전**: 2.2.3
- **NumPy 버전**: 1.23.5
- **Matplotlib 버전**: 3.10.3
- **Seaborn 버전**: 0.13.2
- **Transformers 버전**: 4.52.3
- **CUDA 사용 여부**: False

## 데이터
- **'Fake and Real News Dataset'**: Kaggle에서 공개된 2016-2017년 사이의 **미국 정치 뉴스** 데이터를 활용하였습니다.
- **데이터 파일**:
  - 'True.csv': 실제 뉴스 데이터를 포함한 CSV 파일(파일이 너무 커 압축 파일 사용)
  - 'Fake.csv': 가짜 뉴스 데이터를 포함한 CSV 파일(파일이 너무 커 압축 파일 사용)
- **임베딩 파일**:
  - 'X_test_text_bert.npy': 본문 테스트 셋을 임베딩한 파일(파일이 너무 커 압축 파일 사용)
  - 'X_test_title_bert.npy': 제목 테스트 셋을 임베딩한 파일(파일이 너무 커 압축 파일 사용)
  - **[X_train_text_bert.npy](https://drive.google.com/file/d/1pLCAR_oDBOifYfk3msT2yIa9CPP-k-9_/view?usp=sharing)**: 트레인 데이터 본문 임베딩 파일
  - **[X_train_title_bert.npy](https://drive.google.com/file/d/1bjHtf_v1Tebh06uZwmcIOR2jPh8Whp_B/view?usp=sharing)**: 트레인 데이터 제목 임베딩 파일
- **모델 파일**:
  - 'logreg_model.pkl': 학습된 로지스틱 회귀 모델.
  - 'scaler.pkl': 모델 훈련에 사용된 스케일러.

## 실험 결과
- **Final selection model**: **Logistic Regression**
- **Accuracy**: 0.9945
- **Precision**: 0.9946
- **Recall**: 0.9944
- **F1 Score**: 0.9945

