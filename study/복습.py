#1. linear regression

주피터->machine_learining->복습으로

1.  K-최근접 이웃회귀
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic' 
matplotlib.rcParams['font.size'] = 15 
matplotlib.rcParams['axes.unicode_minus'] = False
from sklearn.neighbors import KNeighborsRegressor

-perch_1v.csv를 읽어와 진행
-농어 그래프로일단 그려서 분포확인
-데이터 전처리
-KNeighborsRegressor학습->R^2확인->과대/과소적합 확인->모델 변경->다시 학습후 확인->[20,50]을 예측->얘들의 이웃확인->전체(파),이웃(빨),데이터는(주)로 그래프

2. linear regression
from sklearn.linear_model import LinearRegression
-위의 데이터 그대로 써 학습->가중치,절편보기->분포와 직선 보기->R^2확인
-다항으로 보기(제곱값 하나더 추가해서)->가중치,절편 확인->그래프로 그리기->R^2확인->[50]예측해보기
-다중회귀:perch_3v.csv사용 ->3v를 더 복잡하게(degree=5)->학습->R^2확인->문제가 무엇인지

3. ridge,lasso
from sklearn.linear_model import Ridge, Lasso
-위의 perch_3v 5degree그대로 사용->최적의 파라미터찾기(그래프 그려서)->그걸로 학습->R^2확->계수몇개가 0됐는지-> test_3v평균낸 데이터로 예측
- 릿지 라쏘 둘다보기


#2. classification
주피터->machine_learining->복습으로
전부 fish.csv로 진행

1. k-neighbor classification
from sklearn.neighbors import KNeighborsClassifier
-fish 전처리 후 kn으로 학습->train, test R^2비교
- test[:5]의 클래스값, 예측값, 확률값(의미는)보기

2. Logistic Regression
from sklearn.linear_model import LogisticRegression
-1. 이진분류
이진분류의 출력함수 그리기
- bream과 smelt로 두가지 클래스의 데이터 준비
- lr로 학습->R^2비교-> 선형방정식 보기
- test[:5]의 클래스값, 예측값, 확률값보기
- z값, 함수를 통한 확률값 구하기

-2. 다중분류
-fish 데이터 전처리 후 lr학습(규제와 횟수 정해서)->R^2비교-> 선형방정식 보기
- test[:5]의 클래스값, 예측값, 확률값보기
- z값, 출력 함수를 통한 확률값 구하기

3. 확률적 경사 하강법(Stochastic Gradient Descent)
from sklearn.linear_model import SGDClassifier
- fish 데이터 전처리 후 sc학습(손실함수와 횟수=10 정해서, seed,끝까지)->R^2비교-> 선형방정식 보기
- 부분학습해보기-> 처음부터 range(300)으로 부분학습하고 score그래프 기르기->epoch횟수 정하기->학습
data={'Species':['King']*20,
     'Weight':np.random.randint(200,350,20),
     'Length':np.random.randint(20,30,20),
     'Diagnol':np.random.randint(22,32,20),
     'Height':np.random.randint(6,9,20),
     'Width':np.random.randint(5,7,20)}
newfish=pd.DataFrame(data)
클래스값, 예측값, 확률값보기- z값, 함수를 통한 확률값 구하기
