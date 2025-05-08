---------------------선형대수학----------------------------
선형회귀:
샘플수 100개라 하고 각 샘플별 변수(특성)이 5개라 하면 입력차원은 (100,5)
그러면 선형방정식의 가중치는(5,1), 절편은 (1,) 이거나 하나의 스칼라값
최종적으로 (100,5)@(5,1) =(100,1)+(1,)는 브로드캐스팅->최종 출력=(100,1) : 각 샘플별 하나의 스칼라 값

분류, 클러스터: 다중 출력
샘플수 100개에 변수(특성)5개로 동일하게-> (100,5)
이때 분류할 클래스가 3개라면(=딥러닝으로 비교하자면 뉴런이 3개) 가중치 크기는 (특성,클래스개수)=(5,3), 절편또한 (3,)
결국 (100,5)@(5,3)+(3,)하면 최종 출력값 크기는 (100,3)으로 100개의 샘플당 하나의 샘플은 벡터길이 3개짜리 출력-> 얘들을 소프트맥스함수로




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


#3. tree
주피터->machine_learining->복습으로
전부 wine.csv로 진행

1. dt
이 데이터의 전처리 차이점 -> dt에 간단한 파라미터 적용 후 학습->검증->트리그리기-> 트리 정보 확인하기
-> 불순도 실제로 계산하기, 불순도 차이 계산하기 -> x_test[:5]로 예측하기 -> 특성 중요도 보기

2. 검증
-cv에 dt넣고 train_score까지도 나오게 해서 검증점수 보기->test_score평균내기

-search 두가지
gs->파라미터 두개만 지정해서->최적파라미터보기->그때의 검증점수보기->최적 파라미터저장값 ddt객체로 불러오기
-> gs와 ddt의 x_test score차이 있는지 보기 -> x_test[:5]예측하고 score점수보기->특성 중요도보기

rs-> 분포범위 불러오는 함수-> 파라미터 4가지 지정->rs 파라미터도 지정-> gs와 rs의 차이는? 파라미터 차이는?
-> 최적파라미터보기->그때의 검증점수보기->최적 파라미터저장값 ddt객체로 불러오기
-> gs와 ddt의 x_test score차이 있는지 보기 -> x_test[:5]예측하고 score점수보기->특성 중요도보기

3. 여러개의 트리
트리 4종류 각각 rs서치로 진행-> 각 트리 기본 파라미터 -> 파라미터 지정하고 rs의 파라미터도 지정하고 학습
-> 이때 3번째 꺼에 새로운 옵션 적용 -> 4번째는 좀 다른 파라미터 4가지 ->  최적파라미터보기->그때의 검증점수보기
-> 최적모델 불러와 특성중요도 보기 -> x_test[:5]예측하고 score보기

4. 다중 분류
from sklearn.datasets import load_iris
data = load_iris()
x = data.data
y = data.target
위 데이터 이용해 여태 트리들 중 가장 정확도 높은 트리로 학습->최적파라미터와 검증점수
-> x_test[:5]예측하고 score보기
