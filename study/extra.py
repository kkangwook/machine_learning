
# 1.regression

1. pandas->numpy는 df.to_numpy()
  numpy->pandas는 pd.Dataframe(arr)

2. 다항회귀에서 lr.coef_의 출력 순서는 들어간 열(변수) 순서대로 출력

3. 머신러닝에서 정규화
   x값은 변수가 여러개 존재할 수 있으므로 정규화가 필요하지만
   y값은 대부분 정규화필요X  (모델이 자동으로 최적의 가중치와 절편을 학습)
 !예외: y가 매우크거나 작을때 or 딥러닝에서 y 정규화


4. 변수가 그리 많지 않고 변수간 상관관계가 낮으며 과대적합이 걱정되지 않을때->그냥 선형회귀사용
   과대적합, 변수 많음, 다중공선성이 있으면 -> 릿지/라쏘 회귀 사용


#사용법

#공통
df  ->  to_numpy()  ->  x,y분리  ->  train_test_split으로 x_train, x_test, y_train, y_test로 분리
->  변수 하나면 x.reshape(-1,1), 아니면 x는 이미 2차원 배열 (y는 그대로)

#kneighborsregressor
from sklearn.neighbors import KNeighborsRegressor
-> standardscaler로 .fit(x_train)  ->  .transform(x_train), .transform(x_test)   (y는 안해도 됨)  
->  knr=KNeighborsRegressor(n_neighbors=n)  ->  knr.fit(x_train,y_train)  ->knr.score로 과대/과소적합 판단  
->  k개수 변경  ->knr.predict(2차원배열)  ->  dist,index=knr.kneighbors(2차원배열)  
-> plt.plot(x_train[index],y_train[index])가 참고한 이웃

# linear regression(1차 or 다항회귀)
from sklearn.linear_model import LinearRegression
-> 1차면 그대로/ 2차하고싶으면 x_train**2, x_test**2를 np.concatenate  ->  ss로 .fit(x_train)
->  .transform(x_train), .transform(x_test)  ->  lr=LinearRegression()  ->  lr.fit(x_train,y_train)
->  lr.score 확인  ->  lr.coef_, lr.intercept_확인  ->  그래프그리기  ->  range=np.arange(0,55)
->  plt.plot(range,coef1*range**2+coef2*range+intercept)  ->  lr.predict(data,data**2)

#다중회귀
from sklearn.preprocessing import PolynomialFeatures
->  poly=PolynomialFeatures(degree=n)  -> poly.fit(x_train)  ->  poly.transform(x_train), poly.transform(x_test)
->  poly.get_feature_names_out()으로 어떻게 증가했는지 확인  ->   ss.fit(x_train)  ->  x_train,x_test을 .transform
->  lr.fit  ->   lr.score  ->  lr.predict(poly_data)

from sklearn.linear_model import Ridge, Lasso
->  alpha_list=[0.001,0.01,0.1,1,10,100]로 for i in alpha_list: Ridge/Lasso(alpha=alpha)
->  .fit(x_train,y_train)  ->  train_score/test_score.append(.score(train/test))
->  plt.plot(np.log10(alpha_list),train_score/test_score) 그리고 제일 가까운 값 확인
-> 그 alpha로 .fit  ->np.sum(.coef_==0)로 몇개 차원 없어졌는지 확인  ->.score  ->.predict(poly+ss_data)



#2. classification

1. kneigborclassification
- 이웃개수k에서 predict_proba에서 나오는 값들은 (각 클래스별 선택된 개수)/k 임-> 이 값이 가장 큰애를 해당 클래스로 예측

2. logistic regression
로지스틱은 선형방정식을 구해 각 데이터별 z값을 구하고 자동으로 활성화 함수를 통과시켜 확률 1에 가장 가까운 클래스를 정답으로

-로지스틱에서의 선형방정식:정답클래스에 대한 손실함수를 최소화하는 방향으로 가중치 정해짐(선형회귀에서의 정답y값을 구하기 위한 방정식과는 다름)
-가중치 찾는법: 경사하강법 사용
  1. 가중치를 랜덤하게 초기화한다.
  2. 현재 가중치로 z 값을 계산하고, 시그모이드 함수로 확률 p를 구한다.
  3. 예측한 p와 실제 정답 y 사이의 오차(손실)를 계산한다.
  4. 이 손실 값을 줄이기 위한 방향(=기울기, gradient)을 계산한다.
  5. 그 방향으로 가중치를 조금씩 업데이트한다.
  6. 이 과정을 반복하면서 손실을 줄여간다.

-z값의 의미:
"내가 z를 계산했을 때, 그게 클래스 1일 확률이 높도록 하고 싶어."
그럼 자연스럽게 데이터 중 클래스 1에 해당하는 x 값들이 z값을 크게 만들도록,
반대로 클래스 0 데이터는 z값이 작게 나오도록,
그에 맞게 가중치들이 조정되는 것임



-이진분류: 클래스가 두개일때 선형방정식을 하나만 구해 시그모이드를 통과시켜 확률높은 값을 정답으로
  -이때 두 클래스를 0,1로 두고 클래스가 1인것에 대한 선형방정식을 구함
    - 클래스 1일 확률을 p라했을때 클래스0일 확률은 자동으로 1-p로 계산. 이중 더 큰 값을 클래스로 선정
      ex) lr.classes_=['bream','smelt']일 경우 smelt가 클래스1-> expit(z)값은 클래스1인 smelt일 확률
-다중분류: 각각의 클래스마다 선형방정식을 구해 소프트맥스함수에 통과시켜 전체합이 1이 되도록하고 그중 가장 큰값을 정답으로
    c값은 양수로 보통 0.01에서 1000사이로 지정
    lr.decision_funtion으로 z값 구하면 각 클래스의 선형방정식 별 모든 z값 구해줌



#손실함수: 실제값과 예측값 사이의 차이
종류: MAE, MSE, RMSE, 이진 교차 엔트로피(Binary Cross-entropy), 카테고리컬 교차 엔트로피(Categorical Cross-entropy)



3. 확률적 경사하강법:이진분류면 선형방정식 하나 클래스 여러개면 각각의 선형방정식을 만듬 
그래프 해석: x축은 모델의 가중치, y축은 손실함수
손실함수 그래프(2차함수모형은 그냥 예시일뿐)에서 손실값이 최소가 될때는 손실함수의 기울기가 0일때(미분값=0)
x의 가중치를 변경해가면서 기울기 0되는곳을 찾는것임 
기울기0인 지점을 m이라 하고 가중치값을a라 하자. 주로 가중치는 왼쪽의 0부터 시작. 
미분계수가 0보다 작다(m의 왼쪽이다)-> a+alpha(학습률)
미분계수가 0보다 크다(m의 오른쪽)-> a+alpha(음의 학습률)
학습률이 너무크면 m지점을 건너뛸수도, 너무 작으면 너무 오래걸림

식: 새로운a=기존a-학습률*(a지점에서의 손실함수 미분값)
-> 기울기 -이면 새로운a는 오른쪽으로, +이면 왼쪽으로
->좌우 왔다갔다 거림-> 이동하면서 학습률은 점점 감소기킴


"그냥 미분 방정식을 풀어서 해를 찾으면 되는거 아니냐!"라고 할수도 있지만 손실함수는 주로 다차함수여서 미분방정식푸는게 더 오래걸림
또한 미분값이 0이 되는곳이 여러군데일수도 있음->목적은 지역 최소값들 중 '전역' 최소값을 찾는 것이 목표
  -> 어느정도 지역최소값을 찾으면 그 값들 중 최소값을 전역 최소값으로 간주
그냥 경사하강법쓰면 지역최소값에 갇힐 수 있음->1. 시작지점을 다양하게 하거나
2. 확률적 경사 하강법 사용
배치 경사 하강법(Batch Gradient Descent)은 전체 데이터를 한 번에 사용하여 평균내고 정확한 평균 기울기를 계산하지만, 
확률적 경사 하강법(SGD)은 단일 데이터 포인트를 사용하여 기울기를 계산
  :데이터 하나 뽑아 손실함수 계산-> 가중치를 업데이트하고 또 다시 다른 데이터 뽑아 손실함수계산-> 가중치를 업데이트
    ->이때 하나의 샘플은 그 전체 데이터를 대표하지 못하므로 계산된 기울기에는 잡음이 포함될수밖에 없음
      ->그러면 이러한 잡음은 지역최소값에 머물지 않고 탈출해 더 좋은 전역 최소값에 도달가능
    ->랜덤하게 뽑히면서 클래스정답이 다른 샘플들이 뽑히지만 그런 부분도 다 일정부분 업데이트하여 각 클래스별 선형방정식을 찾음
    ->시간이 지날수록 평균에 근접해지며, 이러한 하나의 전체과정epoch를 여러번 반복해서 최고의 
    전역 최솟값과 그에따른 선형방정식을 구함
그렇다면 클래스는 무엇을 기준으로 구분하나?
  ->하나의 샘플에 각 클래스별 선형방정식을 적용시켜 각 클래스별 z예측값을 구하고 소프트맥스 함수를 거쳐 
    가장 확률높은 클래스로 분류


이진분류: 손실함수=이진 교차 엔트로피, 출력함수=시그모이드
정답값 1일때: 손실값=-log(예측확률), 정답값 0일때: 손실값=-log(1-예측확률)
다중분류: 손실함수=다중 클래스 교차 엔트로피, 출력함수=소프트맥스

확률적 경사하강법에서 손실함수는 샘플 고를때마다 매번 시행되고 소프트맥스함수는 마지막에 한번 예측할떄 사용

sc.classes_, sc.coef_, sc.intercept_로 클래스별 가중치+절편(선형방정식)을 볼 수 있음



#사용법
fish=pd.read_csv('https://bit.ly/fish_csv_data')

# k-neighbor vlassification
from sklearn.neighbors import KNeighborsClassifier
-> x,y나누기->train, test나누기->x는 정규화->kn=KNeighborsClassifier(n_neighbors=3)->.fit
-> train, test score비교->test[:5]로 kn.predict해보기-> kn.classes_, kn.predict_proba로 확률보기
-> 각 확률은 이웃클래스개수/k

# logistic regression
from sklearn.linear_model import LogisticRegression
1. 이진분류: 출력함수로 시그모이드 함수사용
->x,y나누기->train, test나누기->x는 정규화-> 두 클래스만 가져오기
  x=train_5v_scaled[(train_species=='Bream')|(train_species=='Smelt')]   #빙어/도미 트레인셋
  y=train_species[(train_species=='Bream')|(train_species=='Smelt')] 
-> -5~5z값의 시그모이드 그래프 그리기(z=np.arange(-5,5,0.1), phi=1/(1+np.exp(-z))
->lr=LogisticRegression()->fit->score->[:5]에 대해 lr.classes_, lr.predict_proba로 클래스와 확률 확인
->lr.coef_, lr.intercept_로 하나의 선형방정식 확인->z=lr.decision_function([:5])로 z값 확인
->from scipy.special import expit->expit(z)로 시그모이드함수통한 확률확인->위의 lr.predict_proba와 비교

2. 다중분류: 출력함수로 소프트 맥스 함수 사용
->x,y나누기->train, test나누기->x는 정규화
->lr=LogisticRegression(C=20, max_iter=1000) c:0.01(큰 규제)~1000(작은규제)
-> fit->score로 train, test비교->test[:5]로 predict-> lr.classes_, lr.predict_proba로 확률
-> lr.coef_, lr.intercept로 클래스별 선형방정식 확인->z=lr.decision_function(test[:5])로 각 클래스방정식 별 z확인
->from scipy.special import softmax-> sm=softmax(z,axis=1)로 소프트맥스 함수 확인 후 proba와 비교

#확률적 경사하강(Stochastic Gradient Descent)
from sklearn.linear_model import SGDClassifier
->x,y나누기->train, test나누기->x는 정규화
->sc=SGDClassifier(loss='log_loss', max_iter=10, random_state=123, tol=None)  손실함수는 log_loss, hinge(defalut)두가지 존재/  tol=None하면 멈추지 않고 max_iter까지 epoch도달
->fit->score로 train,test비교->partial_fit으로 추가 학습->score로 얼마나 발전했는지 확인
->epoch회수 정하기
  sc=SGDClassifier(loss='log_loss', random_state=123) ;train_score=[] ;test_score=[] ;classes=np.unique(train_species)   #고유 생선종들
    for _ in range(0,300):
      sc.partial_fit(x,y, classes=classes)    #partial_fit만 사용시 classes넣어줘야함
      train_score.append(sc.score(x,y))
      test_score.append(sc.score(x,y)) 
    plt.plot(train_score);  plt.plot(test_score)로 그리기-> 어느순간 정확도 감소
-> 적절한 epoch찾아 다시학습->fit->score->sc.coef_, sc.intercept_로 클래스별 선형방정식 확인
->predict로 예측->sc.classes_, sc.predict_proba보기
->sc.decision_function, softmax로 위의 확률과 비교
