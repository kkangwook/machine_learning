sklearn의 random_state는 지정한 값에 따라 동일한 샘플일경우 전세계 어디의 누구든 같은 값이 나오게 함



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

------------------------------------------------------------------------------------------------------------

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


!!!손실값은 손실값대로 나오고 이거와 상관없이 손실함수에 대해 가중치별로 편미분해서 각각의 가중치를 업데이트!!!


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


-------------------------------------------------------------------------------------------

#3. 트리
로지스틱 회귀 분류: 빠름, 데이터 특성이 선형적일때 사용, 과적합에 덜 민감, 복잡한 관계는 잘 못잡아냄
트리: 비선형적, 복잡한데이터에 많이 사용, 시각화 쉬움, 과적합에 매우 취약

--cross validate나 grid/randomized search는 자체적으로 .fit하기 때문에 사전에 fit할 필요 없음

-grid search사용할줄알면 이미 cross-validate기능도 포함하기 때문에 cross-validate따로 사용할 필요X
  :파라미터 세세히 정의 필요
  : sklearn의 거의 모든 모델에 사용 가능
-randomized searchcv: 그리드 서치와 유사하지만 파라미터 간격을 지정하기 어려울경우 사용->확률분포를 전달

둘다 파라미터는 같음, 상황에 따라 둘 중 하나만 쓰면 됨
ex) 
ridge = Ridge()
param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error')
# 모델, 파라미터, 폴드수, 손실값
cv=5 or 10주로 사용 : 데이터셋을 몇 조각(fold)으로 나눌지-> 교차검증 수행
scoring: 
        -회귀: 'neg_mean_squared_error'(mse) or 'r2'(default)
        -분류: 'accuracy'(default) or 'f1', 'precision', 'recall'

이때 grid.predict와 model=grid.best_estimator_하고 model.predict는 같음!!!!!
 그냥 model로 명명해주냐 안해주냐의 차이

교차검증용만으로 그리드서치 쓸거면 최소 파라미터 하나는 들어가야됌!!!!!
ex) param_grid = {'alpha': [1.0]}


앙상블
- 랜덤포레스트: 100개의 샘플과 10개의 특징이 있다고 가정하자
    랜덤포레스트는 100개중 랜덤하게 중복허용해서 100개뽑음(부트스트랩)
    또한 특성도 다 쓰지 않고 랜덤하게 선택(ex:5개) 
    n_estimators=100일 경우 위 과정을 100번 반복해 100개 트리 생성해서 모음
    -> 트리들의 예측 결과를 다수결 또는 확률 평균으로 최종 클래스 결정

-엑스트라트리(extreme): 전체데이터 사용하지만 특성 랜덤, 분할기준(부-자 간의 지니차이값이 크게끔)조차도 랜덤
    이 과정을 여러번 반복해 여러 트리 모음
    정확성은 덜하지만 과적합을 줄일 수 있음

-그레디언트 부스팅: 얕은 깊이(주로 깊이3인)사용+ 결정트리 100개사용->과대적합을 예방하고 일반화에 좋음
    경사하강법과 유사: 학습률을 정해주고 결정트리를 계속 추가해 가면서 가장 낮은곳으로 이동하는 방식

-히스토그램 기반 그레디언트 부스팅: 미리 특성에 따른 샘플들 히스토그램 그려서 분할후보를 찾음
     특성은 각 특성값 당 256개의 bin으로 나눔눔



#사용법

!! 정규화는 필요X(분할기준이 특성하나씩만사용+ 비율에 따라 분할하기때문)


1. 결정트리
from sklearn.tree import DecisionClassifier
data ->x,y -> train,test분리 -> dt=DecisionTreeClassifier(max_depth=k, random_state=123) -> dt.fit -> dt.score
-> dt.feature_importances_로 특성 중요도 -> 트리형상화: from sklearn.tree import plot_tree
-> plt.figure + plot_tree(dt,max_depth=k, filled=True, feature_names=['a','b','c'])+plt.show
-> 해석하기: gini란? -> 분할기준은?-> 분류기준은?


2. 교차검증과 최적의 파라미터 찾기

-cross-validate
from sklearn.model_selection import cross_validate
train세트에서 굳이 검증세트 나눌 필요 없음 -> dt새로 만들기(.fit안하기)
-> score=cross_validate(dt,x_train,y_train,cv=5 or 10(fold), return_train_score=True) #자동으로 fit기능 존재
->그냥 print(score)하면 cv개수별로 다보여줌 -> np.mean(scores['train_score']), np.mean(scores['test_score'])

-grid search
from sklearn.model_selection import GridSearchCV
train세트와 새로운 dt(random_state=123)준비-> 파라미터 준비: dt의 경우 params={'min_impurity_decrease':np.arange(0.0001,0.001,0.0001),
 'max_depth':range(5,20,1),'min_samples_split':range(2,100,10)}
-> gs=GridSearchCV(dt,params,n_jobs=-1, cv=5 or 10) ->서치수행 by gs.fit(x_train,y_train)
-> gs.best_params_ (최적 매개변수), gs.cv_results['mean_test_score'] (검증점수)
-> best_dt=gs.best_estimator_ (최적 조건 적용된 모델)-> best_dt.score, predict가능

- randomized search cv
from sklearn.model_selection import RandomizedSearchCV
train세트와 dt(random_state=123)준비->from scipy.stats import randint, uniform필요: 간격 지정하기 어려워서
->params={'min_impurity_decrease':uniform(0.0001,0.001), #실수
       'max_depth':randint(20,50),     #정수
       'min_samples_split':randint(2,25),
       'min_samples_leaf':randint(1,25)}
->gs=RandomizedSearchCV(dt,params,n_iter=100,n_jobs=-1,random_state=123) #랜덤하게 파라미터 조합 100개 뽑아 이중 최적찾음
-> 서치수행 by gs.fit(x_train,y_train)-> gs.best_params_, np.max(gs.cv_results_['mean_test_score'])
->dt=gs.best_estimator_ -> dt.score(), dt.predict()


3. 트리 앙상블: 결정트리여러개 모으기
공통: data -> x,y ->train, test (정규화는 X)
!!!!!모든 모델 공통적으로 max_depth, min_samples_split, min_samples_leaf, min_impurity_decrease
  -> grid나 randomized search사용가능

-랜덤포레스트: 샘플 랜덤, 특성 랜덤
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=100(default->100개의 트)n_jobs=-1,random_state=123, oob_score=True) #부트스트랩에 포함되지 않은 데이터(OOB)를 검증세트로써 사용하는 기능도 존재
->scores=cross_validate(rf, x_train, y_train, return_train_score=True, n_jobs=-1)-> print(score)
->학습하기 rf.fit(x_train,y_train)-> rf.feature_importances, rf.oob_score_->rf.score, rf.predict

- 엑스트라 트리: 전체 데이터 사용+ 특성랜덤+ 분할기준 랜덤
from sklearn.ensemble import ExtraTreesClassifier
et=ExtraTreesClassifier(n_jobs=-1,random_state=123, n_estimators=100(디폴트)) -> scores=cross_validate(et,x_train,y_train, return_train_score=True, n_jobs=-1)
-> np.mean(scores['train_score']), np.mean(scores['test_score']) -> et.fit() -> et.feature_importances_
-> et.score, et.predict

- 그레디언트 부스팅: 경사하강법으로 낮은 depth 여러tree 모음
from sklearn.ensemble import GradientBoostingClassifier
gb=GradientBoostingClassifier(random_state=123) -> scores=cross_validate(gb,x_train,y_train, return_train_score=True, n_jobs=-1)
-> np.mean(scores['train_score']), np.mean(scores['test_score']) 
-> 트리 더모으기 gb=GradientBoostingClassifier(n_estimators=500,learning_rate=0.2,random_state=123)
-> 다시 교차검증 후 score-> gb.fit -> gb.feature_importances_ -> gb.score, gb.predict

-히스토그램 그레디언트 부스팅: 특성 미리 256의 bin으로 나눠 히스토그램으로-> 최적 분할후보 찾음
from sklearn.ensemble import HistGradientBoostingClassifier
hgb=HistGradientBoostingClassifier(random_state=123) -> 교차검증 후 score ->hgb.fit
-> 중요도 보기: from sklearn.inspection import permutation_importance
-> result=permutation_importance(hgb,x_train,y_train, n_repeats=10, random_state=123, n_jobs=-1) #n_repeats는 랜덤하게 섞을 횟수/원래 디폴트값은 5
-> result.importances_mean ->hgb.score, hgb.predict


---------------rs사용버전----------------------
params={'min_impurity_decrease':uniform(0.0001,0.001),
       'max_depth':randint(10,50),
       'min_samples_split':randint(2,25),
       'min_samples_leaf':randint(1,25)}
rf=RandomForestClassifier(random_state=123,n_estimators=200)
rs=RandomizedSearchCV(rf,param_distributions=params,cv=5,n_iter=100,n_jobs=-1,random_state=123)
rs.fit(x_train,y_train)
rs.best_params_, np.max(rs.cv_results_['mean_test_score'])
rf=rs.best_estimator_
rf.feature_importances_


hgb은 다른 파라미터
params={'min_samples_leaf':randint(1,25),
       'max_depth':randint(10,50),
       'max_leaf_nodes':randint(10,50),
       'learning_rate':uniform(0.1,0.2)}   #새로운 옵션
hgc=HistGradientBoostingClassifier(random_state=123,max_iter=200)

rs=RandomizedSearchCV(hgc,param_distributions=params,cv=5,n_iter=100,n_jobs=-1,random_state=123)
rs.fit(x_train,y_train)
rs.best_params_, np.max(rs.cv_results_['mean_test_score'])



4. 랜덤포레스트로 다중분류 해보기
#데이터 준비
from sklearn.datasets import load_iris
data = load_iris()
X = data.data
y = data.target
->train, test로 나누기 -> rf = RandomForestClassifier(n_estimators=100,max_depth=10, random_state=42)
-> rf.fit(X_train, y_train) -> rf.score(X_train,y_train), rf.score(X_test,y_test)
-> rf.feature_importances -> rf.predict(x_test)



-----------------------------------------------------------------------------------------
# 군집화
-kmeans는 n_iter_횟수만큼 여러번 반복하여 최적의 중심지를 찾음, k=n_cluster의 개
->k개 지점 랜덤하게 정하고 이 지점에 가까운 샘플을 하나의 클러스터로->각 클러스터 평균값을 중심으로 정함
->이 중심을 지점으로 다시 가까운 샘플들->클러스터안의 샘플들이 더이상 변화가 없을때 종료

# 주성분 분석 PCA
주성분이란 데이터를 그렸을때 분산(퍼짐)이 큰 방향or 벡터
주성분의 개수는 특성의 개수까지 가능-> 이 중 가장 큰 주성분부터 표현
이 그림은 1차원화해서 (10000,)벡터이므로 차원개수=특성개수는10000->가능한 주성분도 10000개


1. 군집화-Kmeans
fruits_300.npy 배열 사용->불러오기-> 사이즈보기(100,100)짜리가 300개 샘플-> 최소/최대값보기 
-> 첫번째 그림 cmap='gray'로 흑백사진보기(흰색일수록 숫자값큼)-> gray_r로 색반전
->0~99는 사과, 100~199는 파인애플, 200~299는 바나나 -> from sklearn.cluster import KMeans
-> km=KMeans(n_clusters=k, random_state=123, n_init=10) # n_init은 첫 중심지를 랜덤하게 10곳으로 해서 10번의 시행중 
가장 낮은 inertia가 나오는 결과를 불러옴 ->fruits_2d=fruits.reshape(-1,100*100)로 2차원화
-> kn.fit(fruits_2d) -> km.labels_로 군집확인 -> np.unique(km.labels_,return_counts=True) 로 클러스터별 개수 확인
-> 이미지 그리는 함수 만들기 
def draw_fruits(arr,ratio=1):   #arr에 3차원배열인 fruits를 입력받음
    n=len(arr)     #layer개수=3차원개수=샘플개수
    rows=int(np.ceil(n/10))  #한줄에 10개씩 그리기/ rows는 세로개수
    cols=n if rows<2 else 10    #만약 1줄이하면 열의 개수는 샘플개수/ 2줄 이상이면 각 열은 10개씩

    fig, axes=plt.subplots(rows,cols,figsize=(cols*ratio,rows*ratio), squeeze=False)
    for i in range(rows):
        for j in range(cols):
            if i*10+j<n:    #i*10+j는 0,1,2,...,298,299->0에서299까지 그리겠다
                axes[i,j].imshow(arr[i*10+j],cmap='gray_r')
            axes[i,j].axis('off')
    plt.show()
하고 draw_fruits(fruits[km.labels_==0]) 이런식으로 집어넣음
-> km.cluster_centers_로 각 클러스터별 평균 확인 -> draw_fruits(km.cluster_centers_.reshape(-1,100,100),ratio=3)로 그림
-> km.transform(fruits_2d[100:101])으로 2차원 형태로 데이터 넣어줘서 해당 샘플의 모든 클러스터 중심까지의 거리 재기
-> km.predict(fruits_2d[100:101])로 레이블 확인하고 그림그리기 -> km.n_iter_로 알고리즘 반복횟수 확인
-> 최적의 k찾기
inertia=[]  : 모든 거리의 합
for k in range(2,7):
    km=KMeans(n_clusters=k, n_init='auto', random_state=123)
    km.fit(fruits_2d)
    inertia.append(km.inertia_)
plt.plot(range(2,7),inertia) 해서 확꺽이는 부분이 최적의 k값

--이미지 말고 다른 데이터--
->fish데이터 사용해서 Kmeans하기-> x_test[:5]를 predict해서 label_값을 y_test[:5]와 비교해보기

----텍스트 데이터----
전처리+tfidf화가 가장 적합

----클러스터로 나뉘는 기준 보는 법-------
original_centers = ss.inverse_transform(kmeans.cluster_centers_)  #cluster_centers_를 복수 
print(pd.DataFrame(original_centers, columns=df.columns[:-1]))  #각 그룹별 중심값이 보여짐

2. 주성분 분석 PCA

from sklearn.decomposition import PCA ->pca=PCA(n_components=50)   #10000차원을 50차원으로 줄임
-> pca.fit(fruits_2d) 2차원으로 들어감 -> pca.components_로 주성분값 50개 확인 -> 3차원화 후 이미지로 나타내기
-> fruits_pca=pca.transform(fruits_2d)로 차원축소 -> 얘를 km.fit하고 label_확인->차원축소전과 비교
-> fruits_inverse=pca.inverse_transform(fruits_pca)로 다시 원본데이터와 유사한 형태로
-> inverse를 3차원화하고 이미지화해서 보기 -> pca.explained_variance_ratio_ 로 주성분이 분산을 얼마나 잘나타내는지보기
-> np.sum으로 1에 얼만큼 가까운지 보기 -> plt.plot으로 초반 몇개의 주성분이 대부분의 분산을 표현하는지보기
-> 최고의 효율나타내는 n_components개수 찾기: pca=PCA(n_components=0.5) 비율로 나타내 np.sum(분산)이 0.5되는 최소의 n_components개수 찾기
  이때 보통 0.9정도로 하기는 함
-> pca.fit(fruits_2d) -> pca.n_components_하면 개수 알려줌 -> 그냥 이 pca 그대로 써도됌 fruits_pca=pca.transform(fruits_2d)
-> 무려 2개의 차원으로도 50%까지 표현할 수 있음 -> 이 fruits_pca로 kmeans에 적용
-> 차원2~3개면 scatter플롯으로도 그리기 가능:
for label in range(0,3):
    data=fruits_pca[km.labels_==label]
    plt.scatter(data[:,0],data[:,1])   #[:,0]은 첫번째 주성분으로 x축에, [:,1]는 두번째 주성분으로 y축에
plt.legend(['banana','apple','pineapple'])
plt.show()

다른 모델에도 적용
정답 y값: answer=np.array([0]*100+[1]*100+[2]*100)   #0은 사과, 1인 파인애플, 2는 바나나
->lr=LogisticRegression()
->scores=cross_validate(lr,fruits_2d,answer) 하고 np.mean(scores['test_score']),np.mean(scores['fit_time'])확인
->pca한 fruit사진도 cv로 ->np.mean(scores['test_score']),np.mean(scores['fit_time']) 확인
-> 정확도는 비슷한데 시간이 확실히 많이 줄어듬
