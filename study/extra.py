
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
-이진분류: 클래스가 두개일때 선형방정식을 하나만 구해 시그모이드를 통과시켜 확률높은 값을 정답으로
-다중분류: 각각의 클래스마다 선형방정식을 구해 소프트맥스함수에 통과시켜 전체합이 1이 되도록하고 그중 가장 큰값을 정답으로
