
tree하고 앙상블말고 전부 정규화 필요
x들어갈때 2차원으로 !!!
선형모델에서 정답과 변수의 상관계수가 절대값0.2 미만이면 안써도 됌-> but!! 트리모델의 경우는 비선형성이므로 사용가능
    by df.corr()
from scipy.stats import randint, uniform, loguniform
from sklearn.preprocessing import StandardScaler, MinMaxScaler

0. **지도 학습**의 검증+최적의 파라미터



1. 지도학습: x,y
  ---#회귀: 연속숫자형x와 연속숫자형 y, 
      #출력함수 따로 없음
  --손실함수: MSE(mean squared error)- np.mean((y-y_pred)**2)
     from sklearn.metrics import mean_squared_error as mse -> mse(y_true,y_pred) 
  --#scoring: 
      .score(x,y)로 R^2계수

  -- from sklearn.neighbors import KNeighborsRegressor
      -#하이퍼파리미터: 
       params={'n_neighbors':randint(1,100)}  
      -dist, index=knr.kneigbors(x_test[:5])

  --from sklearn.linear_model import LinearRegression
      -#하이퍼파라미터 하나: 
        parmas={'positive'=True}  #시 회귀 계수를 양수로 강제(비용, 수량 등 음수가 말이 안 되는 변수에 유용)
      -lr.coef_, lr.intercept_

  --from sklearn.linear_model import Ridge, Lasso
      from sklearn.preprocessing import PolynomialFeatures(degree=n) #사용 후 정규화
      -#하이퍼파라미터: 
        params={'alpha': loguniform(1e-4, 1e2)} #->지수단위로 균등하게
      -rl.coef_, rl.intercept_, rl.coef==0



  ---#분류: x는 연속숫자 및 원핫인코딩-레이블인코딩-bow-tfidf전부 가능
        # y클래스는 정수(레이블인코딩), 문자열, [T/F], [[1, 0, 1], [0, 1, 1]]과 같은 중복클래스를 같는 다중레이블 가능   
        # 출력함수: 이진분류=시그모이드(scipy.special.expit), 다중분류=softmax(scipy.special.softmax)
    --#손실함수: 이진=log_loss(cross-entropy), 다중= Categorical Cross Entropy
        from sklearn.metrics import log_loss
          log_loss(y_true, y_pred) #알아서 이진인지 다중인지 인식
        #트리의 손실함수: 지니불순도, 엔트로피, MSE (트리는 확률기반 모델이 아님)
    --#scoring:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix, log_loss
          function(y_true,y_pred)로 사용
    
    -- from sklearn.neighbors import KNeighborsClassifier
      -#하이퍼파라미터
        params = {'n_neighbors': randint(1, 31)}
      - kn.classes_, kn.predict_proba

    -- from sklearn.linear_model import LogisticRegression
      -#하이퍼파라미터
        params={'C': loguniform(1e-6, 1e+6)}  #큰 규제~작은규제
      - lr.coef_, lr.intercept_, lr.classes_, lr.predict_proba, z=lr.decision_function 

    -- from sklearn.linear_model import SGDClassifier
      -#하이퍼파라미터:
        param_dist = {
          'alpha': loguniform(1e-6, 1e+1),  # 정규화 강도=규제
          'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],  # 손실 함수
          'max_iter': [100, 200, 500, 1000],}  # 최대 반복 횟수 epoch
      -sc.partial_fit, sc.coef_, sc.intercept_, sc.classes_, sc.predict_proba, z=sc.decision_function

    --
2. 비지도학습: only x
  --군집화 
3. 강화학습 
