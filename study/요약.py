--정수인코딩: tokenizer, label encoding, one-hot encoding
    -tokenizer이용한 정수인코딩은 순서가 중요한 딥러닝에 사용-> 높은 빈도수에 낮은 수 할당 !!!!!!!
        열 위치는 중여X, 어떤 값인지가 중요-> 열개수의 최대값은 샘플들 중 단어가 제일 많은 텍스트->나머지 샘플은 패딩으로 채움
    -label encoding은 y의 범주형 데이터에 (ex 나라이름 0~220까지)
    -onehot encoding은 x의 범주형 데이터를 인코딩할때-> 하나의 컬럼을 인위적으로 여러컬럼으로 확장

--텍스트 벡터화: bow, tfidf
    -bow, tfidf는 나오는 단어순서대로 벡터화+그 텍스트에 나오는 단어빈도수로 표현-> 고전적 머신러닝에 사용
        열값이 나오는 단어순서대로 옆으로->단어가 많으면 열도 많아짐

-tree하고 앙상블은 정규화필요X, ss는 주로 머신러닝에, minmax는 주로 딥러닝에

-x들어갈때 2차원으로 !!!

-선형모델에서 정답과 변수의 상관계수가 절대값0.2 미만이면 안써도 됌-> but!! 트리모델의 경우는 비선형성이므로 사용가능
    by df.corr()

-로그 스케일화는 하나의 컬럼안의 값의 차이가 클때 함/ standardscaler는 다양한 컬럼들 사이값의 차이가 크지 않게 하기 위해 함
    ->로그 스케일 후에 정규화(StandardScaler 등)를 함께 사용하는 경우는 꽤 많음

-x값은 레이블인코딩하기 위험한데(값이 클수록 의미를 가지게됨) y값은 레이블 인코딩해도 상관없음
    ->y를 인코딩한다->분류문제다->클래스로 인식되므로 숫자크기에 의미는 없음

@@@@@회귀에서는 선형일때 릿지/라쏘쓰고 비선형일떄는 xgboostingregressor쓰고 
  분류에서는 고차원이면 svm(svc나 linearsvc)쓰고 그 외는 xgboostingclassifier나 histgbc쓰면 얼추 맞음 @@@@@

from scipy.stats import randint, uniform, loguniform, bernoulli
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder

- tfidf-------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
stemmer=PorterStemmer()
stop_words=stopwords.words('english')
def tokenizer(x):
    tokens=word_tokenize(x.lower())
    tokens=[i for i in tokens if len(i)>2 and i not in stop_words]
    tokens=[stemmer.stem(i) for i in tokens]
    return tokens
------------------------------------------

0-1. **지도 학습**의 검증+최적의 파라미터(search는 따로 fit필요)
    -- 0-1-1 from sklearn.model_selection import cross_validate
            score=cross_validate(model,x_train,y_train,cv=5 or 10(fold), return_train_score=True)
        -np.mean(scores['train_score']), np.mean(scores['test_score']), np.mean(score['fit_time'])

    -- 0-1-2 from sklearn.model_selection import GridSearchCV
            gs=GridSearchCV(model,param_grid=params,cv=5 or 10,n_jobs=-1)
        -gs.fit, gs.best_params_, gs.cv_results['mean_test_score'], best=gs.best_estimator_, gs.score, gs.predict

    --0-1-3 from sklearn.model_selection import RandomizedSearchCV
            rs=RandomizedSearchCV(model,param_distributions=params,n_iter=k,n_jobs=-1,random_state=123) # 랜덤하게 파라미터 조합 k개 뽑아 이중 최적찾음
        -rs.fit, rs.best_params_, np.max(rs.cv_results_['mean_test_score']), best=rs.best_estimator_,  rs.score, rs.predict


0-2. PCA(polynomial과 반대->다중공선성 해결): 2차원 배열로 들어감(주로 x값을 넣음)
    eigenvector와 비슷한 개념: eigenvalue값이 가장 큰 순서대로 주성분 우선적으로 가져옴
    -- from sklearn.decomposition import PCA 
            pca=PCA(n_components=개수 or 비율)-> pca.fit(arr) -> arr_pca=pca.trasform(arr)
        - pca.components_, data_inverse=pca.inverse_transform(data_pca), pca.explained_variance_ratio_, pca.n_components_


1. 지도학습: x,y
  --- 1-1 #회귀: 연속숫자형x와 연속숫자형 y, 
      #출력함수 따로 없음
  --손실함수: MSE(mean squared error)- np.mean((y-y_pred)**2)
     from sklearn.metrics import mean_squared_error as mse -> mse(y_true,y_pred) 
  --#scoring: 
      .score(x,y)로 R^2계수

  -- 1-1-1 from sklearn.neighbors import KNeighborsRegressor
            knr=KNeighborsRegressor()
      -#하이퍼파리미터: 
       params={'n_neighbors':randint(1,100)}  
      -dist, index=knr.kneigbors(x_test[:5])

# 선형회귀에서의 정규,스케일링화
    # 모델성능과 안정성이 중요-> 스케일링 필요
    # 해석가능성이 중요(나이, 연봉등 그자체로 의미) -> 안해도 됨
    # 규제가 들어간 회귀(릿지, 라쏘) or 딥러닝 -> 스케일링 필요
  -- 1-1-2 from sklearn.linear_model import LinearRegression
            lr=LinearRegression()
      -#하이퍼파라미터 하나: 
        parmas={'positive'=True}  #시 회귀 계수를 양수로 강제(비용, 수량 등 음수가 말이 안 되는 변수에 유용)
      -lr.coef_, lr.intercept_

--다중공선성 해결법-> 릿지, 라쏘 사용/ 다중공선성= 독립변수들 간에 강한 상호작용이 존재하는것--

  -- 1-1-3 from sklearn.linear_model import Ridge, Lasso
            r=Ridge(), l=Lasso(max_iter=n) #횟수 늘어날수록 warning안뜸
      from sklearn.preprocessing import PolynomialFeatures(degree=2 or 3 많이 사용) #사용 후 정규화/ 다중공선성 문제 발생가능
        poly.get_feature_names_out()
      -#하이퍼파라미터: 
        params={'alpha': loguniform(1e-4, 1e2)} #->지수단위로 균등하게
      -rl.coef_, rl.intercept_, rl.coef==0

  -- 1-1-4 from statsmodels.formula.api import ols  #-> 통계값 볼 수 있음
            df.columns  # ['x1', 'x2', 'x3', 'x4', 'y']
            ols_obj = ols(formula='y ~ x1 + x2 + x3 + x4', data = df)
            model = ols_obj.fit()
            print(model.summary()) # p>|t|기 p-value값-> 0.05보다 크면 변수 안쓰는것도 고려

        -다중공선성 진단
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            exog = ols_obj.exog
            for idx in range(1,k) : # k-1개의 독립변수
                print(variance_inflation_factor(exog, idx)) #이 값이 10보다 큰 변수는 다중공선성 문제가 있다고 판단

# 트리 회귀 모델: 스케일링, 정규화 필요 X, 비선형적+많은 특성에 주로 사용 
    -- 1-1-5 비선형성 관계의 회귀분석에서는 Gradient Boosting Regressor가 가장 널리 사용
        from sklearn.ensemble import GradientBoostingRegressor
             x_train에는 수치나 원핫인코딩(범주형의 경우) y_train에는 레이블인코딩
            gbr = GradientBoostingRegressor(random_state=123)
           -params = {'n_estimators': [100, 200, 300, 400, 500],           # 트리 개수
                      'learning_rate': np.linspace(0.01, 0.3, 30),        # 학습률
                      'max_depth': [3, 4, 5, 6, 7],                       # 트리 깊이
                      'min_samples_split': [2, 5, 10],                    # 내부 노드 분할 최소 샘플 수
                      'min_samples_leaf': [1, 2, 4],                      # 리프 노드 최소 샘플 수
                      'subsample': [0.6, 0.8, 1.0],                       # 샘플링 비율 (과적합 조절)
                      'max_features': ['auto', 'sqrt', 'log2', None]}      # 각 트리에서 사용하는 특성 비율

    -- 1-1-6 from xgboost import XGBRegressor : 캐글 탑 5 회귀모델
            from xgboost import plot_importance # 중요변수 시각화 
                xr= XGBRegressor(objective='reg:squarederror', random_state=123)  
        # 파라미터:
            params= {'n_estimators': [100, 300, 500], #트리개수
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'subsample': uniform(0.7,1)} #학습에 사용할 데이터 비율
                xr(rs).fit(x_train,y_train) # x는 수치나 원핫인코딩, y는 수치 
        # 중요변수 확인 
        xr.feature_importances_
        fscore = xr.get_booster().get_fscore() 
        # 중요변수 시각화 
        plot_importance(xr, max_num_features=13) # 13개까지 나타냄


  --- 1-2 #분류: x는 연속숫자 및 원핫인코딩-레이블인코딩-bow-tfidf전부 가능
        # y클래스는 정수(레이블인코딩), 문자열, [T/F], [[1, 0, 1], [0, 1, 1]]과 같은 중복클래스를 같는 다중레이블 가능   
        # 출력함수: 이진분류=시그모이드(scipy.special.expit)->0.5이상이면 클래스1로 예측/ 아니면 0으로 예측
                  # 다중분류=softmax(scipy.special.softmax)
    --#손실함수: 이진=log_loss(cross-entropy), 다중= Categorical Cross Entropy
        from sklearn.metrics import log_loss
          log_loss(y_true, y_pred) #알아서 이진인지 다중인지 인식
        #트리의 손실함수: 지니불순도, 엔트로피, MSE (트리는 확률기반 모델이 아님)
    --#scoring:
    from sklearn.metrics import accuracy_score, f1_score,confusion_matrix, classification_report, roc_curve, roc_auc_score
          function(y_true,y_pred)로 사용  #classification_report로 정밀도 , 재현율, f1 score 확인가능
        -roc curve그리기: 이진분류모델 성능 평가 
            x,y,N=roc_curve(y_test,y_pred_proba)
            plt.plot(x,y) -> roc밑의 면적=AUC가 클수록 좋음(0.9이상이면 매우 정확) 
                -auc보는 법 print('auc는: ',roc_auc_score(y_test, y_pred_proba))
    -- 1-2-1 from sklearn.neighbors import KNeighborsClassifier
            kn=KNeighborsClassifier()
      -#하이퍼파라미터
        params = {'n_neighbors': randint(1, 31)}
      - kn.classes_, kn.predict_proba

    -- 1-2-2 from sklearn.linear_model import LogisticRegression #원래는 이진분류용
            lr=LogisticRegression(random_state=123) 
      -#하이퍼파라미터
        params={'C': loguniform(1e-6, 1e+6),  #큰 규제~작은규제
                'max_iter':randint(100,1000), #경사하강법 횟수: 대용량이며 5000까지도 함
                'penalty':['l1', 'l2', 'elasticnet', 'none'], #과적합규
                'solver':['liblinear', 'saga']} #사용하는 알고리즘
                multi_class='multinomial' #다중분류도 가능해짐
      - lr.coef_, lr.intercept_, lr.classes_, lr.predict_proba, z=lr.decision_function 

    -- 1-2-3 from sklearn.linear_model import SGDClassifier
            sc=SGDClassifier(loss='log_loss', random_state=123, tol=None, n_jobs=-1)
      -#하이퍼파라미터:
        params = {'alpha': loguniform(1e-6, 1e+1),  # 정규화 강도=규제
          'loss': ['hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron'],  # 손실 함수
          'max_iter': [100, 200, 500, 1000],}  # 최대 반복 횟수 epoch
      -sc.partial_fit, sc.coef_, sc.intercept_, sc.classes_, sc.predict_proba, z=sc.decision_function

    -- 1-2-4 from sklearn.naive_bayes import MultinomialNB
        텍스트 용도로 사용!!!!!-> x는 tfidf형태로 들어감!!!!!(연속형 데이터는 GaussianNB사용-몰라도됌) 
        params = {'alpha': loguniform(1e-3, 1e3),  # alpha는 0.001과 1000 사이에서 loguniform 분포
                  'fit_prior': bernoulli(0.5)}  # fit_prior는 True/False를 균등한 확률로 샘플링
        - nb.predict_proba(x_test), nb.classes_, nb.coef_


    -- 1-2-5 from sklearn.svm import SVC: 적은 샘플, 고차원 데이터(유전자,유전자서열,아미노산서열,효소EC)에 사용
        직선을 그어 클래스를 나눔 
        x는 무조건 수치형 벡터형태(수치 or one-hot-encoding등), y는 범주형 문자나 레이블인코딩 둘다 가능
            # 선형데이터일때
            svc= SVC(C=1.0, kernel='linear')
            # 비선형일때
            svc = SVC(C=1.0, kernel='rbf', gamma='scale') : 저차원을 고차원화해 선형분리가 가능하게끔
            #고차원일때는 고차원화 할 필요가 없으므로 linear로, 저차원은 선형분리하려면 고차원화가 필요해서 kernel='rbf'
        params={'C' : [0.01, 0.1, 1, 10, 100, 1000], # 오차허용큼(일반화용) ~ 오차허용적음(과대적합 위험, 정확도는 좋음)
                'kernel': ['rbf', 'poly','linear']} #비선형시 kernel 사용-> 주로 rbf 

#분류트리: confusion matrix, classification_report, oob_score로 주로 검증증
                                                # oob_score=True하고 model.oob_score_로 자체 평가
# 부트스트랩 샘플링: 전체데이터 셋으로부터 중복해서 데이터 뽑고 추출되지 않은 나머지 데이터셋으로 모델을 검증 
-배깅방식: randomforest, extratree-> 동일한 알고리즘으로 여러 트리모델 만들고 회귀는 각 모델 평균/
    분류는 투표를 통해 각 클래스에 대해 다수결의 트리가 정한 클래스로 분류 -> 과적합에 강함
-부스팅: hist gradient boosting, xgboosting : 현재 모델을 생성하고 얻은 가중치를 다음 모델로 전달해 순차 학습해나감
    -높은 정확도, learning rate존재 

    -- 1-3-1 from sklearn.tree import DecisionTreeClassifier 
            dt=DecisionTreeClassifier(random_state=123)
      -# 하이퍼파라미터:
        params={'min_impurity_decrease':uniform(0.0001,0.001),
       'max_depth':randint(10,50),
       'min_samples_split':randint(2,25),
       'min_samples_leaf':randint(1,25)}
      - dt.feature_importances_, from sklearn.tree import plot_tree(dt,max_depth=k, filled=True, feature_names=['a','b','c'])

    -- 1-3-2 from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
            re=Ran/Ext(n_estimators=k,n_jobs=-1,random_state=123) # k개의 트리-> 500개까지도 가능 or search사용도 가능 
      -# 하이퍼파라미터:
        params={'min_impurity_decrease':uniform(0.0001,0.001),
       'max_depth':randint(10,50),
       'min_samples_split':randint(2,25),
       'min_samples_leaf':randint(1,25),
       'max_features' : ["sqrt", "log2"]} #최대 사용할 x변수 개수-> 분류는 변수개수의 제곱근, 회귀는 변수개수/3 
      - re.feature_importances

    -- 1-3-3 from sklearn.ensemble import GradientBoostingClassifier
            gb=GradientBoostingClassifier(n_estimators=k,random_state=123) # k개의 트리
      -# 하이퍼파라미터:
        params={'min_impurity_decrease':uniform(0.0001,0.001),
       'max_depth':randint(10,50),
       'min_samples_split':randint(2,25),
       'min_samples_leaf':randint(1,25),
       'learning_rate':uniform(0.05,0.2)}
      -gb.feature_importances

    -- 1-3-4 from sklearn.ensemble import HistGradientBoostingClassifier
            hgb=HistGradientBoostingClassifier(random_state=123,max_iter=k) # k개 트리리
        -# 하이퍼파라미터:
            params={'min_samples_leaf':randint(1,25),
                     'max_depth':randint(10,50),
                     'max_leaf_nodes':randint(10,50),
                     'learning_rate':uniform(0.1,0.2)}   #새로운 옵션
            hgc=HistGradientBoostingClassifier(random_state=123,max_iter=200)
        - 중요도 보기: from sklearn.inspection import permutation_importance
            -> result=permutation_importance(hgb,x_train,y_train, n_repeats=10, random_state=123, n_jobs=-1) #n_repeats는 랜덤하게 섞을 횟수/원래 디폴트값은 5
            -> result.importances_mean ->hgb.score, hgb.predict

    -- 1-3-5  from xgboost import XGBClassifier # kaggle에서 5년 연속 1위 븐류모델-> epoch돌때마다 손실값 뜸
              from xgboost import plot_importance # 중요변수(x) 시각화  
            이진분류: xc=XGBClassifier(objective='binary:logistic',eval_metric='logloss') #활성함수 + 평가방법
            다중분류: xc=XGBClassifier(objective='multi:softprob',eval_metric='mlogloss')
                **또한 early_stopping_rounds=n으로 손실값 다시 증가하면 n까지 가기 전에 멈춤 **
        -#하이퍼파라미터:
            params = {'colsample_bytree': [0.5, 0.7, 1], #각 트리 생성 시 사용하는 feature 비율
                      'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3] #값이 낮을수록 학습이 느리지만 일반화 성능 좋음
                      'max_depth' : randint(5,15),
                      'min_child_weight' : [1, 3, 5], #자식 노드 분할을 결정하는 최소 가중치의 합 -> 작으면 더 만흥 자식 노드 분할
                      'n_estimators' : [100, 200, 300,500]} # 트리개수
             xc(rs).fit(x_train,y_train, eval_set=[(x_test,y_test)], verbose=True) #x는 수치,원핫인코딩 y는 레이블인코딩
                                -> eval_set설정하고 verbose=True하면 매 iter마다 손실값(mlogloss)이 뜸 
        -# 중요변수 시각화
            xc.feature_importances_
            xc.get_booster().get_fscore() # 각 클래스별 fscore 보여줌 
            plot_importance(xc)->plt.show() # 중요변수 시각화

2. 비지도학습: # only x: 정형데이터- 수치형or 원핫인코딩, 텍스트-tfidf, 이미지-벡터형태를 2차원화 
               # 출력함수 없음, 손실함수도 딱히 없음 (KMeans에서는 inertia의 최소값) 
   -이미지 그리는 함수:
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



  --- 2 #군집화(비계층적) 2차원 데이터로 x값만, 정규화 필요  
!!!! 군집수 찾기: 이너시아 그래프 그리고 팔꿈치 꺽이는 부분
    -> 이때 계속 꺽인다면 기울기 감소의 크기가 줄어드는 지점 바로 전꺼로
        ex) 1000->600->400->350 이면 400지점인 k=3  !!!!

    -- 2-1 from sklearn.cluster import KMeans
            km=KMeans(n_clusters=k, random_state=123, n_init=10,max_iter=300)  #search불가 
            #-> km.inertia_ 엘보우 굽어질때를 최적의 k로
        - km.labels_, km.cluster_centers_, km.inertia_, km.transform(f_2d[100:101)), km.n_iter_

    -- 2-2 계층적 군집화: 나누고 덴드로그램 그려 밑에서부터 올라가면서 자를 높이k나 클러스터수k를 결정
        #넘파이, 데이터프레임형태로도 들어감, 주로 표준화해서
        from scipy.cluster.hierarchy import linkage, dendrogram, fcluster 
        clusters = linkage(df, method='single' or 'complete','average','centroid') # 일단 여러개로 나눔
            -single: 데이터 연속적일때, -complete: 군집간 명확한 구분이 필요할때
            -average: 일반적, 균형잡힌, -centroid: 데잍어가 구형분포일때 
            -고차원이면 'ward' 나 'average', -일반적으로는 'single', 'complete' 
        dendrogram(clusters), plt.show()로 덴드로그램으로 나타냄 
            import sys -> sys.setrecursionlimit(10000) # 재귀 깊이 한도 늘려 많은 샘플도 볼 수 있게 
        cut_cluster = fcluster(clusters, t=n, criterion='maxclust' or 'distance') #클러스터개수n or 덴드로그램높이n
        print(cut_cluster)하면 각 샘플별 1~k개의 labels로 표현  -> 얘를 scatter의 c로 하면 색다르게 산점도가능 
            


-이때 한 군집의 개수가 너무 적으면 그 군집은 무시할수도 있음
-시각화하기: df['cluster']=km.labels_ or cut_cluster로 df에 추가
            g=df.groupby('cluster') 이후 g.mean()으로 각 그룹별 특징보기
                    ->g.mean().plot()으로 시각화
    -or a변수에 대해 상관관계 제일높은 변수 b로 plot.scatter(a,b,c=df['cluster']) 



-- 2-3 연관분석(비지도학습): 항목이나 사건 간의 연관성(관계)를 찾는 방법 
            관련분야: 마트, 무역, 마케팅 
    연관규칙 평가척도:
        - 지지도(support) - 동시 구매패턴보기: 상품A,상품B 동시 거래수(A ∩ B) / 전체거래수
        - 신뢰도(confidence) - A구매시 B구매패턴:  A와 B를 포함한 거래수(A ∩ B) / A를 포함한 거래수
        - 향상도(lift) - 상품A와 상품B 간의 상관성: 신뢰도 / B가 포함될 거래율
                향상도>1: 양의 연관성, 향상도=1: 서로 독립(아무관계X), 향상도<1: 음의 연관성(반비례) 

#트랜잭션 데이터로 변환 필요 : 트랜잭션=user(10) vs 아이템-고유값(3)
User Item        Item bread butter milk        Item bread butter milk
                 User                          User
 1 milk           1    1.0   0.0    1.0         1    True False True
 1 bread          2    1.0   1.0    0.0         2    True True False
 2 bread          3    0.0   1.0    1.0         3    False True True
 2 butter         4    1.0   0.0    0.0         4    True False False
 3 milk           5    0.0   1.0    1.0         5    False True True
 3 butter    ->   6    1.0   0.0    0.0   ->    6    True False False
 4 bread          7    0.0   1.0    0.0         7    False True False
 5 milk           8    0.0   0.0    1.0         8    False False True
 5 butter         9    0.0   0.0    1.0         9    False False True
 6 bread          10   1.0   0.0    0.0         10   True False False
 7 butter
 8 milk
 9 milk
 10 bread

# 1. sample data 생성  
data = {'User': [1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 7, 8, 9, 10],
        'Item': ['milk', 'bread', 'bread', 'butter', 'milk', 'butter',
                 'bread', 'milk', 'butter', 'bread', 'butter', 'milk',
                 'milk', 'bread']}
# 데이터프레임 생성
df = pd.DataFrame(data)
# 2. 트랜잭션(transaction) 데이터 만들기 : One-Hot Encoding 변환
group = df.groupby(['User','Item']) 
transaction = group.size().unstack().fillna(0) # 결측치 0 채우기 
# 부울형(True/False) 변환  
transaction = transaction.astype(bool) # 위의 맨 오른쪽 df 

    from mlxtend.frequent_patterns import apriori, association_rules  
      # 지지도(1차)로 아이템선택: 각 아이템별 2개씩 짝지어(스스로도 포함가능) support값 보여줌 
      frequent_itemsets = apriori(transaction, min_support=0.1, max_len=5, use_colnames=True) #최소 support 0.1이상만 보여줌
      # 연관 규칙 생성 : 신뢰도(2차) 기준  ->frequent_itemsets의 각 튜플값 별 jaccard  certainty  kulczynski 값
      rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2) 
      # 다양한 평가 지표들 보기
      final_rules = rules[['antecedents','consequents','support','confidence','lift','zhangs_metric','jaccard']]
'''
antecedents  consequents : 연관 규칙의 선행사건(A)과 후행사건(B) 항목   예) (bread) => (butter) : 빵을 구매한 사람이 버터를 구매할 가능성
antecedent support : 선행사건(A) 발생한 비율 = A / 전체거래수    예) bread = 5 / 10 = 0.5 
consequent support : 후행사건(A) 발생한 비율 = A / 전체거래수     예) butter = 4 / 10 = 0.4  
support : 지지도 = (A ∩ B) / 전체거래수        예) (bread) => (butter) : = 1 / 10 = 0.1 
confidence : 신뢰도(조건부 확률) = (A ∩ B) / LHS 거래수      에) (bread) => (butter) : = 1 / 5 = 0.2 
lift : 향상도(두 항목의 독립성을 고려한 연관관계 강도) = confidence / 지지도(RHS)    예) (bread) => (butter) : 0.2 / 0.4 = 0.5 
zhangs_metric : Zhang의 메트릭, 향상도(left) 평가 지표(left 평가 지표 : -1~1)
jaccard : LHS와 RHS의 교집합을 전체 합집합에 대한 비율(두 항목의 유사도 : 0~1) 
'''

# 연관규칙 시각화
import networkx as nx
def plot_rules(rules, weight, top_n=10): # (rules, n)    
    rules = rules.nlargest(top_n, 'confidence') # 신뢰도 기준, 상위 n개  
    G = nx.DiGraph() # 네트워크 그래프 

    for _, rule in rules.iterrows():
        G.add_edge(rule['antecedents'], rule['consequents'], # 노드 : 타원  
                   weight=rule[weight]) # 엣지 : 가중치

    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 6))
    nx.draw_networkx(G, pos, with_labels=True, node_color="lightblue", node_size=3000, edge_color="gray")
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.title(f"Association Rules Graph : {weight}")
    plt.show()
# 함수호출 
plot_rules(final_rules, 'lift') # 두 항목의 연관관계 지표(0~1)  
plot_rules(final_rules, 'zhangs_metric') # lift 연관관계 평가 지표(-1 ~ 1)
plot_rules(final_rules, 'jaccard') # 두 항목 간의 유사성 측정 지표(0 ~ 1)



Item bread butter milk
User
1 True False True
2 True True False
3 False True True
4 True False False
5 False True True
6 True False False
7 False True False
8 False False True
9 False False True
10 True False False
3. 강화학습 



-------------------------------------------------
회귀분석용 데이터
from sklearn import datasets

--iris = datasets.load_iris()
x = iris.data # x변수 
y = iris.target # y변수
print(iris.feature_names)# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
print(iris.target_names) # ['setosa' 'versicolor' 'virginica']
iris_df = pd.DataFrame(x, columns=iris.feature_names) # X변수 대상 
iris_df['species'] = iris.target #y변수 추가

--diabetes = datasets.load_diabetes()

--from sklearn.datasets import fetch_california_housing
california = fetch_california_housing()
print(california.DESCR)

# X변수 -> DataFrame 변환 
cal_df = pd.DataFrame(california.data, columns=california.feature_names)
# y변수 추가 
cal_df["MEDV"] = california.target

분류용 데이터
-from sklearn.datasets import load_wine
wine = load_wine()

-from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

-from sklearn.datasets import load_digits #숫자이미지
digits = load_digits()

-from sklearn.datasets import fetch_20newsgroups
newsgroups = fetch_20newsgroups(subset='all') # 'train', 'test'
