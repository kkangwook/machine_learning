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


from scipy.stats import randint, uniform, loguniform, bernoulli
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


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


    -- 1-3-1 from sklearn.tree import DecisionTreeClassifier 
            dt=DecisionTreeClassifier(random_state=123)
      -# 하이퍼파라미터:
        params={'min_impurity_decrease':uniform(0.0001,0.001),
       'max_depth':randint(10,50),
       'min_samples_split':randint(2,25),
       'min_samples_leaf':randint(1,25)}
      - dt.feature_importances_, from sklearn.tree import plot_tree(dt,max_depth=k, filled=True, feature_names=['a','b','c'])

    -- 1-3-2 from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
            re=Ran/Ext(n_estimators=k,n_jobs=-1,random_state=123) # k개의 트리
      -# 하이퍼파라미터:
        params={'min_impurity_decrease':uniform(0.0001,0.001),
       'max_depth':randint(10,50),
       'min_samples_split':randint(2,25),
       'min_samples_leaf':randint(1,25)}
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

  --- 2 #군집화 2차원 데이터로 x값만
    -- 2-1 from sklearn.cluster import KMeans
            km=KMeans(n_clusters=k, random_state=123, n_init=10)  #search불가
            #-> km.inertia_ 엘보우 굽어질때를 최적의 k로
        - km.labels_, km.cluster_centers_, km.inertia_, km.transform(f_2d[100:101)), km.n_iter_



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
