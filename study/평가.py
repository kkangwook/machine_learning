# 머신러닝: 주로 train/test만으로 나누고 crossvalidate으로 확인
  -> 이때 grid나 randomizer_search에서 cv설정해주면 cv로 검증해서 최고 모델 return하고 .cv_result['mean_test_score']로 crossvalidate값 확인가능=> cv따로 해줄필요 x
-또한 xgboost같은경우는 eval_set넣어서 epoch마다 eval_set로 검증해서 최적 모델 return

# 딥러닝: train/eval/test로 나눠서 fit할때 eval_set에 집어넣어 eval_set 평가 점수에따라 학습



# 회귀분석
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 로 평가
-> 이때 회귀.score()는 r2값 : 1에 가까울수록좋음(ssr/sst==1-sse/sst)

y_pred=lr.predict(x_test)
print(mean_squared_error(y_test,y_pred))



# 이진분류
.score()는 정확도(accuracy)

from sklearn.metrics import accuracy_score,classification_report, roc_auc_score, confusion_matrix

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:,1]  # Positive 확률

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred)) # precision, recall, f1-score
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred)) # 혼동행렬


# 다중분류
.score()는 정확도(accuracy)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))  
print(confusion_matrix(y_test, y_pred))
print("ROC-AUC (OvR):", roc_auc_score(y_test, y_proba, multi_class="ovr"))  # 다중시 'ovr'로

########################################## classification_report ############################
각 클래스별로 값나옴 (0.8 이상이면 보통 좋다고 평가)
-precision(정밀도): 높아야함(1에 가깝게)-Positive라고 예측한 것 중 실제 Positive인 비율
-recall(재현율): 높아야함(1에 가깝게)-실제 positive중 모델이 positive라 예측한 비율
-f1-score: 높아야함(1에 가깝게)-정밀도와 재현율의 조화평균
-support: 해당 클래스 샘플 개수-> 가중치 계산에 사용
-평균들: macro avg ≈ weighted avg ≈ accuracy → 불균형 없이 잘 맞춘다는 의미



#군집화
from sklearn.metrics import silhouette_score
score = silhouette_score(X, labels) # x는 행렬값, labels는 군집화 예측결과 

Silhouette Score	의미
0.7 ~ 1.0	매우 잘 분리된 군집, 좋은 클러스터링
0.5 ~ 0.7	군집 구조가 어느 정도 있음, 준수한 클러스터
0.25 ~ 0.5	군집이 모호함, 데이터가 잘 섞임
<0.25	군집화 실패, 재검토 필요
