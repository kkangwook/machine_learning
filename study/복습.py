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
