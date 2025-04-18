
# 1.regression

1. pandas->numpy는 df.to_numpy()
  numpy->pandas는 pd.Dataframe(arr)

2. 다항회귀에서 lr.coef_의 출력 순서는 들어간 열(변수) 순서대로 출력

3. 머신러닝에서 정규화
   x값은 변수가 여러개 존재할 수 있으므로 정규화가 필요하지만
   y값은 대부분 정규화필요X  (모델이 자동으로 최적의 가중치와 절편을 학습)
 !예외: y가 매우크거나 작을때 or 딥러닝에서 y 정규화
