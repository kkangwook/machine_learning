분류: 
-------유전자서열에 따른 클래스
sequences = [
    "ATCGTACG",  # 클래스 0
    "CGTACGTA",  # 클래스 0
    "GCGTGCAT",  # 클래스 1
    "TACGTAGC",  # 클래스 1
]
labels = [0, 0, 1, 1]  # 클래스 레이블
데이터 들어갈때 k-mer(하나 샘플의 유전자에서 가능한 모든 부분유전자(길이k))화 : 주로 k는 5~7로 많이함 (4^6=4096차원)
-> countervectorizer를 주로 사용해 벡터화 -> svm사용(적은 샘플수+고차원에 유리): 고차원이면 linear, 저차원이면 rbf


k-mer함수만들기
def kmer_split(seq, k=3):
    return ' '.join([seq[i:i+k] for i in range(len(seq)-k+1)])

모든 샘플 k-mer화
corpus = [kmer_split(seq, k) for seq in sequences]

bow=CountVectorizer().fit_transform(corpus)  


-------아미노산도 가능
sequences = [
    "MVKVYAPASSANMSVGFDVLGAAVTPVDGALLGDVVTVEAAETFSLNNLGQK",  # class 0
    "MELPDVIKGIEVVVADGLSTVQDYAK",                            # class 0
    "MSLLTEVETPIRNEWGCRCSDDRA",                              # class 1
    "MGSSHHHHHHSSGLVPRGSHMSLLTEV",                           # class 1
]
labels = [0, 0, 1, 1]
똑같이 k-mer화(주로 k=2~3: 8000차원) -> countervectorizer -> svm (주로 linear) 


ncbi의 sra에는 개인이 측정해서 올린 유전자서열이 존재-> 다양한 강아지 품종 존재 
-> Canis lupus familiaris breed치면 이름나옴 ->들어가서 sample의 samn~링크 누르면 종 정보

dog10k로 종별 염색체, 유전자등 볼 수 있음

종별 유전자비교: 의미없는 유전자까지 다 비교(의미 없는 유전자에도 종에따른 핵심적인 변이가 있을 수 있음)
종별 기능성 유전자 중심 연구: cds만으로 비교
