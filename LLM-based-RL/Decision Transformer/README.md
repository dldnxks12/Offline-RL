### Decision-Transformer `2021`

---

        * 일반적인 TD 또는 DP 형식이 아니라, Auto-regressive Sequential modeling 을 통해 RL 문제를 해결.

        Credit Assignment 문제 해결 

            - self-attention 덕분에 given trajectory 에서 집중할 부분을 캐치

        So, DT는 CAP 문제가 빈번한 Sparse reward 환경에서 잘 작동할 수 있다.


        * random 한 trajectory 데이터들이 주어져있을 때, 이 경로들을 알맞게 stitching 해서
        새로운 최단 경로를 찾아내는 것.

        * COG 에서는 TD의 backpropagating 을 통해 stitching 의 가능성을 보여줬다면,

        Decision Transformer에서는 Sequential Modeling을 통해 stitching을 구현.

---

        1. offline dataset 이 given
        2. dataset에서 sequence length K의 mini-batch sampling
        3. state st 에서 action at를 예측하도록 학습 
        
        mini-batch sample 들을 학습하면서 ,각 node 들에서 가능한 가장 큰 return을 
        생성한다. -> Decision Transformer 는 optimal-path 를 생성한다. 


