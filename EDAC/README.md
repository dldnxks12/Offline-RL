### Uncertainty-based offline RL - EDAC 

---
    
    
    BCQ, CQL 처럼 ood action 에 대한 penalty 를 주지 않는다.
    다만 앙상블을 통해 각 Q-value의 uncertainty 정도를 계산하고 이용한다.

    clipped-Q-learning을 uncertainty-based penalization term 에 이용.

    Diversifying objective ~ 를 통해서 필요한 앙상블 모델 개수를 줄임.
    

    # Check point 1
        - 어떤 식으로 ood action을 처리하는 지 알아보자.
    
    # Check point 2
        - TD3+BC 의 episodic variance 문제를 해결할 수 있는지 확인해보자. 


