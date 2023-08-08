### Uncertainty-based offline RL - EDAC 

---
    
    
        기존에는 ood action 을 무조건 걸러내는 방향으로 발전해왔는데, 
        High confidence 로 ood action 에 대한 Q value 를 계산할 수 있다면, ood action 이라고 무조건 무시하기보다
        이것도 잘 사용해보아야하지 않겠느냐...!
    
        따라서 EDAC는 BCQ, CQL 처럼 ood action 에 penalty 를 주지 않는다.
        다만 각 Q-value의 uncertainty 에 따라 penalty 를 준다.
            - Clipped Q trick 이 uncertainty 에 따라 Penalty 를 줄 수 있다고 한다.
            - Clipped Q trick 에 들어가는 Q 의 개수를 단순히 늘려서 성능을 많이 높일 수 있다. - Ensemble !
            
        OOD action 은 Q value prediction 에 uncertainty가 크다.
        따라서, OOD action에 implicit 하게 penelty를 줄 수 있다!!
    
        # Q1) OOD action 은 왜 uncertainty 가 큰가?
        # A1)
    
---


        # Q2) OOD action 은 왜 variance가 큰가? 


---
    
        Diversifying objective ~ 를 통해서 필요한 앙상블 모델 개수를 줄임.
    
        # Q3) Variance 에 관한 이야기와 물음 재정리 
    


