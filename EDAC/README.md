### Uncertainty-based offline RL - EDAC 

---
    
        # Q1 ) Why uncertainty base? 
        # A1 ) 
            기존에는 ood action 을 무조건 걸러내는 방향으로 발전해왔는데, 
            High confidence 로 ood action 에 대한 Q value 를 계산할 수 있다면, ood action 이라고 무조건 무시하기보다
            이것도 잘 사용해보아야하지 않겠느냐...!
        
            따라서 EDAC는 BCQ, CQL 처럼 ood action 에 penalty 를 주지 않는다.
            다만 각 Q-value의 uncertainty 에 따라 penalty 를 준다.
                - Clipped Q trick 이 uncertainty 에 따라 Penalty 를 줄 수 있다고 한다.
                
                    - 기존에 Clipped Q trick을 쓴 방법들은 단순히 Q value 를 조금 더 pessimistic 하게 사용하는데에 그쳤는데, 
                    Q function 개수를 늘려서(ensemble), Q value sample 간의 variance 에 따른 penalty 를 주는 방법을 생각해냈다!
                
            OOD action 은 Q value prediction sample 들 간에 variance - uncertainty가 크다.
            따라서, OOD action에 implicit 하게 penelty를 줄 수 있다!!

    
        # Q2 ) OOD action sample 들은 왜 variance / uncertainty 가 큰가?
        # A2 ) 
            dataset 내부의 sample 들은 eqn(2)에 의해서 explicit 하게 어떤 target value로 점점 align된다.
            그래서 Model 1에서 뽑은 Q(s1, a1) 과 Model 2 에서 뽑은 Q(s1, a1) 사이의 차이가 별로 크지 않다.
            하지만 OOD action에 대해서는 이러한 과정이 없기 때문에, 각 모델에서 뽑은 Q value 간의 variance가 크다. 
            

---

        # Q3 ) Clipped-Q trick 이 어떻게 high-variance 관련 penalty 를 줄 수 있어?
        # A3 ) 

            Clipped-Q 를 활용하는 건 Q value 예측의 Confidence bound 를 고려한 방법과 관련이 있다. 
            일반적으로 Online 에서는 ensemble 을 통해서 얻은 Q value들의 평균에 std(양수)를 더해서 optimistic Q value esimation              한다. 
            이 Optimistic Q-value 를 UCB라고도 하고, exploration 을 더 활발히 하도록 하는데 자주 사용된다. 

            반대로 offline RL에서는 exploration 보다는 주어진 static dataset 을 최대한 exploitation하는 게 주 목적이다. 
            따라서, 같은 상황에서는 risky 한 state-action 을 회피하기 위해서 ensemble을 통해 얻은 Q value 들의 평균에 std를 뺀다.
            즉, Pessimistic Q value 또는 LCB 를 사용한다. 

            Clipped Q trick 은 여러가지 모델에서 뽑은 Q value 중에서 가장 value 가 낮은 값을 선택하는데,
            이 방법이 LCB 와 관련이 있다.

            예를 들어, Q value 들이 평균이 m이고 std가 simga인 gaussian 분포를 따른다면, 다음과 같은 식이 성립한다.
            즉, 평균에서 N에 dependant한 상수에 비례한 std를 빼는 LCB 로의 역할을 수행할 수 있다.
                        
            

---
    
        # EDAC

        SAC-N 은 단순히 online algorithm 에 Clipped Q trick 만 사용했음에도 Model 수만 충분히 늘린다면 CQL 보다 좋은 성능을 보여줌.

        하지만 특정 task 에 대해서는 model 수가 굉장히 많이 필요한 경우가 있다.
        
            - Q value sample 들의 gradient 가 align 되있는 정도와 SAC-N 의 performance 가 음의 상관관계가 있고, 
            이게 결국은 모델 수를 늘리게 되는 원인
        
            논문의 figure 4를 통해서 알 수 있 듯이, Q value 들의 gradient 간 코사인 유사도를 낮추기 위해서 모델 수를 늘려야한다.     
            또한 코사인 유사도 값이 작아짐에 따라서, Performance가 좋아지는 것을 확인할 수 있음.

            Q value 간의 variance 를 기준으로 이야기 했었는데, Q value 의 gradient 와의 관계가 조금 의아할 수 있음.
                - 간단히 1차 taylor 근사를 통해 Q value 들의 variance 와 gradient 들 간의 variance 와의 관계가 있음을 알 수 있다.
    
        학습된 Q function 들이 비슷한 local structure 를 가지고 있을 때 policy 성능이 급격히 저하된다.


        
    


