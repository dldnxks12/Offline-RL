### TD3 + BC 

---

        F-BRC 이랑 같거나 조~금 더 나은 성능을 보여준다.
        근데 주목할 점은.. 진짜 TD3에 딱 한 줄 더 쓴 것 뿐.
        Computational cost도 F-BRC에 비해 절반정도.. 

        진짜 간단하게 구현해도 기존의 Offline RL 의 성능을 낼 수 있다는 것이 큰 매력. 

        1. BC term 추가
        2. state 정규화 

        끝.


---
     
        
        학습된 policy 의 불안정성 :
            offline RL 들의 학습이 완료된 policy 를 분석했는데 episode 마다 성능에 variance가 심하다.
    
            즉, average performance 는 좀 합리적이라도 몇몇 episode 에 대해서는 성능이 개똥이다. 
    
        -> offline Benchmark 할 때, mean-value 에만 치중하는 게 문제가 좀 있다.
            -> episode 마다 variance 크니까 ..
    
        뭐 다른게 아니고 저자가 추측하기에 이 문제는 distributional shift 문제고, (unseen action) 
        unobserved state 에 대한 일반화 문제이다. (unseen state)
     
        자기들이 쓴 TD3+BC 에서도 이 문제가 있고, 앞으로 해결해야할 중요한 문제같다. 
    
            -> 이 부분을 내가 해결할 수 있는지 확인해보자... 
            
