##### `Integrating Offline Reinforcement Learning with Transformers for Sequential Recommendation`

---

- `architecture`


      # 2 phase train

        phase 1 : train policy with transformer
        phase 2 : train actor-critic, initialize policy with pre-trained with phase 1.

---

- `idea`

        offline RL 과 transformer 를 합친 추천 시스템 제안.
        환경이랑 interaction 했던 걸 바탕으로 해서 다음 action 을 추천하는 그런 패러다임.
          
        "This recommendation task requires efficient processing of the sequential data
        and aims to provide recommendations that maximize the long-term reward. To this end, we train a far-
        sighted recommender by using an offline RL algorithm with the policy network in our model architecture
        that has been initialized from a pre-trained transformer model."


        "Sequential supervised-learning" -- 이거 완전 offline RL 이자너..

        agent       - recommander (recommand items)
        environment - User (return feedback)

        state  - most recent positive items with a fixed length
          -> rating이 positive일 때만, state transition 이 발생하게 했다. 
      
        action - item (large item space - discrete -> continuous with item embedding)

        

        goal : good recommanding policy 



- `my idea`


        주식은 Cycle 이 있다.
        섹터 별로 평균값을 취하고, 현재 어느 섹터에서 노는게 좋을 지 예측하는 MDP는 어떨까?
        조선 -> 조선 -> 반도체 -> 배터리 -> 반도체 -> 로봇  ...  이렇게?

  

        
    
      
