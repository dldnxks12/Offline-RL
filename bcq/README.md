### Batch-Constraint Deep Q-Learning (BCQ) `2019`

---

    해당 논문은 off-policy로 인해 발생하는 extrapolation error를 극복할 수 있는 BCQ를 제안한다.
    
    - first continuous Deep RL algorithm!

    *keyword : Batch RL / Off policy / Extrapolation Error / VAE

---

- `Offline RL (Batch RL)`

        보통 강화학습은 환경과의 상호작용을 통해 새로운 경험을 얻어내고, 이 경험들을 모아 정책을 발전시킨다. 
        하지만, Offline RL은 사전에 미리 얻어둔 '고정된 데이터셋'을 이용하여 정책을 향상시키는 형태이다.
        
        고정된 데이터셋을 이용하여 정책을 향상시키는 데에는 여러가지 이점이 있다.

            1. Stable learning
            2. Sample efficiency
            3. Exploration Overhead
 
        하지만, 추가적인 exploration이 없기에 주어진 데이터셋에 대해 best policy를 찾을 뿐이다.
        즉, 추가적인 성능을 얻어낼 요소가 부족하다. 우리는 주어진 환경에서의 optimal policy를 찾고 싶다.

---

- `Extrapolation Error`        
        
        Offline RL은 보통 행동 정책과 타겟 정책이 다른 'off-policy' 형태이다.
        (학습하는 데이터 : batch dataset | 환경에서 얻는 데이터 : policy를 통해 얻는 데이터셋)

        off-policy 환경은 sample efficiency를 이유로 많이 선택된다.  
        행동 정책과 타겟 정책이 다른 상태에서 조심스럽게 학습된다면 큰 문제 없이 학습이 가능하다. 

        하지만, off-policy의 특성상 학습에 사용하는 데이터의 분포와 실제 환경에서 얻어내는 데이터의 분포 간의 상관관계가 떨어질 수 있다.
        이로 인해 발생하는 문제를 'Extrapolation Error' 라고 한다.
     
           'Extrapolation Error' : 학습 때 경험하지 못한 State-Action pair가 나타날 경우 비현실적으로 값을 잘못 예측하는 것 

        implicit하게 off-policy의 형태를 띄는 Offline RL의 경우 당연히, 주어진 데이터 분포와 Policy에 의해 얻는 데이터 간의 차이로 EE가 발생한다.
        결국, Batch 내애 포함되어 있지 않은 state-action에 대해서는 value function을 제대로 학습할 수 없다. 

    
        논문에서는 EE가 발생할 수 있는 요소를 크게 3가지로 말한다.

            1. 데이터의 부재
            2. 모델 bias
            3. Training mismatch


---

- `논문 정리`


        #1 : 
          Fixed dataset 환경에서는 extrapolation error 때문에 기존의 DQN, DDPG같은 알고리즘을 사용하기 어렵다.
          해당 논문은 어떠한 fixed dataset에 대해서도 잘~ 동작하는 Continuous Deep RL 알고리즘을 제안한다. (BCQ)

        #2 :
          Imitation Learning의 경우 suboptimal trajectory의 데이터에 노출될 때 학습이 실패함이 알려져있다.
            -> 즉, 주어지는 데이터의 질이 중요하다.

          Offline RL의 경우 데이터의 quality에 제한없이 고정된 데이터셋에서 잘 학습하는 메커니즘을 제공한다.  

----

        #3 :
          대부분의 현대 off-policy 심층 강화학습 알고리즘은 growing batch RL 범주에 들어간다.
            -> Experience replay 가 fixed dataset과 개념상으로는 비슷 

          하지만 off-policy 학습은 다음의 조건
            1. Behavior agent가 학습하는 데이터셋이 고정되어있고 (Batch setting)
            2. Current policy가 환경에서 얻어내는 데이터의 분포가 batch 데이터의 분포와 uncorrelated
              (distribution mismatch)
          일 때, 학습은 대부분 실패하게 된다. ----- 즉, Extrapolation Error가 중요한 쟁점이다!

        #4 : 
          off-policy의 extrapolation error를 해결하기 위해서 BCQ를 제안

          1. Agent는 reward를 최대화한다.
          2. 동시에 Batch 내에 있는 state-action pair와 policy로 인한 state-action 간의 mismatch를 최소화한다.
             (이전에 seen 했던 action 만을 만들어내기 위해 state-conditioned 생성모델을 사용한다. with VAE)

          BCQ는 EE를 고려하므로, 기존 continous control deep RL과 달리 환경과 interaction 없이도 잘 학습된다!!!

----

        #5 :
          finite deterministic MDP에서 이렇게 불완전한 데이터셋으로 unbiased value estimation을 하기 위해서는
          이 논문이 제시하는 방법이 아주 necessary하다는 걸 증명 (Under mild assumption)

        #6 :
          EE factors

          *Absent Data : fixed dataset 안에 없는 s-a pair에 대해 학습할 때. 적어도 비슷한 s-a pair라고 있으면 error가 덜하다.

          *Model Bias  : fixed dataset을 이용한 off-policy Q learning에서 expectation을 stochastic하게 approximation 한다.
           이 때, 무한한 state-action visitation 없이는 transition dynamics에 error를 가져온다. (In stochastic MDP)

          *Training Mismatch : Deep Q learning에서 충분한 데이터가 있다하더라도, transition들은 데이터셋에서 균일하게 sampling 된다.
            그리고 fixed dataset 내에서 그 transition이 나올 가능성 만큼 가중된 loss를 준다.
              -> 즉, 1번 transition의 loss가 10이라고 할 때, 이 데이터가 batch 내에 10 개가 있다면 100의 loss를 준다.
              -> 즉, 2번 transition의 loss가 5 이라고 할 때, 이 데이터가 batch 내에 100개가 있다면 500의 loss를 준다.

              그래서 만약 batch 내에 transition들의 분포가 실제 current policy로 얻어낸 transition들의 분포와 다르다면, 
              value estimation은 poor estimation이 된다.  

              그럼 batch 내의 각 transition의 분포를, current policy의 분포를 고려해서 가중치를 잘 조정하면 ?
                -> high-likelihood를 가진 current policy의 transition이 batch dataset 내에 없다면 똑같이 poor esimate한다.

              즉, 결국 이 세 번째 factor도 데이터 부재와 연결되는 문제!

----



        # 7 : 
          
