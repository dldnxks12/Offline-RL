### Batch-Constraint Deep Q-Learning (BCQ) `2019`

---

    해당 논문은 off-policy로 인해 발생하는 'Extrapolation Error'를 극복할 수 있는 BCQ를 제안한다.

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
        --- 즉, 내가 가지고 있는 data가 환경에서 current policy가 얻어내는 data랑 얼마나 관련이 있나?..
        
        이로 인해 발생하는 문제를 'Extrapolation Error' 라고 한다.

           'Extrapolation Error' : 학습 때 경험하지 못한 State-Action pair가 나타날 경우 비현실적으로 값을 잘못 예측하는 것 

        off-policy의 형태를 띄는 Offline RL의 경우, 
        주어진 데이터 분포와 current policy에 의해 얻어지는 데이터 간의 상관관계가 적다면 EE가 발생한다.
        결국, Batch 내애 포함되어 있지 않은 state-action에 대해서는 value function을 제대로 학습할 수 없게 된다.

---

### `논문 정리`

- `intro`

        #1 : 
          Fixed dataset 환경에서는 extrapolation error 때문에 기존의 DQN, DDPG같은 알고리즘을 사용하기 어렵다.
          해당 논문은 어떠한 fixed dataset에 대해서도 잘~ 동작하는 Continuous Deep RL 알고리즘을 제안한다. (BCQ)

        #2 :
          Imitation Learning의 경우 suboptimal trajectory의 데이터에 노출될 때 학습이 실패함이 알려져있다.
            -> 즉, 주어지는 데이터의 질이 중요하다.

          Offline RL의 경우 데이터의 quality에 제한없이 고정된 데이터셋에서 잘 학습하는 메커니즘을 제공한다.  

----

- `background`

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

        #5 :
          finite-horizon deterministic MDP에서 이렇게 불완전한 데이터셋으로 unbiased value estimation을 하기 위해서는
          이 논문이 제시하는 방법이 아주 necessary하다. (Under mild assumption)

----

`Extrapolation Error`

        #6 : EE factors

          논문에서는 EE가 발생할 수 있는 요소를 크게 3가지로 말한다.

            1. Absent Data
            2. Model Bias
            3. Training Mismatch


          *Absent Data : fixed dataset 안에 없는 s-a pair에 대해 학습할 때. 적어도 비슷한 s-a pair라고 있으면 error가 덜하다.

          *Model Bias  : fixed dataset을 이용한 off-policy Q learning에서 expectation을 stochastic하게 approximation 할 때,
           충분히 많은 state-action visitation 없이는 transition dynamics에 error를 가져온다. (In stochastic MDP)
           --- 즉, Expectation의 approx.에 사용되는 sample mean을 위해 충분히 많은 sample이 없다..

          *Training Mismatch : Deep Q learning에서 충분한 데이터가 있다하더라도, transition들은 데이터셋에서 균일하게 sampling 된다.
            그리고 fixed dataset 내에서 그 transition이 나올 가능성 만큼 가중된 loss를 준다.
              -> 즉, 1번 transition의 loss가 10이라고 할 때, 이 데이터가 batch 내에 10 개가 있다면 100의 loss를 준다.
              -> 즉, 2번 transition의 loss가 5 이라고 할 때, 이 데이터가 batch 내에 100개가 있다면 500의 loss를 준다.

              그래서 만약 batch 내에 transition들의 분포가 실제 current policy로 얻어낸 transition들의 분포와 다르다면, 
              value estimation은 poor estimation이 된다.  

              그럼 batch 내의 각 transition의 분포를, current policy의 분포를 고려해서 가중치를 잘 조정하면 ?
                -> high-likelihood를 가진 current policy의 transition이 batch dataset 내에 없다면 여전히 실패 
                   즉, 결국 이 세 번째 factor도 데이터 부재와 연결되는 문제!


        # 7 : DQN, DDPG, Q-learning

          위 알고리즘들도 off-policy지만 사실 near-on-policy exploration policy를 쓴다.
          ex) soft-greepy + replaybuffer
          
          So, replay buffer가 가지고 있는 dataset은 current policy와 큰 상관관계를 갖는다.
            * Minh. 은 curreny policy가 얻어내는 transition들이 큰 상관관계를 가지기 때문에 replay buffer를 썼었다.
              그래도 완전히 끊어내지는 못했다고 했었다. 그러니까 여전히 어느정도의 상관관계를 가지고 있다는 이야기이다. 

          논문에서는 이제 완전히 상관관계가 없는 dataset을 이용해서 위 off-policy 알고리즘을 구동했을 때,
          결과가 얼마나 똥덩어리인지 설명해준다.

            "off-policy Deep RL algorithms are ineffective when learning 'truly off-policy'"

---

- `Batch constrained RL`


      # 8 : batch-constrained policy  

          OOD action으로 인해 extrapolation error가 발생한다.
          근데, 실제로 batch dataset 내에 있는 data들에 대해서는 문제없이 value estimation이 잘~된다.
          그러니, current policy 가 batch 내에 있는 action과 같거나 유사한 것들로만 선택하도록 유도해보자.
    
            1. 같은 action - 당근 문제없다.
            2. 유사한 action - neural network의 generalization이 있으니 이 또한 문제가 적다.


      # 9 : Batch-constrained deep Q-learning

        BCQ는 생성 모델을 활용한다. (VAE : state conditioned Gw(s))
        State가 주어졌을 때, BCQ는 타당한 몇 개의 action 후보군을 던져준다.
        당연히 이 action들은 batch에 있는 데이터들과 아~주 유사한 것들이지 말이다.
        고 다음에는 학습된 Q-network를 통해서 highest value action을 선택한다.

      # 10 : 

        위 생성모델 Gw(s)는 policy처럼 사용될 수 있다. 
          - ex. Gw(s)에서 action을 n개 sampling해서 가장 높은 값을 갖는 action을 선택하기


      # 11 :

        seen data의 다양성을 또 확보하기 위해력서 perturbation model도 설정한다. 
        action a를 [-k, k] 범위 내에서 조정해서 섭동 후 출력 
        



            
            
          
          
          
          
  
          
