### Offline RL

---

- `Offline RL?`
  
      Offline RL은 환경과의 상호작용 없이 agent를 학습시키는 방법론이다. 
      Exploration이 어렵거나 cost가 높은 환경에서 사용할 수 있다. 


- `Challenge`

  
      가장 명확한 문제점은 OOD (out-of-distribution) action에 대한 문제이다.
      - OOD action에 대해서 agent가 이상할 만큼 크게 Q value를 평가한다. (Q-learning setting 에서)


---


- `Proposed methods`

      # Navie method :

        - FQI : offline batch version of Q-learning. No guaranteed convergence 
        - NFQ : FQI with neural network. 

      # Policy Constraint :

        Constrain the learned policy to be similar to the behavior policy of the dataset.

        - BCQ   : Directly constrain learned policy using perturbation model.
        - BEAR  : With MMD, constrain learned policy via dual GD.
        - BRAC  : Adapts KL divergence to constrain the learned policy.    

      # Regularize Q-function :

          OOD action을 underestimate 하도록 Q function regularization하는 방법.

          - CQL   : OOD action에 페널티를 주며 true Q function보다 더 낮은 Q function을 학습.
          - F-BRC : Fisher-divergence를 이용한 Critic regularization method.
                    this can be interpreted as policy constraint method
                    - fisher-divergence denotes a penalty in terms of distance
                      between the learned policy and the behavior policy in the dataset.
            
      # Advantage-weighted method :

        Utilizing advantage function for policy updates by learning 'good' actions with higher weights.

        - MARWIL : Monotinic advantage re-weighted imitation learning
        - AWR    : Advantage-weighted regression
        - AWAC   : Advantage-weighted actor-critic
        - CRR    : Critic regularized regression
  
          * AW는 Imitation learning 과 online RL에서 준수한 성능을 보인다.
            근데 fixed dataset에 대해서는 성능 향상이 있긴 하지만, 눈에 띌만한 향상은 보이지 못한다고 한다.
            (이유는 차차 공부해보자.)
            So, 이 한계를 극복해서 offline RL에서 써보면 어떨까? 많은 사람들이 이미 그러고 있다.


      # Combination of RL+BC
    
        RL+BC : Simply applies the BC to the policy update of an online RL algorithms.

        - SAC+BC (AWAC)
        - TD3+BC
        - TD3+DABC : 한결

          *Behavior Cloning (BC) : 
  
          BC는 offline RL에서 OOD 문제를 다룰 수 있는 가장 간단하고 실용저인 policy constraint 방법.
          하지만 behavior policy의 행동과 learned policy의 행동 간의 error 만 고려하기 때문에,
          low-quality의 dataset을 가지고 학습할 때에는 좀 문제가 있다. 
        
        
      

##### 출처 : 한결이 졸업 논문
