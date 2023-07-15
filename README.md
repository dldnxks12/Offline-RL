### Offline RL

`한결이 졸업 논문 introduction / related work 참고`

---

- `Offline RL (Data-driven learning)`

        Solving decision-making problems with previously collected datasets.
        Offline RL agent does not interact with environment during training process.                   

            -> Exploration cost가 높거나 불가능할 때 아주 유용하다.

        Most obvious challenge : boostraping from OOD actions (out-of-distribution)
        
            -> 즉, dataset 내에 없는 action을 수행함으로 인해 발생하는 error가 점점 쌓이게 된다.
            결과적으로, Q function을 이상하리 만큼 크게 평가하는 문제가 있다고 한다. (왜 크게 평가하는 지는 차차 알아보자.)


        To resolve this problem, several works have proposed to constraint the learned policy
        to be similar to the behavior policy during the learning process.

            -> policy constraint approach 
        

        # Behavior Cloning
        가장 간단하고 효과적인 policy constaint 방법은 Behavior Cloning (BC)이다.
        하지만 only considers errors of actions between the learned policy and the behavior policy
        -> dataset의 quality가 별로면 성능이 크게 저하된다. 

        # Advantage-weighted methods
        Imitation learning과 online RL 분야에서 advantage-weighted 방법이 제안되고 있다.
        (Distinguish the quality of data by learning 'good' actions with higher weights)

        하지만 이 방법을 offline RL의 fixed dataset에 대해 적용했을 때는 policy 향상이 그렇게 눈에 띄지 않는다. (이 이유도 차차 알아보자.)

        한결이의 논문은 위 두 가지 방법의 limitation을 해소하려는 방향으로 진행됬다. 
        
        'He propose a data-selective advantage-weighted method to address the problem of BC and the limitations of
        advantage-weighted methods in offline RL.'

---

- `Approach 1 : constraint learned policy`

        Constraint learned policy to be 'similar' to the behavior policy of the dataset.

        #BCQ : Batch-constraint Q-learning
            Directly constraint learned policy using a perturbation model.
  
        #BEAR: Bootstrapping error accumulation reduction
            Instead of directly constraint learned policy, it employes MMD (maximum mean discrepancy)
            to update learned policy to be similar to the behavior policy via dual gradient descent.
  
        #BRAC: Behavior regularized actor-critic 
            Adapts KL-divergence to restrict the policy in the step of policy evaluation and improvement.


- `Approach 2 : constraint Q function`

        Regularize Q function to underestimate the values of OOD actions.

        #CQL : Conservative Q-learning
            It learns a Q function lower than the true Q function, by penalizing the values of OOD actions.

        #F-BRC: Fisher-behavior regularized critic
            A critic regularzing method using fisher divergence.
            This can be viewed as policy constraint method, because the fisher divergence denotes a penalty
            in terms of the distance between the learned policy and the behavior policy of the dataset.

        
- `Approach 3 : advantage-weighted methods`

        Advantage-weighted methods utilizing the 'advantage function' for policy updates,
        by learning 'good' actions with higher weighs.
  
        But, the proposed methods strictly prescreened given dataset, rather weighting.
        And uses the advantage-weighed method as a policy constraint. 

        #MARWIL : Monotic advantage re-weighted imitation learning

        #AWR : Advantage-weighted regression

        #AWAC: Advantage-weighted actor critic 

        #CRR : Critic regularized regression



- `Approach 4 : combinations with Behavior Cloning`

        The combination of standard RL and BC.

        #SAC+BC (AWAC - Advantage-weighted actor critic)
        #TD3+BC
        

---
