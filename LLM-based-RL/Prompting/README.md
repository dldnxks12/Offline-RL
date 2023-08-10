### Prompt-based Decision transformer 

---

- `Concept`


        Decision Transformer 를 통해 Transformer 기반으로 Sequential modeling 이 잘 된다는 걸 확인
        -> Transformer 의 inductive bias 와 few-shot learning 능력을 가져와서 prompt-base 로 
        새로운 task 에 대해 모델을 확장시킬 수 있지 않을까?

        Strong meta RL 모델의 성능을 크게 뛰어넘음 

        Prompt-DT : Decision making scene 에서 few-shot learning 효과를 확인. 

        * Meta RL : generalizing an agent's knowledge from one task to another.
        (ex. Model-Agnostic Meta-Learning, MAML)
 

--- 

- `Background` 


        large-scale datset 에 대해 학습이 된 Transformer 는 few-shot, zero-shot learning 을 할 수 있음이 알려져있음.
        NLP 분야에서 prompt-base 로 transformer를 새로운 task 에 대해 확장시키는 방법이 대두
        
            * prompt-based framework
        
            : prompt 는 task 에 대한 중요한 정보를 담고 있고, task input 이전에 넣어준다.
            즉, 이 중요한 정보를 통해 few-shot / zero-shot learning 후 새 task 수행
        
            * few-shot learning
        
            : 아주 적은 dataset 을 이용해서 모델을 학습하는 것. (GPT-3가 아주 좋다고 한다.) 

        
        Inductive bias 를 이용한 구조에 집중
        
            *  Inductive bias
            
            : model이 학습 때 보지 못한 데이터에 대해서 학습을 하게 될 때 사용되는 추가적인 가정들.
            즉, 학습 과정에서 본 적이 없는 데이터에 대해서 판단을 하게 될 때, 
            학습 과정에서 습득된 어떠한 편향을 이용해서 새로운 데이터에 대해 판단을 내리는 것.


---

- `Method`

        
        training 단계는 offline dataset 에 대해 학습.
        각 task 마다, agent 는 same task 에서 샘플링된 prompt trajectory 에 conditioning 되서 target trajectory를 예측하도록 학습됨.
        
        evaluation 단계에서 agent 는 prompt 를 만드는데 사용할 새로운 task 에 대한 trajectory 를 받음 
        fine-tuning X. agent 는 이 trajectory 에서 얻는 정보들로 새 task에 대한 policy 를 생성해냄.

 
        
