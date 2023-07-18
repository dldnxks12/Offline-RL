### Batch RL summary & implementation 

`paper : Batch reinforcement learning, 2012`

---

- `Batch RL`

    
    Batch RL은 다음 3가지를 해결하는데 중점을 둔다.


        1. Exploration overhead
            -> Experience replay 
        2. Inefficiencies due to the stochastic approximation
            -> kernel-based self-approximation
            (use sample transitions rather random supports for approx. value function)
        3. Stability issues when using function approximation
            -> Fitting 

---

- `Algorithms`
        

        1. KADP 
        2. FQI
        3. NFQ - FQI with Neural Network 
        4. DFQ - NFQ with latent space extracted with Auto-Encoder (High Dim to Low Dim)

        
    

        

        