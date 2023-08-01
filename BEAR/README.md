#### Stabilizing off-policy Q-Learning via Bootstrapping Error Reduction `2019`

[BEAR ref](https://sites.google.com/view/bear-off-policyrl)

Key : which polices can be reliably used for backups without backing up OOD actions?

    BEAR의 key idea는 learned policy가 behavior policy distribution의 support 내에 있도록 제한하는 것.
    constraining the learned policy to lie within the support of the behavior policy distribution)

    # Support contraint

    BEAR는 learned policy가 behavior policy과 분포가 비슷하도록 제한하는 것이 아니다.
    Learned policy가 non-negligible behavior policy density를 가진 action에 대해 non-zero 확률을 갖도록 하는 것.
    
---

