### COG `2020 CoRL`


Connecting New Skills to Past Experience with Offline Reinforcement Learning 

---

열린 서랍에서 obj 를 꺼내는 task를 학습한 robot이 있다.
즉, obj 를 집고, 서랍 밖으로 꺼내는 task 가 학습되있다.

근데 만약 로봇이 서랍이 닫힌 환경에 딱 주어졌을 때, 이전에 학습한 그 task 를 수행할 수 있을까?
이 로봇은 그 시나리오나 초기환경을 경험한 적이 없기 때문에 task를 수행하지 못한다. 

Paper의 시작은 로봇이 이런 경험하지 못한 환경에 주어졌을 때,
reasonable 하게 이 상황을 판단하고 행동해서 기존의 task 를 다시금 수행하도록 하는 방법이 없을까?


But what if combining previously collected robot interaction data, which might include drawer opening and 
other behaviors, together with offline RL methods, can allow these behaviors to be combined automatically, 
without any explicit separation into indivisual skills?

we study how model-free RL can utilize prior data to extend and generalize learned policy, at test-time. 

예를 들어, 이전에 서랍을 열었던 경험의 data 를 이용한다면, 서랍이 처음에 닫혀있다하더라도 그 닫힌 서랍을 열음으로써 
기존에 object 를 잡고 꺼낼 수 있도록 학습된 그 initial state 로 갈 수 있다.

그리고 뭐, 서랍 앞에 장애물이 있을 때도, 만약에 이전에 pick and place 같은 task 를 수행한 경험을 사용한다면, 
마찬가지로 장애물을 치우고, 서랍을 열고 원래의 학습된 initial state 로 도달해서 기존의 task 를 수행할 수 있을 것이다. 

중요한 건 이 논문에서 제시한 방법은, 이전에 서랍을 열거나 pick and place 했던 task 를 다시 학습하지 않아도 된다는 것입니다.

그냥 새 task 에 대한 data 는 독립적으로 모아서 어떤 새로운 행동을 구성할 때 과거의 data 를 합쳐서 이용할 수 있다는 것입니다. 

-----

해당 방법은 CQL 을 기반으로 수행됩니다.

먼저 replay buffer 를 prior dataset 으로 채우고, 각 prior data sample 들에 대해서는 모두 0 reward 를 줍니다.

당연히 현재 내가 하려는 task 랑 관련이 없는 data 들이기 때문에 reward 는 없는게 맞습니다. 

그리고 새로운 task 에 대해서 얻은 data 들과 prior dataset 에 대해서 CQL 학습을 수행합니다.

그리고 평가 단계에서는 새로운 task 를 수행할 때, 전혀 보지 못헀던 다양한 초기 환경에서 task 를 수행하도록 합니다. 


이 논문의 main contribution 은 

간단히 얘기해서, 내가 서랍에서 물체를 꺼내는 task 를 수행해야하는데, 학습은 서랍이 열린 상황에서만 학습이 수행됬습니다.

하지만 test 단계에서 서랍을 닫힌 상태, 서랍 앞에 장애물이 있는 상태 등등의 기존의 학습 단계에서 보지 못한 새로운 초기 환경에서도 

이전에 수행했던 여러가지 task 에 관련된 data 를 잘 활용해서, 그런 낯선 초기환경에서도 기존에 서랍에서 object 를 꺼내는 task 가 가능하게 했다는 것입니다. 


---

필요한 건 그냥 두 개의 data source 이다.

prior dataset (no reward) 
task dataset (sparse reward 0, 1)


---

이 방법은 사실 굉장히 간단합니다.

새로운 task 를 학습하는 과정에서 prior data 를 그냥 섞어주기만 하면 됩니다. 

즉, offline setting 이기 때문에 새로운 task 에 대해서 모아둔 datas랑 

prior dataset 을 섞어서 agent 를 학습시키면 됩니다.

이 방법이 왜 initial condition 에 대한 일반화가 잘 되는지에 대해서 간단히 설명이 나와있습니다.

Q learning 의 학습 과정을 보면, update 과정에서 trajectory 를 따라서 backward로 information을 전파합니다.

trajectory 의 가장 끝단의 state-action pair 의 reward는 가장 높을 것이고, 이 정보가 trajectory 가 시작된 곳까지 전파됩니다.

그래서 trajectory가 시작된 곳의 value는 끝단의 value 보다 훨씬 작을 것입니다.

다만, state의 Q value가 0은 아닙니다. 왜냐면, 이 state에서의 Q value가 높은 value 가진 state로 갈 수 있다는 그 가능성을 이해하고 있기 때문입니다.

반대로 어떤 state 든 high-value를 가진 state로 가는 path가 없는 state 는 0의 값을 가지고 있을 것입니다. 

이런 상황에서 만약에 test time 에서 서랍이 닫혀있었던 것 처럼 dataset 에서 관측되지 않은 state 에서 이 task 를 수행하도록 했을 때는

당연히 training distribution 에서 벗어나있는 state 이기 때문에 0 value 를 가지고 있거나 음수 값을 가지고 있습니다.

그래서 이 state 에서 시작된 task는 성공할 수 없습니다.

근데 만약에 훈련단계에서 다양한 task 들에 대한 data들로 기존의 dataset 을 증강한다면 

기존의 trajectory 의 말단 부분의 state 가 prior dataset 내부의 다른 task 로의 trajectory 로 확장되고, 

이걸 통해서 Q value 정보의 propagation 이 추가로 일어날 수 잇습니다. 

그래서, 기존에 보지 못했던 초기 state 들에서, task를 해결할 수 있었던 학습된 state 들로 이어주는 

새로운 trajectory를 증강된 data 들이 제공한다는 것입니다.

---
































