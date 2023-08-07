	# Concept
		DT - 성능 매우 좋다. 다만 느리다.
		Offline RL 하나 딱 고르라면, TD3+BC. 
		왜? 성능이 SOTA는 아니지만,  CQL에 비해 매우 간단.	
		EDAC + TD3+BC 해보면 좋을 것 같다. - 장원형님 의견
		
		+ MCQ가 현재 성능이 압도적으로 좋은데, 이거랑도 비교해보면 좋을 것 같다. 


	# Contribution 후보

		1. offline RL benchmark 성능 향상 
		2. Instability across episode *** -> 이거 해결하면 성능은 알아서 오를 듯.
			- offline RL policy 가 converge 하지 않는 것 같다고 느껴진다. 
			- 학습이 덜 됬다는 느낌? 

		- variance 를 줄이는게 뭐가 중요한가?
			학습이 더 쉽다 - 1 2 1 2 1 보다 8 2 5 6 1 7 99 의 pattern 학습이 더 어렵다. 
		- bias 는? 학습 과정에서는 exploration 관점에서 중요한 factor가 되기도 한다.
			다만 bias가 존재하는 채로 학습이 마무리되면, sub-optimal 로 빠지는 경우가 있다. 

		# for getting true estimate... 
		- error 는 sigma / root N 에 비례 --- sigma : 분산 | N : sample 개수
		- error 줄이려면 분산을 줄이거나, sample 수를 늘리거나 !
		-> offline RL 은 dataset 이 고정 
		-> 즉, true value 를 잘 estimate 하기 위해서는 -> sample 수를 무한정 늘릴 수 없다 -> variance 줄이는 것이 중요 
		-> off-policy setting 에서는 추가적인 variance 가 있다 -> important sampling ratio 곱 -> variance 증가 시키는 요인 


	# Tools to consider 
	
		 1. Ensemble 2. Transformer 3. Attention

		* transformer로 Policy를 pre-train 한 것도 같이 해보면 좋을 것 같다. (2 phase)
		* attention 으로 augment 한 state 에서 주목할 부분을 찾게 해서 unseen state 의 uncertainty 를 줄이도록 해볼까? 
		* ensemble 이 학습 안정성, variance 줄이는 데 성능이 좋으니 한 번 써보자.
		

	# Experiments 	

		Comparing offline RL : 1. OPE 2.Policy replay 

		둘 중 어느것을 쓰던 상관없고, 섞어서 쓰지만 않으면 된다.
		* Fair comparison factors 
		1. Same dataset
		2. Same random seed
		3. Same number of training steps	
		4. Same evaluation protocol (policy replay)
		5. repeat experiment multiple times 
		6. proper metric
		7. Experiment setting 명시


		# TD3+BC 실험

		1 millon step 학습. 5000 time step 마다 evaluate. 각 evaluation 마다 10 episode.
		5개 seed에 대해서 실험.
		마지막 10개의 evaluation의 average normalized score를 표로 명시.
		BRAC - CQL paper의 결과 가져왔음
		AWAC - AWAC paper 의 결과 가져왔음 , BEAR 결과도 있음
		CQL / F-BRC - 7 page github.com 코드 가져와서 돌렸음

	
		* Benchmark 후보들 : BCQ / BEAR / AWAC / CQL / MCQ / F-BRC / BRAC /
					  DT / TD3+BC / EDAC / EDAC+TD3+BC


