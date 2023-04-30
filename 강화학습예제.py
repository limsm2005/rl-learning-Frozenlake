#!/usr/bin/env python
# coding: utf-8

# In[3]:


import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register

#에이전트의 상태, 보상 저장
register(
    id = 'FrozenLake-v3',
    entry_point = 'gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4',
           'is_slippery' : False}
)
#환경 생성
env = gym.make('FrozenLake-v3')
# Q 테이블 생성 및 0으로 초기화
Q = np.zeros([env.observation_space.n, env.action_space.n])
# dicount 정의(학습을 더욱 최적화 시키기 위해 사용)
dis = 0.99
# 시도(에피소드) 횟수 설정
num_episodes = 2000
# 시도(에피소드)당 총 리워드의 합을 저장하는 리스트 정의
rList = []

for i in range(num_episodes):
    # 환경 초기화
    state = env.reset()
    rAll = 0
    done = False
    
    # Q-Table learning 알고리즘
    while not done:
        # 행동(action) 중 가장 보상(reward)이 큰 행동(action) 선택
        # random noise 방식으로 구현함 / noise값 = np.random.randn(1, env.action_space.n)
        action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) / (i+1))
        
        # 환경으로부터 새로운 상태(state)와 보상(reward) 받음
        new_state, reward, done, _ = env.step(action)
        
        # Q-Table 업데이트 (Q = Q + reward)
        Q[state,action] = reward + dis*np.max(Q[new_state,:])
        rAll += reward
        state = new_state
    #시도(에피소드)가 끝난 후 나온 총 리워드의 합 저장
    rList.append(rAll)

# 성공확률 = (에피소드당 총 리워드의 합들의 합) / (에피소드 실행 횟수)
print("success rate : "+str(sum(rList) / num_episodes))
print("Final Q-Table Values")
print(Q)

# bar형태의 그래프로 rList값 시각화
plt.bar(range(len(rList)), rList, color = 'black')
plt.show()


# In[ ]:




