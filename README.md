# PPO pybullet by Keras
透過Tensorflow2 內建 Keras 執行Proximal Policy Optimization (PPO)來學習強化學習，引用 pybullet 套件中的KukaGymEnv作為訓練環境。


# 首要工作  
  安裝 tensorflow  `pip install tensorflow`  
    
  安裝 pybullet、gym、scipy 套件  `pip install pybullet gym scipy`  
    
  安裝繪圖工具套件  `pip install matplotlib`  
  
  (!!注意!!)  
  pybullet官方提供的kukaGymEnv環境有些瑕疵，在此做出些微調整:  
  ※ 執行 `step2` 時確保 `p.stepSimulation()` 執行完全  
  ※ 不因執行 `_termination` 重複給予獎勵  
  ※ 成功夾取後給予獎勵由 `1000` 調整為 `10000` (可自行調整)  
  ※ 執行 `step2` 可以輸出夾取的分數 `blockPos[2]`  
  
  可以透過以下方式更新 kukaGymEnv.py  
  (將 `kukaGymEnv_myself/kukaGymEnv.py` 更換pybullet安裝後提供的kukaGymEnv.py文件 `..(your env)../Lib/site-packages/pybullet_envs/bullet/kukaGymEnv.py` )
  

# 執行文件
`train_PPO_KukaGym.py` : PPO開使訓練KukaGymEnv  
  
`test_PPO_KukaGym.py` : 訓練完成後進行測試 (100次抓取 取成功率%) 

# 超參數說明
```python
env = KukaGymEnv(renders=False, isDiscrete=True, actionRepeat=10)  
```
`renders = (True/False)`  :觀看環境影像  
`isDiscrete = (True/False)`  :是否為離善環境  
`actionRepeat = (int)`  :執行動作選取間隔  

```python
epochs = 2000 
steps = 20 
gamma = 0.9999
clip_ratio = 0.15 
policy_learning_rate = 3e-4
value_function_learning_rate = 1e-3
train_policy_iterations = 80
train_value_iterations = 80
lam = 0.97
target_kl = 0.01 #entropy
hidden_sizes = (64, 64) 
buffer_size = 25000
```  
`epochs`  :迭代終止次數  
`steps`  :一次迭代收集回合數  
`gamma`  :智能體遠見程度(折扣率)  
`clip_ratio`  :clip後比例縮放程度  
`policy_learning_rate`  :Actor Neural Network  學習率  
`value_function_learning_rate`  :Critic Neural Network  學習率  
`target_kl`  :Actor Neural Network更新受到 KL懲罰之閥值  
`hidden_sizes`  :MLP的神經元顆數  
`buffer_size`  :重放區大小  

# Finish
![kukagym_Finsh](https://user-images.githubusercontent.com/94059864/145708278-ed93983c-0451-4625-a6f3-db7cfc1a2a02.gif)

  

