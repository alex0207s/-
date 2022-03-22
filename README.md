# 畢業專題 - Neural Network Backdoor against Transfer Learning


目的: 研究 neural networks 本身的漏洞，並且利用此漏洞達到控制輸出作為研究目標。

示意圖: ![image](https://user-images.githubusercontent.com/52899347/159462637-1b2324cb-373a-44b9-b8c4-33c22f02a3b2.png)

說明: 利用在原始圖片中插入 "trigger"的方式，使得被攻擊過的 neural network 會受到插入 trigger 的影響，而改變模型最後的預測！

結論: 透過我們設計植入惡意 neural network 的方式，我們可以做到攻擊任意神經網路，並且即便 neural network 經過 transfer learning 後，此攻擊仍然會被繼承下來，對新的 neural network 造成攻擊。在我們實驗的數據中，經過 transfer learning 後的 neural network 仍然有高達 95% 攻擊準確率。
