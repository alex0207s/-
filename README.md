# 專題-TrojAI

參考論文: An Embarrassingly Simple Approach for Trojan Attack in Deep Neural Networks

我們選擇一個能夠很好分辨 Oxford flowers 102 資料集的模型作為被攻擊的對象

接著汙染該訓練資料集，並在被攻擊對象中植入 TrojanNet，重新訓練 Backdoor Model 的 Classifier。

此種方式的好處在於，因為不需要重新訓練整個 Backdoor Model（只需訓練後面的 Classifier），所以植入惡意行為的時間成本較低，且不需要大量的資料集就能達到對 Trigger 有極高的辨識率。

除此之外，不同於論文中的方法，此種方式能夠達到抗遷移式學習的好處，且遷移式學習後 Trigger 還是能夠有極高程度的影響力。
