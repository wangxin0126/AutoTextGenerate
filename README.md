# AutoTextGenerate

## 项目说明

- 利用循环神经网络自动生成英文文本
- 通过项目的完成达到1）熟悉pytorch等深度学习框架，2）熟悉循环神经网络等目的

## 简介

- 项目构成比较简单，主要包括数据集生成以及pytorch模型（RNN）
- 数据集的生成是将莎士比亚戏剧根据长度划分成多个训练个例，每一个训练个例包含长度为n-1的输入以及长度为n-1的label
- 通过one-hot编码将输入以及label向量化，利用rnn进行批次训练

## 结果

- 下面是一段生成的文本

  ```
  ROMEO:aTruardal hat shear to rise! the he'en meter.aadom everithes in Hereford in my deoby of Henryay and lovea nagron weedaral's the questing homeates the sweared there.anifeatheralityal Carsarle.aaked the doots thus! God bears you the lastsagive.aak his bistardly man bloody such memeet voice:afism
  ```

- 看起来效果不是特别好，，，不过的确可以学到很多单词

- 这里检查了一下，可能是因为没有吧\n统计到char集合里面导致的，修改后再运行试试看。

- **需要注意**: 在生成文本的时候，不能直接选择概率最高的那个字符，而应该利用multinomial进行抽样选择，否则会重复生成文本(见参考资料)

## 参考资料

- [Andrej Karpathy blog](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [LSTM text generator repeats same words over and over](https://discuss.pytorch.org/t/lstm-text-generator-repeats-same-words-over-and-over/43652)

