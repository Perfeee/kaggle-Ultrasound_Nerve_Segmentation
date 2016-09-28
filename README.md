# kaggle-Ultrasound_Nerve_Segmentation

1. 原始数据都放在上层文件夹: ../../train/或者../../test/ ，生成的数据都放在./generated_data/文件夹
2. 文件名以data开头的都是一些data augmentation的源码，是图像类竞赛当中的重点。文件名以classify和train开头的是模型训练主文件。run_length_encode.py文件是对图像进行编码的文件。
3. 模型如果训练到基本稳定，效果应该在该竞赛Leaderboard的top10%左右。
