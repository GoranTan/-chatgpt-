# 如何提示chatgpt自动画一个合适的流程图
# Awaken的第一篇博客

众所周知，如果直接提示chatgpt来画一个流程图，他使用内置的画图代码画出来十有八九不尽人意。这个时候我们可以使用任务要求提示chatgpt生成流程图的Mermaid代码，然后在支持输入Mermaid代码的网站绘制流程图转化为想要的图片格式。

### 第一步：打开chatgpt
这里使用的模型是chatgpt4o。

### 第二步：输入提示
范式举例：
```
按照要求帮我生成以下任务的Mermaid代码。任务：“以下代码描述了使用Python语言，实现BiLSTM神经网络，并在通用数据集上实现文本情感分类任务。针对本实验任务，画出模型技术框图，如本实验要求画出全连接神经网络模型的模型原理图，要求有输入和输出。” 
要求：
1. 确保流程图的层级结构清晰，节点之间的关系明确，每个步骤和决策点都处于正确的位置，避免过多的交叉线和复杂的连接，逻辑顺畅。
2. 标签应简短且描述性强，格式统一，简洁明了。选择合适的布局方向，如从上到下或从左到右，保持节点之间的合理间距，使图表不显得过于拥挤或松散。
3. 适当使用注释和样式，在代码中添加必要的注释以便理解和维护，使用统一的颜色、边框和字体样式来提高可读性和美观度。
4. 使用颜色区分不同类型的节点（如开始、结束、决策等），确保颜色选择具有足够对比度，提高文本和背景颜色的对比度，避免使用过多的颜色导致混淆。
代码：“
```python
import random
import torch
import numpy as np
from data_preprocess import load_imdb
from torch.utils.data import DataLoader
from model import BiLSTM
import torch.nn as nn

def main():
    set_seed() # 使用随机数 确保在使用随机数时能够获得可重复的结果

    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 10

    train_data, test_data, vocab = load_imdb()
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BiLSTM(vocab).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_accuracy = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f'Epoch {epoch}\n' + '-' * 32)
        avg_train_loss = 0
        model.train()  # 将模型设置为训练模式
        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = criterion(pred, y)
            avg_train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 5 == 0:
                print(f"[{(batch_idx + 1) * BATCH_SIZE:>5}/{len(train_loader.dataset):>5}] train loss: {loss.item():.4f}")

        print(f"Avg train loss: {avg_train_loss / (batch_idx + 1):.4f}\n")

        acc = 0
        model.eval()
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                acc += (pred.argmax(1) == y).sum().item()

        accuracy = acc / len(test_loader.dataset)
        print(f"Accuracy: {accuracy:.4f}")

        # Save the model if it has the highest accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "best_model.pth")
            print("Best model saved!")

    print("Training completed!")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark
”
```

### 第三步：保存你的Mermaid代码，找到一个支持Mermaid格式的网站上传保存图片格式
支持Mermaid格式的网站，以下两个网站都不需要登录，第一个功能比较简洁可以直接转化，第二个网站功能更多：
- [https://www.min2k.com/tools/mermaid/](https://www.min2k.com/tools/mermaid/)
- [https://app.diagrams.net/](https://app.diagrams.net/)

然后得到了以下Mermaid代码：
```
graph TD
    classDef inputOutput fill:#f9f,stroke:#333,stroke-width:2px,color:#000;
    classDef process fill:#0f9,stroke:#333,stroke-width:2px,color:#000;
    classDef decision fill:#ff0,stroke:#333,stroke-width:2px,color:#000;
    classDef endNode fill:#f96,stroke:#333,stroke-width:2px,color:#000;
    
    A[输入数据]:::inputOutput --> B[数据预处理]:::process
    B --> C[加载IMDB数据集]:::process
    C --> D[创建数据加载器]:::process
    D --> E[创建BiLSTM模型]:::process
    E --> F[定义损失函数和优化器]:::process
    F --> G[训练模型]:::process
    G --> H[模型评估]:::process
    H --> I{模型准确率是否提高}:::decision
    I -->|是| J[保存最佳模型]:::process
    I -->|否| K[继续训练]:::process
    J --> L[训练完成]:::endNode
    K --> G
    L
```

网站1的效果：
![image](https://github.com/freesnowmountain/-chatgpt-/assets/102673901/3ca5a6a3-333b-43e5-a609-8020557eeaae)

网站2的效果：
![image](https://github.com/freesnowmountain/-chatgpt-/assets/102673901/680d3de3-a553-4c6b-ba93-bdb860ea3bdf)
![image](https://github.com/freesnowmountain/-chatgpt-/assets/102673901/a2ac3ac8-9eab-4490-87de-7ee19c3df9b0)



