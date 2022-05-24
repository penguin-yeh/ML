class MyCNN(nn.Module): #MyCNN繼承了nn.Module
  # Constructor
  def __init__(self, num_classes=1000):
    super(MyCNN, self).__init__() #super(class, object): object必須為class的type
    # = super().__init__()
    self.features = nn.Sequential( #an ordered container
      #============== 在此區塊新增或減少隱藏層 =================

      # input_shape=(3,224,224)
      #((origin + padding*2 - kernel_size) / stride) + 1(無條件捨去)
      nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2), 
      nn.LeakyReLU(inplace=True), 
      nn.MaxPool2d(kernel_size=3, stride=2), #output = (64, 27, 27)

      nn.Conv2d(64, 192, kernel_size=5, padding=2), #output_shape = (192,27,27)
      nn.LeakyReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2), #output_shape = (192,13,13)
      
      nn.Conv2d(192, 384, kernel_size=3, padding=1), #output_shape = (384,13,13)
      nn.LeakyReLU(inplace=True),

      nn.Conv2d(384, 256, kernel_size=3, padding=1), #output_shape = (256,13,13)
      nn.LeakyReLU(inplace=True),

      nn.Conv2d(256, 256, kernel_size=3, padding=1), #output_shape = (256,13,13)
      nn.LeakyReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2), #output_shape = (256,6,6)
      #==========================================================
    )
    self.features2 = nn.Sequential(
      #============== 可在此區塊新增隱藏層 =====================

      #===========================================================

    )
      
      
    self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
    self.classifier = nn.Sequential(
      #============== 在此區塊新增或減少隱藏層 =================
      nn.Dropout(),
      nn.Linear(256 * 6 * 6, 4096),
      nn.ReLU(inplace=True),
      nn.Dropout(),
      nn.Linear(4096, 4096),
      nn.ReLU(inplace=True),
      # nn.Linear(4096, num_classes), # 原始模型輸出層
      #===========================================================
    )
    self.classifier2 = nn.Sequential(
      #============== 可在此區塊新增隱藏層 =====================
      #===========================================================
      nn.Linear(4096, num_classes),
    )

  def forward(self, x):
    x = self.features(x)
    x = self.features2(x)

    x = self.avgpool(x)
    x = torch.flatten(x, 1)

    x = self.classifier(x)
    x = self.classifier2(x)

    return x
