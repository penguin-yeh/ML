    self.features = nn.Sequential( #an ordered container
      
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
    )
