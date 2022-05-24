#{ key1:value1, key2,value2 }: dict
data_transforms = {
        'train': transforms.Compose([ #Compose: 將transform組合在一起
                                     
            transforms.Resize((224,224)), #(H, W):將圖片轉成224*224

            transforms.CenterCrop(124), #1.4

            transforms.ToTensor(), #將PIL.image/numpy.ndarry轉成torch.FloadTensor,並normalize到[0.0, 1.0]

        ]),
        'val': transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ]),
    }
