#{ key1:value1, key2,value2 }: dict
data_transforms = {
        'train': transforms.Compose([ #Compose: 將transform組合在一起
                                     
            transforms.Resize((224,224)), #(H, W):將圖片轉成224*224

            transforms.ToTensor(), #將PIL.image/numpy.ndarry轉成torch.FloadTensor,並normalize到[0.0, 1.0]

            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), #1.1
        ]),
        'val': transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),

            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), #1.1
        ]),
    }
