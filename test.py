import cv2
import numpy as np
import torch
from BagData import test_dataloader,train_dataloader,val_dataloader,out_dataloader
from PIL import Image
from pinggu import calculate_metric_percase
def test():

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        fcn_model=torch.load('checkpoints/best_unet_model4.pt')
        # fcn_model=torch.load('checkpoints/best_deeplabv3_model2.pt')

        fcn_model.eval()
        with torch.no_grad():
            # l=enumerate(train_dataloader)
            # l = enumerate(test_dataloader)
            l = enumerate(val_dataloader)
            # l = enumerate(out_dataloader)
            for index, (bag, bag_msk,bag_name) in l:

                bag = bag.to(device)
                output = fcn_model(bag)
                output = torch.sigmoid(output)  # output.shape is torch.Size([4, 2, 160, 160])
                # output = torch.sigmoid(output['out'])  # output.shape is torch.Size([4, 2, 160, 160])

                output_np = output.cpu().detach().numpy().copy()  # output_np.shape = (4, 2, 160, 160)
                output_np = np.argmin(output_np, axis=1)

                for index in range(4):
                    img1=output_np[index,None,:,:]
                    img2= img1[0,:,:].astype(np.uint8)
                    img2[img2==1]=255
                    image = Image.fromarray(img2)
                    image=np.array(image)
                    median_image = cv2.medianBlur(image, 5)


                    # cv2.imwrite(f'./predict/deeplab/train/{bag_name[index]}', img2)
                    # cv2.imwrite(f'./predict/deeplab/in_test/{bag_name[index]}', img2)
                    # cv2.imwrite(f'./predict/deeplab/val/{bag_name[index]}', img2)
                    # cv2.imwrite(f'./predict/deeplab/out_test/{bag_name[index]}', img2)

                    # cv2.imwrite(f'./predict/unet/train/{bag_name[index]}', img2)
                    # cv2.imwrite(f'./predict/unet/in_test/{bag_name[index]}', img2)
                    cv2.imwrite(f'./predict/unet/val2/{bag_name[index]}', img2)
                    # cv2.imwrite(f'./predict/unet/out_test/{bag_name[index]}', img2)



if __name__ == "__main__":
    test()
