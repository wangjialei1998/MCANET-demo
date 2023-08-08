import cv2 as cv
from timm.models import create_model
from timm.data import create_transform
import torch
import numpy as np
from PIL import Image
class ODIR_Rating:
    def __init__(self,model_path:str='',threshold:float=0.5) -> None:
        self.use_gpu=torch.cuda.is_available()
        self.transform=create_transform(
        (3,384,384),
        is_training=False,
        interpolation='bicubic',
        mean=(0.4407552, 0.28228086, 0.15446076),
        std=(0.254417, 0.17148255, 0.0995115))
        self.model=create_model(model_name = 'resnet50d',num_classes=8)
        self.model.eval()
        self.model=self.model.to('cuda:0') if self.use_gpu else self.model
        self.threshold=threshold
    def get_odir_rating(self,file_bytes:bytes):
        file_bytes_np = np.frombuffer(file_bytes, dtype=np.uint8)
        img = cv.imdecode(file_bytes_np, cv.IMREAD_ANYCOLOR)
        img=get_the_max_area_image(image=img)
        img=Image.fromarray(img)
        img=self.transform(img)
        img=img.to('cuda:0') if self.use_gpu  else img
        result=self.model(img)
        result=result.detach().cpu().numpy()
        return result>self.threshold



# 用于获取最大眼底区域的图像
def get_the_max_area_image(image:cv.Mat):
    gray_img=cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    ret,binary_img=cv.threshold(gray_img,1,255,cv.THRESH_BINARY)
    contours,hierarchy=cv.findContours(binary_img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    max_contour=max(contours,key=cv.contourArea)
    x,y,w,h=cv.boundingRect(max_contour)
    max_area_img=image[y:y+h,x:x+w]
    return max_area_img

if __name__=='__main__':
    test=ODIR_Rating(model_path='/home/dr/dr-back/pth/odir/checkpoint-24.pth.tar')

    test.get_odir_rating(np.ones((3,384,384),dtype=np.uint8).tobytes())
