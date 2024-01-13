## Imgae matting

Here are a few effects(omitting mask images)：

![image-1](https://github.com/ihmily/image-matting/blob/main/assets/image-1.png)

![image-2](https://github.com/ihmily/image-matting/blob/main/assets/image-2.png)

&emsp;

## How to Run

Firstly, you need to download the project code and install the required dependencies

```
# python 3.10

git clone https://github.com/ihmily/image-matting.git
cd image-matting
pip install -r requirements.txt
```

Next, you can use the following command to run the web interface

```
python app.py
```

Finally, you can visit  http://127.0.0.1:8000

&emsp;

## Use API

Please run it before use API

File upload

```
import requests

server = "http://127.0.0.1:8000"
image_path = "image.png"
model_name = "universal"  # people,universal
files = {"image": (image_path, open(image_path, "rb"))}
data = {"model": model_name}
response = requests.post(server+'/matting', files=files, data=data)
print(response.text)
json_data = response.json()
image_url = json_data['result_image_url']
mask_url = json_data['mask_image_url']
print("image_url:", server + image_url)
print("mask_url:", server + mask_url)
```

Url upload

```
import requests

server = "http://127.0.0.1:8000"
image_url = "http://your-image-url/demo.png"
data = {"image_url": image_url, "model": "universal"}  # people,universal
response = requests.post(server+'/matting/url', json=data)
print(response.text)
json_data = response.json()
image_url = json_data['result_image_url']
mask_url = json_data['mask_image_url']
print("image_url:",server+image_url)
print("mask_url:",server+mask_url)
```

You can freely choose the method you want to upload from above.If you want to get the cropped cutout, you can call `crop_image_by_alpha_channel` function.

&emsp;

## Extended Gallery

![image-3](https://github.com/ihmily/image-matting/blob/main/assets/image-3.png)

![image-4](https://github.com/ihmily/image-matting/blob/main/assets/image-4.png)

&emsp;

## References

[https://modelscope.cn/models/damo/cv_unet_universal-matting/summary](https://modelscope.cn/models/damo/cv_unet_universal-matting/summary)

[https://modelscope.cn/models/damo/cv_unet_image-matting/summary](https://modelscope.cn/models/damo/cv_unet_image-matting/summary)
