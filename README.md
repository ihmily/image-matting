## Imgae matting

Here are a few effects(omitting mask images)：

![image-1](https://github.com/ihmily/image-matting/blob/main/assets/image-1.png)

![image-2](https://github.com/ihmily/image-matting/blob/main/assets/image-2.png)

&emsp;

## How to Run

**Method 1: Run from Source Code**

Firstly, you need to download the project code and install the required dependencies.

```
# Python 3.10

git clone https://github.com/ihmily/image-matting.git
cd image-matting
pip install -r requirements.txt
```

Next, use the following command to run the web interface.

```
python app.py
```

Finally, visit http://127.0.0.1:8000/.

&emsp;

**Method 2: Run with Docker**

Simply run the following commands after entering the project folder.

Pull the Docker image.

```
docker pull ihmily/image-matting:0.0.3
```

After the image is pulled, run the container.

```
docker run -p 8000:8000 image-matting:0.0.3
```

Alternatively, you can build the image yourself.

```
docker build -t image-matting:0.0.3 .
```

Once the build is complete, run the container as before. Visit http://127.0.0.1:8000 to perform online image matting.

Feel free to choose the method that suits your preference.

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
