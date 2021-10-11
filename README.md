# StyleTransfer

This project is based on the VGG19 Network, to extract style features from an image and insert it into another image, below you can see an example, where The Starry Night painting was passed as a style image and as a target image a picture of hoover tower at night, the image on the right is the result after 1000 epochs, using the project's default hyperparameters.

<p float="left">
<img src="https://github.com/pattrickx/StyleTransfer/blob/main/image_sample/The%20Starry%20Night.jpg" height="300" width="300">
<img src="https://github.com/pattrickx/StyleTransfer/blob/main/image_sample/hoovertowernight.jpg" height="300" width="300">
<img src="https://github.com/pattrickx/StyleTransfer/blob/main/image_sample/hoovertowernightLast.png" width="300">
</p>

> :warning: **If you don't have a gpu with cuda, the style transfer execution time will be much longer**

# Prerequisites
Python >=3.8.10
# How to Install
```
sudo pip3 install -r requirements.txt 
```
# How to Use
if you are going to test with the example images, just run:
```
python3 main.py
```
but if you want to use with your own images change the path to the variables shown below in the main.py file:
```
style = "your style image"
img = "your base image"
```
and next run:
```
python3 main.py
```
##### Based on
* [NEURAL TRANSFER USING PYTORCH](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)
* [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)
* [Aladdin Persson Pytorch Neural Style Transfer](https://www.youtube.com/watch?v=imX4kSKDY7s)
