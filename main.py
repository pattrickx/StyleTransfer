from src.StyleTransfer import StyleTransfer

st = StyleTransfer()
style = "./image_sample/The Starry Night.jpg"
img = "./image_sample/hoovertowernight.jpg"
st.Transfer(img,style,steps=200)
