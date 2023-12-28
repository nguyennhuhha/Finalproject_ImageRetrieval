import time
import faiss
from PIL import Image
import os

import streamlit as st
from streamlit_cropper import st_cropper

from src.feature_extraction import MyVGG16, MyResnet50, MyXception, MyEfficient

st.set_page_config(layout="wide")

def get_image_list(image_root):
    image_root=image_root
    image_list=[]
    file_names = os.listdir(image_root)
    for file_name in file_names:
        # Kiểm tra nếu là tệp ảnh
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            full_path = os.path.join(image_root, file_name)
            image_list.append(full_path)
    return image_list

def retrieve_image(img, data,feature_extractor,number_k):
    if (feature_extractor == 'VGG16'):
        extractor = MyVGG16()
    elif (feature_extractor == 'Resnet50'):
        extractor = MyResnet50()
    elif (feature_extractor == 'Xception'):
        extractor = MyXception()
    elif (feature_extractor == 'Efficient'):
        extractor = MyEfficient()
    
    if (data == 'Oxford'):
        image_root = './dataset/oxford'
        feature_root= './dataset/feature' + '/oxford/'
    else:
        image_root = './dataset/paris'
        feature_root= './dataset/feature' + '/paris/'


    feat = extractor.extract_features1(img)

    indexer = faiss.read_index(feature_root + feature_extractor + '.index.bin')

    _, indices = indexer.search(feat, k=number_k)

    return indices[0], image_root

def main():
    st.title('IMAGE RETRIEVAL')
    
    with st.container():
        #st.header('QUERY')
        with st.container():
            cola, colb = st.columns(2)
            with cola:
                st.subheader('Dataset')
                data = st.selectbox('', ( 'Paris', 'Oxford'))
            with colb:
                st.subheader('Model')
                option = st.selectbox('', ( 'Resnet50', 'VGG16', 'Xception', 'Efficient'))

        st.subheader('Upload image')
        img_file = st.file_uploader(label='', type=['png', 'jpg'])
        colc, cold = st.columns(2)
        with colc:
            if img_file:
                img = Image.open(img_file)
                # Get a cropped image from the frontend
                cropped_img = st_cropper(img, realtime_update=True, box_color='#FF0004')
                # Manipulate cropped image at will
                with cold:
                    st.write("Preview")
                    _ = cropped_img.thumbnail((300,300))
                    st.image(cropped_img)
        
        with st.form("query"):
            number_k = st.text_input("Number of outcomes: ")
            submitted = st.form_submit_button("Submit")
    
    with st.container():
        st.header('RESULT')
        col1, col2, col3 = st.columns(3)
        if submitted:
            if img_file:
                text_placeholder=st.markdown('**Retrieving .......**')
                start = time.time()

                retriev, image_root = retrieve_image(cropped_img, data, option, int(number_k))
                image_list = get_image_list(image_root)

                end = time.time()
                for i in range(0, int(number_k), 1):
                    print(image_list[retriev[i]])
                with col1: 
                    st.image(cropped_img)
                    st.markdown('**Finish in ' + str(end - start) + ' seconds**')
                    text_placeholder.empty()

                with col2:
                    for i in range(0,int(number_k), 2):
                        image = Image.open(image_list[retriev[i]])
                        st.image(image, use_column_width = 'always')

                with col3:
                    for i in range(1,int(number_k), 2):
                        image = Image.open(image_list[retriev[i]])
                        st.image(image, use_column_width = 'always')

if __name__ == '__main__':
    main()