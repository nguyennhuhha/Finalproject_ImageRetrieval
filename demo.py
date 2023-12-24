import time
import faiss
from PIL import Image
import os

import streamlit as st
from streamlit_cropper import st_cropper

from src.feature_extraction import MyVGG16, MyResnet50, MyXception

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

def retrieve_image(img, data,feature_extractor):
    if (feature_extractor == 'VGG16'):
        extractor = MyVGG16()
    elif (feature_extractor == 'Resnet50'):
        extractor = MyResnet50()
    elif (feature_extractor == 'Xception'):
        extractor = MyXception()
    
    if (data == 'Oxford'):
        image_root = './dataset/oxford'
        feature_root= './dataset/feature' + '/oxford/'
    else:
        image_root = './dataset/paris'
        feature_root= './dataset/feature' + '/paris/'


    feat = extractor.extract_features1(img)

    indexer = faiss.read_index(feature_root + feature_extractor + '.index.bin')

    _, indices = indexer.search(feat, k=21)

    return indices[0], image_root

def main():
    st.title('CONTENT-BASED IMAGE RETRIEVAL')
    
    col1, col2 = st.columns(2)

    with col1:
        st.header('QUERY')

        st.subheader('Choose feature extractor')
        data = st.selectbox('Dataset', ( 'Paris', 'Oxford'))

        st.subheader('Choose feature extractor')
        option = st.selectbox('Model', ( 'Resnet50', 'VGG16', 'Xception'))

        st.subheader('Upload image')
        img_file = st.file_uploader(label='', type=['png', 'jpg'])

        if img_file:
            img = Image.open(img_file)
            # Get a cropped image from the frontend
            cropped_img = st_cropper(img, realtime_update=True, box_color='#FF0004')
            
            # Manipulate cropped image at will
            st.write("Preview")
            _ = cropped_img.thumbnail((150,150))
            st.image(cropped_img)

    with col2:
        st.header('RESULT')
        if img_file:
            st.markdown('**Retrieving .......**')
            start = time.time()

            retriev, image_root = retrieve_image(cropped_img, data, option)
            image_list = get_image_list(image_root)

            end = time.time()
            st.markdown('**Finish in ' + str(end - start) + ' seconds**')

            col3, col4 = st.columns(2)

            with col3:
                image = Image.open(image_list[retriev[0]])
                st.image(image, use_column_width = 'always')

            with col4:
                image = Image.open(image_list[retriev[1]])
                st.image(image, use_column_width = 'always')

            col5, col6, col7 = st.columns(3)

            with col5:
                for u in range(2, 11, 3):
                    image = Image.open(image_list[retriev[u]])
                    st.image(image, use_column_width = 'always')

            with col6:
                for u in range(3, 11, 3):
                    image = Image.open(image_list[retriev[u]])
                    st.image(image, use_column_width = 'always')

            with col7:
                for u in range(4, 11, 3):
                    image = Image.open(image_list[retriev[u]])
                    st.image(image, use_column_width = 'always')

if __name__ == '__main__':
    main()