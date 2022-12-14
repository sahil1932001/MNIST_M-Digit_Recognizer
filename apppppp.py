import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow import keras
import cv2
import numpy as np
from PIL import Image


@st.cache(allow_output_mutation = True)
def load_model():
    
    model = keras.models.load_model('MNIST-M.h5')
    return model

model = load_model()


def main():

    PAGES = {
        "About": about,
        "Prediction": app,

    }
    page = st.sidebar.selectbox("Pages :", options=list(PAGES.keys()))
    PAGES[page]()

    with st.sidebar:
        st.markdown("---")
        
        st.markdown("**Stay Connected With Me On**")
        
        st.markdown(
            '<div style="margin: 0.75em 0;"><a href="https://www.linkedin.com/in/sahil-shaikh-707474204/" target="_blank"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" alt="Buy Me A Coffee" height="40" width="40"></a></div>',
            unsafe_allow_html=True,
        )
        
        st.markdown(
            '<div style="margin: 0.75em 0;"><a href="https://github.com/sahil1932001" target="_blank"><img src="https://i.pinimg.com/736x/b5/1b/78/b51b78ecc9e5711274931774e433b5e6.jpg" alt="Buy Me A Coffee" height="40" width="40"></a></div>',
            unsafe_allow_html=True,
        )



def about():
    
    st.write("---")
    
    st.markdown(
        """
        I Loaded this MNIST_M Dataset Using Torchvision dataset.
        
        You can see the [source code](https://github.com/liyxi/mnist-m/blob/main/mnist_m.py) through wich you can download the MNIST_M dataset
        """
        )
    
    
    st.write("---")

    st.markdown(
        
        """ 
        **MNIST-M** is created by combining **MNIST** digits with the patches randomly extracted from color photos of BSDS500 as their background. 
        
        It contains 60,000 training and 10,000 test images.
        """
        )
    
    st.write("---")
        
    st.markdown(
        
        """
        
        The **MNIST** database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits that is commonly used for training various image processing systems. 
        
        The database is also widely used for training and testing in the field of machine learning.
        
        """
        )


def app():
    st.sidebar.header("Configuration")
    st.markdown(
        """
        ##### Draw a digit in the range of 0-9 in the box below
    """
    )

    with st.echo("below"):
        # Specify canvas parameters in application
        drawing_mode='freedraw'
        stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 24)
        stroke_color = st.sidebar.color_picker("Stroke color hex: ")
        bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
        bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
        realtime_update = st.sidebar.checkbox("Update in realtime", True)

        # Create a canvas component
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            background_image=Image.open(bg_image) if bg_image else None,
            update_streamlit=realtime_update,
            height=200,width=200,
            drawing_mode=drawing_mode,
            display_toolbar=st.sidebar.checkbox("Display toolbar", True),
            key="app",
        )

        st.write("---")
        
        if canvas_result.image_data is not None:
            
            img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
            #input_numpy_array = np.array(img)
            #input_image = Image.fromarray(input_numpy_array.astype('uint8'), 'RGBA')
            
            img_rescaling = cv2.resize(img, (200, 200), interpolation=cv2.INTER_NEAREST)
            st.markdown(""" ##### Input Image """)
            st.image(img_rescaling)


        if st.button('Predict'):
            test_x = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            input_numpy_array = np.array(test_x)
            x_train = input_numpy_array.reshape((1,28,28,3))
            x_train = x_train.astype('float32')/255
            
            
            pred = model.predict(x_train)
            st.write(f'Result is : {np.argmax((pred[0]))}')
            st.bar_chart(pred[0])
            st.image(x_train)
            

if __name__ == "__main__":

    st.title("MNIST_M Digit Recognizer")
    main()
    
