import streamlit as st
from skimage.io import imread
from skimage.transform import resize
import pickle
from PIL import Image
st.title("Imaeg classifier")
st.text("upload image")
