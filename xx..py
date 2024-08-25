import streamlit as st
import matplotlib.pyplot as plt

# Simple test plot
plt.figure()
plt.plot([1, 2, 3], [4, 5, 6])
st.pyplot(plt.gcf())
