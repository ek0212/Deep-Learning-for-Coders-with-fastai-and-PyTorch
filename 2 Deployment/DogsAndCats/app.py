from fastai.vision.all import*
import gradio as gr

def is_cat(x): return x[0].isupper()
learn = load_learner('model.pkl')
categories = ('Dog', 'Cat')
def classify_img(img):
    pred,pred_idx,probs = learn.predict(img)
    return dict(zip(categories, map(float,probs)))
image = gr.Image()  # No shape argument needed
label = gr.Label()  # Use gr.Label instead of gr.outputs.Label
examples = ['dog.jpg', 'cat.jpg', 'dunno.jpg']

inf= gr.Interface(fn=classify_img, inputs=image, outputs=label, examples=examples)
inf.launch(inline=False)