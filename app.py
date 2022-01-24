
from gramformer import Gramformer
import torch
import gradio as gr

def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

set_seed(1212)

"""# Instantiate Gramformer"""

gf = Gramformer(models = 1, use_gpu=False) # 1=corrector, 2=detector



"""# Creating Gradio Interface"""

def func(text):
  lst = []
  lst.append(text)
  for text in lst:
    corrected_sentences = gf.correct(text, max_candidates=1)
    for corrected_sentence in corrected_sentences:
      return("[Edits] ", gf.highlight(text, corrected_sentence[0]))

app = gr.Interface(func, inputs="textbox", description="To use this model, you put in a piece of text and the model provides correction to the text as output", outputs="textbox")

#Initiating gradio app
app.launch()



