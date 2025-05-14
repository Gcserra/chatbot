from transformers import pipeline
import gradio as gr

# Daha güçlü ama yine küçük bir model
generator = pipeline("text-generation", model="distilgpt2")

# Few-shot örnekler
few_shot_examples = (
    "Soru: Python nedir?\nCevap: Python, açık kaynaklı ve çok yönlü bir programlama dilidir.\n"
    "Soru: Yapay zeka nedir?\nCevap: Makinelerin düşünme ve öğrenme yeteneği kazanmasını sağlayan teknolojidir.\n"
    "Soru: Veri bilimi nedir?\nCevap: Verileri analiz ederek anlamlı bilgi elde etme bilimidir.\n"
)

# Chat fonksiyonu (uyumlu hale getirildi)
def chat_function(message, history):
    prompt = few_shot_examples + f"Soru: {message}\nCevap:"
    output = generator(prompt, max_new_tokens=60, do_sample=True, temperature=0.7)
    answer = output[0]['generated_text'].split("Cevap:")[-1].strip()
    return answer

# Gradio arayüzü
gr.ChatInterface(
    fn=chat_function,
    title="🧠 Few-shot Slaybot",
    theme="soft",
    examples=["Python nedir?", "Veri bilimi nedir?", "En büyük gezegen nedir?"]
).launch()

