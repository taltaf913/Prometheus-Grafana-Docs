from transformers import pipeline

class TranslationModel:
    def __init__(self, model_name="Helsinki-NLP/opus-mt-en-fr"):  # Example: English to French
        self.translator = pipeline("translation", model=model_name)

    def translate(self, text):
        result = self.translator(text)
        return result[0]['translation_text']
