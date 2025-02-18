import logging
import traceback
import spacy

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class NERProcessor:
    def __init__(self, model_path):
        try:
            self.trained_ner = spacy.load(model_path)
            logger.info(f"Loaded NER model from {model_path}")
        except Exception as e:
            logger.error(f"Error initializing NERProcessor with model path {model_path}: {e}")
            logger.debug(traceback.format_exc())
            raise

    def apply_custom_ner_model(self, text):
        try:
            logger.info(f"Applying NER model to text of length {len(text)}")
            doc = self.trained_ner(text)
            entities = [(ent.label_, ent.text) for ent in doc.ents]
            logger.info(f"Extracted {len(entities)} entities")
            return entities, doc.text
        except Exception as e:
            logger.error(f"Error applying custom NER model to text: {e}")
            logger.debug(traceback.format_exc())
            return [], text

