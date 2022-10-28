coimport unittest
from simpletransformers.classification import ClassificationModel
from transformers import DistilBertModel, DistilBertTokenizer

class TestCompatibility(unittest.TestCase):

    #could generalize to test for different BERT models    

    #compatibility with simpletransformers
    def test_load_simpletransformers(self, modeltype="distilbert", path="outputs/test_output"):
        ClassificationModel(modeltype, path, use_cuda=False)

    def test_load_tokenizer(self, path="outputs/test_output"):
        DistilBertTokenizer.from_pretrained(path)
    
    def test_load_model(self, path="outputs/test_output"):
        DistilBertModel.from_pretrained(path)

if __name__ == '__main__':
    unittest.main()