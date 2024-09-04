import requests
from preprocess import Preprocessor

class Embedder():
    def embed(self, prompt):
        res = requests.post('http://localhost:11434/api/embeddings', json = {
                                'model': 'nomic-embed-text',
                                'prompt': prompt
                            }
                        )

        embedding = res.json()['embedding']
        print(embedding)

Preprocessor().clean_assembly_file(input_file = "goodware/br_com_tapps_bidwars2.txt")