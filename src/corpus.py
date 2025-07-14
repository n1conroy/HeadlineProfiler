import json
import datetime
import utils

class Corpus:
    def __init__(self):
        self.index = {}
        self.documents = []

    def build_index(self):
        self.documents = sorted(self.documents, key=lambda k: datetime.datetime.strptime(
            k["date"], "%Y-%m-%d %H:%M:%S"))
        self.index = {}
        i = -1
        for sorted_document in self.documents:
            rem = []
            for fn, fv in sorted_document["features"].items():
                if not fv:
                    rem.append(fn)
            for fr in rem:
                del(sorted_document["features"][fr])

            i += 1
            self.index[sorted_document["id"]] = i

    def get_document(self, id):
        return self.documents[self.index[id]]

     
    def load_corpora(self, dataset_path, dataset_tok_ner_path, languages):
        corpus_index = {}
        with open(dataset_path, errors="ignore", mode="r") as data_file:
            corpus_data = json.load(data_file)
            for d in corpus_data:
                corpus_index[d["id"]] = d
        with open(dataset_tok_ner_path, errors="ignore", mode="r") as data_file:
            for l in data_file:
                tok_ner_document = json.loads(l)
                corpus_index[tok_ner_document["id"]
                             ]["features"] = tok_ner_document["features"]

        self.documents = []
        for archive_document in corpus_index.values():
            if archive_document["lang"] not in languages:
                continue
            self.documents.append(archive_document)

        self.build_index()

class Document:
    def __init__(self, archive_document):
        self.id = archive_document["id"]
        self.title = archive_document["title"]
        self.body = archive_document['text']
        if ('cluster' in archive_document):
            self.cluster = archive_document["cluster"]
        else:
            self.cluster = 'NA'
        for fmt in ("%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M"):
            try:
                self.timestamp = datetime.datetime.strptime(archive_document["date"], fmt)
            except ValueError:
                pass
        self.reprs = utils.build_features(self)



    