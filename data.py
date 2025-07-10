import csv

def get_msmarco_data():
    id2doc, id2query = {}, {}
    g = open("./data/msmarco/sampled500.txt", "w", encoding="utf-8")
    with open("~/retrieval/beir/datasets/msmarco/queries.jsonl", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            json_dict = eval(line)
            id2query[json_dict['_id']] = json_dict['text']
    with open("~/retrieval/beir/datasets/msmarco/corpus.jsonl", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            json_dict = eval(line)
            id2doc[json_dict['_id']] = json_dict['text']
    with open("~/retrieval/beir/datasets/msmarco/qrels/dev.tsv", encoding="utf-8") as f:
        lines = f.readlines()[:501]
        for i, line in enumerate(lines):
            if i == 0:
                continue
            query_id, doc_id, _ = line.split("\t")
            query = id2query[query_id]
            doc = id2doc[doc_id]
            g.write(f"{query}\t{doc}\n")
    g.close()


def get_sts_data(data_name="sts12"):
    if data_name == "sts12":
        data_path = "./data/sts/STS12-en-test/STS.input.surprise.SMTnews.txt"
        score_path = "./data/sts/STS12-en-test/STS.gs.surprise.SMTnews.txt"
        texts = []
        with open(data_path, encoding="utf-8") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                fs, sc = line.replace(" .", "").replace(" ?", "").split("\t")
                texts += [(fs.strip(), sc.strip())]
        scores = []
        with open(score_path, encoding="utf-8") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                score = line.strip()
                scores += [score]

        new_texts, new_scores = [], []
        for text, score in zip(texts, scores):
            if float(score) > 4.0:
                new_texts.append(text)
                new_scores.append(score)
        texts_a = [text[0] for text in new_texts]
        texts_b = [text[1] for text in new_texts]
    elif data_name == "sts-b":
        data_path = "./data/STS/STSBenchmark/sts-train.csv"
        texts_a, texts_b = [], []
        with open(data_path, encoding="utf-8") as f:
            for row in csv.reader(f):
                data_list = row[0].split("\t")
                print(data_list)
                score, text_a, text_b = data_list[4], data_list[5], data_list[6]
                if float(score) > 4.0:
                    texts_a.append(text_a)
                    texts_b.append(text_b)
    else:
        raise NotImplementedError
    return texts_a, texts_b


def get_text_similarity_data(data_name):
    if data_name == "sts":
        texts_a, texts_b = get_sts_data()
    elif data_name == "msmarco":
        texts_a, texts_b = [], []
        with open("./data/msmarco/sampled500.txt", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                query, doc = line.strip().split("\t")
                texts_a.append(query)
                texts_b.append(doc)
    elif data_name == "nli":
        texts_a, texts_b = [], []
        with open("./data/nli_for_simcse.csv") as f:
            text_list = []
            lines = f.readlines()
            for i, line in enumerate(lines):
                if i == 0:
                    continue
                text_list = list(line.strip().split(","))
                texts_a.append(text_list[0])
                texts_b.append(text_list[1])
    elif data_name == "wiki":
        with open("./data/wiki/wiki1m_for_simcse.txt", "r") as f:
            lines = f.readlines()
        return lines
    else:
        raise NotImplementedError
    return texts_a, texts_b