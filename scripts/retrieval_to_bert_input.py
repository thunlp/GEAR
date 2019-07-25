import json
import sqlite3
from tqdm import tqdm
from drqa.retriever import utils
ENCODING = 'utf-8'
DATABASE = '../data/fever/fever.db'

conn = sqlite3.connect(DATABASE)
cursor = conn.cursor()

'''
    Build train/dev/test set from retrieval results for BERT.
'''
def process(input, output):
    fin = open(input, 'rb')
    instances = []
    index = 0
    for line in fin:
        object = json.loads(line.decode(ENCODING).strip('\r\n'))
        if 'label' in object:
            label = ''.join(object['label'].split(' '))
        else:
            label = 'REFUTES'
        evidences = object['predicted_evidence']
        claim = object['claim']
        instances.append([index, label, claim, evidences])
        index += 1
    fin.close()
    print(index)

    fout = open(output, 'wb')
    for instance in tqdm(instances):
        index, label, claim, evidences = instance
        for evidence in evidences:
            article = evidence[0]
            location = evidence[1]
            evidence_str = None
            cursor.execute(
                "SELECT * FROM documents WHERE id = ?",
                (utils.normalize(article),)
            )
            for row in cursor:
                sentences = row[2].split('\n')
                for sentence in sentences:
                    if sentence == '': continue
                    arr = sentence.split('\t')
                    if not arr[0].isdigit():
                        # print(('Warning: this line from article %s for claim %d is not digit %s\r\n' % (article, i, sentence)).encode(ENCODING))
                        continue
                    line_num = int(arr[0])
                    if len(arr) <= 1: continue
                    sentence = ' '.join(arr[1:])
                    if sentence == '':
                        continue
                    if line_num == location:
                        evidence_str = sentence
                        break
            if evidence_str:
                fout.write(('%s\t%s\t%s\t%s\t%s\t%d\t%s\r\n' % (label, evidence_str, claim, index, evidence[0], evidence[1], evidence[2])).encode(ENCODING))
            else:
                print('Error: cant find %s %d for %s' % (article, location, index))
    fout.close()


'''
    Build support/refute samples of train dataset for BERT.
'''
def build_bert_train_sr_set(data_dir, output_dir):
    fin = open(data_dir, 'rb')
    fout = open(output_dir, 'wb')
    cnt = -1
    for line in fin:
        cnt += 1
        if cnt % 1000 == 0:
            print(cnt)

        data = json.loads(line)
        claim = data['claim']
        evidences = data['evidence']
        label = data['label']

        if label == 'NOT ENOUGH INFO':
            continue

        for evidence_set in evidences:
            # text_set = []
            for evidence in evidence_set:
                article = evidence[2]
                article_index = evidence[3]
                try:
                    cursor.execute("select * from documents where id='%s';" % article.replace("'", "''"))
                except Exception as e:
                    print(e)
                    continue
                for row in cursor:
                    lines = row[2].split('\n')
                    items = lines[article_index].split('\t')

                    sentence = ' '.join(items[1:])
                    # words = sentence.split(' ')
                    # if len(words) > 100:
                    #     sentence = ' '.join(words[:100])
                    # text_set.append(sentence)

                    fout.write(('%s\t%s\t%s\t%d\t%s\t%s\r\n' % (label, sentence, claim, cnt, article, article_index)).encode(
                            ENCODING))

            # total_evidence = ' '.join(text_set)
            # total_evidence_words = total_evidence.split(' ')
            # if len(total_evidence_words) > 100:
            #     total_evidence = ' '.join(total_evidence_words[:100])
            # if len(total_evidence_words) + len(claim.split(' ')) > 120:
            #    total_evidence = ' '.join(total_evidence_words[:120 - len(claim.split(' '))])

            # fout.write(('%s\t%s\t%s\r\n' % (label, claim, total_evidence)).encode(ENCODING))

    fin.close()
    fout.close()


if __name__ == '__main__':
    build_bert_train_sr_set('../data/fever/train.jsonl', '../data/bert/bert-nli-train-sr-set.tsv')
    process('../data/retrieved/train.ensembles.s10.jsonl', '../data/bert/bert-nli-train-retrieve-set.tsv')
    process('../data/retrieved/dev.ensembles.s10.jsonl', '../data/bert/bert-nli-dev-retrieve-set.tsv')
    process('../data/retrieved/test.ensembles.s10.jsonl', '../data/bert/bert-nli-test-retrieve-set.tsv')
