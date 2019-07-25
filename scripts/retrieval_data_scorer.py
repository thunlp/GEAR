import json
from fever.scorer import fever_score
ENCODING = 'utf-8'


def check(file, threshold, max_evidence=5):
    fin = open(file, 'rb')
    instances = []

    for line in fin:
        instance = json.loads(line.decode(ENCODING))
        evidences = []
        for evidence in instance['predicted_evidence']:
            if float(evidence[2]) < threshold:
                continue
            evidence = [evidence[0], evidence[1]]
            evidences.append(evidence)
        instance['predicted_evidence'] = evidences
        instances.append(instance)

    fin.close()

    strict_score, label_accuracy, precision, recall, f1 = fever_score(instances, actual=None, max_evidence=max_evidence)
    print('Evidence precision:', precision)
    print('Evidence recall:', recall)
    print('Evidence f1:', f1)


if __name__ == '__main__':
    check('../data/retrieved/dev.ensembles.s10.jsonl', 0.01)
