ENCODING = 'utf-8'
SENTENCE_NUM = 5


def is_evidence_exist(evidence_set, evid):
    for evidence in evidence_set:
        if evid[1] == evidence[1] and evid[2] == evidence[2]:
            return True
    return False

'''
    Build GEAR train set with truth evidence and retrieval evidence.
'''
def build_with_truth_and_threshold(truth_file, athene_file, output_file, label_file, threshold):
    fin = open(truth_file, 'rb')
    instances = {}
    for line in fin:
        label, evidence, claim, index, article, article_index = line.decode(ENCODING).strip('\r\n').split('\t')

        evidence_tuple = [evidence, article, article_index, 1]

        if index not in instances:
            instances[index] = {}
            instances[index]['claim'] = claim
            instances[index]['label'] = label
            instances[index]['evidences'] = [evidence_tuple]
        else:
            assert instances[index]['claim'] == claim
            if not is_evidence_exist(instances[index]['evidences'], evidence_tuple):
                instances[index]['evidences'].append(evidence_tuple)
    fin.close()
    print('Finish reading truth file...')

    retrieve_instances = {}
    fin = open(athene_file, 'rb')
    for line in fin:
        label, evidence, claim, index, article, article_index, confident = line.decode(ENCODING).strip('\r\n').split('\t')

        evidence_tuple = [evidence, article, article_index, float(confident)]

        if index not in retrieve_instances:
            retrieve_instances[index] = {}
            retrieve_instances[index]['claim'] = claim
            if evidence_tuple[3] >= threshold:
                retrieve_instances[index]['evidences'] = [evidence_tuple]
            else:
                retrieve_instances[index]['evidences'] = []
        else:
            assert retrieve_instances[index]['claim'] == claim
            if not is_evidence_exist(retrieve_instances[index]['evidences'], evidence_tuple):
                if evidence_tuple[3] >= threshold:
                    retrieve_instances[index]['evidences'].append(evidence_tuple)
    fin.close()
    print('Finish reading retrieve file...')

    total_keys = list(instances.keys())
    total_keys.extend(list(retrieve_instances.keys()))
    total_keys = list(set(total_keys))
    total_keys = sorted(total_keys, key=lambda x: int(x))
    print(len(retrieve_instances.keys()), len(total_keys))

    for index in total_keys:
        if index not in instances:
            if index not in retrieve_instances:
                print("Cannot find the index: %s" % index)
                continue
            instances[index] = retrieve_instances[index]
            instances[index]['label'] = 'NOTENOUGHINFO'

        instance = instances[index]
        if len(instance['evidences']) < SENTENCE_NUM:
            if index in retrieve_instances:
                pos = 0
                while len(instance['evidences']) < SENTENCE_NUM and pos < len(retrieve_instances[index]['evidences']):
                    evidence = retrieve_instances[index]['evidences'][pos]
                    if not is_evidence_exist(instance['evidences'], evidence):
                        instance['evidences'].append(evidence)
                    pos += 1
            else:
                print('Warning: %s' % index)
    print('Finish adding evidences...')

    fout = open(output_file, 'wb')
    # flog = open(label_file, 'wb')
    for index in total_keys:
        instance = instances[index]
        output_line = '%s\t%s\t%s' % (index, instance['label'], instance['claim'])
        label_line = '%s\t' % index
        label_list = []
        try:
            assert len(instance['evidences']) >= SENTENCE_NUM
        except Exception as _:
            pass
        for evidence in instance['evidences'][:SENTENCE_NUM]:
            output_line += ('\t%s' % evidence[0])
            label_list.append(str(evidence[3]))
        output_line += '\r\n'
        while len(label_list) < SENTENCE_NUM:
            label_list.append('0')

        label_line += '\t'.join(label_list) + '\r\n'
        fout.write(output_line.encode(ENCODING))
        # flog.write(label_line.encode(ENCODING))
    fout.close()
    # flog.close()

'''
    Build GEAR dev/test set with retrieval evidence.
'''
def build_with_threshold(input, output, threshold):
    fin = open(input, 'rb')
    instances = {}
    for line in fin:
        label, evidence, claim, index, _, _, confidence = line.decode(ENCODING).strip('\r\n').split('\t')
        confidence = float(confidence)

        if not index in instances:
            instances[index] = {}
            instances[index]['claim'] = claim
            instances[index]['label'] = label
            if confidence >= threshold:
                instances[index]['evidences'] = [evidence]
            else:
                instances[index]['evidences'] = []
        else:
            assert instances[index]['label'] == label
            assert instances[index]['claim'] == claim
            if confidence >= threshold:
                instances[index]['evidences'].append(evidence)
    fin.close()
    instances = sorted(instances.items(), key=lambda x: int(x[0]))

    fout = open(output, 'wb')
    for instance in instances:
        output_line = '%s\t%s\t%s' % (instance[0], instance[1]['label'], instance[1]['claim'])
        if len(instance[1]['evidences']) == 0:
            print(0)
        for evidence in instance[1]['evidences']:
            output_line += ('\t%s' % evidence)
        output_line += '\r\n'
        fout.write(output_line.encode(ENCODING))
    fout.close()


if __name__ == '__main__':
    print('Start building gear train set...')
    build_with_truth_and_threshold('../data/bert/bert-nli-train-sr-set.tsv',
                                   '../data/bert/bert-nli-train-retrieve-set.tsv',
                                   '../data/gear/gear-train-set-0_001.tsv',
                                   'none.tsv', 0.001)

    print('Start building gear dev set...')
    build_with_threshold('../data/bert/bert-nli-dev-retrieve-set.tsv',
                         '../data/gear/gear-dev-set-0_001.tsv', 0.001)
    # build_with_threshold('../data/bert/bert-nli-dev-retrieve-set.tsv', '../data/gear/gear-dev-set-0_1.tsv', 0.1)
    # build_with_threshold('../data/bert/bert-nli-dev-retrieve-set.tsv', '../data/gear/gear-dev-set-0_01.tsv', 0.01)
    # build_with_threshold('../data/bert/bert-nli-dev-retrieve-set.tsv', '../data/gear/gear-dev-set-0_0001.tsv', 0.0001)

    print('Start building gear test set...')
    build_with_threshold('../data/bert/bert-nli-test-retrieve-set.tsv',
                         '../data/gear/gear-test-set-0_001.tsv', 0.001)
    # build_with_threshold('../data/bert/bert-nli-test-retrieve-set.tsv', '../data/gear/gear-test-set-0_1.tsv', 0.1)
    # build_with_threshold('../data/bert/bert-nli-test-retrieve-set.tsv', '../data/gear/gear-test-set-0_01.tsv', 0.01)
    # build_with_threshold('../data/bert/bert-nli-test-retrieve-set.tsv', '../data/gear/gear-test-set-0_0001.tsv', 0.0001)
