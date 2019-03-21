import sys
import collections

model_file="hmmmodel.txt"

def extract_tags_words(training_data):
    word_dict = []
    tag_dict = []
    word_tag_dict = []
    tag_tag_dict = []
    initial_state_dict = []
    i = 0
    training_data = training_data.split("\n")

    for item in training_data[:-1]:
        state = "initial_state"
        item2 = item.split(" ")
        for element in item2:
            word_tag = element.rsplit('/', 1)

            word_dict.append(word_tag[0])
            tag_dict.append(word_tag[1])

            word_tag_dict.append((word_tag[1], word_tag[0]))

            if state == "initial_state":
                initial_state_dict.append((state, word_tag[1]))
                state = word_tag[1]
            else:
                tag_tag_dict.append((state, word_tag[1]))
                state = word_tag[1]

    initial_state_dict = dict(collections.Counter(initial_state_dict))
    word_dict = dict(collections.Counter(word_dict))
    tag_dict = dict(collections.Counter(tag_dict))
    word_tag_dict = dict(collections.Counter(word_tag_dict))
    tag_tag_dict = dict(collections.Counter(tag_tag_dict))
    len_training_data = len(training_data)
    len_tag_dict = len(tag_dict)

    for key, value in initial_state_dict.items():
        # initial_state_dict[key] = math.log((initial_state_dict[key])/len_training_data)
        initial_state_dict[key] = float(initial_state_dict[key] + 1) / float(len_training_data)

    smoothen2 = float(1) / float(len_training_data)
    for i in tag_dict:
        if ("initial_state", i) not in initial_state_dict:
            initial_state_dict.update({("initial_state", i): smoothen2})

    for key, value in tag_tag_dict.items():
        tag_tag_dict[key] = float(tag_tag_dict[key] + 1)/float(tag_dict[key[0]] + len(tag_dict))

    for i in tag_dict:
        for j in tag_dict:
            if (i, j) not in tag_tag_dict:
                tag_tag_dict.update({(i, j): float(1)/float(tag_dict[i] + len(tag_dict))})

    for key, value in word_tag_dict.items():
        word_tag_dict[key] = float(word_tag_dict[key])/float(tag_dict[key[0]])

    return word_dict, tag_dict, word_tag_dict, tag_tag_dict, initial_state_dict


def export_to_file(word_dict, tag_dict, word_tag_dict, tag_tag_dict, initial_state_dict):
    f = open(model_file, mode="w", encoding="utf-8")

    for key, value in word_dict.items():
        sentence = str(key) + "****" + str(value) + "\n"
        f.write(sentence)

    f.write("\n\n")

    for key, value in tag_dict.items():
        sentence = str(key) + "****" + str(value) + "\n"
        f.write(sentence)

    f.write("\n\n")

    for key, value in word_tag_dict.items():
        sentence = key[0] + "  " + key[1] + "****" + str(value) + "\n"
        f.write(sentence)

    f.write("\n\n")

    for key, value in tag_tag_dict.items():
        sentence = key[0] + "  " + key[1] + "****" + str(value) + "\n"
        f.write(sentence)

    f.write("\n\n")

    for key, value in initial_state_dict.items():
        sentence = key[0] + "  " + key[1] + "****" + str(value) + "\n"
        f.write(sentence)


if __name__=="__main__":
    train_file = sys.argv[1]
    data = open(train_file, mode="r", encoding="utf-8").read()
    word_dict, tag_dict, word_tag_dict, tag_tag_dict, initial_state_dict = extract_tags_words(data)
    export_to_file(word_dict, tag_dict, word_tag_dict, tag_tag_dict, initial_state_dict)
