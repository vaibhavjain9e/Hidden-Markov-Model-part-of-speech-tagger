import sys
import numpy as np
import math

output_file="hmmoutput.txt"
model_file="hmmmodel.txt"


def extract_data(data):
    data = data.split("\n\n\n")
    word_dict = {}
    tag_dict = {}
    word_tag_dict = {}
    tag_tag_dict = {}
    initial_state_dict = {}
    for item in data[0].split("\n"):
        key_value = item.rsplit("****", 1)
        word_dict.update({key_value[0]: int(key_value[1])})

    for item in data[1].split("\n"):
        key_value = item.rsplit("****", 1)
        tag_dict.update({key_value[0]: int(key_value[1])})

    for item in data[2].split("\n"):
        key_value = item.rsplit("****", 1)
        key = key_value[0].split("  ")
        word_tag_dict.update({(key[0], key[1]): float(key_value[1])})

    for item in data[3].split("\n"):
        key_value = item.rsplit("****", 1)
        key = key_value[0].split("  ")
        tag_tag_dict.update({(key[0], key[1]): float(key_value[1])})

    for item in data[4].split("\n")[:-1]:
        key_value = item.rsplit("****", 1)
        key = key_value[0].split("  ")
        initial_state_dict.update({(key[0], key[1]): float(key_value[1])})

    return word_dict, tag_dict, word_tag_dict, tag_tag_dict, initial_state_dict


def find_tags(word_dict, tag_dict, word_tag_dict, tag_tag_dict, initial_state_dict, input_data):
    tagged_data = []

    for item in input_data.split("\n")[:-1]:
        word_list = item.split(" ")
        tag_list = VITERBI(word_dict, tag_dict, word_tag_dict, tag_tag_dict, initial_state_dict, item.split(" "))
        sentence = ""
        for i in range(0, len(word_list)):
            sentence = sentence + word_list[i] + "/" + tag_list[i] + " "
        tagged_data.append(sentence.strip())
    return tagged_data


def export_to_file(tagged_data):
    f = open(output_file, mode="w", encoding="utf-8")

    for item in tagged_data:
        f.write(item)
        f.write("\n")


def VITERBI(word_dict, tag_dict, word_tag_dict, tag_tag_dict, initial_state_dict, sentence):
    T = len(sentence)
    N = len(tag_dict)
    viterbi = np.zeros(shape=(N, T))
    backpointer = np.zeros(shape=(N, T))

    tag_list = []
    for item in tag_dict:
        tag_list.append(item)

    if sentence[0] in word_dict:
        for i in range(0, len(tag_list)):
            b = (tag_list[i], sentence[0])
            if b in word_tag_dict:
                b = math.log(float(word_tag_dict[b]))
            else:
                b = 0

            viterbi[i][0] = math.log(float(initial_state_dict[('initial_state', tag_list[i])])) + b

            if b == 0:
                viterbi[i][0] = 0

            backpointer[i][0] = 0

    else:
        for i in range(0, len(tag_list)):
            viterbi[i][0] = math.log(float(initial_state_dict[('initial_state', tag_list[i])]))
            backpointer[i][0] = 0

    for t in range(1, len(sentence)):
        if sentence[t] in word_dict:
            for s, state in enumerate(tag_list):
                max_value = -10000000
                arg_max_value = -1
                b = (tag_list[s], sentence[t])
                if b in word_tag_dict:
                    b = math.log(float(word_tag_dict[b]))
                else:
                    b = 0
                for i, s_dash in enumerate(tag_list):
                    val = viterbi[i, t-1] + b + math.log(float(tag_tag_dict[s_dash, state]))

                    if b == 0:
                        val = 0

                    if val > max_value:
                        max_value = val
                        arg_max_value = i

                viterbi[s][t] = max_value
                backpointer[s][t] = arg_max_value


        else:
            for s, state in enumerate(tag_list):
                max_value = -10000000
                arg_max_value = -1
                for i, s_dash in enumerate(tag_list):
                    val = viterbi[i, t-1] + math.log(float(tag_tag_dict[s_dash, state]))

                    if val > max_value:
                        max_value = val
                        arg_max_value = i

                viterbi[s][t] = max_value
                backpointer[s][t] = arg_max_value

    viterbi = viterbi.T
    backpointer = backpointer.T

    index = np.argmax(viterbi[len(sentence) - 1])

    count = len(sentence) - 1
    tagged_sentence = []
    while count >= 0:
        tagged_sentence.append(tag_list[index])
        index = int(backpointer[count][index])
        count = count - 1

    tagged_sentence.reverse()

    return tagged_sentence


if __name__=="__main__":
    input_file = sys.argv[1]
    model_data = open(model_file, mode="r", encoding="utf-8").read()
    word_dict, tag_dict, word_tag_dict, tag_tag_dict, initial_state_dict = extract_data(model_data)
    input_data = open(input_file, mode="r", encoding="utf-8").read()
    tagged_data = find_tags(word_dict, tag_dict, word_tag_dict, tag_tag_dict, initial_state_dict, input_data)
    export_to_file(tagged_data)





