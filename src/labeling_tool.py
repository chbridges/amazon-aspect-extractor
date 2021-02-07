import datetime
import json
import sys
from collections import OrderedDict
from time import sleep
from tkinter import *

label_file = sys.argv[1]

rev_dict = []

with open(label_file) as json_data:
    rev_dict = json.load(json_data)["data"]
    json_data.close()

current_rev = 0


root = Tk()
f1 = Frame(root, bg="lightgray")
f2 = Frame(root, bg="lightgray")
f1.grid(sticky="nsew")
f2.grid(sticky="nsew")
Grid.columnconfigure(root, 0, weight=1)
Grid.rowconfigure(root, 0, weight=1)
root.title("Labeling tool")

# This is the first Question
Bar = Scrollbar(f1)
T = Text(f1, fg="black", bg="white", yscrollcommand=Bar.set, width=50, height=20)
Bar.config(command=T.yview)
T.grid(row=0, column=0, columnspan=4, sticky="nsew")
Bar.grid(row=0, column=2, sticky="ns")
Grid.columnconfigure(f1, 0, weight=1)
Grid.rowconfigure(f1, 0, weight=1)

T.tag_config("positive", background="green")
T.tag_config("neutral", background="yellow")
T.tag_config("negative", background="red")


def strip_emojis(text):
    char_list = [
        text[j] if ord(text[j]) in range(65536) else "?" for j in range(len(text))
    ]
    return "".join(char_list)


def get_next_rev():
    global current_rev
    if current_rev < len(rev_dict) - 1:
        current_rev += 1
    else:
        current_rev = 0
    return rev_dict[current_rev]


def get_last_rev():
    global current_rev
    if current_rev > 0:
        current_rev -= 1
    else:
        current_rev = len(rev_dict) - 1
    return rev_dict[current_rev]


def delete_selected():
    global current_rev
    index = T.index(INSERT)
    to_delete = []
    line = int(index.split(".")[0])
    col = int(index.split(".")[1])
    for i, annotation in enumerate(rev_dict[current_rev]["opinions"][line - 1]):
        start_ind = "{}.{}".format(line, annotation["target_position"][0])
        end_ind = "{}.{}".format(line, annotation["target_position"][1])
        if (
            int(annotation["target_position"][0]) <= col
            and int(annotation["target_position"][1]) > col
        ):
            to_delete.append(i)
            T.tag_remove(annotation["polarity"], start_ind, end_ind)

    # to_delete.sort() #shouldnt be necessary, just for safety
    for ind in to_delete[::-1]:
        del rev_dict[current_rev]["opinions"][line - 1][ind]


def add_annotation(polarity):
    global current_rev
    if T.tag_ranges(SEL):
        content = T.get(SEL_FIRST, SEL_LAST)
        start, end = T.index(SEL_FIRST), T.index(SEL_LAST)
        line = int(start.split(".")[0])
        rev_dict[current_rev]["opinions"][line - 1].append(
            {
                "target_position": (start.split(".")[1], end.split(".")[1]),
                "polarity": polarity,
                "target": content,
                "category": "",
                "subcategory": "",
            }
        )
        T.tag_add(polarity, SEL_FIRST, SEL_LAST)


def del_all_tags():
    global current_rev
    for line, annotations in enumerate(rev_dict[current_rev]["opinions"]):
        for annotation in annotations:
            start_ind = "{}.{}".format(line + 1, annotation["target_position"][0])
            end_ind = "{}.{}".format(line + 1, annotation["target_position"][1])
            T.tag_remove(annotation["polarity"], start_ind, end_ind)


def add_all_tags():
    global current_rev
    for line, annotations in enumerate(rev_dict[current_rev]["opinions"]):
        for annotation in annotations:
            start_ind = "{}.{}".format(line + 1, annotation["target_position"][0])
            end_ind = "{}.{}".format(line + 1, annotation["target_position"][1])
            T.tag_add(annotation["polarity"], start_ind, end_ind)


def save():
    with open(label_file, "w") as json_data:
        json.dump({"data": rev_dict}, json_data, indent=4)
        json_data.close()


def prev_rev():
    del_all_tags()
    T.delete("1.0", END)
    rev = get_last_rev()
    T.insert(END, strip_emojis("\n".join(rev["text"])))
    add_all_tags()


def next_rev():
    del_all_tags()
    T.delete("1.0", END)
    rev = get_next_rev()
    T.insert(END, strip_emojis("\n".join(rev["text"])))
    add_all_tags()


def pos_rev():
    add_annotation("positive")
    save()


def neut_rev():
    add_annotation("neutral")
    save()


def neg_rev():
    add_annotation("negative")
    save()


def del_op():
    delete_selected()
    save()


rev = rev_dict[current_rev]
T.insert(END, strip_emojis("\n".join(rev["text"])))
add_all_tags()


# Make a Send button to send the answer from the entry box to the definition of answer1
prev = Button(f2, text="<<", command=prev_rev)
prev.grid(row=0, column=0, sticky="nswe")
nex = Button(f2, text=">>", command=next_rev)
nex.grid(row=0, column=5, sticky="nswe")
pos = Button(f2, text="positive", command=pos_rev)
pos.grid(row=0, column=1, sticky="nswe")
neutral = Button(f2, text="neutral", command=neut_rev)
neutral.grid(row=0, column=2, sticky="nswe")
neg = Button(f2, text="negative", command=neg_rev)
neg.grid(row=0, column=3, sticky="nswe")
delete = Button(f2, text="delete", command=del_op)
delete.grid(row=0, column=4, sticky="nswe")

root.mainloop()
