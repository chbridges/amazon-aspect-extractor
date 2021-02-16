import datetime
import json
import sys
from collections import OrderedDict
from time import sleep
from tkinter import *
from typing import List

label_file = sys.argv[1]

rev_dict = []

with open(label_file) as json_data:
    # load the dataset
    rev_dict = json.load(json_data)["data"]
    json_data.close()


# index of currently displayed review
current_rev = 0

# Main window
root = Tk()
f1 = Frame(root, bg="lightgray")
f2 = Frame(root, bg="lightgray")
f1.grid(sticky="nsew")
f2.grid(sticky="nsew")
Grid.columnconfigure(root, 0, weight=1)
Grid.rowconfigure(root, 0, weight=1)
root.title("Labeling tool")

# Text window, which shows the review
Bar = Scrollbar(f1)
T = Text(f1, fg="black", bg="white", yscrollcommand=Bar.set, width=50, height=20)
Bar.config(command=T.yview)
T.grid(row=0, column=0, columnspan=4, sticky="nsew")
Bar.grid(row=0, column=2, sticky="ns")
Grid.columnconfigure(f1, 0, weight=1)
Grid.rowconfigure(f1, 0, weight=1)


# Tags for each of the sentiments (highlighting color)
T.tag_config("positive", background="green")
T.tag_config("neutral", background="yellow")
T.tag_config("negative", background="red")

# From a list of words, replace all characters over u+65536 with ?
# (can not be displayed by tkinter) and unify into one review text
def strip_emojis(text: List[str]):
    char_list = [
        text[j] if ord(text[j]) in range(65536) else "?" for j in range(len(text))
    ]
    return "".join(char_list)


# Advance to next review, wrap if end of file is reached
def get_next_rev():
    global current_rev
    if current_rev < len(rev_dict) - 1:
        current_rev += 1
    else:
        current_rev = 0
    return rev_dict[current_rev]


# Go back to previous review, wrap if start of file is reached
def get_last_rev():
    global current_rev
    if current_rev > 0:
        current_rev -= 1
    else:
        current_rev = len(rev_dict) - 1
    return rev_dict[current_rev]


# delete the selected annotations, where the cursor is located
def delete_selected():
    global current_rev
    index = T.index(INSERT)  # index of cursor
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


# add an annotation at the marked text
def add_annotation(polarity: str):
    global current_rev
    if T.tag_ranges(SEL):
        content = T.get(SEL_FIRST, SEL_LAST)  # get highlighted text
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
        T.tag_add(polarity, SEL_FIRST, SEL_LAST)  # add highlighting in the text


# delete all highlighting from the text
def del_all_tags():
    global current_rev
    for line, annotations in enumerate(rev_dict[current_rev]["opinions"]):
        for annotation in annotations:
            start_ind = "{}.{}".format(line + 1, annotation["target_position"][0])
            end_ind = "{}.{}".format(line + 1, annotation["target_position"][1])
            T.tag_remove(annotation["polarity"], start_ind, end_ind)


# add all highlighting to the text
def add_all_tags():
    global current_rev
    for line, annotations in enumerate(rev_dict[current_rev]["opinions"]):
        for annotation in annotations:
            start_ind = "{}.{}".format(line + 1, annotation["target_position"][0])
            end_ind = "{}.{}".format(line + 1, annotation["target_position"][1])
            T.tag_add(annotation["polarity"], start_ind, end_ind)


# save the current dataset dictionary
def save():
    with open(label_file, "w") as json_data:
        json.dump({"data": rev_dict}, json_data, indent=4)
        json_data.close()


# switch the text field back one review
def prev_rev():
    del_all_tags()
    T.delete("1.0", END)
    rev = get_last_rev()
    T.insert(END, strip_emojis("\n".join(rev["text"])))
    add_all_tags()


# switch the text field forward one review
def next_rev():
    del_all_tags()
    T.delete("1.0", END)
    rev = get_next_rev()
    T.insert(END, strip_emojis("\n".join(rev["text"])))
    add_all_tags()


# add positive annotation
def pos_rev():
    add_annotation("positive")
    save()


# add neutral annotation
def neut_rev():
    add_annotation("neutral")
    save()


# add negative annotation
def neg_rev():
    add_annotation("negative")
    save()


def del_op():
    delete_selected()
    save()


# insert the first review when program is first started
rev = rev_dict[current_rev]
T.insert(END, strip_emojis("\n".join(rev["text"])))
add_all_tags()


# Buttons to control the labeling
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

root.mainloop()  # keep window alive until closing
