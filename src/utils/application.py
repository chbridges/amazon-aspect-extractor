import datetime
import json
import sys
from time import sleep
import tkinter as tk
from typing import List

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from wordcloud import WordCloud
from .reviewextractor import extract_product_title_and_jpg

class plotWindow:

    def __init__(self, frame, box, P):
        self.frame = frame
        self.box = box
        self.box.bind("<Return>", self.plot)
        self.P = P

    def plot(self, event):
        url = self.box.get()
        data = self.P(url)
        title, jpg = extract_product_title_and_jpg(url)

        fig = Figure(figsize=(6,24))
        fig.suptitle(title, fontsize=18)
        a = fig.add_subplot(221)
        #Piechart
        n_asp = 30

        topx = data["all"][:n_asp]
        self.df = topx

        pos = topx.loc[topx['sentiment_text'] == "positive"]["count"].sum()
        neu = topx.loc[topx['sentiment_text'] == "neutral"]["count"].sum()
        neg = topx.loc[topx['sentiment_text'] == "negative"]["count"].sum()

        a.pie([pos, neu, neg], labels=["positive", "neutral", "negative"], colors=["green", "yellow", "red"])

        a = fig.add_subplot(222)

        n_asp = 15

        pos = int(n_asp*len(topx.loc[topx['sentiment_text'] == "positive"])/30)
        neu = int(n_asp*len(topx.loc[topx['sentiment_text'] == "neutral"])/30)
        neg = n_asp - pos - neu


        counts = []
        counts.extend(data["pos"][:pos]["count"])
        counts.extend(data["pos"][:neu]["count"])
        counts.extend(data["neg"][:neg]["count"])

        labels = []
        labels.extend(data["pos"][:pos]["aspect"])
        labels.extend(data["pos"][:neu]["aspect"])
        labels.extend(data["neg"][:neg]["aspect"])

        colors = []
        colors.extend(["green"]*pos)
        colors.extend(["yellow"]*neu)
        colors.extend(["red"]*neg)

        a = fig.add_subplot(223)

        a.imshow(jpg)



        a = fig.add_subplot(224)


        n_asp = 20

        wordcloud = WordCloud(background_color="white", max_font_size=60, max_words=n_asp, repeat=False)

        wordcloud.generate_from_frequencies(topx.loc["aspect", "count"])
        wordcloud.recolor(color_func=self.color_by_sentiment)


        a.imshow(wordcloud, interpolation="bilinear")

        canvas = FigureCanvasTkAgg(fig, master=self.frame)
        canvas.get_tk_widget().grid(row=1, column=0, columnspan=2)
        canvas.draw()

    def color_by_sentiment(self, word):
        self.df.loc[df["aspect"] == word]
        if self.df["sentiment_value"][0] == 0:
            return "red"
        elif self.df["sentiment_value"][0] == 1:
            return "yellow"
        elif self.df["sentiment_value"][0] == 2:
            return "green"
        else:
            return "grey"

def run_app(P):
    # Main window
    root = tk.Tk()
    f1 = tk.Frame(root, bg="lightgray")
    f2 = tk.Frame(root, bg="lightgray")
    f1.grid(sticky="nsew")
    f2.grid(sticky="nsew")
    tk.Grid.columnconfigure(root, 0, weight=1)
    tk.Grid.rowconfigure(root, 0, weight=1)
    root.title("Amazon Aspect Extraction")


    e1 = tk.Entry(f1)
    e2 = tk.Entry(f2)

    tk.Label(f1, text="Amazon URL 1").grid(row=0)
    tk.Label(f2, text="Amazon URL 2").grid(row=0)

    e1.grid(row=0, column=1)
    e2.grid(row=0, column=1)


    Bar = tk.Scrollbar(root)

    plot1 = plotWindow(f1, e1, P)
    plot2 = plotWindow(f2, e2, P)

    root.mainloop()
