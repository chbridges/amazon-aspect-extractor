import datetime
import json
import sys
from time import sleep
import tkinter as tk
from typing import List

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

try:
    from matplotlib.backends.backend_tkagg import NavigationToolbar2TkAgg
except ImportError:
    from matplotlib.backends.backend_tkagg import (
        NavigationToolbar2Tk as NavigationToolbar2TkAgg,
    )
from matplotlib.figure import Figure
from wordcloud import WordCloud
from .reviewextractor import extract_product_title_and_jpg
from PIL import Image
from io import BytesIO


class ResizingCanvas(tk.Canvas):
    def __init__(self, parent, **kwargs):
        tk.Canvas.__init__(self, parent, **kwargs)
        self.bind("<Configure>", self.on_resize)
        self.height = self.winfo_reqheight()
        self.width = self.winfo_reqwidth()

    def on_resize(self, event):
        # determine the ratio of old width/height to new width/height
        wscale = float(event.width) / self.width
        hscale = float(event.height) / self.height
        self.width = event.width
        self.height = event.height
        # resize the canvas
        self.config(width=self.width, height=self.height)
        # rescale all the objects tagged with the "all" tag
        self.scale("all", 0, 0, wscale, hscale)


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
        stream = BytesIO(jpg)
        image = Image.open(stream).convert("RGBA")
        stream.close()
        image = np.asarray(image)

        fig = Figure(figsize=(16, 6))
        fig.suptitle(title, fontsize=18)
        a = fig.add_subplot(221)
        # Piechart
        n_asp = 30

        topx = data["all"][:n_asp]
        self.df = topx

        print(topx.head())

        pos = topx.loc[topx["sentiment_text"] == "positive"]["counts"].sum()
        neu = topx.loc[topx["sentiment_text"] == "neutral"]["counts"].sum()
        neg = topx.loc[topx["sentiment_text"] == "negative"]["counts"].sum()

        a.pie(
            [pos, neu, neg],
            labels=[
                "positive" if pos > 0 else "",
                "neutral" if neu > 0 else "",
                "negative" if neg > 0 else "",
            ],
            colors=["green", "yellow", "red"],
        )
        a.axis("off")

        a = fig.add_subplot(222)

        n_asp = 15

        pos = int(n_asp * len(topx.loc[topx["sentiment_text"] == "positive"]) / 30)
        neu = int(n_asp * len(topx.loc[topx["sentiment_text"] == "neutral"]) / 30)
        neg = n_asp - pos - neu

        counts = []
        counts.extend(data["pos"][:pos]["counts"])
        counts.extend(data["pos"][:neu]["counts"])
        counts.extend(data["neg"][:neg]["counts"])

        labels = []
        labels.extend(data["pos"][:pos]["aspect"])
        labels.extend(data["pos"][:neu]["aspect"])
        labels.extend(data["neg"][:neg]["aspect"])

        colors = []
        colors.extend(["green"] * pos)
        colors.extend(["yellow"] * neu)
        colors.extend(["red"] * neg)

        a.bar(np.arange(len(counts)), counts, color=colors)
        a.set_xticks(np.arange(len(counts)))
        a.set_xticklabels(labels)

        a = fig.add_subplot(223)

        a.imshow(image)
        a.axis("off")

        a.set_xticks([])
        a.set_yticks([])

        a = fig.add_subplot(224)

        n_asp = 20

        wordcloud = WordCloud(
            background_color="white", max_font_size=60, max_words=n_asp, repeat=False
        )
        inp = {}
        for index, row in topx[["aspect", "counts"]].iterrows():
            inp[row["aspect"]] = row["counts"]

        wordcloud.generate_from_frequencies(inp)
        wordcloud.recolor(color_func=self.color_by_sentiment)

        a.imshow(wordcloud, interpolation="bilinear")
        a.set_xticks([])
        a.set_yticks([])

        fig.tight_layout()
        a.axis("off")

        canvas = FigureCanvasTkAgg(fig, master=self.frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=1, column=0, columnspan=2, sticky="nswe")

    def convert_string_to_bytes(self, string):
        bytes = b""
        for i in string:
            bytes += struct.pack("B", ord(i))
        return bytes

    def color_by_sentiment(self, word, **kwargs):
        sentiment = self.df.loc[self.df["aspect"] == word]["sentiment_text"].iloc[0]
        if sentiment == "negative":
            return "red"
        elif sentiment == "neutral":
            return "yellow"
        elif sentiment == "positive":
            return "green"
        else:
            return "grey"


def onFrameConfigure(canvas):
    """Reset the scroll region to encompass the inner frame"""
    canvas.configure(scrollregion=canvas.bbox("all"))


def run_app(P):
    # Main window
    root = tk.Tk()

    canvas = tk.Canvas(root, borderwidth=0, background="lightgray")
    frame = tk.Frame(canvas, background="lightgray")
    vsb = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=vsb.set)

    vsb.grid(row=0, column=1, sticky="ns")
    hsb = tk.Scrollbar(root, orient="horizontal", command=canvas.xview)
    canvas.configure(xscrollcommand=hsb.set)

    hsb.grid(row=1, column=0, sticky="we")
    canvas.grid(row=0, column=0, sticky="nswe")
    canvas.create_window((4, 4), window=frame, anchor="nw")

    frame.bind("<Configure>", lambda event, canvas=canvas: onFrameConfigure(canvas))
    window = frame

    f1 = tk.Frame(window, bg="lightgray")
    f2 = tk.Frame(window, bg="lightgray")
    f1.grid(row=0, column=0, sticky="nsew")
    f2.grid(row=1, column=0, sticky="nsew")
    tk.Grid.columnconfigure(root, 0, weight=1)
    tk.Grid.rowconfigure(root, 0, weight=1)
    root.title("Amazon Aspect Extraction")

    e1 = tk.Entry(f1)
    e2 = tk.Entry(f2)

    tk.Label(f1, text="Amazon URL 1").grid(row=0)
    tk.Label(f2, text="Amazon URL 2").grid(row=0)

    e1.grid(row=0, column=1)
    e2.grid(row=0, column=1)

    plot1 = plotWindow(f1, e1, P)
    plot2 = plotWindow(f2, e2, P)

    root.mainloop()
