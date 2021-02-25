import threading
import tkinter as tk
from io import BytesIO

import numpy as np
from PIL import Image

import matplotlib
from matplotlib import gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from wordcloud import WordCloud

from .reviewextractor import extract_product_title_and_jpg

matplotlib.use("TkAgg")


try:
    from matplotlib.backends.backend_tkagg import NavigationToolbar2TkAgg
except ImportError:
    from matplotlib.backends.backend_tkagg import (
        NavigationToolbar2Tk as NavigationToolbar2TkAgg,
    )



def executePipeline(P, url, data):
    """Pipeline execution function for threading
    Arguments:
    - P: Pipeline for extracting aspects and sentiments from a URL
    - url: The URL to crawl
    - data: the array to save the results in"""
    data.append(P(url))


class plotWindow:
    """ "A class that binds itself to a tkinter entry box and plots a link when-
    ever Enter is pressed.
    Arguments:
    - frame: The tkinter frame to draw the plots into
    - box: The tkinter entry box to monitor
    - P: The pipeline to use for processing the URL"""

    def __init__(self, frame, box, P):
        self.frame = frame
        self.box = box
        self.box.bind("<Return>", self.plot)
        self.P = P

    def plot(self, event):
        """Called whenever enter is pressed in an entry box
        Arguments:
        - event: Unused tkinter event"""
        url = self.box.get()
        data = []
        # Assign pipeline execution to a new thread
        thread = threading.Thread(
            target=executePipeline,
            args=(
                self.P,
                url,
                data,
            ),
        )
        thread.start()
        while thread.is_alive():
            self.frame.winfo_toplevel().update()
        thread.join()
        data = data[0]  # extracted data from URL

        title, jpg = extract_product_title_and_jpg(url)  # Get metadata
        if len(title) > 50:
            title = title[:47] + "..."
        stream = BytesIO(jpg)
        image = Image.open(stream).convert("RGBA")
        stream.close()
        image = np.asarray(image)

        fig = Figure(figsize=(18, 6))
        gs = gridspec.GridSpec(ncols=2, nrows=2, figure=fig, width_ratios=[1, 3])

        fig.suptitle(title, fontsize=18)
        a = fig.add_subplot(gs[0, 0])
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

        a = fig.add_subplot(gs[0, 1])

        # Bar Plot

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

        a = fig.add_subplot(gs[1, 0])

        # Product image

        a.imshow(image)
        a.axis("off")

        a.set_xticks([])
        a.set_yticks([])

        a = fig.add_subplot(gs[1, 1])

        # Wordcloud

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
        canvas.get_tk_widget().grid(row=1, column=0, columnspan=3, sticky="nswe")

    def color_by_sentiment(self, word, **kwargs):
        """Coloring function for the wordcloud
        Arguments:
        - word: the string to color
        Returns:
        A string contatining the color based on the sentiment in the dataset"""
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
    """Define window layout and bind plotting listeners
    Arguments:
    - P: Pipeline for fetching data from a URL"""
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

    font = tk.font.Font(family="Calibri", size=14)

    e1 = tk.Entry(f1, font=font)
    e2 = tk.Entry(f2, font=font)

    tk.Label(f1, text="Amazon URL 1", font=font).grid(row=0)
    tk.Label(f2, text="Amazon URL 2", font=font).grid(row=0)

    e1.grid(row=0, column=1, sticky="we", columnspan=2)
    e2.grid(row=0, column=1, sticky="we", columnspan=2)

    plot1 = plotWindow(f1, e1, P)
    plot2 = plotWindow(f2, e2, P)

    root.mainloop()
