import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import seaborn as sns

def playlist(data, legend=[], file="", title=""):
    fig, ax= plt.subplots(dpi=600)
    fig.set_figheight(4.8)
    fig.set_figwidth(6.4)

    # add formatted labels
    titleFont = fm.FontProperties(fname="./static/fonts/KievitOffc-Bold.ttf",size='x-large')
    axisFont = fm.FontProperties(fname="./static/fonts/KievitOffc.ttf",size='x-large')
    legendFont = fm.FontProperties(fname="./static/fonts/KievitOffc-Ita.ttf",size='medium')
    ax.set_xlabel("valence", fontproperties=axisFont)
    ax.set_ylabel("arousal", fontproperties=axisFont)
    ax.set_title(title, fontproperties=titleFont)

    count = len(data)
    if (count == 1):
        points = np.transpose(data)
        ax.plot(points[0], points[1], marker='.', color="#D73F09", linestyle='-')
    else:
        for i in range(count):
            points = np.transpose(data[i])
            ax.plot(points[0], points[1], marker='.', linestyle='-')
    
    if (legend != []):
        ax.legend(legend, prop=legendFont, fontsize='small')

    if (title != ""):
        ax.set_title(title, fontproperties=titleFont)

    if (file != ""):
        plt.savefig(file, dpi=600)
    else:
        plt.show(block=False)
    
    plt.clf()
    plt.close()

def hist(label, data, title="", file=""):
    figsize = (12.8, 9.6)
    plt.figure(figsize=figsize)
    fig, ax= plt.subplots(dpi=600)

    # add formatted labels
    titleFont = fm.FontProperties(fname="./static/fonts/KievitOffc-Bold.ttf",size='x-large')
    axisFont = fm.FontProperties(fname="./static/fonts/KievitOffc.ttf",size='x-large')
    legendFont = fm.FontProperties(fname="./static/fonts/KievitOffc-Ita.ttf",size='medium')
    ax.set_xlabel(label, fontproperties=axisFont)
    ax.set_ylabel("Count", fontproperties=axisFont)

    n, bins, patches = ax.hist(data, bins=int(1+3.3*np.log10(len(data))), facecolor="#D73F09")    

    if (title != ""):
        plt.title(ax.set_title(title, fontproperties=titleFont))

    if (file != ""):
        plt.savefig(file, dpi=600)
    else:
        plt.show(block=False)
    
    plt.clf()
    plt.close()

def boxplots(df, x, y, legend=[], file="", title=""):
    fig, ax= plt.subplots(dpi=600)

    # add formatted labels
    titleFont = fm.FontProperties(fname="./static/fonts/KievitOffc-Bold.ttf",size='x-large')
    axisFont = fm.FontProperties(fname="./static/fonts/KievitOffc.ttf",size='x-large')
    legendFont = fm.FontProperties(fname="./static/fonts/KievitOffc-Ita.ttf",size='medium')
    # ax.set_xlabel(xlabel, fontproperties=axisFont)
    # ax.set_ylabel(ylabel, fontproperties=axisFont)

    df.boxplot(column=x, by=y, figsize = (len(df.columns) * 1.5,10), ax=ax)
    fig.tight_layout()

    if (legend != []):
        ax.legend(legend, prop=legendFont, fontsize='small')

    if (title != ""):
        ax.set_title(title, fontproperties=titleFont)

    if (file != ""):
        plt.savefig(file, dpi=600)
    else:
        plt.show(block=False)
    
    plt.clf()
    plt.close()

def line(xlabel, ylabel, data, dim = 1, count = 1, marker=',', linestyle='-', legend = [], file = "", title="", point_annotations=None):

    fig, ax= plt.subplots(dpi=600)
    fig.set_figheight(4.8)
    fig.set_figwidth(6.4)

    # add formatted labels
    titleFont = fm.FontProperties(fname="./static/fonts/KievitOffc-Bold.ttf",size='x-large')
    axisFont = fm.FontProperties(fname="./static/fonts/KievitOffc.ttf",size='x-large')
    legendFont = fm.FontProperties(fname="./static/fonts/KievitOffc-Ita.ttf",size='medium')
    ax.set_xlabel(xlabel, fontproperties=axisFont)
    ax.set_ylabel(ylabel, fontproperties=axisFont)
    

    if dim == 1:
        if count == 1:
            ax.plot(data, marker=marker, linestyle=linestyle)
        else:
            for i in range(count):
                ax.plot(data[i], marker=marker, linestyle=linestyle)
    if dim == 2:
        if count == 1:
            ax.plot(data[0], data[1], marker=marker, linestyle=linestyle)
        else:
            for i in range(count):
                ax.plot(data[i][0], data[i][1], marker=marker, linestyle=linestyle)
        
    
    if point_annotations != None:
        for i in range(len(point_annotations)):
            if (i != 0 and i != len(point_annotations) - 1):
                ha = 'left' if (i % 2 == 0) else 'right'
                yoffset = -4 if (i % 2 == 0) else 4
                plt.annotate(point_annotations[i], # this is the text
                    (data[0][i],data[1][i]), # these are the coordinates to position the label
                    textcoords="offset points", # how to position the text
                    xytext=(4,yoffset), # distance from text to points (x,y)
                    size='small',
                    fontweight='bold',
                    color='blue',
                    ha=ha) # horizontal alignment can be left, right or center

    if (legend != []):
        ax.legend(legend, prop=legendFont, fontsize='small')

    if (title != ""):
        ax.set_title(title, fontproperties=titleFont)

    if (file != ""):
        plt.savefig(file, dpi=600)
    else:
        plt.show(block=False)
    
    plt.clf()
    plt.close()

def av_box(plots, labels, title="test", file="./test.png", plt_size=10, vert=True, showfliers=True):
    fig, ax= plt.subplots(dpi=600)
    fig.set_figheight(plt_size)
    fig.set_figwidth(plt_size)

    
    titleFont = fm.FontProperties(fname="./static/fonts/Stratum2-Bold.otf")
    ax.set_title(title, fontproperties=titleFont)    
    
    plt.boxplot(plots, labels=labels, showmeans=True, meanline=True, vert=vert, showfliers=showfliers)
    plt.tight_layout()
    
    plt.savefig(file, dpi=600)
    plt.show(block=False)
    plt.clf() 

def av_circle(v, a, title="", colors="#D73F09", file="./test.png", plt_size=10, alpha=.5):
    plt.figure(figsize=(plt_size,plt_size))
    plt.scatter(v, a, s=20, c=colors, alpha=alpha)
    # plt.xlim(-1.25,1.25)
    # plt.ylim(-1.25,1.25)  

    # draw the unit circle
    fig = plt.gcf()
    ax = fig.gca()
    circle1 = plt.Circle((0, 0), 1.0, color='0.25', fill=False)
    ax.add_artist(circle1)

    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["top"].set_position(("data",0))
    ax.spines["right"].set_position(("data",0))

    # add formatted labels
    titleFont = fm.FontProperties(fname="./static/fonts/Stratum2-Bold.otf")
    axisFont = fm.FontProperties(fname="./static/fonts/Stratum2-Medium.otf")
    emotionFont = fm.FontProperties(fname="./static/fonts/KievitOffc-BoldIta.ttf", size='xx-large')

    ax.set_xlabel("Valence", fontproperties=axisFont, size=plt_size*3)
    ax.set_ylabel("Arousal", fontproperties=axisFont, size=plt_size*3)
    ax.set_title(title, fontproperties=titleFont, size=plt_size*4)
    # ax.axes.xaxis.set_ticks([])
    # ax.axes.yaxis.set_ticks([])
    
    # print emotion labels
    ax.text(0.98, 0.35, 'Happy', fontproperties=emotionFont, size=int(plt_size*2.5))
    ax.text(0.5, 0.9, 'Excited', fontproperties=emotionFont, size=int(plt_size*2.5))
    ax.text(-1.16, 0.35, 'Afraid', fontproperties=emotionFont, size=int(plt_size*2.5))
    ax.text(-0.7, 0.9, 'Angry', fontproperties=emotionFont, size=int(plt_size*2.5))
    ax.text(-1.05, -0.25, 'Sad', fontproperties=emotionFont, size=int(plt_size*2.5))
    ax.text(-0.9, -0.9, 'Depressed', fontproperties=emotionFont, size=int(plt_size*2.5))
    ax.text(0.98, -0.25, 'Content', fontproperties=emotionFont, size=int(plt_size*2.5))
    ax.text(0.7, -0.9, 'Calm', fontproperties=emotionFont, size=int(plt_size*2.5)) 

    plt.savefig(file, dpi=600)
    plt.show(block=False)
    plt.clf() 
