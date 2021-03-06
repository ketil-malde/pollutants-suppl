import matplotlib.pyplot as plt
import pandas
from pandas.plotting import scatter_matrix

pandas.set_option('display.width', 190)

output_format = ".tiff"

cols = ['Year','7PCBs','ΣDDTs','ΣHCHs','HCB','TNC','weight']


def load_data(file):
    data = pandas.read_csv(file,header=0,sep='\t')
    data['weight'] = data['7PCBs']/data['PCB7/wish weight']
    return(data[cols])

# correlations = raw.corr()
# print(correlations)

############################################################

# linear regression?
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from numpy import log, exp


def mk_regr(raw,ws):
    for c in [1,2,4,5]:  # missing value in 3 causes NaN
        print(' ** ',cols[c],' ** ')
        vs = log(raw[cols[c]])
        regr = LinearRegression().fit(ws,vs)
        pred = regr.predict(ws)

        print('Coefficients: ', regr.coef_)
        print('Intercept: ', regr.intercept_)
        print("Mean squared error: %.2f" % mean_squared_error(vs, pred))
        print('R²: %.2f' % r2_score(vs, pred))

# mk_regr(pandas.DataFrame({'W': log(raw['weight']), 'Y': raw['Year']}))
# print()
# mk_regr(pandas.DataFrame({'Y': raw['Year']}))


############################################################
def smatrix(raw):
    scatter_matrix(raw, alpha=0.3, figsize=(7,7), diagonal='kde')
    plt.show()


# plot vs weight by year
def plot_corr(df, size=10):
    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns);
    plt.yticks(range(len(corr.columns)), corr.columns);


############################################################
def plot_all(raw):
    # plot_corr(df=raw,size=7)
    # plt.show()
    fig, ax = plt.subplots(2, 5)
    for c in range(1, 6):
        for name, group in raw.groupby('Year'):
            ax[0,c-1].scatter(log(group['weight']), log(group[cols[c]]), label=name, alpha=0.75)
        ax[0,c-1].set_title(cols[c])
        ax[0,c-1].set_xlabel('ln weight (g)')
        ax[0,0].set_ylabel('ln concentration (µg/kg ww)')

        ax[1,c-1].scatter(raw['Year'], log(raw[cols[c]]), alpha=0.65)
        ax[1,c-1].set_xlabel('Year')
        ax[1,0].set_ylabel('ln concentration (µg/kg ww)')

    # seaborn has smoothing: sns.lmplot(x="total_bill", y="tip", data=tips, lowess=True);

    ax[0,0].legend()
    plt.show()

medians = { 'cod' : 1150, 'haddock' : 935 }

# plot year by weight, color for contaminant levels
def plot3(fname,raw,c):
    r = raw[raw[cols[c]].notnull()]
    ws = pandas.DataFrame({'W': log(r['weight']), 'Y': r['Year']})
    vs = log(r[cols[c]])
    fig = plt.figure(figsize=(6,4.5))
    plt.scatter(ws.Y,ws.W,c=(1-vs),alpha=0.7)  # todo: normalize column
    plt.show()

   
# Generate a single plot by year using non-log y-axis, regression for median size
def plot2(fname,raw,c):
    r = raw[raw[cols[c]].notnull()]
    ws = pandas.DataFrame({'W': log(r['weight']), 'Y': r['Year']})
    vs = log(r[cols[c]])
    regr = LinearRegression().fit(ws,vs)
    pred = regr.predict(ws)
    r['resid'] = vs-pred
    print(fname,cols[c],'Coef: ', regr.coef_, 'Incpt: ', regr.intercept_, "MSE: %.2f" % mean_squared_error(vs, pred), 'R²: %.2f' % r2_score(vs, pred))

    fig = plt.figure(figsize=(6,4.5))
    plt.scatter(r['Year'],r[cols[c]],alpha=0.7)  # todo: normalize column
    i=0
    xs = []
    ys = []
    for x in r.Year.unique():
        xs.append(x)
        ys.append(exp(regr.predict([[log(medians[fname]),x]])[0]))
    plt.plot(xs,ys,alpha=0.7)
    plt.suptitle(cols[c]+" ("+fname+")")
    plt.xlabel('Year')
    plt.ylabel("Concentration (µg/kg ww)")
    plt.savefig(fname+"-"+cols[c]+output_format, dpi=600)

def multiplot2():
    i = 0
    fig, ax = plt.subplots(5,2,figsize=(12,20))
    for fname, raw in [("cod",cod), ("haddock",had)]:
        for c in range(1,6):
            r = raw[raw[cols[c]].notnull()]
            ws = pandas.DataFrame({'W': log(r['weight']), 'Y': r['Year']})
            vs = log(r[cols[c]])
            regr = LinearRegression().fit(ws,vs)
            pred = regr.predict(ws)
            print(c-1,i)
            ax[c-1,i].scatter(ws.Y, r[cols[c]], alpha=0.7)
            xs = []
            ys = []
            for x in r.Year.unique():
                xs.append(x)
                ys.append(exp(regr.predict([[log(medians[fname]),x]])[0]))
            ax[c-1,i].plot(xs,ys,alpha=0.7)
            # ax[c-1,i].set_title(cols[c]+" ("+fname+")")
            # ax[c-1,i].set_xlabel('Year')
            ax[c-1,0].set_ylabel(cols[c]+" concentration (µg/kg ww)")
        ax[0,i].set_title(fname)
        i=i+1
    plt.savefig("multiplot"+output_format,dpi=300)
    
def plot1(fname,raw,c):
    r = raw[raw[cols[c]].notnull()]
    ws = pandas.DataFrame({'W': log(r['weight']), 'Y': r['Year']})
    vs = log(r[cols[c]])
    regr = LinearRegression().fit(ws,vs)
    pred = regr.predict(ws)
    r['resid'] = vs-pred
    print('%',fname,cols[c],'Coef: ', regr.coef_, 'Incpt: ', regr.intercept_, "MSE: %.2f" % mean_squared_error(vs, pred), 'R²: %.2f' % r2_score(vs, pred))
    
    fig, ax = plt.subplots(2,2,figsize=(12,9))
    for name, group in r.groupby('Year'):
        ax[0,0].scatter(log(group['weight']), log(group[cols[c]]), label=name, alpha=0.75)
        w0 = min(log(group['weight']))
        w1 = max(log(group['weight']))
        [c0, c1] = regr.predict([[w0,name],[w1,name]])
        # print(name,w0,w1,y0,y1)
        ax[0,0].plot([w0,w1],[c0,c1],linewidth=1,linestyle='dotted')
        ax[0,1].scatter(log(group['weight']), group['resid'], label=name, alpha=0.75)
        ax[0,1].axhline(linewidth=0.7,color='black',linestyle='dotted')
        ax[1,0].scatter(group['Year'], log(group[cols[c]]), alpha=0.65)
        ax[1,1].scatter(group['Year'], group['resid'], alpha=0.65)

    ax[0,0].set_title(cols[c])
    ax[0,0].set_xlabel('ln weight (g)')
    ax[0,0].set_ylabel('ln concentration (µg/kg ww)')
    
    ax[0,1].set_title('Residuals')
    
    ax[1,0].set_xlabel('Year')
    ax[1,0].set_ylabel('ln concentration (µg/kg ww)')

    y0 = min(r['Year'])
    y1 = max(r['Year'])
    [c0,c1] = regr.predict([[log(medians[fname]),y0],[log(medians[fname]),y1]])  # log w = 7
    ax[1,0].plot([y0,y1],[c0,c1],linewidth=1,linestyle='dotted')
    
    ax[1,1].axhline(linewidth=0.7,color='black',linestyle='dotted')
    ax[1,1].set_xlabel('Year')

    ax[0,0].legend()
    # fig.show()
    fig.savefig(fname+"-all-"+cols[c]+".pdf", bbox_inches='tight', dpi=300)

cod = load_data("cod.csv")
had = load_data("had.csv")

def genplots():
    for x in range(1,6):
        # plot1("cod", cod, x)
        # plot1("had", had, x)
        plot2("cod", cod, x)
        plot2("haddock", had, x)        

import numpy as np
import statsmodels.api as sm


def linreg2(raw,c):
    r = raw[raw[cols[c]].notnull()]
    ws = pandas.DataFrame({'Weight': log(r['weight']), 'Year': (r['Year']-1992)})
    vs = log(r[cols[c]])
    X = sm.add_constant(ws)
    mod = sm.OLS(vs,X)
    reg = mod.fit()
    # print(reg.summary())
    # Alternatively:
    regr = LinearRegression().fit(ws,vs)
    pred = regr.predict(ws)
    r['resid'] = vs-pred
    # print('% Coef: ', regr.coef_, 'Incpt: ', regr.intercept_, "MSE: %.2f" % mean_squared_error(vs, pred), 'R²: %.2f' % r2_score(vs, pred))
    return(reg)

def linregs():    
    for x in range(1,6):
        print(" *** cod: "+cols[x]+" *** ")
        print(linreg2(cod, x).summary())
        print(" *** haddock: "+cols[x]+" *** ")
        print(linreg2(had, x).summary())

import sys

def make_supplementary():
    keep = sys.stdout
    sys.stdout = open("supplementary.tex","w")
    print("\\input{preamble}")
    for name,data in [("cod",cod), ("haddock",had)]:
        print("\\subsection*{Correlation matrix for "+name+" data}")
        # print("Correlation matrix for the "+name+" data.")
        print("\\begin{verbatim}")
        print(data.corr())
        print("\\end{verbatim}")
    for c in range(1,6):
        for name,data in [("cod",cod), ("haddock",had)]:
            tex_add_fig(name,cols[c])
            plot1(name, data, c)
            reg=linreg2(data,c)
            tex_summary(name,cols[c],reg)
    print("\\end{document}")
    sys.stdout = keep
    
def tex_add_fig(fname,cname):
    print("\\subsection*{"+cname+" in "+fname+"}")
    #print("\\begin{figure}")
    print("  \\begin{center}\\includegraphics[scale=0.6]{\""+fname+"-all-"+cname+"\"}\\end{center}")
    # print("  \\label{fig:"+fname+":"+cname+"}")
    print("Diagram showing the log-transformed concentration of "+cname+"  measured in "+fname+" by fish weight (top left) and by year (bottom left).  Regression lines are plotted for each year class (above) and for a fish of median size (below).  The respective residuals are shown on the right.\n")
    print("\\newpage")
    #print("\\end{figure}")

from math import ceil
    
def tex_summary(fname,cname,reg):
    print("\\subsection*{Regression summary, "+cname+" in "+fname+"}")
    print("Regression formula: ")
    print("$ c = %.2f \cdot %.2f^{y-1992} \cdot w^{%.2f} $\n" % (exp(reg.params.const),exp(reg.params.Year),reg.params.Weight))

    print("Linear regression statistics from fitting the log-transformed concentration of "+cname+" in "+fname+".")
    ci=reg.conf_int(alpha=0.05, cols=None)
    ydec=100*(1-exp(reg.params.Year))
    ymin=100*(1-exp(ci[1]['Year']))
    ymax=100*(1-exp(ci[0]['Year']))
    winc=100*(2**reg.params.Weight-1)
    wmin=100*(2**ci[0]['Weight']-1)
    wmax=100*(2**ci[1]['Weight']-1)

    pval=("$p<10^{%d}$" % ceil(log(reg.pvalues['Year'])/log(10)))
    print("There is an observed decline in "+cname+" levels over time ("+pval+"), at a rate is estimated to be %.2f\\%% (95\\%% CI: %.2f-%.2f) per year." % (ydec,ymin,ymax))
    print("The time required for a 50\\%% reduction in levels is estimated to be %.1f years (%.1f-%.1f)." % (log(0.5)/log(1-ydec/100), log(0.5)/log(1-ymax/100), log(0.5)/log(1-ymin/100)))
    print("The estimated increase in "+cname+" level from a doubling in fish weight is %.2f\\%% (95\\%% CI: %.2f-%.2f)." % (winc,wmin,wmax))
    #print("\\begin{figure}")
    print("\\begin{verbatim}")
    print(reg.summary())
    print("\\end{verbatim}")
    # print("\\end{figure}")
    print("\\newpage")

make_supplementary()

# genplots()
# plot3("cod",cod,1)
# multiplot2()
