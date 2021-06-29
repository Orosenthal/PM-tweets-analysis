# Import packages
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import matplotlib.patheffects as path_effects
from matplotlib.patches import Rectangle
from  matplotlib.ticker import PercentFormatter
from sklearn.feature_extraction.text import CountVectorizer
from bidi.algorithm import get_display
from nltk.probability import FreqDist
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")




def monthly_count_plot(df):
    '''
    Returns monthly heatmap since 2015 with highlights months
    '''

    df["Year"] = df.index.year
    df["Month"] = df.index.month
    months = dict(zip(range(1,13),["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]))
    a = df['2015':].pivot_table(index="Month",columns="Year",values="tweet", aggfunc=lambda x: len(x))
    a = a.T.rename(mapper=months, axis=1)

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.set(font_scale=1.4);
    sns.heatmap(a,mask= a.isnull(),cmap="Blues",linewidth=0.3, cbar=False, annot=True,fmt='g');
    plt.xlabel('')
    plt.ylabel('')
    ax.xaxis.tick_top()
    ax.tick_params(left=False, top=False)
    plt.yticks(rotation=0); 
    ax.add_patch(Rectangle((0, 0), 3, 1, fill=False, edgecolor='red', lw=3))
    ax.add_patch(Rectangle((6, 4), 3, 1, fill=False, edgecolor='red', lw=3))
    ax.add_patch(Rectangle((1, 4), 3, 1, fill=False, edgecolor='red', lw=3))
    ax.add_patch(Rectangle((0, 6), 3, 1, fill=False, edgecolor='red', lw=3))
    ax.add_patch(Rectangle((0, 5), 3, 1, fill=False, edgecolor='red', lw=3))
    plt.title('Monthly tweets count (since 2015)',pad=50, fontweight = 'semibold');



def gi_plot(df_main):
    '''
    Returns general information plot for pm_summary
    Benjamin Netanyahu twitter 
    '''

    plt.style.use('seaborn')
    fig,ax = plt.subplots(figsize=(20, 14))
    # number of tweets by language
    plt.subplot(2, 2, 1);
    df_main['tweet_n_words'] = df_main['tweet'].map(lambda x: len(x.split()))
    im = plt.imread('twitter_logo_big.jpg')
    plt.imshow(im, extent=[-0.05,1,0.1,1.1], zorder=0, aspect='auto', alpha=0.5)
    plt.text(0, 0.95,f'General information about the data set:', fontsize=20)
    plt.text(0, 0.8,f'Start date: {df_main.index.min().date()}'\
             ,color='black', fontsize=30,path_effects=[path_effects.withSimplePatchShadow()]);
    plt.text(0, 0.67,f'End date: {df_main.index.max().date()}'\
             ,color='black', fontsize=30, path_effects=[path_effects.withSimplePatchShadow()]);
    plt.text(0, 0.5,f'Number of tweets: {len(df_main):,}'\
             ,color='b', fontsize=30, path_effects=[path_effects.withSimplePatchShadow()]);
    plt.text(0, 0.3,f'Avg number of words per tweet: {round(df_main["tweet_n_words"].mean())}'\
             ,color='g', fontsize=30, path_effects=[path_effects.withSimplePatchShadow()]);
    plt.axis('off');
    plt.subplot(2, 1, 2);
    yearly_tweets = df_main.resample('Y')['tweet'].count()
    yearly_tweets.index = yearly_tweets.index.year
    yearly_tweets.plot(kind='bar', edgecolor='k',linewidth=2);plt.xlabel("");
    plt.title('@netanyahu - Number of tweets per year', size=25,fontweight='bold');plt.xticks(size=25,rotation=0);

    l_pie = (
         df_main['language'].value_counts(normalize=True)
        .to_frame()
        .reset_index()
        .rename(columns={'index':'l_code','language':'percentage'})
        .replace({'l_code':{'iw':'Hebrew','en':'English'}})
        .assign(language = lambda x: np.select([x['percentage'] > 0.05,x['percentage']<=0.05],[x['l_code'], "Others"]))
        .groupby('language')['percentage'].sum())

    plt.subplot(2, 2, 2);
    plt.pie(l_pie, explode=(0.1,0.1,0.1),autopct='%1.1f%%', labels=l_pie.index,colors = [ '#e74c3c', '#56738f','#eb4d4b'],
           shadow=True, startangle=90,pctdistance=0.8, labeldistance=1.1, textprops={'fontsize': 18, 'weight':'bold'});
    plt.title("%Tweets by language", size=20,color='navy',pad=20);


def most_common(data, sw, ngram=3):
    
    '''
    Returns trigrams and bigrams
    '''
    if ngram == 3:
        
        top=50
        firstword=""
        c = CountVectorizer(stop_words=sw,ngram_range=(3,3))
        X=c.fit_transform(data['clean_tweet'].tolist())
        words=pd.DataFrame(X.sum(axis=0),columns=c.get_feature_names()).T.sort_values(0,ascending=False).reset_index()
        res=words[words['index'].apply(lambda x: firstword in x)].head(top)
        t, dup = res.copy(), []
        # eliminate overlaping trigram
        for n,i in enumerate(t['index']):
            temp = i.split()
            for i2 in t['index'][n+1:]:
                if temp[0] and temp[1] in i2 or temp[1] and temp[2] in i2 or temp[0] and temp[2] in i2:
                    dup.append(i2)
        res = res.loc[~res['index'].isin(set(dup))]    
        terms = [get_display(x) for x in res['index']]; plt.figure(figsize=(12,8));sns.set(font_scale = 1.5);
        sns.barplot(x=res[0], y=terms,edgecolor='k'); 
        plt.title('Most common phrases (trigrams) 2008 -> today', size=20); plt.xlabel("");

    elif ngram == 2:

        election_dates = ['2015-03-17', '2019-04-09', '2019-09-17', '2020-03-02','2021-03-23']
        three_month_bfore_election =[str(dt.datetime.fromisoformat(i).date() - dt.timedelta(days=90)) for i in election_dates]
        time_range_b4_elections = list(zip(three_month_bfore_election,election_dates))

        fig = plt.figure(figsize=(20,18))
        for index, itr in enumerate(time_range_b4_elections):
            
            top, firstword = 20,""
            d = data.loc[itr[0]:itr[1],'clean_tweet'].to_frame()    
            c = CountVectorizer(stop_words=sw,ngram_range=(2,2))
            X=c.fit_transform(d['clean_tweet'].tolist())
            words=pd.DataFrame(X.sum(axis=0),columns=c.get_feature_names()).T.sort_values(0,ascending=False).reset_index()
            res=words[words['index'].apply(lambda x: firstword in x)].head(top)
            t, dup = res.copy(), []
            # eliminate overlaping trigram
            for n,i in enumerate(t['index']):
                temp = i.split()
                for i2 in t['index'][n+1:]:
                    if temp[0] in i2 or temp[1] in i2 :
                        dup.append(i2)
            res = res.loc[~res['index'].isin(set(dup))]    
            res['pct'] = (res[0]/len(d))
            ax = fig.add_subplot(3, 2, index+1);
            terms = [get_display(x) for x in res['index']]; sns.set(font_scale = 1.5);
            sns.set(font_scale = 1.8);plt.tight_layout();
            ax = sns.barplot(x=res['pct'][:10], y=terms[:10], palette='Blues_r',edgecolor='k');
            plt.yticks(fontsize=(20),fontweight='semibold');
            plt.title(f'Most common bigrams  ({itr[0]} -> {itr[1]})', size=20);
            plt.xlabel(""); ax.xaxis.set_major_formatter(mtick.PercentFormatter(1,decimals=0));

def sentiment_all(df_heBert):
    '''
    Returns sentiment distribution for all Hebrew tweets 
    '''

    df_overall_sent = round(df_heBert['s_heBert_label'].value_counts(normalize=True)*100).to_frame(name='Percent')
    g = sns.catplot(x=df_overall_sent.index,y='Percent',kind='bar',data=df_overall_sent, height=7, aspect=1.5,
                    palette ={"pos": "green", "neu": "grey", "neg": "red"}, edgecolor='k',legend_out = False);
    g.ax.set_ylim(0,100); plt.title('Sentiment distribution for all tweets (since 2008)',size=20);
    sns.set(font_scale = 1.5); 
    for ax in g.axes.flat:
        ax.yaxis.set_major_formatter(PercentFormatter(100))
    for p in g.ax.patches:
        txt = str(round(p.get_height())) + '%'
        txt_x, txt_y = p.get_x(), p.get_height()
        g.ax.text(txt_x+0.25,txt_y+5,txt,fontweight='semibold', fontsize=23)


def sentiment_m_time(df_heBert):
    '''
    Returns tweets sentiment time series since 2019 on a monthly basis (HeBERT model)
    '''

    election_dates = ['2015-03-17', '2019-04-09', '2019-09-17', '2020-03-02','2021-03-23']
    since='2019'
    time='M'
    strftime_c = '%Y-%m' if time=='M' else '%Y-%m-%d'
    df_since2019 = df_heBert[since:].groupby('s_heBert_label').resample(time).size().T
    df_since2019.index = pd.to_datetime(df_since2019.index, format = '%Y-%m')
    neg = df_since2019['neg'].values
    pos = df_since2019['pos'].values

    plt.style.use('seaborn-dark')
    ax = df_since2019.plot(marker='o',color=['red','grey','green'] ,figsize=(19,10),lw=2);
    ax.set_xlim(588,615.5)
    plt.fill_between(df_since2019.index, neg, pos, where=neg >= pos, interpolate=True, facecolor='red',alpha=0.7)
    plt.fill_between(df_since2019.index, neg, pos, where=neg <= pos, interpolate=True, facecolor='green',alpha=0.7)
    for d in election_dates[1:]:
        plt.axvline(dt.datetime.fromisoformat(d), color='k',lw=3)
        plt.text(dt.datetime.fromisoformat(d)+dt.timedelta(10),120,f'elections\n date',size=20)
    plt.title(f'Tweets sentiment time series since 2019 on a monthly basis (HeBERT model)',size=25, pad=30,
              fontweight='semibold');
    plt.legend(loc="upper left"); plt.ylabel('Number of tweets');plt.xlabel("");plt.tight_layout();




def sentiment_dist_plot(df_heBert):
    '''
    Returns sentiment distribution for all Hebrew tweets
    '''
    

    l_pie = (
         df_heBert['s_heBert_label'].value_counts(normalize=True)
        .reset_index()
        .rename(columns={'index':'sentiment','s_heBert_label':'percentage'})
        .set_index('sentiment'))

    fig,ax = plt.subplots(figsize=(8, 8))
    plt.pie(l_pie.percentage, explode=(0.05,0.05,0.05),autopct='%1.1f%%', colors = ['green', 'red','grey'],
            shadow=True, startangle=90,pctdistance=0.7,
            labels=['Positive','Negative','Neutral'], labeldistance=1.1, textprops={'fontsize': 18, 'weight':'bold'});
    plt.title(f"Sentiment distribution for all tweets (since 2008, n = {len(df_heBert):,})", size=20,pad=20);




def sentiment_plot(df_heBert,election_dates,since='2019', time='M'):
    
    strftime_c = '%Y-%m' if time=='M' else '%Y-%m-%d'
    if time =='M':
        df_since2019 = df_heBert[since:].groupby('s_heBert_label').resample(time).size().T
        df_since2019.index = df_since2019.index.strftime(strftime_c) 
    else:
        df_since2019 = df_heBert[since:].groupby('s_heBert_label').resample(time).size().unstack(0, fill_value=0)
        df_since2019.index = df_since2019.index.strftime(strftime_c)
        
    print('Elections Dates: ' ,election_dates[1:])
    plt.style.use('seaborn-dark')
    df_since2019.plot(kind='bar',width=0.8,color=['red','grey','green'] ,figsize=(19,10));
    for d in pd.to_datetime(election_dates):
        if d >= pd.to_datetime(since):
            plt.axvline(d, color='k',lw=10)
    time_interval = 'monthly' if time=='M' else 'weekly' if time=='W' else 'daily'
    plt.title(f'Sentiment summarization of tweets since {since} on a {time_interval} basis (HeBERT model)',size=25);
    plt.legend(loc="upper left");plt.xticks(rotation=45); plt.ylabel('Number of tweets');plt.xlabel("");plt.tight_layout();

def sentiment_days(df_heBert,election_dates, n=90):
    '''
    Returns tweets sentiment summary before and after elections
    '''

    sentiment , sentiment_after = dict(), dict()
    one_month_bfore_election =[str(dt.datetime.fromisoformat(i).date() - dt.timedelta(days=90)) for i in election_dates[:-1]]
    time_range_b4_elections_1m = list(zip(one_month_bfore_election,election_dates[:-1]))

    for i, index in enumerate(time_range_b4_elections_1m):
        sentiment[f'Before {index[1][:-3]} Elections'] = df_heBert.loc[time_range_b4_elections_1m[i][0]:\
                                                         time_range_b4_elections_1m[i][1]]['s_heBert_label'].value_counts()
    one_month_after_election =[str(dt.datetime.fromisoformat(i).date() + dt.timedelta(days=n)) for i in election_dates[:-1]]
    time_range_after_elections_1m = list(zip(election_dates,one_month_after_election))

    for i, index in enumerate(time_range_after_elections_1m):
        sentiment_after[f'After {index[0][:-3]} Elections'] = df_heBert.loc[time_range_after_elections_1m[i]\
                                                       [0]:time_range_after_elections_1m[i][1]]['s_heBert_label'].value_counts()
    pd.DataFrame(sentiment).T.plot(kind='bar',width=0.5,color=['red','green','grey'] ,figsize=(14,5), edgecolor='k');
    plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left",prop={'size': 20});plt.xticks(size=15,rotation=0);
    plt.title(f'Tweets sentiment during the {n} Days before the elections Vs. {n} Days after',fontweight='bold');
    pd.DataFrame(sentiment_after).T.plot(kind='bar',width=0.5,color=['red','grey','green'] ,figsize=(14,5), edgecolor='k');
    plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left",prop={'size': 20});plt.xticks(size=15,rotation=0);
    before = pd.DataFrame(sentiment).sum(1)
    after = pd.DataFrame(sentiment_after).sum(1)
    election_sum_sent = pd.DataFrame({'Before elections': before,'After elections':after})
    g = (election_sum_sent.T.div(election_sum_sent.T.sum(1), axis=0).round(3)*100)
    ax = g.plot(kind='bar',width=0.8,color=['red','grey','green'], figsize=(8,6), edgecolor='k',stacked=True);
    plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left",prop={'size': 20});plt.xticks(size=15,rotation=0);
    ax.set_ylim(0,100); plt.title('Tweets sentiment summary before and after elections',size=17,fontweight='bold', pad=20);
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy() 
        ax.text(x+width/2, y+height/2,'{:.1f} %'.format(height), 
                horizontalalignment='center', verticalalignment='center',fontweight='semibold')


if __name__ == '__main__':
    print('utils for pm_summary.ipynb')
