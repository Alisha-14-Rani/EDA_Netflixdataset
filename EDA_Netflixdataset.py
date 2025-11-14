#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
#get_ipython().run_line_magic('matplotlib', 'inline')
md=pd.read_csv('netflix.csv')
tvd=md[md['type']=='TV Show']


# In[2]:


df = pd.read_csv('netflix.csv')


# In[3]:


df.head(3)


# In[4]:


df.tail(4)


# In[5]:


df.columns


# In[6]:


df.shape


# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


df.describe(include='object')


# In[10]:


df.describe(include='all')


# In[11]:


df.sample()


# In[12]:


df.dtypes


# In[13]:


df[df.duplicated()]


# In[14]:


unnesting = ['director', 'cast', 'listed_in','country']
for column in unnesting:
    df[column] = df[column].str.split(', ')
    df = df.explode(column)


# In[15]:


df.shape


# In[16]:


df.dtypes


# In[17]:


df.reset_index(drop=True,inplace=True)


# In[18]:


df


# In[19]:


plt.figure(figsize=(14,8))
sns.heatmap(df.isnull())
plt.title('Visual Check of Nulls',fontsize=20)
plt.show()


# In[20]:


df.isna().sum().sort_values(ascending=False)


# In[21]:


for i in df.columns:
    null_pct = (df[i].isna().sum() / df.shape[0]) *100
    if null_pct > 0 :
        print(f'Null_pct of {i} is {round(null_pct,3)} %')


# In[22]:


df[df.date_added.isna()]


# In[23]:


df['date_added'] = pd.to_datetime(df['date_added'] ,format="%B %d, %Y" , errors='coerce')


# In[24]:


df['date_added'].fillna(df['date_added'].mode()[0], inplace=True)


# In[25]:


df.isna().sum().sort_values(ascending=False)


# In[26]:


df.dtypes


# In[27]:


df['year_added'] = df['date_added'].dt.year


# In[28]:


df.sample()


# In[29]:


df.dtypes


# In[30]:


df.shape


# In[31]:


df.info()


# In[32]:


df.isna().sum().sort_values(ascending=False)


# In[33]:


df[df.rating.isna() | df.duration.isna()]


# In[34]:


df["country"].fillna("Unknown",inplace=True)
df["cast"].fillna("Unknown actors",inplace=True)
df["director"].fillna("Unknown director",inplace=True)
df["rating"].fillna("Unknown",inplace=True)


# In[35]:


df.isna().sum()


# In[36]:


df[df.duration.isna()]


# In[37]:


df.rating.value_counts()


# In[38]:


rvc = df.rating.value_counts(dropna=False).reset_index()


# In[39]:


plt.figure(figsize=(14,5))
a = sns.barplot(rvc , x='rating' , y='count' , color='red' , width=0.3)
plt.title('Raw analysis of Ratings',fontsize=20,fontweight='bold')
a.bar_label(a.containers[0], label_type='edge')
plt.show()


# In[40]:


df[df.director=='Louis C.K.'] 


# In[41]:


df.loc[df['director']=='Louis C.K.', 'duration']=df.loc[df['director']=='Louis C.K.','duration'].fillna(df.loc[df['director'] == 'Louis C.K.', 'rating'])


# In[42]:


df[df.director=='Louis C.K.'] 


# In[43]:


df.loc[df['director'] == 'Louis C.K.', 'rating'] = 'Unknown'


# In[44]:


df[df.director=='Louis C.K.'] 


# In[45]:


df.shape


# In[46]:


df.dtypes


# In[47]:


df.isna().sum()


# In[48]:


df.type.value_counts()


# In[49]:


movies_data = df[df.type=='Movie']


# In[50]:


movies_data.shape


# In[51]:


tvshows_data = df[df.type=='TV Show']


# In[52]:


tvshows_data.shape


# In[53]:


movies_data.sample()


# In[54]:


movies_data.isna().sum()


# In[55]:


tvshows_data.sample()


# In[56]:


tvshows_data.isna().sum()


# In[57]:


movies_data['runtime_in_mins'] = movies_data['duration'].str.split(' ').str[0]
tvshows_data['no_of_seasons'] = tvshows_data['duration'].str.split(' ').str[0]


# In[58]:


movies_data.sample()


# In[59]:


movies_data.dtypes


# In[60]:


movies_data.runtime_in_mins = movies_data.runtime_in_mins.astype(int)


# In[61]:


movies_data.dtypes


# In[62]:


movies_data = movies_data.drop(columns=['description','duration']).reset_index(drop=True)


# In[63]:


movies_data.shape


# In[64]:


tvshows_data.tail()


# In[65]:


tvshows_data.no_of_seasons.value_counts()


# In[66]:


tvshows_data.dtypes


# In[67]:


tvshows_data.no_of_seasons = tvshows_data.no_of_seasons.astype(int)


# In[68]:


tvshows_data.no_of_seasons.dtypes


# In[69]:


tvshows_data = tvshows_data.drop(columns=['description','duration']).reset_index(drop=True)


# In[70]:


tvshows_data.sample(3)


# In[71]:


df = df.drop(columns=['description']).reset_index(drop=True)


# In[72]:


print(f'Cleaned Netflix data has {df.shape[0]} Rows and {df.shape[1]} Columns')
print(f'Netflix Movies data has {movies_data.shape[0]} Rows and {movies_data.shape[1]} Columns')
print(f'Netflix TV shows data has {tvshows_data.shape[0]} Rows and {tvshows_data.shape[1]} Columns')


# In[73]:


df.shape


# In[74]:


df.type.value_counts()


# In[75]:


plt.figure(figsize=(25,8), layout='tight').suptitle('Visual checks of Nulls',fontsize=20,fontweight="bold",fontfamily='serif')


plt.subplot(1,3,1)
sns.heatmap(df.isnull())
plt.title('cleaned Netflix data Nulls',fontsize=12)
plt.xlabel('metrics',fontsize=12)
plt.ylabel('row_numbers',fontsize=12)

plt.subplot(1,3,2)
sns.heatmap(movies_data.isnull())
plt.title('Movies data Nulls',fontsize=12)
plt.xlabel('metrics',fontsize=12)
plt.ylabel('row_numbers',fontsize=12)


plt.subplot(1,3,3)
sns.heatmap(tvshows_data.isnull())
plt.title('Tv Shows data Nulls',fontsize=12)
plt.xlabel('metrics',fontsize=12)
plt.ylabel('row_numbers',fontsize=12)

plt.show()


# In[76]:


df.to_csv('netflix_cleaned_data.csv',sep=',',index=False)
movies_data.to_csv('cleaned_movies_data.csv',sep=',',index=False)
tvshows_data.to_csv('cleaned_tvshows_data.csv',sep=',',index=False)


# In[77]:


nx = pd.read_csv('netflix_cleaned_data.csv')
md = pd.read_csv('cleaned_movies_data.csv')
tvd = pd.read_csv('cleaned_tvshows_data.csv')


# In[78]:


pg = nx.groupby('type')['show_id'].nunique()
pg


# In[79]:


pgdf = pg.reset_index()
pgdf


# In[80]:


plt.figure(figsize=(13.5,4))
font = {'weight':'bold',
        'family':'serif'}
plt.suptitle("Netflix Contents Distribution",fontdict=font,fontsize=20)

plt.style.use('seaborn-v0_8-bright')

plt.subplot(1,2,1)
plt.pie(pg,
        labels=pg.index,
        startangle=80, explode=(0.08,0), colors=['red','#dedede'],
        shadow=True, autopct='%1.1f%%',textprops={'color':"k"})

plt.subplot(1,2,2)
a = sns.barplot(y=pgdf.show_id , data=pgdf , x=pgdf.type , palette=['red','#dedede'])
a.bar_label(a.containers[0], label_type='edge')
sns.despine(left=True,bottom=True)
plt.yticks([])
plt.ylabel('')


plt.show()


# In[81]:


plt.figure(figsize=(16,6))
font = {'weight':'bold',
        'family':'serif'}
plt.suptitle("Netflix Contents Distribution",fontweight='bold',fontsize=20)

plt.style.use('seaborn-v0_8-bright')

plt.subplot(1,2,1)
sns.violinplot(nx,x='type',y='release_year',palette=['red','#dedede'])
sns.despine()
plt.xlabel('')
plt.title(" Violin Distribution",fontdict=font,fontsize=14)

plt.subplot(1,2,2)
sns.boxplot(nx,x='type',y='release_year',palette=['red','#dedede'])
sns.despine()
plt.xlabel('')
plt.title("Box & Whisker Distribution",fontdict=font,fontsize=14)

plt.show()


# In[82]:


ryvc = nx.release_year.value_counts()[:20]


# In[83]:


plt.figure(figsize=(13,5))
plt.style.use('seaborn-v0_8-bright')
sns.countplot(nx , y='release_year' , order = ryvc.index , palette=['red','dimgrey'] , width=0.2)
sns.despine(bottom=True)
plt.xticks([])
plt.xlabel('')
plt.title('Years With Maximum contents Released',fontsize=16,fontweight='bold',fontfamily='serif')
plt.show()


# In[84]:


cm = md.groupby('country')[['show_id']].nunique().sort_values(by='show_id',ascending=False)
cm = cm[:15]
cwm = cm[cm.index!=('Unknown')]
cwm


# In[85]:


# countrywise content count with tvshows_data
ctv = tvd.groupby('country')[['show_id']].nunique().sort_values(by='show_id',ascending=False)
cwtv = ctv[:15]
cwtv = cwtv[cwtv.index!=('Unknown')]
cwtv


# In[86]:


# Graphical Analysis
plt.figure(figsize=(16,6))
plt.suptitle('Countries consuming Movies & TV Shows',
             fontsize=20,fontweight="bold",fontfamily='serif')
plt.style.use('seaborn-v0_8-bright')

c1 = sns.barplot(cwm, x=cwm.index , y=cwm.show_id,
                 color='red' , width=0.4 , label='Movies_count')
#c1.bar_label(c1.containers[0], label_type='edge',color='r')
plt.xlabel('Country',fontsize=12)
plt.ylabel('Content count',fontsize=12)
plt.legend(loc='upper right')
plt.xticks(rotation=30)

c2 = sns.barplot(cwtv, x=cwtv.index , y=cwtv.show_id,
                 color='dimgray' , width=0.2 , label='Tvshows_count')
plt.xlabel('Country',fontsize=12)
plt.ylabel('Content count',fontsize=12)
plt.legend(loc='upper right')
plt.xticks(rotation=30)

top_n = 14
for i in range(top_n):
    c1.annotate(cwm.show_id[i], (i+0.12, cwm.show_id[i]+50),
                ha='left', va='baseline',color='red')
    
for i in range(top_n):
    c2.annotate(cwtv.show_id[i], (i+0.22, cwtv.show_id[i]),
                ha='left', va='baseline', color='dimgrey')
    
plt.show()


# In[87]:


yc = nx.groupby(['year_added','type'])[['show_id']].nunique().reset_index()
yc.sort_values(by='show_id',ascending=False)


# In[88]:


yc['show_id'].sum()


# In[89]:


ycm = md.groupby(['year_added','type'])[['show_id']].nunique().reset_index()
ycm


# In[90]:


ycm['show_id'].sum()


# In[91]:


yctv = tvd.groupby(['year_added','type'])[['show_id']].nunique().reset_index()
yctv


# In[92]:


yctv['show_id'].sum()


# In[93]:


plt.figure(figsize=(16,6))
plt.style.use('default')
plt.style.use('seaborn-v0_8-bright')
c = sns.barplot(data = yc, x = 'year_added' , y = 'show_id' ,
                hue = 'type', palette=['red','dimgrey'] , width=0.35)
plt.title('Contents added to Netflix Yearwise',
          fontsize=16,fontweight="bold",fontfamily='serif')
c.bar_label(c.containers[0], label_type='edge',color='red')
c.bar_label(c.containers[1], label_type='edge',color='dimgray')
plt.legend(loc='upper left')
plt.show()


# In[94]:


mr = md.groupby('release_year')[['title']].nunique()
mr = mr.reset_index()
mr


# In[95]:


mr.title.sum()


# In[96]:


tvr = tvd.groupby('release_year')[['title']].nunique()
tvr = tvr.reset_index()
tvr


# In[97]:


tvr.title.sum()


# In[98]:


plt.figure(figsize=(16,6))
plt.style.use('seaborn-v0_8-darkgrid')
sns.lineplot(data=mr , x='release_year' , y='title' , color='r' ,
             label = 'Movies', marker='d')
sns.lineplot(data=tvr , x='release_year' , y='title' , color='dimgrey',
             label='Tv Shows' , marker='d')
plt.title('Contents Released count Yearwise',fontsize=16,
                      fontweight="bold",fontfamily='serif')
plt.ylabel('contents uploaded count')
plt.legend(loc='upper left')
plt.show()


# In[99]:


plt.figure(figsize=(30,10) , dpi=250)
plt.suptitle('Yearly Release of Contents',
             fontsize=20,fontweight="bold",fontfamily='serif')
plt.style.use('default')
plt.style.use('seaborn-v0_8-bright')

mr= mr[mr.release_year>2000]
plt.subplot(1,2,1)
c = sns.barplot(mr , x = 'release_year' , y='title', color='tomato',width=0.98)
c.bar_label(c.containers[0], label_type='edge',color='r')
sns.pointplot(mr , x='release_year' , y='title' , color='r')
plt.xlabel("Year",fontsize=12)
plt.ylabel("Movies Counts", fontsize=12)
plt.title("Year of Movies Release", fontsize=16,fontweight="bold",fontfamily='serif')

plt.subplot(1,2,2)
d = sns.histplot(x = tvr.release_year, bins = 10, kde = True, 
             color='dimgrey' , edgecolor ='dimgrey')
d.bar_label(d.containers[0], label_type='edge',color='dimgrey')
plt.xlabel('Year',fontsize=12)
plt.ylabel("TV Show Counts", fontsize=12)
plt.title("Year of TV show Release", fontsize=16,fontweight="bold",fontfamily='serif')

plt.show()


# In[100]:


mr.dtypes


# In[101]:


plt.figure(figsize=(30,10) , dpi=250)
plt.suptitle('Yearly Release of Contents',
             fontsize=20,fontweight="bold",fontfamily='serif')
plt.style.use('default')
plt.style.use('seaborn-v0_8-bright')

plt.subplot(1,2,1) 
sns.boxplot(md , x= 'release_year', color='red')
sns.despine()
plt.title('Movie Releases',fontsize=16,fontfamily='serif')

plt.subplot(1,2,2)
sns.boxplot(tvr , x= 'release_year', color='dimgrey')
sns.despine(left=True)
plt.title('Tvshow Releases',fontsize=16,fontfamily='serif')

plt.show()


# In[102]:


sns.jointplot(nx , x='year_added' , y='release_year' , hue='type' , 
                      palette=['red','dimgrey'])
plt.show()


# In[103]:


mg = md.groupby(['listed_in'])[['title']].nunique().sort_values(by='title',ascending=False)
mg = mg.reset_index()
mg


# In[104]:


tvg = tvd.groupby(['listed_in'])[['title']].nunique().sort_values(by='title',ascending=False)
tvg = tvg.reset_index()
tvg


# In[105]:


plt.figure(figsize=(25,10))
plt.suptitle('Popular Genre Contents count',fontsize=20,
             fontweight="bold",fontfamily='cursive')
plt.style.use('default')
plt.style.use('seaborn-v0_8-bright')

plt.subplot(1,2,2)
sns.barplot(mg , x='title' , y='listed_in' , color='red' , width=0.3)
sns.despine(left=True,bottom=True,trim=True)
plt.title('Movie Genre',fontsize=16)
plt.xlabel('Movies count')
plt.xticks([])
n=20
for i in range(n):
    plt.annotate(mg.title[i], (mg.title[i]+75,i+0.2),
                 ha='center' , va='bottom' , color='r')

plt.subplot(1,2,1)
sns.barplot(tvg , x='title' , y='listed_in' , color='dimgrey' , width=0.3)
sns.despine(left=True,bottom=True,trim=True)
plt.title('Tv Show Genre',fontsize=16)
plt.xlabel('TvShows count')
plt.xticks([])
nn=22
for i in range(nn):
    plt.annotate(tvg.title[i], (tvg.title[i]+45,i+0.2),
                 ha='center' , va='bottom' , color='dimgrey')

plt.show()


# In[106]:


from wordcloud import WordCloud


# In[107]:


plt.figure(figsize=(16,4))
plt.suptitle('Popular Genre Contents in Word Cloud',
             fontsize=16,fontweight="bold",fontfamily='fantasy')
plt.style.use('default')
plt.style.use('dark_background')

plt.subplot(1,2,1)
mgwc = WordCloud(width=1600, height=800, background_color='black',
                 colormap='Reds').generate(md.listed_in.to_string())
plt.imshow(mgwc)
plt.axis('off')
plt.title("Movie Genre",fontsize=14,fontweight='bold',fontfamily='serif')

plt.subplot(1,2,2)
tvgwc = WordCloud(width=1600, height=800, background_color='black',
                  colormap='Greys').generate(tvd.listed_in.to_string())
plt.imshow(tvgwc)
plt.axis('off')
plt.title("Tv Shows Genre",fontsize=14,fontweight='bold',fontfamily='serif')

plt.show()


# In[108]:


mdgc = md.groupby('listed_in')['director'].nunique().sort_values(ascending=False)
mdgc


# In[109]:


tvdgc = tvd.groupby('listed_in')['director'].nunique().sort_values(ascending=False)
tvdgc


# In[110]:


plt.figure(figsize=(25, 12))
plt.suptitle('Directors popular Genre Contents',
                fontsize=20,fontweight="bold",fontfamily='cursive')
plt.style.use('seaborn-v0_8-bright')

plt.subplot(1,2,1)
a = sns.barplot(y=mdgc.index, x=mdgc.values, color='r',width=0.3)
a.bar_label(a.containers[0], label_type='edge',color='r')
plt.title('Movie Directors comfy Genre\'s',fontsize=20,
                  fontweight="bold",fontfamily='serif')
sns.despine(left=True,bottom=True,trim=True)
plt.ylabel('Genre')
plt.xticks([])

plt.subplot(1,2,2)
a = sns.barplot(y=tvdgc.index, x=tvdgc.values, color='dimgrey',width=0.3)
a.bar_label(a.containers[0], label_type='edge',color='dimgrey')
plt.title('TvShow Directors comfy Genre\'s',fontsize=20,
                  fontweight="bold",fontfamily='serif')
sns.despine(left=True,bottom=True,trim=True)
plt.ylabel('')
plt.xticks([])
plt.show()


# In[111]:


plt.figure(figsize=(20, 8))
plt.style.use('default')
plt.style.use('seaborn-v0_8-bright')


given_country = input("Enter your preferred Choice of Country : ")

mcountry = md[md["country"] == given_country]
tvcountry = tvd[tvd["country"] == given_country]

mc_data = mcountry.groupby(['listed_in','type'])[['show_id']].nunique()
mc_data = mc_data.sort_values(by=["show_id"],ascending = False).reset_index()

mtv_data = tvcountry.groupby(['listed_in','type'])[['show_id']].nunique()
mtv_data = mtv_data .sort_values(by=["show_id"],ascending = False).reset_index()


plt.suptitle('Genre Distribution across the selected Country'
             ,fontsize=20,fontweight="bold",fontfamily='serif')

plt.subplot(1,2,1)
a = sns.barplot(mc_data , y='listed_in', x='show_id',color='red',width = 0.2)
a.bar_label(a.containers[0], label_type='edge',color='red')
plt.title('Movie Genre Distribution')
plt.xlabel('')
plt.xticks([])
plt.ylabel('Count of Contents')
plt.xticks(rotation=90, ha = 'center',fontsize = 8)
plt.yticks(fontsize =8)

plt.subplot(1,2,2)
b = sns.barplot(mtv_data , y='listed_in', x='show_id',color='dimgrey',width = 0.2)
b.bar_label(b.containers[0], label_type='edge',color='dimgrey')
sns.despine(bottom=True,left=True)
plt.title('TvShows Genre Distribution')
plt.xlabel('')
plt.xticks([])
plt.ylabel('')
plt.xticks(rotation=90, ha = 'center',fontsize = 8)
plt.yticks(fontsize =8)

plt.show()


# In[112]:


tvd.groupby(['no_of_seasons'])[['title']].nunique().sum()


# In[113]:


md.groupby(['runtime_in_mins'])[['title']].nunique().sum()


# In[114]:


mrt = md.groupby(['runtime_in_mins'])[['title']].nunique().sort_values(by='title',ascending=False)
mrt = mrt.reset_index()
mrt


# In[115]:


tvs = tvd.groupby(['no_of_seasons'])[['title']].nunique().sort_values(by='title',ascending=False)
tvs = tvs.reset_index()
tvs


# In[116]:


plt.figure(figsize=(25,13))
plt.suptitle('Length of Contents',
             fontsize=20,fontweight="bold",fontfamily='serif')
plt.style.use('default')
plt.style.use('seaborn-v0_8-bright')

plt.subplot(2,1,1)
sns.lineplot(mrt , y='title' , x='runtime_in_mins' , color='red' , marker='d')
sns.despine()
plt.grid(True, linestyle='--', alpha=0.6)
plt.title('Movie runtime\n' ,fontsize=12,fontweight="bold")
plt.ylabel('Movies count')
plt.text(205,123,'It is seen that the most optimum\nduration for a content is',
         fontsize=14,fontfamily='sans-serif')
plt.text(248,122,'90-120 Minutes',color='r',
         fontsize=14,fontfamily='fantasy',fontweight='bold')
max_value = mrt.title.max()
max_x = mrt[mrt.title == max_value]['runtime_in_mins'] 
sns.scatterplot(x=max_x, y=max_value, color='#dedede', marker='s', s=10000)

plt.subplot(2,1,2)
sns.barplot(tvs , y='title' , x='no_of_seasons' , color='dimgrey' , width=0.3)
sns.despine()
plt.title('------------------------------\n Tv Shows Seasons',
          fontsize=12,fontweight="bold")
plt.ylabel('TvShows count')
n=15
for i in range(n):
     plt.annotate(tvs.title[i], (i+0.2,tvs.title[i]+45),
                  ha='center' , va='bottom' , color='dimgrey')

plt.show()


# In[117]:


plt.figure(figsize=(15,8))
plt.suptitle('Length of Contents',
             fontsize=20,fontweight="bold",fontfamily='serif')
plt.style.use('default')
plt.style.use('seaborn-v0_8-darkgrid')

plt.subplot(1,2,1)
sns.histplot(x = md.runtime_in_mins, bins = 200, color='red',
            kde = True, edgecolor = 'salmon')
plt.xlabel("Movie Duration in mins",fontsize=12)
plt.ylabel("Movies Counts", fontsize=12)
plt.title("Duration of Movies", fontsize=14)

plt.subplot(1,2,2)
b = sns.histplot(x = tvd.no_of_seasons, bins = 10, kde = True, 
             color='dimgrey' , edgecolor ='k')
b.bar_label(b.containers[0], label_type='edge',color='dimgrey')
plt.xlabel('No.of Seasons',fontsize=12)
plt.ylabel("TV Show Counts", fontsize=12)
plt.title("Duration of TV shows", fontsize=14)

plt.show()


# In[118]:


plt.figure(figsize=(18,8) , dpi=250)
plt.suptitle('Contents Duration',
             fontsize=20,fontweight="bold",fontfamily='serif')
plt.style.use('default')
plt.style.use('seaborn-v0_8-bright')

plt.subplot(2,2,1)
sns.boxplot(md , x ='runtime_in_mins', color='red',showfliers=True)
plt.title('Movies',fontsize=16,fontfamily='serif')

plt.subplot(2,2,3)
sns.boxplot(mrt , x='title', color='red')

plt.subplot(2,2,2)
sns.boxplot(tvd , x= 'no_of_seasons', color='dimgrey')
plt.title('TvShows',fontsize=16,fontfamily='serif')

plt.subplot(2,2,4)
sns.boxplot(tvs , x='title', color='dimgrey')
sns.despine(left=True,trim=True)

plt.show()


# In[119]:


sns.relplot(nx , x='release_year' , 
              y='duration' , hue='type' , 
              palette=['red','dimgrey'])
plt.text(1948,320,'Overall Duration Comparison \n with release year',color='dimgrey',
         fontsize=12,fontweight='bold')
plt.show()


# In[120]:


sns.relplot(nx , x='year_added' , 
              y='duration' , hue='type' , 
              palette=['red','dimgrey'])
plt.text(2007,300,'Overall Duration Comparison \n with uploaded year',color='red',
         fontsize=12,fontweight='bold')
plt.show()


# In[121]:


sns.relplot(md , y='release_year' , x='runtime_in_mins' , color='red')
plt.text(35,2022.5,'Movies Duration Comparison with release year',
                 color='dimgrey',fontsize=9,fontweight='bold')
plt.show()


# In[122]:


sns.jointplot(md , x='date_added' , y='runtime_in_mins' , color='red')
plt.text(50,320,'Movies Duration Comparison with uploaded date',
                 color='dimgrey',fontsize=9,fontweight='bold')
plt.show()


# In[123]:


sns.jointplot(md , y='release_year' , x='runtime_in_mins' , color='red')
plt.text(30,2022.4,'TvShows Comparison by release year',color='dimgrey',fontsize=12,fontweight='bold')
plt.show()


# In[124]:


sns.jointplot(md , x='date_added' , y='runtime_in_mins' , color='red')
plt.text(35,290,'TvShows Comparison by uploaded_date',color='dimgrey',fontsize=12,fontweight='bold')
plt.show()


# In[125]:


sns.jointplot(tvd , x='release_year' , y='year_added' , color='dimgrey')
plt.text(1930,2021.3,'TvShows Comparison by release year',color='red',fontsize=12,fontweight='bold')
plt.show()


# In[126]:


sns.jointplot(tvd , y='release_year' , x='no_of_seasons' , color='dimgrey')
plt.text(2,2022.8,'TvShows Comparison by release year',
             color='red',fontsize=12,fontweight='bold')
plt.show()


# In[127]:


md.groupby(['rating'])[['title']].nunique().sum()


# In[128]:


tvd.groupby(['rating'])[['title']].nunique().sum()


# In[129]:


movie_rating = md.groupby(['rating'])[['title']].nunique().reset_index()
movie_rating = movie_rating.sort_values(by='title',ascending=False)
movie_rating


# In[130]:


tv_rating = tvd.groupby(['rating'])[['title']].nunique().reset_index()
tv_rating = tv_rating.sort_values(by='title',ascending=False)
tv_rating


# In[131]:


plt.figure(figsize=(16,6))
plt.style.use('ggplot')
sns.lineplot(data=movie_rating , x='rating' , y='title' , color='r' , label = 'Movies', marker='s')
sns.lineplot(data=tv_rating , x='rating' , y='title' , color='dimgrey', label='Tv Shows' , marker='o')
plt.title('Ratings of the Contents Released',fontsize=16,fontweight="bold",fontfamily='serif')
plt.ylabel('contents uploaded count')
plt.legend(loc='upper right')
plt.show()


# In[132]:


movies_cast = md.groupby('cast')[['title']].nunique().sort_values(by='title',ascending=False)[1:20]
movies_cast


# In[133]:


tv_cast = tvd.groupby('cast')[['title']].nunique().sort_values(by='title',ascending=False)[1:20]
tv_cast


# In[134]:


plt.figure(figsize=(20,12))
plt.suptitle('Actors with more Contents',
             fontsize=20,fontweight="bold",fontfamily='serif',color='k')
plt.style.use('Solarize_Light2')

plt.subplot(2,1,1)
c1 = sns.barplot(movies_cast, y=movies_cast.index , x='title',color='red',width=0.3)
sns.despine(left=True,bottom=True,trim=True)
plt.title('Actors with more Movie Contents',
          fontsize=16,fontweight="bold",fontfamily='serif',color='r')
plt.xticks([])
plt.yticks(fontweight='bold')
plt.xlabel('')
plt.ylabel('Actors',fontsize=12)
for i in range(19):
    c1.annotate((str(movies_cast.title[i])+' movies'), (movies_cast.title[i]+1,i+0.3),
                ha='center' , va='bottom' , color='red')

plt.subplot(2,1,2)
c2 = sns.barplot(tv_cast, y=tv_cast.index , x='title',color='dimgrey',width=0.3)
sns.despine(left=True,bottom=True,trim=True)
plt.title('\n Actors with more TvShows Contents',
          fontsize=16,fontweight="bold",fontfamily='serif',color='dimgray')
plt.xticks([])
plt.xlabel('')
plt.yticks(fontweight='bold')
plt.ylabel('Actors',fontsize=12)
for i in range(19):
    c2.annotate((str(tv_cast.title[i])+' shows'), (tv_cast.title[i]+0.63,i+0.3),
                ha='center' , va='bottom' , color='dimgrey')

plt.show()


# In[135]:


fmd = md.groupby('director')[['show_id']].nunique()
fmd = fmd.sort_values(by='show_id',ascending=False)[1:21]
fmd


# In[136]:


ftvd = tvd.groupby(['director'])[['show_id']].nunique()
ftvd= ftvd.sort_values(by='show_id',ascending=False)[1:21]
ftvd


# In[137]:


plt.figure(figsize=(20,12))
plt.suptitle('Directors with more Contents',
             fontsize=20,fontweight="bold",fontfamily='serif',color='k')
plt.style.use('Solarize_Light2')

plt.subplot(2,1,1)
c1 = sns.barplot(fmd, y=fmd.index , x='show_id',color='red',width=0.3)
sns.despine(left=True,bottom=True,trim=True)
plt.title('Directors with more Movie Contents',
          fontsize=16,fontweight="bold",fontfamily='serif',color='r')
plt.xticks([])
plt.yticks(fontweight='bold')
plt.xlabel('')
plt.ylabel('Directors',fontsize=12)
for i in range(20):
    c1.annotate((str(fmd.show_id[i])+' movies'), (fmd.show_id[i]+0.53,i+0.3),
                ha='center' , va='bottom' , color='red')

plt.subplot(2,1,2)
c2 = sns.barplot(ftvd, y=ftvd.index , x='show_id',color='dimgrey',width=0.3)
sns.despine(left=True,bottom=True,trim=True)
plt.title('\n Directors with more TvShows Contents',
          fontsize=16,fontweight="bold",fontfamily='serif',color='dimgray')
plt.xticks([])
plt.xlabel('')
plt.yticks(fontweight='bold')
plt.ylabel('Directors',fontsize=12)
for i in range(20):
    if ftvd.show_id[i]>1:
        c2.annotate((str(ftvd.show_id[i])+' shows'),(ftvd.show_id[i]+0.07,i+0.3),
                    ha='center' , va='bottom' , color='dimgrey')
    else:
        c2.annotate((str(ftvd.show_id[i])+' show'),(ftvd.show_id[i]+0.07,i+0.3),
                    ha='center' , va='bottom' , color='dimgrey')
plt.show()


# In[138]:


ad = nx[['cast','show_id','director','type']]
ad = ad[ad.cast!='Unknown actors']
ad = ad[ad.director!='Unknown director']
ad = ad.drop_duplicates().reset_index(drop=True)
ad


# In[139]:


nad = ad.groupby(['cast','director','type'])[['show_id']].nunique()
new_ad = nad.reset_index().sort_values(by='show_id',ascending=False)
new_ad['ad_pair'] = new_ad['cast']+'-'+new_ad['director']
new_ad


# In[140]:


mad = new_ad[new_ad.type=='Movie']
tvad = new_ad[new_ad.type=='TV Show']


# In[141]:


mad.info()


# In[142]:


mad = mad[['ad_pair','show_id']]
mad


# In[143]:


tvad = tvad[['ad_pair','show_id']]
tvad


# In[144]:


mad.dtypes


# In[145]:


fmad = mad[:25].set_index('ad_pair')
ftvad = tvad[:25].set_index('ad_pair')


# In[146]:


fmad


# In[147]:


ftvad


# In[148]:


plt.figure(figsize=(19, 15))
plt.suptitle('Actor - Director pairs',fontsize=20,
                 fontweight="bold",fontfamily='serif')
plt.style.use('default')
plt.style.use('Solarize_Light2')

plt.subplot(2,1,1)
a1 = sns.barplot(y=fmad.index, x=fmad.show_id, color='red',width=0.3)
plt.title('Movie Directors-Actors Combo',fontsize=12,fontweight="bold")
sns.despine(left=True,bottom=True,trim=True)
plt.yticks(fontweight='bold')
plt.xticks([])
plt.xlabel('No.of times worked together')
for i in range(25):
    a1.annotate((str(fmad.show_id[i])+' times'), (fmad.show_id[i]+0.47,i+0.5),
                ha='center' , va='bottom' , color='red')

plt.subplot(2,1,2)
a2 = sns.barplot(ftvad , y=ftvad.index, x=ftvad.show_id, color='dimgrey',width=0.3)
plt.title('TvShow Directors-Actors Combo',fontsize=12,fontweight="bold")
sns.despine(left=True,bottom=True,trim=True)
plt.yticks(fontweight='bold')
plt.xticks([])
plt.xlabel('No.of times worked together')
for i in range(25):
    if ftvad.show_id[i]>1:
        a2.annotate((str(ftvad.show_id[i])+' times'), (ftvad.show_id[i]+0.07,i+0.2),
                ha='center' , va='bottom' , color='dimgrey')
    else:
        a2.annotate((str(ftvad.show_id[i])+' time'), (ftvad.show_id[i]+0.07,i+0.3),
                ha='center' , va='bottom' , color='dimgrey')
    
plt.show()


# In[149]:


md.columns


# In[150]:


movie_release = md[['show_id','title','date_added']]
movie_release = movie_release.reset_index(drop=True)
movie_release


# In[151]:


movie_release.dtypes


# In[152]:


movie_release['date_added'] = pd.to_datetime(movie_release['date_added'])


# In[153]:


movie_release.dtypes


# In[154]:


movie_release.isna().sum()


# In[155]:


movie_release['week_uploaded'] = movie_release['date_added'].dt.isocalendar().week
movie_release['uploaded_weekday'] = movie_release['date_added'].dt.strftime('%A')
movie_release['uploaded_month'] = movie_release['date_added'].dt.strftime('%B')


# In[156]:


month_order = ['January', 'February', 'March', 'April', 'May',
               'June', 'July', 'August', 'September', 
               'October', 'November', 'December']
movie_release['uploaded_month']= pd.Categorical(movie_release['uploaded_month'],
                                            categories=month_order, ordered=True)


# In[157]:


movie_release


# In[158]:


week_movie_release=movie_release.groupby('week_uploaded')['show_id'].nunique()
week_movie_release=week_movie_release.reset_index()
week_movie_release


# In[159]:


week_movie_release.sum()


# In[160]:


monthly_movie_release = movie_release.groupby('uploaded_month').agg({
    'title': 'nunique',
    'show_id': 'nunique'
}).reset_index().sort_values(by='uploaded_month').reset_index(drop=True)

monthly_movie_release


# In[161]:


monthly_movie_release.title.sum()


# In[162]:


movies_release_pivot = movie_release.pivot_table(index='uploaded_month', 
                                                 columns='uploaded_weekday', 
                                                 values='show_id', 
                                                 aggfunc=pd.Series.nunique)

day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
             'Friday', 'Saturday','Sunday']
movies_release_pivot = movies_release_pivot[day_order]
movies_release_pivot


# In[163]:


plt.figure(figsize=(15, 10))
plt.style.use('default')
plt.style.use('seaborn-v0_8-bright')
sns.heatmap(movies_release_pivot, cmap='Reds',
                annot=True, fmt='d' , linewidth=0.1)
plt.title("Movie Releases by Weekday and Month",
              fontfamily='serif',fontsize=16,fontweight='bold')
plt.tick_params(axis='both', which='both', left=False, bottom=False)
plt.show()


# In[164]:


movies_release_pivot.sum().sort_values(ascending=False)


# In[165]:


movies_release_pivot.sum().sum()


# In[166]:


tvs_release = tvd[['show_id','title','date_added']]
tvs_release = tvs_release.reset_index(drop=True)
tvs_release


# In[167]:


tvs_release.dtypes


# In[168]:


tvs_release.isna().sum()


# In[169]:


tvs_release['date_added'].fillna(tvs_release['date_added'].mode()[0],inplace=True)


# In[170]:


tvs_release['date_added'] = pd.to_datetime(tvs_release['date_added'])


# In[171]:


tvs_release['date_added'].dtypes


# In[172]:


tvs_release['week_uploaded'] = tvs_release['date_added'].dt.isocalendar().week
tvs_release['uploaded_weekday'] = tvs_release['date_added'].dt.strftime('%A')
tvs_release['uploaded_month'] = tvs_release['date_added'].dt.strftime('%B')


# In[173]:


month_order = ['January', 'February', 'March', 'April', 'May',
               'June', 'July', 'August', 'September', 
               'October', 'November', 'December']
tvs_release['uploaded_month']= pd.Categorical(tvs_release['uploaded_month'],
                                    categories=month_order, ordered=True)


# In[174]:


tvs_release


# In[175]:


tvs_release.groupby('week_uploaded')['show_id'].nunique().sum()


# In[176]:


week_release = tvs_release.groupby('week_uploaded')['show_id'].nunique()
week_release = week_release.reset_index()
week_release


# In[177]:


month_release = tvs_release.groupby('uploaded_month')['show_id'].nunique()
month_release = month_release.reset_index()
month_release


# In[178]:


month_release.show_id.sum()


# In[179]:


tvs_release_pivot = tvs_release.pivot_table(
            index='uploaded_month' , 
            columns='uploaded_weekday' , 
            values='show_id' , 
            aggfunc=pd.Series.nunique
    ) 

day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                 'Friday', 'Saturday','Sunday']
tvs_release_pivot = tvs_release_pivot[day_order]
tvs_release_pivot


# In[180]:


tvs_release_pivot.sum(axis=1)


# In[181]:


tvs_release_pivot.sum().sort_values(ascending=False)


# In[182]:


tvs_release_pivot.sum().sum()


# In[183]:


plt.figure(figsize=(15, 10))
plt.style.use('seaborn-v0_8-bright')
sns.heatmap(tvs_release_pivot, cmap='Greys', annot=True, 
                fmt='d' , linewidth=0.1)
plt.title("TvShows Releases by Weekday and Month",
                  fontfamily='serif',fontsize=16,fontweight='bold')
plt.tick_params(axis='both', which='both', left=False, bottom=False)
plt.show()


# In[184]:


plt.figure(figsize=(30,15))
plt.suptitle('Releases Broader view',fontfamily='serif',
                         fontsize=20,fontweight='bold')

plt.subplot(2,2,1)
sns.pointplot(week_movie_release ,x='week_uploaded' , y='show_id' ,color='r')
plt.title('Movie Weekly Releases count',fontfamily='serif',
                      fontsize=16,fontweight='bold')
plt.xlabel('Week Number')
plt.ylabel('no.of contents released')

plt.subplot(2,2,3)
sns.pointplot(monthly_movie_release ,x='uploaded_month' , y='show_id' ,color='r')
plt.title('Movie Monthly Releases count',fontfamily='serif',
                      fontsize=16,fontweight='bold')
plt.xlabel('Month')
plt.ylabel('no.of contents released')

plt.subplot(2,2,2)
sns.pointplot(week_release ,x='week_uploaded' , y='show_id' ,color='dimgrey')
plt.title('Tv Shows Weekly Releases count',fontfamily='serif',
                      fontsize=16,fontweight='bold')
plt.xlabel('Week Number')
plt.ylabel('no.of contents released')

plt.subplot(2,2,4)
sns.pointplot(month_release ,x='uploaded_month' , y='show_id' ,color='dimgrey')
plt.title('Tv Shows Monthly Releases count',fontfamily='serif',
                          fontsize=16,fontweight='bold')
plt.xlabel('Month')
plt.ylabel('no.of contents released')

plt.show()


# In[185]:


plt.figure(figsize=(20,11) , dpi=400)
plt.suptitle('Releases by Weekday and Month',fontfamily='serif',
                     fontsize=20,fontweight='bold')
plt.style.use('seaborn-v0_8-bright')

plt.subplot(1,2,1)
sns.heatmap(movies_release_pivot, cmap='Reds', annot=True, 
                    fmt='d' , linewidth=0.1)
plt.title("Movie Releases by Weekday and Month",fontfamily='serif',
                  fontsize=16,fontweight='bold')
plt.yticks(rotation=0)
plt.tick_params(axis='both', which='both', left=False , bottom=False)

plt.subplot(1,2,2)
sns.heatmap(tvs_release_pivot, cmap='Greys', annot=True, fmt='d' ,
                    linewidth=0.1)
plt.title("TvShows Releases by Weekday and Month",fontfamily='serif',
                  fontsize=16,fontweight='bold')
plt.yticks(rotation=0)
plt.tick_params(axis='both', which='both', left=False, bottom=False)

plt.show()


# In[186]:


sns.jointplot(md , x='release_year' , y='year_added' , color='red')
plt.text(1940,2021.6,'Movies Comparison' , color='dimgrey' , fontsize=12 , fontweight='bold')
plt.show()


# In[187]:


sns.jointplot(tvd , x='release_year' , y='year_added' , color='dimgrey')
plt.text(1930,2021.4,'TvShows Comparison',color='red',fontsize=12,fontweight='bold')
plt.show()


# In[188]:


filtered_md = md[['show_id','title','release_year','year_added']].drop_duplicates()


# In[189]:


filtered_md.shape


# In[190]:


filtered_md.sample()


# In[191]:


filtered_md['time_diff_in_yrs']=filtered_md['year_added']-filtered_md['release_year']


# In[192]:


filtered_md.head()


# In[193]:


filtered_md.time_diff_in_yrs.mode()[0]


# In[194]:


filtered_md['time_diff_in_yrs'].value_counts()


# In[195]:


fmdg=filtered_md.groupby(['time_diff_in_yrs'])[['title']].agg(numbers_released = ('title','count'))


# In[196]:


rtf = fmdg.sort_values(by='numbers_released',ascending=False)


# In[197]:


rtf = rtf.reset_index()


# In[198]:


rtf


# In[199]:


rtf[rtf.time_diff_in_yrs==-1]


# In[200]:


filtered_md[filtered_md.time_diff_in_yrs==-1]


# In[201]:


filtered_tvd = tvd[['show_id','title','release_year','year_added']].drop_duplicates()


# In[202]:


filtered_tvd


# In[203]:


filtered_tvd.shape


# In[204]:


filtered_tvd['time_diff_in_yrs']=filtered_tvd['year_added']-filtered_tvd['release_year']


# In[205]:


filtered_tvd.tail()


# In[206]:


filtered_tvd['time_diff_in_yrs'].mode()


# In[207]:


filtered_tvd['time_diff_in_yrs'].mode()[0]


# In[208]:


ftv = filtered_tvd.groupby(['time_diff_in_yrs'])[['title']].agg(numbers_released = ('title','count'))


# In[209]:


rtv = ftv.sort_values(by='numbers_released',ascending=False)


# In[210]:


rtv


# In[211]:


rtv = rtv.reset_index()


# In[212]:


rtv[(rtv.time_diff_in_yrs==-1)|(rtv.time_diff_in_yrs==-2)|(rtv.time_diff_in_yrs==-3)]


# In[213]:


filtered_tvd[filtered_tvd.time_diff_in_yrs<0]


# In[214]:


plt.figure(figsize=(25,12))
plt.suptitle('Comparision of TimeFrame Gap for contents getting uploaded on Netflix',
             fontfamily='serif',fontweight='bold',fontsize=20)
plt.style.use('default')
plt.style.use('ggplot')

plt.subplot(2,1,1)
# plt.figure(figsize=(20,6))
sns.pointplot(rtf , x='time_diff_in_yrs' , y='numbers_released' , color='r')
plt.title('Movies uploaded in Netflix TimeFrame(Years)',
                  fontfamily='serif',fontweight='bold',fontsize=16)
plt.text(43,1450,'''Mode of time_diff is the uploading factor \n
                and it is evident that the contents \n',
                 fontsize=12,fontfamily='cursive''')
plt.text(43,1450,'are uploaded at the earliest possible',
                 fontsize=12,fontfamily='cursive')
plt.text(43,1150,'It is seen that movies now a days are being released',
                 fontsize=12,fontfamily='cursive')
plt.text(43,1050,'in OTT platforms with the time gap of',
                 fontsize=12,fontfamily='cursive')
plt.text(54.2,1030,'50-60 days (4 weeks).',
                 fontsize=16,fontfamily='fantasy',fontweight='bold',color='red')

plt.subplot(2,1,2)
# plt.figure(figsize=(20,6))
sns.pointplot(rtv , x='time_diff_in_yrs' , y='numbers_released' , color='dimgrey')
plt.title('Tvshows uploaded in Netflix TimeFrame(Years)',
                    fontfamily='serif',fontweight='bold',fontsize=16)
plt.text(27,1100,'''Mode of time_diff is the uploading factor \n
                    and it is evident that the contents \n',
                 fontsize=12,fontfamily='cursive''')
plt.text(27,1100,'are uploaded at the earliest possible',fontsize=12,fontfamily='cursive')
plt.text(27,950,'It is seen that TvShows now a days are being released',
                 fontsize=12,fontfamily='cursive')
plt.text(27,860,'in OTT platforms with the time gap of',
                 fontsize=12,fontfamily='cursive')
plt.text(34.5,840,'24 Hrs from airing on Television',
                 fontsize=16,fontfamily='fantasy',fontweight='bold',color='dimgrey')
plt.show()


# In[215]:


nx.release_year.dtypes


# In[216]:


cu = nx.copy
cu = nx.drop_duplicates(subset='show_id')
cu['date_added'] = pd.to_datetime(cu['date_added'])
cu['release_date'] = pd.to_datetime(cu['release_year'].astype(str))
cu


# In[217]:


cu['days_to_add'] = (cu['date_added'] - cu['release_date']).dt.days
cu


# In[218]:


# considering entire data (both tvshows and movies)
day_mode = cu['days_to_add'].mode()[0]
day_mode 


# In[219]:


cu.dtypes


# In[220]:


# filtering the recent past data (after 2018) for movies
fcum = cu[(cu.release_year>2018) & (cu.type=='Movie')]
fcum


# In[221]:


upload_date_interval_movie = fcum['days_to_add'].mode()[0]
upload_date_interval_movie


# In[222]:


# filtering the recent past data (after 2018) for tvshows
fcutv = cu[(cu.release_year>2018) & (cu.type=='TV Show')]
fcutv


# In[223]:


fcutv['days_to_add'].value_counts()


# In[224]:


upload_date_interval_tvs = fcutv['days_to_add'].mode()
upload_date_interval_tvs


# In[225]:


upload_date_interval_tvs.mean()


# In[226]:


upload_date_interval_tvs.median()


# In[227]:


upload_date_interval_tvs[0]


# In[228]:


plt.figure(figsize=(15,8))
plt.suptitle('TimeFrame Gap for contents getting uploaded on Netflix',
             fontfamily='serif',fontweight='bold',fontsize=20)
plt.style.use('default')
plt.style.use('seaborn-v0_8-whitegrid')

plt.subplot(1,2,1)
plt.hist(fcum['days_to_add'], bins=30, color='red', edgecolor='white')
plt.title('Days to Add to Netflix After Theatrical Movie Release',
         fontfamily='serif',fontweight='bold',fontsize=12)
plt.xlabel('Days to Add')
plt.ylabel('Frequency')
plt.axvline(upload_date_interval_movie, color='k', linestyle='-.',
            linewidth=3, label=f'Mode: {upload_date_interval_movie} days')
plt.legend(fontsize=16)

plt.subplot(1,2,2)
plt.hist(fcutv['days_to_add'], bins=30, color='silver', edgecolor='white')
plt.title('Days to Add to Netflix After Tvshow Air',
         fontfamily='serif',fontweight='bold',fontsize=12)
plt.xlabel('Days to Add')
plt.ylabel('Frequency')
plt.axvline(upload_date_interval_tvs[0], color='red', linestyle='-.',
            linewidth=3, label=f'Mode: {upload_date_interval_tvs[0]} days')
plt.legend(fontsize=16)

sns.despine()
plt.show()


# In[229]:


md.columns


# In[230]:


#Shortest Movie
shortest_movie = md.loc[(md['runtime_in_mins']==np.min(md.runtime_in_mins))] [['title','runtime_in_mins']].drop_duplicates()
shortest_movie


# In[231]:


#Longest Movie
longest_movie = md.loc[(md['runtime_in_mins']==np.max(md.runtime_in_mins))] [['title','runtime_in_mins']].drop_duplicates()
longest_movie


# In[232]:


md.sample()


# In[233]:


df = md.drop_duplicates(subset='title')


# In[234]:


df.shape


# In[235]:


df = df[['show_id','title','date_added']]


# In[236]:


df.shape


# In[237]:


df.sample()


# In[238]:


df.dtypes


# In[239]:


df['date_added'] = pd.to_datetime(df['date_added'])


# In[240]:


df.dtypes


# In[241]:


df['year_added'] = df['date_added'].dt.year


# In[242]:


df['month_added'] = df['date_added'].dt.month_name()


# In[243]:


df.sample()


# In[244]:


upload_rate = df.groupby('year_added')['month_added'].value_counts()


# In[245]:


upload_rate


# In[246]:


month_order = ['January', 'February', 'March', 'April', 'May',
               'June', 'July','August', 'September', 
               'October', 'November', 'December']
upload_rate = upload_rate.unstack()[month_order]
upload_rate


# In[247]:


upload_rate = upload_rate.fillna(0)
upload_rate


# In[248]:


plt.figure(figsize=(16,8) , dpi=500)
plt.style.use('default')
plt.style.use('seaborn-v0_8-bright')
sns.heatmap(upload_rate, cmap='Reds', edgecolors='beige', linewidths=2)
plt.title('Monthly Netflix Contents Update Rate',
          fontsize=16, fontfamily='calibri', fontweight='bold')
plt.yticks(rotation=0)
plt.xticks(rotation=0)
plt.xlabel('Uploaded Month',fontsize=12)
plt.ylabel('Uploaded Year',fontsize=12)
plt.show()


# In[249]:


sns.jointplot(upload_rate)
plt.show()


# In[250]:


sns.pairplot(nx)


# In[251]:


sns.pairplot(md)


# In[252]:


sns.pairplot(tvd)


# In[253]:


sns.pairplot(upload_rate, kind='scatter')
plt.show()


# In[254]:


sns.pairplot(movies_release_pivot)
plt.show()


# In[255]:


sns.pairplot(tvs_release_pivot)
plt.show()


# In[256]:


movies_release_pivot.corr()


# In[257]:


tvs_release_pivot.corr()


# In[258]:


plt.figure(figsize=(25,10))
plt.suptitle('Correlation',fontsize=30,fontfamily='serif',fontweight='bold')

plt.subplot(1,2,1)
sns.heatmap(movies_release_pivot.corr() ,cmap='Reds',annot=True)
plt.title('Movies corr',fontsize=16,fontweight='bold')
plt.xticks(rotation=0)
plt.yticks(rotation=0)

plt.subplot(1,2,2)
sns.heatmap(movies_release_pivot.corr() ,cmap='Greys',annot=True)
plt.title('Tvshows corr',fontsize=16,fontweight='bold')
plt.xticks(rotation=0)
plt.yticks(rotation=0)

plt.show()


# In[259]:


mdc = md[['release_year','year_added','runtime_in_mins']].corr()
mdc


# In[260]:


tvdc = tvd[['release_year','year_added','no_of_seasons']].corr()
tvdc


# In[261]:


plt.figure(figsize=(15,5))
plt.suptitle('Correlation',fontsize=20,fontfamily='serif',fontweight='bold')

plt.subplot(1,2,1)
sns.heatmap(mdc,cmap='Reds',annot=True)
plt.title('Movies corr',fontsize=16,fontweight='bold')
plt.xticks(rotation=0)
plt.yticks(rotation=0)

plt.subplot(1,2,2)
sns.heatmap(tvdc,cmap='Greys',annot=True)
plt.title('Tvshows corr',fontsize=16,fontweight='bold')
plt.xticks(rotation=0)
plt.yticks(rotation=0)

plt.show()


# In[ ]:





# In[ ]:




