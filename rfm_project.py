from pathlib import Path
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import json, warnings
warnings.filterwarnings('ignore')
BASE=Path(__file__).parent
INPUT=BASE/'OnlineRetail.xlsx'
OUT=BASE
sns.set_theme(style='whitegrid', context='notebook')
# 1. Load and inspect actual uploaded source
raw=pd.read_excel(INPUT, sheet_name=0)
inspection={'source_file':INPUT.name,'sheet':'Online Retail','rows':int(len(raw)),'columns':int(len(raw.columns)),'column_names':list(raw.columns),'missing_values':{k:int(v) for k,v in raw.isna().sum().items()},'duplicate_rows':int(raw.duplicated().sum()),'dtypes':{k:str(v) for k,v in raw.dtypes.items()},'date_min':str(raw['InvoiceDate'].min()),'date_max':str(raw['InvoiceDate'].max()),'unique_customer_ids_before_cleaning':int(raw.CustomerID.nunique()),'head':raw.head().astype(object).where(pd.notna(raw.head()),None).to_dict('records')}
# Cleaning stage counts, explicitly sequential
x=raw.copy(); counts={'input_rows':len(x)}
x=x.drop_duplicates(); counts['duplicates_removed']=len(raw)-len(x); counts['after_deduplication']=len(x)
missing_id=int(x.CustomerID.isna().sum()); x=x.dropna(subset=['CustomerID']); counts['missing_customer_id_rows_removed']=missing_id; counts['after_customer_id_filter']=len(x)
x['InvoiceDate']=pd.to_datetime(x['InvoiceDate'],errors='coerce'); baddate=int(x.InvoiceDate.isna().sum()); x=x.dropna(subset=['InvoiceDate']); counts['invalid_dates_removed']=baddate
cancel=x.InvoiceNo.astype(str).str.upper().str.startswith('C'); counts['cancelled_invoice_rows_removed']=int(cancel.sum()); x=x.loc[~cancel].copy(); counts['after_cancelled_filter']=len(x)
invalid=(x.Quantity<=0)|(x.UnitPrice<=0); counts['nonpositive_quantity_or_price_rows_removed']=int(invalid.sum()); x=x.loc[~invalid].copy(); counts['after_valid_purchase_filter']=len(x)
x['CustomerID']=x.CustomerID.astype('int64'); x['TotalAmount']=x.Quantity*x.UnitPrice
counts['cleaned_rows']=len(x); counts['unique_customers']=int(x.CustomerID.nunique()); counts['cleaned_date_min']=str(x.InvoiceDate.min()); counts['cleaned_date_max']=str(x.InvoiceDate.max())
x.to_csv(OUT/'Cleaned_Online_Retail.csv',index=False)
# RFM reference date = one day after last purchase in the cleaned data
snapshot=x.InvoiceDate.max().normalize()+pd.Timedelta(days=1)
rfm=x.groupby('CustomerID').agg(Recency=('InvoiceDate',lambda s:int((snapshot-s.max().normalize()).days)),Frequency=('InvoiceNo','nunique'),Monetary=('TotalAmount','sum')).reset_index()
# Quantile ranking (ties handled using rank(method='first')); high recency means worse score
rfm['R_Score']=pd.qcut(rfm.Recency.rank(method='first'),5,labels=[5,4,3,2,1]).astype(int)
rfm['F_Score']=pd.qcut(rfm.Frequency.rank(method='first'),5,labels=[1,2,3,4,5]).astype(int)
rfm['M_Score']=pd.qcut(rfm.Monetary.rank(method='first'),5,labels=[1,2,3,4,5]).astype(int)
rfm['RFM_Score']=rfm[['R_Score','F_Score','M_Score']].astype(str).agg(''.join,axis=1)
# log-transform skewed positive metrics; compare k using silhouette on deterministic sample
features=np.log1p(rfm[['Recency','Frequency','Monetary']].astype(float))
scaler=StandardScaler(); scaled=scaler.fit_transform(features)
ks=range(2,9); inertias=[]; silhouettes=[]
rng=np.random.RandomState(42); sil_idx=rng.choice(len(rfm),min(10000,len(rfm)),replace=False)
for k in ks:
 model=KMeans(n_clusters=k,random_state=42,n_init=10); labels=model.fit_predict(scaled); inertias.append(float(model.inertia_)); silhouettes.append(float(silhouette_score(scaled[sil_idx],labels[sil_idx])))
# Selection: maximize sampled silhouette among k=2..8, then fit final model
bestk=int(list(ks)[int(np.argmax(silhouettes))]); final=KMeans(n_clusters=bestk,random_state=42,n_init=20); rfm['Cluster']=final.fit_predict(scaled)
prof=rfm.groupby('Cluster').agg(Customers=('CustomerID','size'),Avg_Recency=('Recency','mean'),Median_Recency=('Recency','median'),Avg_Frequency=('Frequency','mean'),Median_Frequency=('Frequency','median'),Avg_Monetary=('Monetary','mean'),Median_Monetary=('Monetary','median'),Revenue=('Monetary','sum')).reset_index()
prof['Revenue_Share']=prof.Revenue/prof.Revenue.sum()
# Behaviour-derived labels: sort engagement/value; label based on actual cluster medians and relative metrics
# rank cluster profile by recency ascending and spend/frequency descending
medr=prof.Median_Recency; medf=prof.Median_Frequency; medm=prof.Median_Monetary
# Label cluster based on rankings in the actual chosen solution; each cluster label unique
profiles=prof.set_index('Cluster')
low_r=profiles.Median_Recency.rank(method='min',ascending=True); high_f=profiles.Median_Frequency.rank(method='min',ascending=False); high_m=profiles.Median_Monetary.rank(method='min',ascending=False)
# classify by recency quartile and F/M relative to medians
labels={}
for c,row in profiles.iterrows():
 if row.Median_Recency<=profiles.Median_Recency.quantile(.25) and row.Median_Frequency>=profiles.Median_Frequency.median() and row.Median_Monetary>=profiles.Median_Monetary.median(): name='Champions'
 elif row.Median_Recency<=profiles.Median_Recency.median() and row.Median_Frequency>=profiles.Median_Frequency.median(): name='Loyal Customers'
 elif row.Median_Recency>=profiles.Median_Recency.median() and (row.Median_Frequency>=profiles.Median_Frequency.median() or row.Median_Monetary>=profiles.Median_Monetary.median()): name='At-Risk Customers'
 else: name='Lost / Low-Engagement Customers'
 labels[int(c)]=name
# disambiguate ties/duplicate segment labels, preserve behavior-based base names
for name in set(labels.values()):
 ids=[c for c,v in labels.items() if v==name]
 if len(ids)>1:
  ids=sorted(ids,key=lambda c:(profiles.loc[c,'Median_Recency'], -profiles.loc[c,'Median_Frequency'], -profiles.loc[c,'Median_Monetary']))
  for i,c in enumerate(ids):
   if i: labels[c]=name+f' (Group {i+1})'
rfm['Segment']=rfm.Cluster.map(labels)
rfm.to_csv(OUT/'RFM_Customer_Scores.csv',index=False)
rfm.to_csv(OUT/'Customer_Clusters.csv',index=False)
prof['Segment']=prof.Cluster.map(labels); prof.to_csv(OUT/'Cluster_Profile.csv',index=False)
# charts
palette=sns.color_palette('viridis',n_colors=bestk); color_map={c:palette[i] for i,c in enumerate(sorted(prof.Cluster))}
fig,ax=plt.subplots(figsize=(10,6)); order=rfm.Segment.value_counts().index; sns.countplot(data=rfm,y='Segment',order=order,palette={s:color_map[int(rfm.loc[rfm.Segment==s,'Cluster'].iloc[0])] for s in order},ax=ax); ax.set(title='Customer Distribution by Behaviour-Based Segment',xlabel='Customers',ylabel='Segment'); fig.tight_layout(); fig.savefig(OUT/'segment_distribution.png',dpi=180); plt.close(fig)
fig,ax=plt.subplots(figsize=(10,7)); sc=ax.scatter(rfm.Recency,rfm.Frequency,c=rfm.Cluster,cmap='viridis',s=24,alpha=.7); ax.set_xscale('symlog'); ax.set_yscale('symlog'); ax.set(title='RFM Customer Clusters (log-like axes)',xlabel='Recency (days; lower is more recent)',ylabel='Unique invoices (frequency)'); cb=fig.colorbar(sc,ax=ax); cb.set_label('Cluster'); fig.tight_layout(); fig.savefig(OUT/'RFM_Cluster_Visualization.png',dpi=180); plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(16,5)); metrics=['Avg_Recency','Avg_Frequency','Avg_Monetary']; titles=['Mean Recency (days)','Mean Purchase Frequency','Mean Spend'];
for ax,m,t in zip(axs,metrics,titles):
 sns.barplot(data=prof,x='Segment',y=m,ax=ax,palette={s:color_map[int(prof.loc[prof.Segment==s,'Cluster'].iloc[0])] for s in prof.Segment}); ax.set_title(t); ax.set_xlabel(''); ax.tick_params(axis='x',rotation=75)
fig.suptitle('Average RFM Metrics by Cluster'); fig.tight_layout(); fig.savefig(OUT/'rfm_comparison.png',dpi=180); plt.close(fig)
fig,ax=plt.subplots(figsize=(10,6)); q=prof.sort_values('Revenue',ascending=False); sns.barplot(data=q,x='Segment',y='Revenue',palette={s:color_map[int(prof.loc[prof.Segment==s,'Cluster'].iloc[0])] for s in q.Segment},ax=ax); ax.set(title='Revenue Contribution by Customer Segment',xlabel='Segment',ylabel='Revenue'); ax.tick_params(axis='x',rotation=25); fig.tight_layout(); fig.savefig(OUT/'revenue_by_segment.png',dpi=180); plt.close(fig)
fig,ax=plt.subplots(figsize=(8,5)); ax.plot(list(ks),inertias,marker='o',label='Inertia'); ax.set(xlabel='Number of clusters (k)',ylabel='Within-cluster sum of squares',title='K-Means Elbow Method'); ax2=ax.twinx(); ax2.plot(list(ks),silhouettes,color='darkorange',marker='s',label='Silhouette (sampled)'); ax2.set_ylabel('Silhouette score'); fig.legend(loc='upper right',bbox_to_anchor=(.9,.88)); fig.tight_layout(); fig.savefig(OUT/'elbow_method.png',dpi=180); plt.close(fig)
# machine-readable actual facts
summary={'inspection':inspection,'cleaning_counts':counts,'reference_date':str(snapshot.date()),'rfm_customers':len(rfm),'cluster_count':bestk,'cluster_selection':'highest silhouette score on deterministic sample among k=2..8; elbow chart also included','silhouette_sample_n':len(sil_idx),'elbow':{str(k):v for k,v in zip(ks,inertias)},'silhouette':{str(k):v for k,v in zip(ks,silhouettes)},'profiles':prof.to_dict('records'),'cluster_labels':{str(k):v for k,v in labels.items()}}
(OUT/'analysis_summary.json').write_text(json.dumps(summary,indent=2,default=str))
print(json.dumps(summary,indent=2,default=str))
