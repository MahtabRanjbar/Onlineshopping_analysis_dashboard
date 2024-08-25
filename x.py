fig, axs = plt.subplots(2,2, figsize=(30,20))

# month vs pagevalues wrt revenue
sns.boxplot(x = 'Month', y = 'PageValues', hue ='Revenue', data=df ,palette = 'inferno',ax=axs[0,0])
axs[0,0].set_title('Mon. vs PageValues w.r.t. Rev.', fontsize = 20)

# month vs exitrates wrt revenue
sns.boxplot(x = 'Month', y = 'ExitRates', hue ='Revenue', data=df ,palette = 'Reds',ax=axs[0,1])
axs[0,1].set_title('Mon. vs ExitRates w.r.t. Rev.', fontsize = 20)

# month vs bouncerates wrt revenue
sns.boxplot(x = 'Month', y = 'BounceRates', hue ='Revenue', data=df ,palette = 'Oranges',ax=axs[1,0])
axs[1,0].set_title('Mon. vs BounceRates w.r.t. Rev.', fontsize = 20)
axs[1,0].legend(loc='upper left',fancybox=True, shadow=True)

# visitor type vs exit rates w.r.t revenue
sns.boxplot(x = 'VisitorType', y = 'BounceRates', hue ='Revenue', data=df, palette = 'Purples',ax=axs[1,1])
axs[1,1].set_title('Visitors vs BounceRates w.r.t. Rev.', fontsize = 20)

plt.tight_layout()


fig, axs = plt.subplots(2,2, figsize=(30,20))

# month vs pagevalues wrt revenue
sns.barplot(x = 'VisitorType', y = 'ExitRates', hue ='Revenue', data=df ,palette = 'inferno',ax=axs[0,0])
axs[0,0].set_title('Mon. vs PageValues w.r.t. Rev.', fontsize = 20)

# month vs exitrates wrt revenue
sns.boxplot(x = 'VisitorType', y = 'PageValues', hue ='Revenue', data=df ,palette = 'Reds',ax=axs[0,1])
axs[0,1].set_title('Mon. vs ExitRates w.r.t. Rev.', fontsize = 20)

# month vs bouncerates wrt revenue
sns.boxplot(x = 'Region', y = 'PageValues', hue ='Revenue', data=df ,palette = 'Oranges',ax=axs[1,0])
axs[1,0].set_title('Mon. vs BounceRates w.r.t. Rev.', fontsize = 20)
axs[1,0].legend(loc='upper left',fancybox=True, shadow=True)

# visitor type vs exit rates w.r.t revenue
sns.boxplot(x = 'Region', y = 'ExitRates', hue ='Revenue', data=df, palette = 'Purples',ax=axs[1,1])
axs[1,1].set_title('Visitors vs BounceRates w.r.t. Rev.', fontsize = 20)

plt.tight_layout()

matrix = np.triu(df.corr())
fig, ax = plt.subplots(figsize=(12,12))
sns.heatmap(df.corr(), annot=True, ax=ax, fmt='.1g', vmin=-1, vmax=1, center= 0, mask=matrix, cmap='RdBu_r')
plt.show()

g1 = sns.pairplot(df[['Administrative', 'Informational', 'ProductRelated', 'PageValues', 'Revenue']], hue='Revenue')
g1.set_title('Feature Relations')
plt.show()

fig, axs = plt.subplots(3,1, figsize=(10,10))

sns.countplot(x='Informational', color = "crimson",data=df, ax=axs[0]);
axs[0].set_title("Number of Informational pages visited by user", fontsize = 15);
axs[0].set_xlabel('No of pages', fontsize = 10);
axs[0].set_ylabel('count', fontsize = 10);


sns.countplot(x='Administrative', color = "crimson",data=df, ax=axs[1]);
axs[1].set_title("Number of Adminstrative pages visited by user", fontsize = 15);
axs[1].set_xlabel('No of pages', fontsize = 10);
axs[1].set_ylabel('count', fontsize = 10);




plt.tight_layout()

plt.figure(figsize = (12,4))
sns.histplot(x='ProductRelated',color ='crimson',data=df,bins=100,stat='probability')
plt.show()

fig, axs = plt.subplots(1,1, figsize=(7,5))

sns.countplot(x='SpecialDay', palette='inferno',data=df, ax=axs);
axs.set_title("Closeness to special day", fontsize = 15);
axs.set_xlabel('rate', fontsize = 10);
axs.set_ylabel('count', fontsize = 10);