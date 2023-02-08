from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt



if __name__ == '__main__':
    firstdata = pd.read_csv('dataset.csv')
    data = firstdata.sample(8000,random_state=2)
    X = MinMaxScaler().fit_transform(data[['cwc_min', 'cwc_max', 'csc_min', 'csc_max' , 'ctc_min' , 'ctc_max' , 'last_word_eq', 'first_word_eq' , 'abs_len_diff' , 'mean_len' , 'token_set_ratio' , 'token_sort_ratio' ,  'fuzz_ratio' , 'fuzz_partial_ratio' , 'longest_substr_ratio']])
    y = data['is_duplicate'].values  
    tsne2d = TSNE(
        n_components=2,
        init='random', # pca
        random_state=101,
        method='barnes_hut',
        n_iter=1000,
        verbose=2,
        angle=0.5
    ).fit_transform(X)
    x_df = pd.DataFrame({'x':tsne2d[:,0], 'y':tsne2d[:,1] ,'label':y})

    # draw the plot in appropriate place in the grid
    sns.lmplot(data=x_df, x='x', y='y', hue='label', fit_reg=False, palette="Set1",markers=['s','o'])
    # plt.savefig('3d.pdf')
    plt.show()