"""Програма для знаходження відсотку схожости слів у текстах"""
# імпортуємо всі необхідні бібліотеки
import glob
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns

# Стягуємо всі файли з папки diff_texts
filenames = glob.glob("diff_texts/*.txt")
# Назвами стовпців у майбутній таблиці робимо назви файлів папки diff_texts
filekeys = [f.split("\\")[-1].split('.')[0] for f in filenames]

# Створюємо функцію, яка векьоризуватиме наші дані, знаходимо 1000 найпоширеніших слів
# Нейчастіші(по типу "і", "та" не впраховуємо)
vectorizer = CountVectorizer(input="filename", max_features=1000, max_df=0.7)
# Створюємо функцію для пошуку 1000 найпоширеніших слів
# wordcounts = vectorizer.fit_transform(filenames).toarray()
# На основі нашої таблички, що містить інформацію про наші файли,
# Створюємо наче шахову таблицю(квадратну матрицю,
# де колонка і стовпець - назви файлів з текстами, які у нашій першопочатковій таблиці
# розташовані у колонці Doc_name)
# metadata = pd.read_csv('Authors.csv', sep=";", keep_default_na=False, index_col="Doc_name")
def count_and_show(filekey:list, name_for_top_5:str, metadata, wordcounts, filename_1, filename_2, euclidean:bool=True, cosine:bool=True):
    """Створюємо функцію, котра дам відображатиме евклідову віддаль, а також косинусну для відповідних даних"""
    # прораховуємо "пряму" віддаль між текстами та внесімо ці дані до таблиці
    if euclidean:
        euclidean_distances = pd.DataFrame(squareform(pdist(wordcounts)), index=filekey, columns=filekey)
        # print(euclidean_distances)
        euclidean_distances = euclidean_distances.sort_index()
        # тут ми можемо для кожної відобразити топ 5 найближчих, для цього розкоментійте 31-32 лінійки
        top5_euclidean = euclidean_distances.nsmallest(6, name_for_top_5)[name_for_top_5][1:]
        # print(top5_euclidean)
        # print(metadata.loc[top5_euclidean.index, ['Author','Title','Data']])
        plt.figure(figsize=(15, 13))
        # Щоб змінити на назви файлів
        ticklabels = metadata['Title'].tolist()
        # Побудова теплової карти з використанням Seaborn
        heatmap_1 = sns.heatmap(euclidean_distances, annot=False, cmap='flare_r', fmt=".2f", vmin=0, vmax=160, center=80)
        # Сортування списку підписів за алфавітом або іншим критерієм
        heatmap_1.set_xticklabels(ticklabels, fontsize=8, rotation=90)
        heatmap_1.set_yticklabels(ticklabels, fontsize=8, rotation=0)
        # Додавання назви графіку
        plt.title('Залежність за Евклідовою віддалю (оберненопропорційна)')
        plt.subplots_adjust(left=0.3, bottom=0.2)
        # Показ графіка
        plt.show()
        # зберігаємо у файл
        plt.savefig(filename_1)
        print(plt.show)
    if cosine:
        # виконуємо такі ж дії, тільки вже з кутовою віддалю
        cosine_distances = pd.DataFrame(squareform(pdist(wordcounts, metric='cosine')), index=filekey, columns=filekey)
        cosine_distances = cosine_distances.sort_index()
        top5_cosine = cosine_distances.nsmallest(6, name_for_top_5)[name_for_top_5][1:]
        # print(top5_cosine)
        # print(metadata.loc[top5_cosine.index, ['Author','Title','Data']])

        plt.figure(figsize=(15, 13))
        heatmap_2 = sns.heatmap(cosine_distances, annot=False, cmap='flare', fmt=".2f", vmin=0, vmax=1, center=0.5)
        heatmap_2.set_xticklabels(ticklabels, fontsize=8, rotation=90)
        heatmap_2.set_yticklabels(ticklabels, fontsize=8, rotation=0)
        plt.title('Залежність за кутовою віддалю (прямопропорційна)')
        plt.subplots_adjust(left=0.3, bottom=0.2)
        plt.show()
        plt.savefig(filename_2)


if __name__ == "__main__":
    # ЗАГАЛЬНА СТАТИСТИКА
    filenames = glob.glob("diff_texts/*.txt")
    # Назвами стовпців у майбутній таблиці робимо назви файлів папки diff_texts
    filekeys = [f.split("\\")[-1].split('.')[0] for f in filenames]
    (metadata := pd.read_csv('Authors.csv', sep=";", keep_default_na=False, index_col="Doc_name"))
    # Створюємо функцію для пошуку 1000 найпоширеніших слів
    wordcounts = vectorizer.fit_transform(filenames).toarray()
    count_and_show(filekeys, 'file_15', metadata, wordcounts, 'heatmap_evclid_all.png', 'heatmap_cosine_all.png', True, True)


    # СТАТИСТИКА ПО ПОЕЗІЇ
    poetry_metadata = metadata[metadata['Ganra'] == 'поезія']
    poetry_filekeys = poetry_metadata.index
    poetry_filnames = ['diff_texts\\'+str(el)+'.txt' for el in poetry_filekeys]
    poetry_wordcounts = vectorizer.fit_transform(poetry_filnames).toarray()
    count_and_show(poetry_filekeys, 'file_01', poetry_metadata, poetry_wordcounts, 'heatmap_evclid_poetry.png', 'heatmap_cosine_poetry.png')


    # СТАТИСТИКА ПО ПРОЗІ
    poetry_metadata = metadata[metadata['Ganra'] == 'проза']
    poetry_filekeys = poetry_metadata.index
    poetry_filnames = ['diff_texts\\'+str(el)+'.txt' for el in poetry_filekeys]
    poetry_wordcounts = vectorizer.fit_transform(poetry_filnames).toarray()
    count_and_show(poetry_filekeys, 'file_06', poetry_metadata, poetry_wordcounts, 'heatmap_evclid_fiction.png', 'heatmap_cosine_fiction.png')
