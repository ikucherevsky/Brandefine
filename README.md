# Brandefine
Решение задачи определения бренда на соревновании https://boosters.pro/championship/data_fusion/overview
Тренировка моделей осуществлялась в https://colab.research.google.com/
Ссылка на ноутбук https://colab.research.google.com/drive/1qsYUm9MStEXsnfR_zfovUXtXVE2n01jw?usp=sharing
В файле script.py поиск бренда происходит гибридным методом, с помощью tfidf_matcher и 2-х моделей для NER SpaCy,
наибольший вес при поиске отдается предсказаниям сделанным tfidf_matcher 'tfidf_pred', также имеется предсказание
'tfidf_alter_pred', также полученное tfidf_matcher, но с меньшим значением косинусного  подобия, чем в первом случае.
Предсказание делается по мажоритарной схеме с весами у голосующих моделей, веса указываются в функции total_pred()

