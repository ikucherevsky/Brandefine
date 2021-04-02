# Brandefine
Решение задачи определения бренда на соревновании https://boosters.pro/championship/data_fusion/overview
Тренировка моделей осуществлялась в https://colab.research.google.com/
Ссылка на ноутбук https://colab.research.google.com/drive/1qsYUm9MStEXsnfR_zfovUXtXVE2n01jw?usp=sharing
В файле script.py поиск бренда происходит гибридным методом, с помощью tfidf_matcher и 2-х моделей для NER SpaCy,
наибольший вес при поиске отдается предсказаниям сделанным tfidf_matcher 'tfidf_pred', также имеется предсказание
'tfidf_alter_pred', также полученное tfidf_matcher, но с меньшим значением косинусного  подобия, чем в первом случае.
Предсказание делается по мажоритарной схеме с весами у голосующих моделей, веса указываются в функции total_pred()

В папках pymorphy2, pymorphy2_dicts_ru, pymorphy2_dicts_ru-2.4.417127.4579844.dist-info, pymorphy2-0.9.1.dist-info,
DAWG_Python-0.7.2.dist-info, dawg_python находится библиотека pymorphy2 и необходимая для pymorphy2 DAWG. Pymorphy2 
необходима для работы русской языковой модели spaCy.


