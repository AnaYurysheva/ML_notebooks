# ML_notebooks
Учебные проекты: 

* data_preprocessing: примеры обработки даты (pandas, numpy, matplotlin)
 - извлечение данных
 - создание новых признаков (при необходимости)
 - визуализация

* lineral: настроенные линейная и логистическая регрессии (sklearn)
 - прогноз стоимости машины (car_prediction), кастомная линейная регрессия (на основе матричных вычислений + среднеквадратичная ошибка отклонения)
 - прогноз стоимости квартиры в Бруклине(host_price_prediction), линейная регрессия + rmse
 - Титаник, логистическая регрессия + accuracy
* decision_tree: проект на основе дерева решений и леса решений, XGBoost + scklearn
 - прогноз кредитных рисков (credit_risk_scoring), сравнение моделей descision tree | random forest | XGBoost + roc auc curve
  - прогноз стоимости квартиры в Бруклине (Homework_trees), сравнение моделей descision tree | random forest | XGBoost + rmse
* model_deploy: проект с деплоем линейной модели на локальный сервер ( Docker, flask, pipfile)
* text_classification: анализ описанияч объявлений Авито, предсказание категории, сравнение SGDClassifier и LSTM (keras), предобработка текста пакетом nltk (удаление пунктуации, стоп слов, разделение на токены и стемминг)

