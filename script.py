import pandas as pd 
import numpy as np
import spacy
import pickle
import re
import pymorphy2
print('pymorphy2 succesfully installed')
import tfidf_matcher as tm

with open('brands.pkl', 'rb') as input:
    all_brands = pickle.load(input)
print('Brands are loaded ... ')


df = pd.read_parquet('data/task2_test_for_user.parquet')
df.fillna('UNKNOWN',inplace = True)

def TFIDF_predict_brand(DF,k_matches = 5, ngram_length = 3, chunk_size = 1000):
  pred_brands = []
  alter_pred = []

  brands = all_brands
  brands = set(brands)
  if None in brands:
    brands.remove(None)

  items = DF.item_name.values

  num_chunks = len(items) // chunk_size

  for i in range(num_chunks + 1):

    items_chunk = items[i * chunk_size : (i + 1) * chunk_size]
    if len(items_chunk) < 1:
      break

    data = tm.matcher(original=items_chunk,lookup=brands,k_matches=k_matches, ngram_length=ngram_length)

    cols = [col for col in data.columns if 'Lookup' in col]
    for j in range(data.shape[0]):

      applicants = data[cols].iloc[j].to_list()
      original_name = data['Original Name'].iloc[j].lower()
      pred = [pr for pr in applicants if (pr in original_name) or (re.sub(' ','',pr) in original_name)]
    
      if pred:

        prediction = max(pred, key=len)
        pred_brands.append(prediction)
        try:
          pred.remove(prediction)
          alter_pred.append(max(pred, key=len))

        except:
          alter_pred.append('UNKNOWN')
      else:
        pred_brands.append('UNKNOWN')
        alter_pred.append('UNKNOWN')
  return pred_brands, alter_pred

brands_with_hyphen = []

for b in all_brands:
    if '-' in b:
        brands_with_hyphen.append((re.sub('-',' ',b),b))
print('brands_with_hyphen created....')

# ------------------FUNCTIONS---------------------------
def is_a_in_x(A, X):
  for i in range(len(X) - len(A) + 1):
    if A == X[i:i+len(A)]: return True
  return False

def filter_item(txt):
    
    if pd.isna(txt):
        return 'UNKNOWN'

    string = re.sub('ё','е',txt)
    string = re.sub('Ё','Е',string)
    patterns = r"[!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"
    res = re.sub(patterns, ' ', string)
    result_list = res.split(' ')

    for i in range(len(result_list)):
        if not result_list[i].isupper() and not result_list[i].isdigit():
            
            result_list[i] = ' '.join(re.split(r'(\d+)', result_list[i]))
            result_list[i] = ' '.join(re.findall(r'[A-ZА-Яa-zа-я\d+][^A-ZА-Я]*', result_list[i]))
        if any(char.isdigit() for char in result_list[i]):
            result_list[i] = ' '.join(re.findall(r'[A-ZА-Яa-zа-я\d+][^A-ZА-Я]*', result_list[i]))
    res = ' '.join(result_list)
    res = re.sub('- ', '-', res)
    res = re.sub(' -', '-', res)
    res = re.sub(' +',' ',res)
    res = res.lower()

    if brands_with_hyphen:
        for br in brands_with_hyphen:
            if br[0] in res:
                if is_a_in_x(br[0].split(' '), res.split(' ')):
                    res = re.sub(br[0],br[1], res) 
    
    return  res

def predict_brand(nlp,CLEAR_TEXTS):
    pred = []
    for doc in nlp.pipe(CLEAR_TEXTS):
        brand = [ent.text for ent in doc.ents if ent.label_ == 'BRAND']
        try:

            pred.append(brand[0])
        except:
            pred.append('UNKNOWN')
    return pred

def total_pred(data, weights):
    final_pred = []
    weights = np.array(weights)
    pred_cols = ['tfidf_pred','tfidf_alter_pred','model1_pred','model2_pred']
    for i in range(data.shape[0]):
        
        predictions = data[pred_cols].iloc[i,:].to_list()
        set_word = set(predictions)
        try:
            
            set_word.remove('UNKNOWN')
        except:
            pass
        result = {}
        
        for w in set_word:
            indices = [i for i, x in enumerate(predictions) if x == w]
            result[w] = sum(weights[[indices]])
        try:
            final = max(result, key=result.get)
            final_pred.append(final)
        except:
            final_pred.append('UNKNOWN')
    
    return final_pred
 
        


df['tfidf_pred'],df['tfidf_alter_pred'] = TFIDF_predict_brand(df)
print('TFIDF is done!')

#----------------PREDICTION--------------------
# prediction by spacy NER model
# load the models
nlp1 = spacy.load('models/NER_WITH_PRODUCT_450000_1st_part')
nlp2 = spacy.load('models/NER_WITH_PRODUCT_450000_2nd_part')
print('Models are loaded...')
# text clearing
df['clear_text'] = df.item_name.apply(filter_item)
CLEAR_TEXTS = df.clear_text.to_list()
print('item_name is cleared and loaded to dataframe')
# prediction by model
pred = predict_brand(nlp1,CLEAR_TEXTS)
df['model1_pred'] = pred
df['model1_pred'].fillna('UNKNOWN', inplace = True)
print('Model 1 done ...')
pred = predict_brand(nlp2,CLEAR_TEXTS)
df['model2_pred'] = pred
df['model2_pred'].fillna('UNKNOWN', inplace = True)
print('Model 2 done ...')

df['pred'] = total_pred(df,[1.5,1,1,1])
df[['id', 'pred']].to_csv('answers.csv', index=None)

#print(1 - df[df.brands != df.pred].shape[0] / df.shape[0])