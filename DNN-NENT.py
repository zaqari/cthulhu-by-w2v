import tensorflow as tf
import gensim
import nltk
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np
import codecs

####################
#################DATA PRE-PROCESSING & FUNCTIONS
####################

lem=WordNetLemmatizer()
corpus_path='./Cthulhu.txt'
vec_dims=100
model=gensim.models.Word2Vec.load('./_models/kbase.bin')
comparison_nns=pd.read_csv('./?-csvs/nn.csv', skipinitialspace=True)['nn'].values.tolist()



class doc:
        #######CONSTRUCT TOKENIZED DOCUMENTS######
        #These functions are here in order to create tokenized data.
        # This data can then be used to build a w2v model useful for
        # research purposes, as exemplified in the class knowledge.

        def tokenize(corpus):
                doc = codecs.open(corpus, 'r', 'utf-8')
                searchlines = doc.readlines()
                doc.close()
                
                data=[]

                for i, line in enumerate(searchlines):
                        readable = line.replace('\\', '').replace('}', '').replace('uc0u8232', '').replace('\'92', '\'').replace('a0', '').replace('\'93', '\"').replace('\'94', '\"').replace('\'96', ',').replace('\'97', ',').replace('f0fs24 ', '').replace('cf0 ', '').replace('< ', '').replace(' >', '').replace('\r\n', '').replace('Mr.', 'mr').replace('Ms.', 'ms').replace('Mrs.', 'mrs').replace('Dr.', 'dr').replace('mr.', 'mr').replace('ms.', 'ms').replace('mrs.', 'ms').replace('dr.', 'dr')
                        tok_words=nltk.word_tokenize(line)
                        pos=nltk.pos_tag(tok_words)
                        for word in pos:
                                data.append(word)

                return data


        def convert_to_w2v(listin, w2vmode=model):
                data=[]

                for word in listin:
                        try:
                                data.append(w2vmode.wv[lem.lemmatize(str(word[0]))])
                        except KeyError:
                                data.append(np.array([0.0 for k in range(vec_dims)]))

                return data


        def w2vFromPOR(word_list_in, nominal_comparison, w2v_model):
                data=[]
                err=0

                cols=[str(k) for k in range(vec_dims)]

                POR_array=np.array([0.0 for k in range(vec_dims)])
                POR=[]
                for term in nominal_comparison:
                	try:
                		POR.append(w2v_model.wv[lem.lemmatize(str(term))])
                	except KeyError:
                		pass
                for vec in POR:
                        POR_array=np.add(POR_array, vec)
                POR_avg=POR_array/len(POR)

                for word in word_list_in:
                        word_vec=list(POR_avg-word)
                        data.append(word_vec)
        
                df_out=pd.DataFrame(np.array(data).reshape(-1, len(cols)), columns=cols)
                df_out['Labels']=3
                return df_out


####################
#################DNN SET-UP
####################

word_list=doc.tokenize(corpus_path)
w2vs=doc.convert_to_w2v(word_list)
df_pred=doc.w2vFromPOR(w2vs, comparison_nns, model)


#####
##IMPORTS
#####
train_data='./test_data-nent.csv' #Original file, train_datav2.csv is too big to upload. Contact me for the file.
test_data='./test_datav-nent.csv'
features=[str(k) for k in range(vec_dims)]
DNN_COLUMNS=list(features)+['Labels']

df_train = pd.read_csv(train_data, names=DNN_COLUMNS, skipinitialspace=True)
df_test = pd.read_csv(test_data, names=DNN_COLUMNS, skipinitialspace=True)

#####
##VARIABLES PART II
#####
drop_out_x=.4
num_tr_steps=len(df_train)*2
hd_units=[100]
model_dir=[
        './_models/nent',
        ]
early_stop=500
nClasses=len(set(df_train['Labels'].values.tolist()))


#####
##INPUT FN
#####
CONTINUOUS_COLUMNS=list(df_train)[:-1]
LABELS_COLUMN = ['Labels']
def input_fn(df):
        # Creates a dictionary mapping from each continuous feature column name (k) to
        # the values of that column stored in a constant Tensor.
        continuous_cols = {k: tf.constant(df[k].values)
                                  for k in CONTINUOUS_COLUMNS}
        # Creates a dictionary mapping from each categorical feature column name (k)
        # to the values of that column stored in a tf.SparseTensor.
        #categorical_cols = {k: tf.SparseTensor(
                #indices=[[i, 0] for i in range(df[k].size)],
                #values=df[k].values,
                #dense_shape=[df[k].size, 1])
                        #for k in CATEGORICAL_COLUMNS}
        # Merges the two dictionaries into one.
        feature_cols = dict(continuous_cols.items())
        # Converts the label column into a constant Tensor.
        label = tf.constant(df['Labels'].values.astype(int))
        # Returns the feature columns and the label.
        return feature_cols, label

def train_input_fn():
        return input_fn(df_train)

def eval_input_fn():
        return input_fn(df_test)

def pred_input_fn():
        return input_fn(df_pred)


#####
##FEATURE GENERATION
#####
def embeddings_in(df):
        embeds=[]
        for it in list(df)[:-1]:
                embeds.append(tf.contrib.layers.real_valued_column(it, dimension=vec_dims))
        return embeds


#####
##MODEL SPECS
#####
deep_columns=embeddings_in(df_train)
wide_columns=[]

validation_metrics = {
        #The below is the best bet to run accuracy in here, but we need to
        # somehow run labels as a full-blown tensor of some sort.
        'accuracy': tf.contrib.metrics.streaming_accuracy,
        'precision': tf.contrib.metrics.streaming_precision,
        'recall': tf.contrib.metrics.streaming_recall
        }

validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
        #df_test[feature_columns].values,
        #df_test['Labels'].values,
        input_fn=eval_input_fn,
        every_n_steps=len(df_train),
        metrics=validation_metrics,
        early_stopping_rounds=early_stop,
        early_stopping_metric='loss',
        early_stopping_metric_minimize=False
        )

pred_m = tf.contrib.learn.DNNLinearCombinedClassifier(
        model_dir=model_dir[0],
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hd_units,
        n_classes=nClasses,
        #config=tf.contrib.learn.RunConfig(save_checkpoints_secs=10),
        fix_global_step_increment_bug=True
        )


class DNN:

        def predict():
                predictions = pred_m.predict_classes(input_fn=pred_input_fn)
                return predictions
        
        def convert(classes, wordlist, document):
                doc = codecs.open(document, 'r', 'utf-8')
                searchlines = doc.readlines()
                doc.close()

                text=''

                for i, line in enumerate(searchlines):
                        text+=line
                
                convert_dic=list(zip(wordlist, classes))

                for entry in convert_dic:
                        if entry[1]==1:
                                text.replace(entry[0], 'Apple')
                        if entry[1]==2:
                                text.replace(entry[0], 'Banana')

                return text

        def train_model(dropout,  train_steps=num_tr_steps, calc_results=True, resultsteps=10):
                f_m = tf.contrib.learn.DNNLinearCombinedClassifier(
                        model_dir=model_dir[0],
                        linear_feature_columns=wide_columns,
                        dnn_feature_columns=deep_columns,
                        dnn_hidden_units=hd_units,
                        n_classes=nClasses,
                        dnn_dropout=dropout,
                        config=tf.contrib.learn.RunConfig(save_checkpoints_secs=40),
                        fix_global_step_increment_bug=True)
                f_m.fit(input_fn=train_input_fn, steps=train_steps)
                if calc_results==True:
                        results = pred_m.evaluate(input_fn=eval_input_fn, steps=resultsteps)
                        return results


#####
##IMPLEMENTATION
#####
pred=DNN.predict()
new_text=DNN.convert(pred, word_list, corpus_path)

doc=open('./Cthulhu-Protagonist-Antagonist.txt', 'w')
doc.write(new_text)
doc.close()
