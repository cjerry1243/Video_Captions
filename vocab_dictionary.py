import json
from keras.preprocessing import text


def Vocabs_Tokenizer(vocab_num):
    tokenizer = text.Tokenizer(num_words=vocab_num)
    f = open('MLDS_hw2_data/training_label.json','r')
    train_json = json.load(f)
    f.close()
    allcaptions = []
    for i in range(len(train_json)):
        captions = train_json[i]['caption']

        for sentence in captions:
            allcaptions.append('BOS '+ sentence + ' EOS')

    ###  Delete the caption:
    ###  "id": "_sJ_09Mf1HY_49_72.avi"
    ###  "\u090f\u0915 \u0932\u0921\u093c\u0915\u093e \u092b\u0941\u091f\u092c\u094b\u0932 \u092a\u0948\u0930 \u0938\u0947 \u0932\u094b\u0917\u094b\u0915\u0947 \u092c\u0940\u091a\u092e\u0947 \u0916\u0947\u0932\u093e \u0930\u0939\u093e \u0939\u0948.",
    tokenizer.fit_on_texts(allcaptions)
    # print(allcaptions)
    reverse_tokenizer = dict(map(reversed, tokenizer.word_index.items()))
    return tokenizer, reverse_tokenizer

# tokenizer, reverse= Vocabs_Tokenizer(10)
# print(tokenizer.word_index)