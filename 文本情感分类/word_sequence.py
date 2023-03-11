#构建词典，实现方法把句子转化位数字序列何其翻转

class Word2Sequence:
    UNK_TAG="UNK"
    PAD_TAG="PAD"

    UNK=0
    PAD=1

    def __init__(self):
        self.dict={
            self.UNK_TAG:self.UNK,
            self.PAD_TAG:self.PAD
        }
        self.count={}

    def fit(self,sentence):
        #把单个句子保存到dict中
        for word in sentence:
            self.count[word]=self.count.get(word,0)+1

    def build_vocab(self,min=5,max=None,max_features=None):
        if min is not None:
            self.count={word:value for word,value in self.count if value>min}
        if  max is not None:
            self.count = {word: value for word, value in self.count if value <max}
        #限制保留的词语数
        if max_features is not None:
            temp=sorted(self.count.items(),key=lambda x:x[-1],reverse=True)[:max_features]
            self.count=dict(temp)

        for word in self.count:
            self.dict[word]=len(self.dict)

        self.inversed_dict=dict(zip(self.dict.values(),self.dict.keys()))

    def transform(self,sentence,max_len=None):
        if max_len<len(sentence):

        if max_len>len(sentence):
            sentence=sentence+[self.PAD_TAG]*(max_len(sentence))
        return [self.dict.get(word,self.UNK) for word in sentence]

    def inverse_transform(self,indices):
        return [self.inversed_dict.get(idx) for idx in indices]




