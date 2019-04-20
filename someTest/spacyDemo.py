import spacy
nlp = spacy.load('en')

doc = nlp(u'Apple is looking at buying U.K. startup for $1 billion')
# 词性标注
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
          token.shape_, token.is_alpha, token.is_stop)

# 依存分析
doc = nlp(u'Autonomous cars shift insurance liability toward manufacturers')
for token in doc:
    print(token.text, token.dep_, token.head.text, token.head.pos_,
          [child for child in token.children])
# 实体识别
doc = nlp(u'Apple is looking at buying U.K. startup for $1 billion')
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)

# 标签化
for token in doc:
    print(token.text)