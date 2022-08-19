import csv
import nltk
import re
import string
import math
from collections import Counter
inpTweets = csv.reader(open('dataset_v1.csv', 'rb'),  delimiter=',', quotechar='"', escapechar='\\')
inpTweet = csv.reader(open('dataset_v1.csv', 'rb'),  delimiter=',', quotechar='"', escapechar='\\')

    
def processTweet(tweet):
    tweet = tweet.lower()
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',tweet)
    tweet = re.sub('@[^\s]+','',tweet)
    tweet = re.sub('[\s]+', ' ', tweet)
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    tweet= tweet.translate(string.maketrans("",""), string.punctuation)    
    tweet = tweet.strip('\'"')
    return tweet
    
def getStopWordList(stopWordListFileName):
    stopWords = []
    stopWords.append('rt')
    stopWords.append('https')
    fp = open(stopWordListFileName, 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = fp.readline()
    fp.close()
    return stopWords
sw = getStopWordList('stopwords.txt')

def choose_attribute(data,attributes,target_attr):
	gain_ = 0
	best =""
	for i in attributes:
		temp_gain = gain(data,i,target_attr)
		if(temp_gain > gain_):
			gain_,best = temp_gain,i
	attributes.remove(best)
	return best


def get_examples(data, best, i):
    exam = []
    if i == 0:
        for record in data:
            if best in record:
                exam.append(record)
    elif i == 1:
        for record in data:
            if best not in record:
                exam.append(record)
    return exam

def majority_value(data, target_attr):
    l = [i[target_attr] for i in data]
    num = Counter(l)
    cls = num.most_common(1)[0][0]
    return cls

def create_decision_tree(data, attributes, target_attr):
    """
    Returns a new decision tree based on the examples given.
    """
    data    = data[:]
    vals    = [record[target_attr] for record in data]
    default = majority_value(data, target_attr)

    # If the dataset is empty or the attributes list is empty, return the
    # default value. When checking the attributes list for emptiness, we
    # need to subtract 1 to account for the target attribute.
    if not data or (len(attributes) - 1) <= 0:
        return default
    # If all the records in the dataset have the same classification,
    # return that classification.
    elif vals.count(vals[0]) == len(vals):
        return vals[0]
    else:
        # Choose the next best attribute to best classify our data
        best = choose_attribute(data, attributes, target_attr)
        
        # Create a new decision tree/node with the best attribute and an empty
        # dictionary object--we'll fill that up next.
        tree = {best:{}}

        # Create a new decision tree/sub-node for each of the values in the
        # best attribute field
        for val in get_values(data, best):
            # Create a subtree for the current value under the "best" field
            subtree = create_decision_tree(
                get_examples(data, best, val),
                [attr for attr in attributes if attr != best],
                target_attr)

            # Add the new subtree to the empty dictionary object in our new
            # tree/node we just created.
            tree[best][val] = subtree

    return tree
def get_values(data, best):
	return [0,1]
	    
def entropy(data, target_attr):
    """
    Calculates the entropy of the given data set for the target attribute.
    """
    val_freq     = {}
    data_entropy = 0.0

    # Calculate the frequency of each of the values in the target attr
    for record in data:
        if (val_freq.has_key(record[target_attr])):
            val_freq[record[target_attr]] += 1.0
        else:
            val_freq[record[target_attr]]  = 1.0

    # Calculate the entropy of the data for the target attribute
    for freq in val_freq.values():
        data_entropy += (-freq/len(data)) * math.log(freq/len(data), 2) 
        
    return data_entropy
    
def gain(data, attr, target_attr):
    """
    Calculates the information gain (reduction in entropy) that would
    result by splitting the data on the chosen attribute (attr).
    """
    val_freq       = {1:0,0:0}
    subset_entropy = 0.0
    for record in data:
    	if(attr in record):
    		val_freq[1]+=1

    	else:
    		val_freq[0]+=1
    # Calculate the frequency of each of the values in the target attribute
    '''for record in data:
        if (val_freq.has_key(record[attr])):
            val_freq[record[attr]] += 1.0
        else:
            val_freq[record[attr]]  = 1.0'''

    # Calculate the sum of the entropy for each subset of records weighted
    # by their probability of occuring in the training set.
    for val in val_freq.keys():
        val_prob        = val_freq[val] / sum(val_freq.values())
        data_subset =[]
        if(val == 1):
        	for record in data:
        		if(attr in record):
        			data_subset.append(record)
        elif(val == 0):
        	for record in data:
        		if(attr not in record):
        			data_subset.append(record)
        subset_entropy += val_prob * entropy(data_subset, target_attr)

    # Subtract the entropy of the chosen attribute from the entropy of the
    # whole data set with respect to the target attribute (and return it)
    return (entropy(data, target_attr) - subset_entropy)
c=0
d=[]
se=[] 
data=[] 
attributes=[] 
for row in inpTweets:
    s=row[1]
    t=row[0]
    c+=1
    s=int(s)
    if c>1200:
        break
    else:
        pt = processTweet(t)
        st=nltk.tokenize.wordpunct_tokenize(pt)
        for i in st[:]:
            if i in sw or len(i)<3:
                st.remove(i)
        attributes+=st
        st.append("sent")
        d.append(st)
        se.append(s)


def word_feats(words):
    return dict([(word, 1) for word in words])
    
for i in range(len(se)):
    di=word_feats(d[i])
    di['sent']=se[i]
    data.append(di)
    
tr = create_decision_tree(data,attributes,'sent') 


classes = []

def classify(root, tokens):

	if isinstance(root, int):
		classes.append(root)
		return

	keys = list(root.keys())
	key = keys[0]
	if key in tokens:
		d = root[key][0]
		classify(d, tokens)

	else:
		d = root[key][1]
		classify(d, tokens)

L = []
#Testing Phase
c=0
c_test=0
c_ini=0
for row in inpTweet:
    c_ini+=1
    if c_ini>1200:
	k=1			#represents posterior prob for class1
	l=1			#represents posterior prob for class2
	m=1			#represents posterior prob for class3
	cls = int(row[1])
	t = row[0]
	t1 = processTweet(t)
	twt = nltk.tokenize.wordpunct_tokenize(t1)
        for i in twt[:]:
            if i in sw:
                twt.remove(i)	
        s=row[1]
        s=int(s)
        classify(tr,twt)
        L.append(s)
count = 0
for i in range(len(L)):
	if L[i] == classes[i]:
		count += 1

acc=float(count)/len(classes)
acc*=100
print "Accuracy : "+str(acc)
            